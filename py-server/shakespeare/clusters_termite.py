import logging, sys, math, time
from operator import itemgetter

# From compute_saliency.py

class Model(object):
    def __init__(self):
        self.topic_index = []
        self.term_index = []
        self.topic_count = 0
        self.term_count = 0
        self.term_topic_matrix = []
    def load(self, lda, lda_ctxt):
        self.term_topic_matrix = lda.state.sstats.T 
        self.topic_count = lda.num_topics
        self.topic_index = map(lambda n: 'Topic %d' % (n+1), range(self.topic_count))
        self.term_index = lda_ctxt.get_terms()
        self.term_count = len(self.term_index)
        
class Saliency(object):
    def __init__(self):
        self.term_info = {}
        self.topic_info = {}

class Client(object):
    def __init__(self):
        self.seriated_parameters = {}
        self.filtered_parameters = {}
        self.global_term_freqs = {}

class Seriation(object):
    def __init__(self):
        self.term_ordering = []
        self.term_iter_index = []

class Similarity(object):
    def __init__(self):
        self.document_occurrence = {}
        self.document_cooccurrence = {}
        self.window_occurrence = {}
        self.window_cooccurrence = {}
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.document_g2 = {}
        self.window_g2 = {}
        self.collcation_g2 = {}
        self.combined_g2 = {}

class ComputeSaliency(object):
    """
    Distinctiveness and saliency.
    
    Compute term distinctiveness and term saliency, based on
    the term probability distributions associated with a set of
    latent topics.
    
    Input is term-topic probability distribution, stored in 3 separate files:
        'term-topic-matrix.txt' contains the entries of the matrix.
        'term-index.txt' contains the terms corresponding to the rows of the matrix.
        'topic-index.txt' contains the topic labels corresponding to the columns of the matrix.
    
    Output is a list of term distinctiveness and saliency values,
    in two duplicate formats, a tab-delimited file and a JSON object:
        'term-info.txt'
        'term-info.json'
    
    An auxiliary output is a list topic weights (i.e., the number of
    tokens in the corpus assigned to each latent topic) in two
    duplicate formats, a tab-delimited file and a JSON object:
        'topic-info.txt'
        'topic-info.json'
    """
    
    def __init__(self, logging_level=logging.DEBUG):
        self.logger = logging.getLogger( 'ComputeSaliency' )
        self.logger.setLevel( logging_level )
        handler = logging.StreamHandler( sys.stderr )
        handler.setLevel( logging_level )
        self.logger.addHandler( handler )
    
    def execute(self, lda, lda_ctxt):
        
        #assert data_path is not None
        
        self.logger.info( '--------------------------------------------------------------------------------' )
        self.logger.info( 'Computing term saliency...'                                                       )
        #self.logger.info( '    data_path = %s', data_path                                                    )
        
        self.logger.info( 'Connecting to data...' )
        self.model = Model()
        self.model.load(lda, lda_ctxt)
        
        self.saliency = Saliency()
        
        self.logger.info( 'Reading data from disk...' )
        #self.model.read()
        
        self.logger.info( 'Computing...' )
        self.computeTopicInfo()
        self.computeTermInfo()
        self.rankResults()
        
        self.logger.info( 'Writing data to disk...' )
        #self.saliency.write()
        
        self.logger.info( '--------------------------------------------------------------------------------' )
    
    def computeTopicInfo(self):
        topic_weights = [ sum(x) for x in zip( *self.model.term_topic_matrix ) ]
        topic_info = []
        for i in range(self.model.topic_count):
            topic_info.append( {
                'topic' : self.model.topic_index[i],
                'weight' : topic_weights[i]
            } )
        
        self.saliency.topic_info = topic_info
    
    def computeTermInfo(self):
        """Iterate over the list of terms. Compute frequency, distinctiveness, saliency."""
        
        topic_marginal = self.getNormalized( [ d['weight'] for d in self.saliency.topic_info ] )
        term_info = []
        for i in range(self.model.term_count):
            term = self.model.term_index[i]
            counts = self.model.term_topic_matrix[i]
            frequency = sum( counts )
            probs = self.getNormalized( counts )
            distinctiveness = self.getKLDivergence( probs, topic_marginal )
            saliency = frequency * distinctiveness
            term_info.append( {
                'term' : term,
                'saliency' : saliency,
                'frequency' : frequency,
                'distinctiveness' : distinctiveness,
                'rank' : None,
                'visibility' : 'default'
            } )
        self.saliency.term_info = term_info
    
    def getNormalized(self, counts):
        """Rescale a list of counts, so they represent a proper probability distribution."""
        tally = sum( counts )
        if tally == 0:
            probs = [ d for d in counts ]
        else:
            probs = [ d / tally for d in counts ]
        return probs
    
    def getKLDivergence(self, P, Q):
        """Compute KL-divergence from P to Q"""
        divergence = 0
        #print 'len(P):', len(P), 'len(Q):', len(Q)
        
        assert len(P) == len(Q)
        for i in range(len(P)):
            p = P[i]
            q = Q[i]
            assert p >= 0
            assert q >= 0
            if p > 0:
                divergence += p * math.log( p / q )
        return divergence
    
    def rankResults(self):
        """Sort topics by decreasing weight. Sort term frequencies by decreasing saliency."""
        self.saliency.topic_info = sorted( self.saliency.topic_info, key = lambda topic_weight : -topic_weight['weight'] )
        self.saliency.term_info = sorted( self.saliency.term_info, key = lambda term_freq : -term_freq['saliency'] )
        for i, element in enumerate( self.saliency.term_info ):
            element['rank'] = i

class ComputeSeriation(object):
    """Seriation algorithm.

    Re-order words to improve promote the legibility of multi-word
    phrases and reveal the clustering of related terms.
    
    As output, the algorithm produces a list of seriated terms and its 'ranking'
    (i.e., the iteration in which a term was seriated).
    """
    
    DEFAULT_NUM_SERIATED_TERMS = 100
    
    def __init__(self, logging_level=logging.DEBUG):
        self.logger = logging.getLogger( 'ComputeSeriation' )
        self.logger.setLevel( logging_level )
        handler = logging.StreamHandler( sys.stderr )
        handler.setLevel( logging_level )
        self.logger.addHandler( handler )
    
    def execute(self, saliency, similarity, numSeriatedTerms=None):
        
        if numSeriatedTerms is None:
            numSeriatedTerms = ComputeSeriation.DEFAULT_NUM_SERIATED_TERMS
        
        self.logger.info( '--------------------------------------------------------------------------------' )
        self.logger.info( 'Computing term seriation...'                                                      )
        #self.logger.info( '    data_path = %s', data_path                                                    )
#         self.logger.info( '    number_of_seriated_terms = %d', numSeriatedTerms                              )
        
        self.logger.info( 'Connecting to data...' )
        self.saliency = saliency
        self.similarity = similarity

        self.seriation = Seriation()
        
#         self.logger.info( 'Reading data from disk...' )
#         self.saliency.read()
#         self.similarity.read()
        
        self.logger.info( 'Reshaping saliency data...' )
        self.reshape()
        
        self.logger.info( 'Computing seriation...' )
        self.compute( numSeriatedTerms )
        
        self.logger.info( 'Writing data to disk...' )
        #self.seriation.write()
        
        self.logger.info( '--------------------------------------------------------------------------------' )
    
    def reshape( self ):
        self.candidateSize = 100
        self.orderedTermList = []
        self.termSaliency = {}
        self.termFreqs = {}
        self.termDistinct = {}
        self.termRank = {}
        self.termVisibility = {}
        for element in self.saliency.term_info:
            term = element['term']
            self.orderedTermList.append( term )
            self.termSaliency[term] = element['saliency']
            self.termFreqs[term] = element['frequency']
            self.termDistinct[term] = element['distinctiveness']
            self.termRank[term] = element['rank']
            self.termVisibility[term] = element['visibility']
    
    def compute( self, numSeriatedTerms ):
        # Elicit from user (1) the number of terms to output and (2) a list of terms that should be included in the output...
        # set in init (i.e. read from config file)
        
        # Seriate!
        start_time = time.time()
        candidateTerms = self.orderedTermList
        self.seriation.term_ordering = []
        self.seriation.term_iter_index = []
        self.buffers = [0,0]
        
        preBest = []
        postBest = []
        
        for iteration in range(numSeriatedTerms):
            print "Iteration no. ", iteration
            
            addedTerm = 0
            if len(self.seriation.term_iter_index) > 0:
                addedTerm = self.seriation.term_iter_index[-1]
            if iteration == 1:
                (preBest, postBest) = self.initBestEnergies(addedTerm, candidateTerms)
            (preBest, postBest, self.bestEnergies) = self.getBestEnergies(preBest, postBest, addedTerm)
            (candidateTerms, self.seriation.term_ordering, self.seriation.term_iter_index, self.buffers) = self.iterate_eff(candidateTerms, self.seriation.term_ordering, self.seriation.term_iter_index, self.buffers, self.bestEnergies, iteration)
            
            print "---------------"
        seriation_time = time.time() - start_time
        
        # Output consists of (1) a list of ordered terms, and (2) the iteration index in which a term was ordered
        #print "term_ordering: ", self.seriation.term_ordering
        #print "term_iter_index: ", self.seriation.term_iter_index   # Feel free to pick a less confusing variable name
        
        #print "similarity matrix generation time: ", compute_sim_time
        #print "seriation time: ", seriation_time
        self.logger.debug("seriation time: " +  str(seriation_time))

#-------------------------------------------------------------------------------#
# Helper Functions
    
    def initBestEnergies(self, firstTerm, candidateTerms):
        
        preBest = []
        postBest = []
        for candidate in candidateTerms:
            pre_score = 0
            post_score = 0
            
            # preBest
            if (candidate, firstTerm) in self.similarity.combined_g2:
                pre_score = self.similarity.combined_g2[(candidate, firstTerm)]
            # postBest
            if (firstTerm, candidate) in self.similarity.combined_g2:
                post_score = self.similarity.combined_g2[(firstTerm, candidate)]
            
            preBest.append((candidate, pre_score))
            postBest.append((candidate, post_score))
        
        return (preBest, postBest)
    
    def getBestEnergies(self, preBest, postBest, addedTerm):
        if addedTerm == 0:
            return (preBest, postBest, [])
        
        term_order = [x[0] for x in preBest]
        # compare candidate terms' bests against newly added term
        remove_index = -1
        for existingIndex in range(len(preBest)):
            term = term_order[existingIndex]
            if term == addedTerm:
                remove_index = existingIndex
            
            # check pre energies
            if (term, addedTerm) in self.similarity.combined_g2:
                if self.similarity.combined_g2[(term, addedTerm)] > preBest[existingIndex][1]:
                    preBest[existingIndex] = (term, self.similarity.combined_g2[(term, addedTerm)])
            # check post energies
            if (addedTerm, term) in self.similarity.combined_g2:
                if self.similarity.combined_g2[(addedTerm, term)] > postBest[existingIndex][1]:
                    postBest[existingIndex] = (term, self.similarity.combined_g2[(addedTerm, term)])
        
        # remove the added term's preBest and postBest scores
        if remove_index != -1:
            del preBest[remove_index]
            del postBest[remove_index]
        
        #create and sort the bestEnergies list
        energyMax = [sum(pair) for pair in zip([x[1] for x in preBest], [y[1] for y in postBest])]
        bestEnergies = zip([x[0] for x in preBest], energyMax)
        
        return (preBest, postBest, sorted(bestEnergies, key=itemgetter(1), reverse=True))
    
    def iterate_eff( self, candidateTerms, term_ordering, term_iter_index, buffers, bestEnergies, iteration_no ):
        maxEnergyChange = 0.0;
        maxTerm = "";
        maxPosition = 0;
        
        if len(bestEnergies) != 0:
            bestEnergy_terms = [x[0] for x in bestEnergies]
        else:
            bestEnergy_terms = candidateTerms
        
        breakout_counter = 0
        for candidate_index in range(len(bestEnergy_terms)):
            breakout_counter += 1
            candidate = bestEnergy_terms[candidate_index]
            for position in range(len(term_ordering)+1):
                current_buffer = buffers[position]
                candidateRank = self.termRank[candidate]
                if candidateRank <= (len(term_ordering) + self.candidateSize):
                    current_energy_change = self.getEnergyChange(candidate, position, term_ordering, current_buffer, iteration_no)
                    if current_energy_change > maxEnergyChange:
                        maxEnergyChange = current_energy_change
                        maxTerm = candidate
                        maxPosition = position
            # check for early termination
            if candidate_index < len(bestEnergy_terms)-1 and len(bestEnergies) != 0:
                if maxEnergyChange >= (2*(bestEnergies[candidate_index][1] + current_buffer)):
                    print "#-------- breaking out early ---------#"
                    print "candidates checked: ", breakout_counter
                    break;
        
        print "change in energy: ", maxEnergyChange
        print "maxTerm: ", maxTerm
        print "maxPosition: ", maxPosition
        
        candidateTerms.remove(maxTerm)
        
        # update buffers
        buf_score = 0
        if len(term_ordering) == 0:
            buffers = buffers
        elif maxPosition >= len(term_ordering):
            if (term_ordering[-1], maxTerm) in self.similarity.combined_g2:
                buf_score = self.similarity.combined_g2[(term_ordering[-1], maxTerm)]
            buffers.insert(len(buffers)-1, buf_score)
        elif maxPosition == 0:
            if (maxTerm, term_ordering[0]) in self.similarity.combined_g2:
                buf_score = self.similarity.combined_g2[(maxTerm, term_ordering[0])]
            buffers.insert(1, buf_score)
        else:
            if (term_ordering[maxPosition-1], maxTerm) in self.similarity.combined_g2:
                buf_score = self.similarity.combined_g2[(term_ordering[maxPosition-1], maxTerm)]
            buffers[maxPosition] = buf_score
            
            buf_score = 0
            if (maxTerm, term_ordering[maxPosition]) in self.similarity.combined_g2:
                buf_score = self.similarity.combined_g2[(maxTerm, term_ordering[maxPosition])]
            buffers.insert(maxPosition+1, buf_score)
        
        # update term ordering and ranking
        if maxPosition >= len(term_ordering):
            term_ordering.append(maxTerm)
        else:
            term_ordering.insert(maxPosition, maxTerm)
        term_iter_index.append(maxTerm)
            
        
        return (candidateTerms, term_ordering, term_iter_index, buffers)
    
    def getEnergyChange(self, candidateTerm, position, term_list, currentBuffer, iteration_no):
        prevBond = 0.0
        postBond = 0.0
        
        # first iteration only
        if iteration_no == 0:
            current_freq = 0.0
            current_saliency = 0.0
            
            if candidateTerm in self.termFreqs:
                current_freq = self.termFreqs[candidateTerm]
            if candidateTerm in self.termSaliency:
                current_saliency = self.termSaliency[candidateTerm]
            return 0.001 * current_freq * current_saliency
        
        # get previous term
        if position > 0:
            prev_term = term_list[position-1]
            if (prev_term, candidateTerm) in self.similarity.combined_g2:
                prevBond = self.similarity.combined_g2[(prev_term, candidateTerm)]
        
        # get next term
        if position < len(term_list):
            next_term = term_list[position]
            if (next_term, candidateTerm) in self.similarity.combined_g2:
                postBond = self.similarity.combined_g2[(candidateTerm, next_term)]
        
        return 2*(prevBond + postBond - currentBuffer)


class PrepareDataForClient(object):
    """
    Reformats data necessary for client to run. 
    
    Extracts a subset of the complete term list and term-topic matrix and writes
    the subset to a separate file. Also, generates JSON file that merges/packages term
    information with the actual term.
    
    Input is term-topic probability distribution and term information, stored in 4 files:
        'term-topic-matrix.txt' contains the entries of the matrix.
        'term-index.txt' contains the terms corresponding to the rows of the matrix.
        'topic-index.txt' contains the topic labels corresponding to the columns of the matrix.
        'term-info.txt' contains information about individual terms.
    
    Output is a subset of terms and matrix, as well as the term subset's information.
    Number of files created or copied: 5
        'submatrix-term-index.txt'
        'submatrix-topic-index.txt'
        'submatrix-term-topic.txt'
        'term-info.json'
        'term-info.txt'
    """
    
    def __init__(self, logging_level=logging.DEBUG):
        self.logger = logging.getLogger( 'PrepareDataForClient' )
        self.logger.setLevel( logging_level )
        handler = logging.StreamHandler( sys.stderr )
        handler.setLevel( logging_level )
        self.logger.addHandler( handler )
    
    def execute(self, model, saliency, seriation):
        self.logger.info( '--------------------------------------------------------------------------------' )
        self.logger.info( 'Preparing data for client...'                                                     )
        #self.logger.info( '    data_path = %s', data_path                                                    )
        
        self.logger.info( 'Connecting to data...' )
        self.model = model
        self.saliency = saliency
        self.seriation = seriation
        self.client = Client()
        
#         self.logger.info( 'Reading data from disk...' )
#         self.model.read()
#         self.saliency.read()
#         self.seriation.read()

        self.logger.info( 'Preparing parameters for seriated matrix...' )
        self.prepareSeriatedParameters()
        
        self.logger.info( 'Preparing parameters for filtered matrix...' )
        self.prepareFilteredParameters()
        
        self.logger.info( 'Preparing global term freqs...' )
        self.prepareGlobalTermFreqs()
        
        self.logger.info( 'Writing data to disk...' )
        self.client.write()

    def prepareSeriatedParameters( self ):
        topic_index = self.model.topic_index
        term_index = self.model.term_index
        term_topic_matrix = self.model.term_topic_matrix
        term_ordering = self.seriation.term_ordering
        term_topic_submatrix = []
        term_subindex = []
        for term in term_ordering:
            if term in term_index:
                index = term_index.index( term )
                term_topic_submatrix.append( term_topic_matrix[ index ] )
                term_subindex.append( term )
            else:
                self.logger.info( 'ERROR: Term (%s) does not appear in the list of seriated terms', term )

        self.client.seriated_parameters = {
            'termIndex' : term_subindex,
            'topicIndex' : topic_index,
            'matrix' : term_topic_submatrix
        }
    
    def prepareFilteredParameters( self ):
        term_rank_map = { term: value for value, term in enumerate( self.seriation.term_iter_index ) }
        term_order_map = { term: value for value, term in enumerate( self.seriation.term_ordering ) }
        term_saliency_map = { d['term']: d['saliency'] for d in self.saliency.term_info }
        term_distinctiveness_map = { d['term'] : d['distinctiveness'] for d in self.saliency.term_info }

        self.client.filtered_parameters = {
            'termRankMap' : term_rank_map,
            'termOrderMap' : term_order_map,
            'termSaliencyMap' : term_saliency_map,
            'termDistinctivenessMap' : term_distinctiveness_map
        }

    def prepareGlobalTermFreqs( self ):
        topic_index = self.model.topic_index
        term_index = self.model.term_index
        term_topic_matrix = self.model.term_topic_matrix
        term_ordering = self.seriation.term_ordering
        term_topic_submatrix = []
        term_subindex = []
        for term in term_ordering:
            if term in term_index:
                index = term_index.index( term )
                term_topic_submatrix.append( term_topic_matrix[ index ] )
                term_subindex.append( term )
            else:
                self.logger.info( 'ERROR: Term (%s) does not appear in the list of seriated terms', term )

        term_freqs = { d['term']: d['frequency'] for d in self.saliency.term_info }

        self.client.global_term_freqs = {
            'termIndex' : term_subindex,
            'topicIndex' : topic_index,
            'matrix' : term_topic_submatrix,
            'termFreqMap' : term_freqs
        }

