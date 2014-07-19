import plays_n_graphs
import numpy as np
import matplotlib.pyplot as plt
import json, sys, os
import logging
import pandas as pd

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess #, SaveLoad
from os.path import join
import helper

logging.basicConfig(level=logging.DEBUG)
logger = helper.setup_sysout_handler(__name__)

rootdir = helper.get_root_dir()

if join(rootdir, 'py-external') not in sys.path:
    sys.path.append(join(rootdir, 'py-external'))

from pcibook import nmf

from datetime import datetime
import time
def get_ts():
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def get_lda_base_dir():
    return join(helper.get_dynamic_rootdir(), 'lda')

class LDAContext(object):
    def __init__(self, doc_nms, doc_contents, from_cache=None, stopwds=None):
        self.doc_names = doc_nms
        self.doc_contents = doc_contents
        self.doc_contents_tokenized = [simple_preprocess(doc) for doc in doc_contents]
        
        if from_cache is None:
            dictionary = Dictionary(self.doc_contents_tokenized)
            
            if not stopwds:
                logger.warn('No stopwords were specified, will use the entire vocabulary!')
                stopwds = []
            
            # remove stop words and words that appear only once
            stop_ids = [dictionary.token2id[stopword] for stopword in stopwds
                        if stopword in dictionary.token2id]
            once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]

            dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
            #dictionary.filter_tokens(stop_ids)
            dictionary.compactify()
            # MANDATORY! to trigger the id2token creation
            dictionary[0]
            self.dictionary = dictionary
            
            # would be interesting to get bigram collocations here
            
            self.corpus = [dictionary.doc2bow(doc) for doc in self.doc_contents_tokenized]
            self.stopwords = stopwds
        else:
            self.dictionary = from_cache['dictionary']
            self.corpus = from_cache['corpus']
            self.stopwords = from_cache['stopwords']
    
    def get_terms(self):
        return self.dictionary.id2token.values()
    def find_term(self, t):
        return self.dictionary.id2token[t]
        #return self.get_terms().index(t)
    def term_cnt_matrix(self):
        pass

    lda_dict_fname = 'lda.dict'
    lda_corpus_data = 'corpus_data.json'

    def save_corpus(self, basedir=get_lda_base_dir()):
        # corpus is just a BOW, do i need to save that? or just the doc_titles and content?
        self.dictionary.save(join(basedir, self.lda_dict_fname))
        
        data = \
        {
         'corpus'       : self.corpus,
         'stopwords'    : list(self.stopwords),
         'doc_titles'   : self.doc_names,
         'doc_contents' : self.doc_contents
        }
        
        json_rslt = json.dumps(data, ensure_ascii=False, #cls=PlayJSONMetadataEncoder, 
                               indent=True)
        fname = join(basedir, self.lda_corpus_data) 
        with open(fname, 'w') as fh:
            fh.write(json_rslt)

    @classmethod
    def load_corpus(cls, basedir=get_lda_base_dir()):
        fname = join(basedir, cls.lda_corpus_data)
        if not os.path.exists(fname):
            logger.warn('File path [%s] doesn\'t exist!', fname)
        lda_json = json.loads(open(fname, 'r').read())
        dictionary = Dictionary.load(join(basedir, cls.lda_dict_fname))
        
        doc_nms = lda_json['doc_titles']
        doc_contents = lda_json['doc_contents']
        
        lda_json['dictionary'] = dictionary
        return LDAContext(doc_nms, doc_contents, from_cache=lda_json)

CACHED_LDA_RSLTS = {}
def get_lda_rslt(lda_label, reload_ctx=False):
    global CACHED_LDA_RSLTS
    if lda_label not in CACHED_LDA_RSLTS or reload_ctx:
        basedir = join(get_lda_base_dir(), lda_label)
        lda_ctxt = LDAContext.load_corpus(basedir=basedir)
        
        lda_model_loc = join(get_lda_base_dir(), lda_label, 'run.lda')
        lda = LdaModel.load(lda_model_loc)
        lda_result = LDAResult(lda_label, lda_ctxt, lda)
        CACHED_LDA_RSLTS[lda_label] = lda_result
    return CACHED_LDA_RSLTS[lda_label]

# def get_docs_per_topic(lda, lda_ctxt):
#     # http://stackoverflow.com/questions/20984841/topic-distribution-how-do-we-see-which-document-belong-to-which-topic-after-doi
#     corpus_scores = lda[lda_ctxt.corpus]
#     from itertools import chain
#     # Find the threshold, let's set the threshold to be 1/#clusters,
#     # To prove that the threshold is sane, we average the sum of all probabilities:
#     all_scores = list(chain(*[[score for _topic, score in topic] \
#                               for topic in [doc_score for doc_score in corpus_scores]]))
#     threshold = np.mean(all_scores)
#     print threshold
#     docs_per_cluster = \
#     dict([
#      (n, [x for x in n]) 
#      for n in range(lda.num_topics)
#     ])
    #doc_nms = lda_ctxt.doc_names
#     cluster1 = [j for i,j in zip(corpus_scores, doc_nms) if i[0][1] > threshold]
#     cluster2 = [j for i,j in zip(corpus_scores, doc_nms) if i[1][1] > threshold]
#     cluster3 = [j for i,j in zip(corpus_scores, doc_nms) if i[2][1] > threshold]

class LDAResult(object):
    def __init__(self, label, lda_ctxt, lda=None, ntopics=None, npasses=None):
        corpus = lda_ctxt.corpus 
        dictionary = lda_ctxt.dictionary
        
        if lda:
            self.lda = lda
            self.label = label
        else:
            lda = LdaModel(corpus, num_topics=ntopics, id2word=dictionary.id2token, passes=npasses)
            t = datetime.now()
            #self.baselabel = label
            self.label = '%s_%s_%s_%s_lda' % (label, t, ntopics, npasses)
            self.lda = lda

        self.lda_ctxt = lda_ctxt
        self._docs_per_topic = None
        self.termite_data = TermiteData(self)

    def generate_viz(self):
        self.termite_data.data_for_client()

    def save(self):
        basedir = join(get_lda_base_dir(), self.label)
        os.makedirs(basedir)
        self.lda_ctxt.save_corpus(basedir=basedir)
        
        self.lda.save(join(basedir, 'run.lda'))
        # should also save some of the state
        
    def as_dataframe(self):
        return pd.DataFrame(self.lda.state.sstats, columns=self.lda_ctxt.get_terms())
    # get index location of term:
    # df.columns.get_loc('jew')

    @property
    def doc_names(self):
        return self.lda_ctxt.doc_names
    @property
    def corpus(self):
        return self.lda_ctxt.corpus
    @property
    def docs_per_topic(self):
        if self._docs_per_topic is None:
            d = {}
            corpus_scores = self.lda[self.corpus]
            for scores, doc_nm in zip(corpus_scores, self.doc_names):
                #print 'j:', doc_nm, 'i:', i
                for score in scores:
                    topic = score[0]
                    d.setdefault(topic, []).append((doc_nm, score[1]))
            self._docs_per_topic = d
        return self._docs_per_topic
    
    def print_lda_results(self):
        doc_results = self.lda[self.corpus]
        for idx, doc_rslt in enumerate(doc_results):
            character = self.doc_names[idx]
            print '*'*80
            print character
            for topic, score in doc_rslt:
                print '\ttopic %d, score: %s' % (topic, score)
                print '\t', self.lda.show_topic(topic)

from termite import Model, Tokens, ComputeSaliency, ComputeSimilarity, ComputeSeriation, \
    ClientRWer, SaliencyRWer, SimilarityRWer, SeriationRWer

class TermiteData(object):
    def __init__(self, lda_rslt, from_cache=False):
        #assert(isinstance(lda_rslt, LDAResult))
        lda = lda_rslt.lda
        lda_ctxt = lda_rslt.lda_ctxt
        
        self.lda = lda
        self.lda_ctxt = lda_ctxt
        self.basepath = join(get_lda_base_dir(), lda_rslt.label, 'termite')

        model = Model()
        model.term_topic_matrix = lda.state.sstats.T

        #model.topic_count = lda.num_topics
        #model.term_count = len(model.term_index)
        
        model.topic_index = map(lambda n: 'Topic %d' % (n+1), range(lda.num_topics))
        model.term_index = lda_ctxt.get_terms()
        self.model = model

        tokens = Tokens()
        tokens.data = dict(self.docs_tokenized_zipped())
        self.tokens = tokens
        
        self._saliency = None
        self._similarity = None
        self._seriation = None
    
    def docs_zipped(self):
        return [(t, c) for t,c in zip(self.lda_ctxt.doc_names, self.lda_ctxt.doc_contents)]

    def docs_tokenized_zipped(self):
        return [(t, c) for t,c in zip(self.lda_ctxt.doc_names, self.lda_ctxt.doc_contents_tokenized)]

    @property
    def saliency(self):
        if self._saliency is None:
            saliency_calc = ComputeSaliency(self.model)
            saliency_calc.execute()
            self._saliency = saliency_calc.saliency
            SaliencyRWer.write(self._saliency, self.basepath) 
        return self._saliency
    @property
    def similarity(self):
        if self._similarity is None:
            similarity_calc = ComputeSimilarity()
            similarity_calc.execute(self.tokens)
            self._similarity = similarity_calc.similarity
            SimilarityRWer.write(self._similarity, self.basepath)
        return self._similarity
    @property
    def seriation(self):
        if self._seriation is None:
            seriation_calc = ComputeSeriation()
            seriation_calc.execute(self.saliency, self.similarity)
            self._seriation = seriation_calc.seriation
            SeriationRWer.write(self._seriation, self.basepath)
        return self._seriation
    def data_for_client(self):
        from termite.prepare_data_for_client import Client 
        client = Client()
        client.prepareSeriatedParameters(self.model, self.seriation)
        client.prepareFilteredParameters(self.seriation, self.saliency)
        client.prepareGlobalTermFreqs(self.saliency)
        ClientRWer.write(client, self.basepath)
        return client

    def load(self, which):
        if which=='saliency':
            self._saliency = SaliencyRWer.read(self.basepath)
        elif which=='seriation':
            self._seriation = SeriationRWer.read(self.basepath)
        elif which=='similarity':
            self._similarity = SimilarityRWer.read(self.basepath)

def runs_affine_prop(mat):
    from sklearn import cluster, covariance, manifold

    # EDGES - Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()
    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = mat.values.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)
    
    # Cluster using affinity propagation
    plays = mat.index
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()
    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(plays[labels == i])))

    # NODES - Find a low-dimension embedding for visualization: find the 
    # best position of the nodes (the stocks) on a 2D plane
    node_position_model = manifold.LocallyLinearEmbedding(n_components=2, 
                                                          eigen_solver='dense', 
                                                          n_neighbors=6)
    node_embedding = node_position_model.fit_transform(X.T).T
    
    # Visualization
    import pylab as pl
    from matplotlib.collections import LineCollection
    pl.figure(1, facecolor='w', figsize=(10, 8))
    pl.clf()
    ax = pl.axes([0., 0., 1., 1.])
    pl.axis('off')
    
    # EDGES - Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero_edges = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    
    # NODES - Plot the nodes using the coordinates of our embedding
    pl.scatter(node_embedding[0], node_embedding[1], 
               s=100 * d ** 2, 
               c=labels, 
               cmap=pl.cm.spectral)
    
    # EDGES - Plot the edges
    start_idx, end_idx = np.where(non_zero_edges)
    #a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[node_embedding[:, start], node_embedding[:, stop]]
                    for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero_edges])
    lc = LineCollection(segments,
                        zorder=0, cmap=pl.cm.hot_r,
                        norm=pl.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)
    
    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(zip(plays, labels, node_embedding.T)):
        dx = x - node_embedding[0]
        dx[index] = 1
        dy = y - node_embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        pl.text(x, y, name, size=10,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                bbox=dict(facecolor='w',
                          edgecolor=pl.cm.spectral(label / float(n_labels)),
                          alpha=.6))
    
    pl.xlim(node_embedding[0].min() - .15 * node_embedding[0].ptp(),
            node_embedding[0].max() + .10 * node_embedding[0].ptp(),)
    pl.ylim(node_embedding[1].min() - .03 * node_embedding[1].ptp(),
            node_embedding[1].max() + .03 * node_embedding[1].ptp())
    
    pl.show()

#def runs_lingo(mat):
#    import carrot_alg_lingo as lingo
#    from org.carrot2.core import ControllerFactory
#    controller = ControllerFactory.createSimple();
#    byTopicClusters = controller.process(documents, "data mining", LingoClusteringAlgorithm.class);
#    final List<Cluster> clustersByTopic = byTopicClusters.getClusters();

def create_lda_corpus_with_mat(mat):
    class MyCorpus(object):
        def __iter__(self):
            for idx in range(len(mat.index)):
                #doc_nm = mat.index[idx]
                yield mat.ix[idx]
    return MyCorpus()

def runs_multi_nmf(mat, nruns=5, pc=16, iters=100):
    runs = []
    for _n in range(nruns):
        print 'Start:', get_ts() 
        w,h = nmf.factorize(mat.values, pc=pc, iters=iters)
        print 'End:', get_ts()
        print w.shape, h.shape
        runs.append((w,h))
    return runs

def plot_wd_cnts(mat, pct_min=0, pct_max=100, mincnt=10):
    """ Useful to see the top words """
    cnts = mat.sum()
    cnts.sort()
    cnts = cnts[cnts>mincnt]
    wdcnts = cnts.values
    wds = cnts.keys()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    pos = np.arange(len(wds))+0.5    #Center bars on the Y-axis ticks

    _rects = ax.barh(pos, wdcnts, align='center', height=.5, color='m')
    plt.yticks(pos, wds)
    
    ax.set_xlabel('Counts')
    ax.set_ylabel('Word Bins')
    ax.grid(True)
    plt.show()

def plot_cnt_distrib(cnts, cnt_min=5, cnt_max=100):
#    import matplotlib.mlab as mlab
    
    vals = cnts.values()
    nbins = 100
    step = (cnt_max-cnt_min) / nbins
    bin_lbls = range(cnt_min, cnt_max, step)

#    for k,v in cnts:
#        if v > cnt_min and v <= cnt_max:
#            pass

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(vals, bins=bin_lbls, 
                               #normed=1, facecolor='green', alpha=0.75,
                               log=True,
                               range=[cnt_min, cnt_max]
                               )

    #bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters, mu, sigma)
    #l = ax.plot(bincenters, y, 'r--', linewidth=1)

    ax.set_xlabel('Word Bins')
    ax.set_ylabel('Counts')
    #ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #ax.set_xlim(40, 160)
    #ax.set_ylim(0, 0.03)
    ax.grid(True)
    plt.show()

def analyze_factors(mat, runs):
    for n in range(len(runs)):
        w,h = runs[n]
        # top patterns, pattern names
        tps, ptns = nmf.showfeatures(w, h, mat.index, mat.columns, out='output/features_%d.txt'%n)
        nmf.showarticles(mat.index, tps, ptns, out='output/articles_%d.txt'%n)
    # it would be good to be able to get the difference between different runs

def plot_factors(plays, dp, tp): #mat
    """ 
    Taken largely from http://matplotlib.org/examples/api/radar_chart.html
    """
    import radar_chart
    
    #nruns = len(dp.items)
    nfactors = len(dp.minor_axis)
    #spoke_angles = range(nfactors)
    
    #docs = mat.index
    docs = set([p.type for p in plays.values()])

    theta = radar_chart.radar_factory(nfactors, frame='polygon')

    #data = radar_chart.example_data()
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    pc = plays_n_graphs.play_classifications
    colors = ['b', 'r', 'g']
    #colors = ['b', 'r', 'g', 'm', 'y']
    #colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _n in range(len(docs))]

    # Plot the four cases from the example data on separate axes
    for n in range(2):
        title = 'Run %d'%n
        ax = fig.add_subplot(2, 2, 2*n+1, projection='radar')
        #plt.rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')

        spoke_labels = []
        for f in range(nfactors):
            top_factors = tp.ix[n][f].copy()
            top_factors.sort()
            top_factors = top_factors[::-1].head(5).keys()
            #print top_factors
            #spoke_labels += ','.join(top_factors)
            spoke_labels.append(','.join(top_factors))
        
        # strength of factors
        #factor_str = dp.ix[n].values
        grpd = dp.ix[n].groupby(lambda p: pc[p])
        factor_str = grpd.sum().values
        
        for d, color in zip(factor_str, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    plt.subplot(2, 2, 1)
    legend = plt.legend(docs, 
                        loc=(1.2, -.95),
                        #loc='center right', 
                        labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')

#    plt.figtext(0.5, 0.965, 
#                '5-Factor Solution Profiles Across Four Scenarios',
#                ha='center', color='black', weight='bold', size='large')
    plt.show()

def make_matricesXXX(ngdf, min_cnt=2, max_threshold=1.0, ret_skipped=False):
    """ 
    in some respects can think of this as the kernel function, 
    the magic sauce to get the right mix of words to retain
    
    min_cnt : should be in at least this many plays
    """
    ndocs = len(ngdf)
    # Words which are in less than x% of the documents
    doc_threshold = ndocs * max_threshold
    #print 'doc_threshold: %d' % doc_threshold
    # word document frequency
    wddf = ngdf.count(axis=0)
    X = ngdf.T
    skipped_wds = X[wddf > doc_threshold]
    #print 'skipped_wds: %s' % skipped_wds.ix
    X = X[wddf.between(min_cnt, doc_threshold)] # inclusive
    # tf idf , idf = total docs * # docs in which word appears
    #X = X.apply(lambda S: S*(ndocs/wddf))
    X.fillna(0, inplace=True)
    if ret_skipped:
        return X.T, skipped_wds
    return X.T
   
    # actually this doesn't make sense...
#    if min_cnt:
#        X = X[X>min_cnt]
#        wddf = X.count(axis=0)
#        X = X.T[wddf > 0]
#        return X
#    wordvec = [] # Only take words that are common but not too common
#    word_df = [] # word document frequency
#    for w, c in cnts.iteritems():
#        _ndocs = np.sum([1 for f in ngrams_per_doc if w in f])
#        if c > 3 and _ndocs > 1 and _ndocs < doc_threshold:
#            wordvec.append(w)
#            word_df.append(_ndocs) 
#    for f in ngrams_per_doc:
#        x = []
#        # tf idf , idf = total docs * # docs in which word appears
#        for idx, word in enumerate(wds):
#            print f, word
#            #x.append(word in f and f[word] or 0)
#            x.append(word in f and f[word]*(ndocs/wddf[idx]) or 0)
#        mat.append(x)
#     return mat, wds
    #' '.join([li.spoken_line for li in plays['Hamlet'].characters['HAMLET'].clean_lines])
    #docs[k] = {x:y for x,y in zip(wds, cnts) }
    #print wds.shape
    #cnts = np.concatenate((cnts, cnts2)) # , axis=1 ?
#        docs[k].update({x:y for x,y in zip(ngs, cnts)})

