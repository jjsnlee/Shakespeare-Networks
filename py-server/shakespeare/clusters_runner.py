from shakespeare import clusters as sc
from shakespeare.clusters import LDAContext, get_lda_base_dir
import plays_n_graphs as png
from pcibook import nmf, clusters
from os.path import join
#import json
from gensim.models.ldamodel import LdaModel

def main():
    """
    maybe:
        - reduce the sample size by removing characters with x # of lines
    """
    play_ctx = png.get_plays_ctx('shakespeare')
    
    prc_ctx = sc.ProcessCtxt(play_ctx)
    #sc.preproc_data(prc_ctx, by='Play') # by='Char'
    
    prc_ctx.preproc(by='Char') # by='Char'
    
    lda_ctxt = LDAContext.load_corpus()
    lda = LdaModel.load('../data/dynamic/lda/2014-05-13 00:50:36.652535_50_50.lda')
    #prepare_json(lda, lda_ctxt)
    td = TermiteData(lda, lda_ctxt)
    #td.saliency()
    #td.similarity()
    td.seriation()
    

def doLDA(prc_ctx):
    doc_titles, docs_content = sc.get_doc_content(prc_ctx)
    
    import logging
    #logger = logging.getLogger('gensim.models.ldamodel')
    #logger.setLevel(logging.DEBUG)
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    
    lda_ctxt = LDAContext(doc_titles, docs_content)
    lda_ctxt.save_corpus()
    
    # Do this N number of times
    lda = run_n_save_lda(lda_ctxt)
    sc.print_lda_results(lda, lda_ctxt.corpus, doc_titles)

from termite import Model, Tokens, ComputeSaliency, ComputeSimilarity, \
    ComputeSeriation, PrepareDataForClient, \
    ClientRWer, SaliencyRWer, SimilarityRWer, SeriationRWer

class TermiteData(object):
    def __init__(self, lda, lda_ctxt, from_cache=False):
        self.lda = lda
        self.lda_ctxt = lda_ctxt
        self.basepath = get_lda_base_dir()

        model = Model()
        model.term_topic_matrix = lda.state.sstats.T 
        model.topic_count = lda.num_topics
        model.topic_index = map(lambda n: 'Topic %d' % (n+1), range(model.topic_count))
        model.term_index = lda_ctxt.get_terms()
        model.term_count = len(model.term_index)
        self.model = model

        tokens = Tokens()
        tokens.data = dict(self.docs_zipped())
        self.tokens = tokens
        
        self._saliency = None
        self._similarity = None
        self._seriation = None
    
    def docs_zipped(self):
        return [(t, c) for t,c in zip(self.lda_ctxt.doc_names, self.lda_ctxt.doc_content)]

    @property
    def saliency(self):
        if self._saliency is None:
            saliency_calc = ComputeSaliency()
            saliency_calc.execute(self.model)
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
            SeriationRWer.write()
        return self._seriation
    def data_for_client(self):
        prep_client = PrepareDataForClient()
        prep_client.execute(self.model, self.saliency, self.seriation)
        ClientRWer.write(prep_client, self.basepath)

#def prepare_json(lda, lda_ctxt):
    #lda = LdaModel.load(dataset_nm)
    #lda_ctxt = LDAContext.load_corpus()
    #import pandas as pd
    #df = pd.DataFrame(lda.state.sstats, columns=lda_ctxt.get_terms())

    # N = topics
    # T - top terms (~400)
    # V - vocabulary
    #ntopics = lda.num_topics
    # topicIndex (1..N)
    # topicMapping (1..N)
    # termIndex (1..T)
    # matrix (T x N)
    #topic_mapping = range(ntopics)
    #topic_index = map(lambda n: 'Topic %d' % n+1, topic_mapping) 
#     term_index = None
#     matrix = None
# 
#     # termOrderMap (1..T)
#     # termRankMap (1..T)
#     # termDistinctivenessMap (1..V?)
#     # termSailiencyMap (1..V)
#     term_order_map = None
#     term_rank_map = None
#     term_distinctiveness_map = None
#     term_saliency_map = None
#     
#     # termFreqMap (1..V)
#     term_frequency_map = None
#     
#     if which_json == 'seriated-parameters.json':
#         json_out = \
#         {
#         'topicIndex'   : topic_index,
#         'topicMapping' : topic_mapping,
#         'termIndex'    : term_index,
#         'matrix'       : matrix
#         }
#     elif which_json == 'filtered-parameters.json':
#         json_out = \
#         {
#         'termOrderMap' : term_order_map,
#         'termRankMap'  : term_rank_map,
#         'termDistinctivenessMap' : term_distinctiveness_map,
#         'termSailiencyMap'       : term_saliency_map 
#         }
#     elif which_json == 'global-term_freqs.json':
#         # same as in the seriated-parameters.json
#         # topicIndex (1..N)
#         # topicMapping (1..N)
#         # termIndex (1..T)
#         # matrix (T x N)
#         
#         json_out = \
#         {
#         'topicIndex'   : topic_index,
#         'topicMapping' : topic_mapping,
#         'termIndex'    : term_index,
#         'matrix'       : matrix,
#         'termFreqMap'  : term_frequency_map
#         }
#     return json.dumps(json_out, ensure_ascii=False)

def run_n_save_lda(lda_ctxt, ntopics=50, npasses=50):
    corpus = lda_ctxt.corpus 
    dictionary = lda_ctxt.dictionary
    lda = LdaModel(corpus, num_topics=ntopics, id2word=dictionary.id2token, passes=npasses)
    from datetime import datetime
    t = datetime.now()
    fname = '%s_%s_%s.lda' % (t, ntopics, npasses)
    lda.save(join(sc.get_lda_base_dir(), fname))
    return lda
    
#     doc_results = lda[corpus]
#     from gensim.models.tfidfmodel import TfidfModel
#     tfidf_model = TfidfModel( )

def doNMF(prc_ctx):
    #-- NMF
    mat = sc.process_data(prc_ctx, max_df=.8) # ngram data frame
    #mat = sc.process_data(prc_ctx, max_df=.8, raw=True) # ngram data frame
    clust = clusters.hcluster(mat.values)
    # some error here
    clusters.drawdendrogram(clust, map(str, mat.index), jpeg='shakespeare.jpg')
    #rdata = clusters.rotatematrix(mat)
    #wordclust = clusters.hcluster(rdata)
    #w,h = nmf.factorize(a*b, pc=3, iters=100)
    #-- 
    w,h = nmf.factorize(mat.values, pc=16, iters=100)
    tps, ptns = nmf.showfeatures(w, h, mat.index, mat.columns)
    nmf.showarticles(mat.index, tps, ptns)
    #-- 
    runs = sc.runs_multi_nmf(mat=mat, nruns=5)
    #runs = sc.runs_lda(mat=mat, nruns=5)
    
if (__name__=="__main__"):
    main()

