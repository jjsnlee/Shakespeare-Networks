from shakespeare import clusters as sc
from shakespeare.clusters import LDAContext
import plays_n_graphs as png
from pcibook import nmf, clusters
from os.path import join
import json
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

import pandas as pd
def prepare_json(dataset_nm, which_json):
    # N = topics
    # T - top terms (~400)
    # V - vocabulary
    lda = LdaModel.load(dataset_nm)

    lda_ctxt = LDAContext.load_corpus()
    terms = lda_ctxt.dictionary.id2token.values()
    df = pd.DataFrame(lda.state.sstats, columns=terms)
    
    ntopics = lda.num_topics

    # topicIndex (1..N)
    # topicMapping (1..N)
    # termIndex (1..T)
    # matrix (T x N)
    topic_mapping = range(ntopics)
    topic_index = map(lambda n: 'Topic %d' % n+1, topic_mapping) 

    term_index = None
    matrix = None

    # termOrderMap (1..T)
    # termRankMap (1..T)
    # termDistinctivenessMap (1..V?)
    # termSailiencyMap (1..V)
    term_order_map = None
    term_rank_map = None
    term_distinctiveness_map = None
    term_saliency_map = None
    
    # termFreqMap (1..V)
    term_frequency_map = None
    
    if which_json == 'seriated-parameters.json':
        json_out = \
        {
        'topicIndex'   : topic_index,
        'topicMapping' : topic_mapping,
        'termIndex'    : term_index,
        'matrix'       : matrix
        }
    elif which_json == 'filtered-parameters.json':
        json_out = \
        {
        'termOrderMap' : term_order_map,
        'termRankMap'  : term_rank_map,
        'termDistinctivenessMap' : term_distinctiveness_map,
        'termSailiencyMap'       : term_saliency_map 
        }
    elif which_json == 'global-term_freqs.json':
        # same as in the seriated-parameters.json
        # topicIndex (1..N)
        # topicMapping (1..N)
        # termIndex (1..T)
        # matrix (T x N)
        
        json_out = \
        {
        'topicIndex'   : topic_index,
        'topicMapping' : topic_mapping,
        'termIndex'    : term_index,
        'matrix'       : matrix,
        'termFreqMap'  : term_frequency_map
        }
    
    return json.dumps(json_out, ensure_ascii=False)

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

