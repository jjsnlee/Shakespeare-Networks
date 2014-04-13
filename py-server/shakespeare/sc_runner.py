import shakespeare_clusters as sc
import plays_n_graphs as png
from pcibook import nmf, clusters

def main():
    """
    maybe:
        - reduce the sample size by removing characters with x # of lines
    """
    play_ctx = png.get_plays_ctx('shakespeare')
    
    prc_ctx = sc.ProcessCtxt(play_ctx)
    #sc.preproc_data(prc_ctx, by='Play') # by='Char'
    
    prc_ctx.preproc(by='Char') # by='Char'
    mat = sc.process_data(prc_ctx, max_df=.8) # ngram data frame
    
    #mat = sc.process_data(prc_ctx, max_df=.8, raw=True) # ngram data frame

    #-- 
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

def doLDA(prc_ctx):
    from gensim.models.ldamodel import LdaModel
    _doc_titles, docs_content = sc.get_doc_content(prc_ctx)
    
    import logging
    #logger = logging.getLogger('gensim.models.ldamodel')
    #logger.setLevel(logging.DEBUG)
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    
    corpus, dictionary  = sc.create_lda_corpus(docs_content)
    # trigger the id2token creation
    dictionary[0]
    lda = LdaModel(corpus, num_topics=10, id2word=dictionary.id2token)
    
    from gensim.models.tfidfmodel import TfidfModel
    tfidf_model = TfidfModel( )

    
if (__name__=="__main__"):
    main()

