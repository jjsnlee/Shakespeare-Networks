import shakespeare_clusters as sc
import clusters, nmf

def main():
    """
    maybe:
        - reduce the sample size by removing characters with x # of lines
    """
    plays = sc.plays_n_graphs.get_plays()
    prc_ctx = sc.get_ctx_from_plays(plays)
    sc.preproc_data(prc_ctx, by='Char') # by='Play'
    mat = sc.process_data(prc_ctx, max_df=.8) # ngram data frame

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
    
    runs = sc.runs_lda(mat=mat, nruns=5)
    
if (__name__=="__main__"):
    main()

