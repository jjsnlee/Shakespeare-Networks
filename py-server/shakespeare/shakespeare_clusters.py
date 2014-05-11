import plays_n_graphs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helper
from os.path import join
rootdir = helper.get_root_dir()

import sys
if join(rootdir, 'py-external') not in sys.path:
    sys.path.append(join(rootdir, 'py-external'))

from pcibook import nmf

import datetime, time
def get_ts():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

#def get_ctx_from_plays(play_ctx):
#    #print pformat(p.characters.values())
#    prc_ctx = ProcessCtxt(plays)
#    prc_ctx.chars_per_play = chars_per_play
#    return prc_ctx

def _get_stopwords():
    from nltk.corpus import stopwords
    stopwds = set(stopwords.words('english'))
    addl_stopwords_file = join(helper.get_root_dir(), 'data/stopwords')
    with open(addl_stopwords_file) as fh:
        more_stopwds = [s.strip() for s in fh.readlines() if not s.startswith('#')]
        stopwds = stopwds.union(more_stopwds)
    
    addl_stopwds = set([
        'thee', 'thy', 'thou', 'hath', 'shall', 'doth', 'dost', 'prithee', 'tis', 'ye', 'ay', 'hast',
        'says', 'good', 'sir',
        'give', 'put', #'speak', 'leave',
        #"'s", '!', '?', ':', ';', 'i', '.', ',', "'", 
        "ll", "d", "em", "n't",
        'edward', 'henry', 'jack', 'john', 'richard', 'harry', 'anne', 'hal', 'kate',
        #'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 
        'ten'
    ])

    stopwds = stopwds.union(addl_stopwds)
    return stopwds

class ProcessCtxt:
    def __init__(self, play_ctx):
        chars_per_play = {}
        for play_alias in play_ctx.map_by_alias:
            p = play_ctx.load_play(play_alias)
            chars_per_play[play_alias] = set(p.characters.keys())
        
        self.plays = play_ctx.play_details
        self.reset()
        self.chars_per_play = chars_per_play
        self.documents = [] # plays, characters, etc

    def reset(self):
        self.pruned_characters = {}
        self.pruned_max_terms = []

    def preproc(self, plays_to_filter=None, by='Play'):
        """
        get all the characters, the key should be name and play
        then all their lines, and relationships?
        it could be an interesting game of clustering
        """
        plays = self.plays.values()
        if plays_to_filter:
            plays_to_filter = set(plays_to_filter)
            plays = [k for k in plays if k.title in plays_to_filter]
        self.reset()
        
        if by == 'Play':
            self.documents = plays
        
        if by == 'Char':
            clines = []
            for p in plays:
                clines.extend(p.characters.values())
            self.documents = clines

def get_character_names(prc_ctx):
    #name_d = helper.init_name_dict()
    all_c_in_play = set()
    for play_name in prc_ctx.plays.keys():
        # Only characters in ALL CAPS are considered major, do not 
        # include minor characters in the list of stopwords.
        # There may be minor characters in the play
        # such as "Lord" in Hamlet. Do not want those terms to be removed. 
        c_in_play = prc_ctx.chars_per_play[play_name]
        c_in_play = set([c.lower() for c in c_in_play if c.isupper()])
        for c in c_in_play:
            v = prc_ctx.pruned_characters.setdefault(c, set())
            v.add(play_name)
        all_c_in_play.update(c_in_play)
    return all_c_in_play

def get_doc_content(prc_ctx, minlines=10):
    doc_titles   = []
    docs_content = []
    for doc in prc_ctx.documents:
        lines = doc.clean_lines
        # remove documents: scenes/characters with very few lines
        if len(lines) < minlines:
            print 'Skipping', str(doc)
            continue
        lines = ' '.join([li.spoken_line for li in lines])
        lines = lines.replace('--', ' ') # for now just replace these...
        #print lines+"|"
        lines = lines.lower()
        docs_content.append(lines)
        doc_titles.append(str(doc))
    return doc_titles, docs_content
    
def process_data(prc_ctx,
                 min_df=2, # in at least 2 documents
                 max_df=1.0,
                 minlines=10,
                 raw = False
                 ):
    stopwds = _get_stopwords()
    all_c_in_play = get_character_names(prc_ctx)
    
    #import PorterStemmer
    
    doc_titles, docs_content = get_doc_content(prc_ctx, minlines=minlines)

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    if raw: 
        vectorizer = CountVectorizer
    else:
        vectorizer = TfidfVectorizer
        
    # Do unigrams
    cv = vectorizer(min_df=min_df,
                    max_df=max_df,
                    charset_error="ignore",
                    stop_words=stopwds|all_c_in_play, 
                    )
    
    cnts = cv.fit_transform(docs_content).toarray()
    uni = pd.DataFrame(cnts, columns=cv.get_feature_names(), index=doc_titles)
    ngdf = uni

#    cv = CountVectorizer(min_df=2,  # in at least 2 documents
#                         charset_error="ignore",
#                         #stop_words=stopwds|c_in_play, 
#                         ngram_range=(2, 3)
#                         )
#    cnts = cv.fit_transform(docs).toarray()
#    ngs = pd.DataFrame(cnts, columns=cv.get_feature_names(), index=plays)
#    # Filter ngrams which end with a stopword        
#    keep = np.array([n.split(' ')[-1] not in stopwds for n in ngs.columns])
#    ngs = ngs.T[keep]
#    ngdf = pd.DataFrame.join(uni, ngs.T)
    # Remove terms which show up frequently
#    ngdf = ngdf.replace(0, float('Nan'))
#    nzero_cnts = ngdf.count()
#    rm_max = nzero_cnts > max_df*len(doc_titles)
#    prc_ctx.pruned_max_terms = ngdf.columns[rm_max]
#    ngdf = ngdf.T[rm_max==False]

    prc_ctx.pruned_max_terms = cv.stop_words_
    ngdf.fillna(0, inplace=True)
    
    # Transpose to have plays/documents on the index
    # Keep as is to have the terms on the index
    return ngdf

def create_lda_corpus(docs):
    from gensim.corpora import Dictionary
    from gensim.utils import simple_preprocess
    prcd_docs = [simple_preprocess(doc) for doc in docs]
    dictionary = Dictionary(prcd_docs)

    stopwds = _get_stopwords()
    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stopwds
                if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
    #dictionary.filter_tokens(stop_ids)
    dictionary.compactify()
    
    # odd API
    return [[dictionary.doc2bow(doc) for doc in prcd_docs], dictionary]

def print_lda_results(lda, corpus, docs):
    doc_results = lda[corpus]
    for idx, doc_rslt in enumerate(doc_results):
        character = docs[idx]
        print '*'*80
        print character
        for topic, score in doc_rslt:
            print '\ttopic %d, score: %s' % (topic, score)
            print '\t', lda.show_topic(topic)

def create_lda_corpus_with_mat(mat):
    class MyCorpus(object):
        def __iter__(self):
            for idx in range(len(mat.index)):
                #doc_nm = mat.index[idx]
                yield mat.ix[idx]
    return MyCorpus()

def runs_lda(corpus):
    from gensim.models.ldamodel import LdaModel
    lda = LdaModel(corpus, num_topics=10)
    return lda

def runs_multi_nmf(mat, nruns=5, pc=16, iters=100):
    runs = []
    for _n in range(nruns):
        print 'Start:', get_ts() 
        w,h = nmf.factorize(mat.values, pc=pc, iters=iters)
        print 'End:', get_ts()
        print w.shape, h.shape
        runs.append((w,h))
    return runs

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

