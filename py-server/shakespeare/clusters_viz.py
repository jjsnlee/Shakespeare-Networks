import matplotlib.pyplot as plt
import plays_n_graphs
import numpy as np
import pandas as pd
import clusters as sc
from os.path import join
import os

def close_plots(func):
    def func_wrapper(*a, **k):
        try:
            func(*a, **k)
        finally:
            plt.close('all') 
    return func_wrapper

@close_plots
def plot_char_wd_cnts(ctx, binwidth=1000, as_hist=True):
    _fig, ax = plt.subplots()
    #plt.subplot(1, 1, 1)
    y = np.array([len(c) for c in ctx.doc_contents])
    y.sort()
    
    if as_hist:
        ax.hist(y, bins=np.arange(min(y), max(y) + binwidth, binwidth))
    else:
        idx = range(len(ctx.doc_contents))
        ax.bar(idx, y, alpha=0.5)
    
    #plt.xticks(idx, x, rotation=70)
    plt.title(r'Words per Character')
    ax.grid(True)
    plt.show()

def create_char_data(ctx):
    wds = pd.Series([len(c) for c in ctx.doc_contents], index=ctx.doc_names)
    return wds

def distrib_of_terms(ctx):
    corpus = ctx.corpus
    wd_occurs = (corpus>0).sum()
    chars = corpus.shape[0]
    

@close_plots
def plot_perplexity_scores():
    ppx_scores = _perplexity_scores()
    fig, ax = plt.subplots()

    for k,v in ppx_scores.iteritems():
        v = map(float, v)
        print k,v[-1]
        #line,_ = plt.plot(v, label=k)
        ax.plot(v, label=k)
        #print line
        #labels.append(k)
        #lines.append(line)
    ax.legend(loc=1,prop={'size':8})
    plt.yscale('log')
    #ax.legend(handles=lines, loc=3)
    plt.title(r'Perplexity Scores')
    ax.grid(True)
    plt.show()

def _perplexity_scores():
    basedir = sc.get_models_base_dir()
    rslts = {}
    for d in os.listdir(basedir):
        if d.startswith('.') or d=='old':
            continue
        try:
            logfile = join(basedir, d, 'gensim.log')
            rslts[d] = sc.perplexity_score(logfile)
        except:
            # some of these may not have the logs
            pass
    return rslts

@close_plots
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

@close_plots
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
    
    