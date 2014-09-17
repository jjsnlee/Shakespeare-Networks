from shakespeare import clusters as sc
from shakespeare.clusters import LDAContext, LDAResult, get_lda_rslt
import plays_n_graphs as png
from pcibook import nmf, clusters
from os.path import join
import pandas as pd

from datetime import datetime
import time
import os
import logging
import helper
logger = helper.setup_sysout_handler(__name__)
logging.basicConfig(level=logging.DEBUG)

# 2014-05-13 00:50:36.652535_50_50.lda
# 2014-06-01 12:55:34.874782_20_50.lda

def main(label=None, train_new=False):
    """
    maybe:
        - reduce the sample size by removing characters with x # of lines
    """
    #import shakespeare.clusters as sc
    #import shakespeare.clusters_runner as scr
    if train_new:
#         play_ctx = png.get_plays_ctx('shakespeare')
#         prc_ctx = ClustersCtxt(play_ctx)
#         prc_ctx.preproc(by='Char/Scene') # by='Char'
#         lda_rslt = doLDA(prc_ctx)
        lda_rslt = doLDA(label, ntopics=50, npasses=50)
    else:
        #lda_key = '../data/dynamic/lda/2014-05-13 00:50:36.652535_50_50.lda'
        # char_scene_2014-06-29 19.49.11.703618_100_50_lda
        lda_key = '2014-05-13 00:50:36.652535_50_50.lda'
        lda_rslt = get_lda_rslt(lda_key)

    td = sc.TermiteData(lda_rslt)
    #td.saliency
    #td.similarity
    #td.seriation
    td.data_for_client()
    return td

def doLDA(baselabel, ntopics=50, npasses=50, ctx='shakespeare', by='Char/Scene', as_bow=False):
    play_ctx = png.get_plays_ctx(ctx)
    prc_ctx = ClustersCtxt(play_ctx)
    prc_ctx.preproc(by=by) # by='Char'
    
    doc_titles, docs_content = get_doc_content(prc_ctx)

    t = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    label = '%s_%s_%s_%s_lda' % (baselabel, t, ntopics, npasses)
    basedir = join(sc.get_lda_base_dir(), label)
    os.makedirs(basedir)
    logfile = join(basedir, 'gensim.log')
    
    # Need this to analyze the perplexity
    logger = logging.getLogger('gensim.models.ldamodel')
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    logger.addHandler(fh)
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    lda_ctxt = LDAContext(doc_titles, docs_content, stopwds=_get_stopwords(), as_bow=as_bow)
    #lda_ctxt.save_corpus()
    
    # Do this N number of times
    lda_rslt = LDAResult(label, lda_ctxt, ntopics=ntopics, npasses=npasses)
    lda_rslt.save()

    return lda_rslt

    #sc.print_lda_results(lda, lda_ctxt.corpus, doc_titles)
#     doc_results = lda[corpus]
#     from gensim.models.tfidfmodel import TfidfModel
#     tfidf_model = TfidfModel( )

def create_model_ctxt(ctx='shakespeare', by='Char/Scene', ):
    play_ctx = png.get_plays_ctx(ctx)
    prc_ctx = ClustersCtxt(play_ctx)
    prc_ctx.preproc(by=by) # by='Char'
    doc_titles, docs_content = get_doc_content(prc_ctx)
    from clusters import ModelContext
    ctxt = ModelContext(doc_titles, docs_content)
    return ctxt

def doAffProp(ctx='shakespeare', by='Char/Scene', ):
    from clusters import AffinityPropagationResult, ModelContext
    pass

def perplexity_scores():
    basedir = sc.get_lda_base_dir()
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

class ClustersCtxt(object):
    def __init__(self, play_ctx):
        from plays_n_graphs import RootPlayCtx
        assert(isinstance(play_ctx, RootPlayCtx))
        
        chars_per_play = {}
        for play_alias in play_ctx.map_by_alias:
            p = play_ctx.get_play(play_alias)
            chars_per_play[play_alias] = set(p.characters.keys())
        
        self.plays = play_ctx.play_details
        self.reset()
        self.chars_per_play = chars_per_play
        self.documents = [] # plays, characters, etc
        # remove documents: scenes/characters with very few lines
        #self.min_lines_per_doc = 10

    def reset(self):
        self.pruned_characters = {}
        self.pruned_max_terms = []

    def preproc(self, plays_to_filter=None, by='Play'):
        """
        get all the characters, the key should be name and play
        then all their lines, and relationships?
        it could be an interesting game of clustering
        """
        
        assert(by in ['Play', 'Char', 'Char/Scene'])
        plays = self.plays.values()
        if plays_to_filter:
            plays_to_filter = set(plays_to_filter)
            plays = [k for k in plays if k.title in plays_to_filter]
        self.reset()
        
        if by == 'Play':
            self.documents = plays
        
        elif by == 'Char':
            clines = []
            for p in plays:
                clines.extend(p.characters.values())
            self.documents = clines
        
        elif by == 'Char/Scene':
            from plays_n_graphs import Character
            clines = []
            for p in plays:
                chars = p.characters.values()
                # create artificial characters
                for c in chars:
                    char_lines = {}
                    for li in c.clean_lines:
                        char_lines.setdefault((li.act, li.scene), []).append(li)
                    for k in char_lines.keys():
                        char_name = '%s, Act %s, Sc %s' % (c.name, k[0], k[1])
                        artif_char = Character(char_name, c.play)
                        artif_char._cleaned_lines = char_lines[k] 
                        clines.append(artif_char)
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
            logger.info('Skipping [%s] since it had too few lines.', str(doc))
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
                 raw = False,
                 stopwords=_get_stopwords()
                 ):
    
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
                    stop_words=stopwords|all_c_in_play, 
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

def doNMF(prc_ctx):
    #-- NMF
    mat = process_data(prc_ctx, max_df=.8) # ngram data frame
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

