import json, os, re
import logging
import pandas as pd
import numpy as np
from gensim.models import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess #, SaveLoad
from os.path import join
import helper

from sklearn.externals import joblib

logging.basicConfig(level=logging.DEBUG)
logger = helper.setup_sysout_handler(__name__)

from datetime import datetime
import time
def get_ts():
  ts = time.time()
  return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def get_models_base_dir():
  return join(helper.get_dynamic_rootdir(), 'models')

class ModelContext(object):
  def __init__(self, doc_nms, doc_contents, 
               from_cache=None, stopwds=None, as_bow=True, 
               min_df=2, # in at least 2 documents
               max_df=1.0
               ):
    self.doc_names = doc_nms
    self.doc_contents = doc_contents
    
    if from_cache is None:
      self.stopwords = stopwds
    else:
      self.stopwords = from_cache['stopwords']
    
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    self.as_bow = as_bow # for the sake of serializing
    if as_bow:
      vectorizer = CountVectorizer
    else:
      vectorizer = TfidfVectorizer
    cv = vectorizer(
        min_df=min_df,
        max_df=max_df,
        charset_error="ignore",
        stop_words=stopwds, 
        )
    self.cnts = cv.fit_transform(doc_contents).toarray()
    unigrams = pd.DataFrame(self.cnts, columns=cv.get_feature_names(), index=doc_nms)
    self.corpus = unigrams

  def get_terms(self):
    return list(self.corpus.columns)

  corpus_data = 'corpus_data.json'

  def save_corpus(self, basedir=get_models_base_dir()):
    data = \
    {
     'vectorizer'   : 'BOW' if self.as_bow else 'TF-IDF',
     'stopwords'    : list(self.stopwords) if self.stopwords else [],
     'doc_titles'   : self.doc_names,
     'doc_contents' : self.doc_contents
    }
    
    # probably should also serialize the corpus, so we are dealing with the same exact object
    
    json_rslt = json.dumps(data, ensure_ascii=False, #cls=PlayJSONMetadataEncoder, 
                           indent=True)
    fname = join(basedir, self.corpus_data) 
    with open(fname, 'w') as fh:
      fh.write(json_rslt)

  @classmethod
  def load_corpus(cls, basedir=get_models_base_dir()):
    fname = join(basedir, cls.corpus_data)
    if not os.path.exists(fname):
      logger.warn('File path [%s] doesn\'t exist!', fname)
    ctxjson = json.loads(open(fname, 'r').read())
    doc_nms = ctxjson['doc_titles']
    doc_contents = ctxjson['doc_contents']
    stopwds = ctxjson['stopwords']
    as_bow = True if ctxjson['vectorizer']=='BOW' else False
    return ModelContext(doc_nms, doc_contents, from_cache=ctxjson, stopwds=stopwds, as_bow=as_bow)

def tokenize_documents(doc_contents):
  return [simple_preprocess(doc) for doc in doc_contents]

class LDAContext(object):
  def __init__(self, doc_nms, doc_contents, 
              from_cache=None, stopwds=None, as_bow=True,
	            terms_min=5, terms_max=0.5):

    self.doc_names = doc_nms
    #self.doc_contents = doc_contents
    self.doc_contents_tokenized = [simple_preprocess(doc) for doc in doc_contents]
    
    if from_cache is None:
      dictionary = Dictionary(self.doc_contents_tokenized)
      
      if not stopwds:
        logger.warn('No stopwords were specified, will use the entire vocabulary!')
        stopwds = []
      
      # remove stop words and words that appear only once
      stop_ids = [dictionary.token2id[stopword] for stopword in stopwds
                  if stopword in dictionary.token2id]
      once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() 
                  if docfreq == 1]
      
      dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
      
      dictionary.filter_extremes(no_below=terms_min, no_above=terms_max)
      #dictionary.filter_tokens(stop_ids)
      dictionary.compactify()
      # MANDATORY! to trigger the id2token creation
      dictionary[0]
      self.dictionary = dictionary
      
      # would be interesting to get bigram collocations here
      # this should use TF/IDF?
      
      bow_corpus = [dictionary.doc2bow(doc) for doc in self.doc_contents_tokenized]
      
      if as_bow:
        self.corpus = bow_corpus
      else:
        tfidf = TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        self.corpus = [doc for doc in corpus_tfidf]
      
      self.stopwords = stopwds
    else:
      self.dictionary = from_cache['dictionary']
      self.corpus = from_cache['corpus']
      self.stopwords = from_cache['stopwords']
  
  # can't seem to realize the transformed corpus 
  
  def get_terms(self):
    return self.dictionary.id2token.values()
  def find_term(self, t):
    return self.dictionary.id2token[t]
  
  lda_dict_fname = 'lda.dict'
  lda_corpus_data = 'corpus_data.json'
  
  def save_corpus(self, basedir=get_models_base_dir()):
    # corpus is just a BOW, do i need to save that? or just the doc_titles and content?
    self.dictionary.save(join(basedir, self.lda_dict_fname))
    
    data = \
    {
     'corpus'       : self.corpus,
     'stopwords'    : list(self.stopwords),
     #'doc_titles'   : self.doc_names,
     #'doc_contents' : self.doc_contents
    }
    
    json_rslt = json.dumps(data, ensure_ascii=False, #cls=PlayJSONMetadataEncoder, 
                           indent=True)
    fname = join(basedir, self.lda_corpus_data) 
    with open(fname, 'w') as fh:
      fh.write(json_rslt)
    
  @classmethod
  def load_corpus(cls, basedir=get_models_base_dir()):
    fname = join(basedir, cls.lda_corpus_data)
    if not os.path.exists(fname):
      logger.warn('File path [%s] doesn\'t exist!', fname)
    lda_json = json.loads(open(fname, 'r').read())
    dictionary = Dictionary.load(join(basedir, cls.lda_dict_fname))
    
    doc_nms = lda_json['doc_titles']
    doc_contents = lda_json['doc_contents']
    
    lda_json['dictionary'] = dictionary
    return LDAContext(doc_nms, doc_contents, from_cache=lda_json)

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
#     cluster1 = [j for i,j in zip(corpus_scores, doc_nms) if i[0][1] > threshold]
#     cluster2 = [j for i,j in zip(corpus_scores, doc_nms) if i[1][1] > threshold]
#     cluster3 = [j for i,j in zip(corpus_scores, doc_nms) if i[2][1] > threshold]

class ModelResult(object):
  def __init__(self, label, ctxt, model=None, init_model=None, ntopics=None, npasses=None):
    if model:
      self.model = model
    else:
      init_model()
    self.label = label
    self.model_ctxt = ctxt
    self.__termite_data = None
  
  @property
  def term_topic_matrix(self):
    return self.model.components_
  
  @property
  def num_topics(self):
    return self.model.n_components
  
  @property
  def termite_data(self):
    if self.__termite_data is None:
      from clusters_termite import TermiteData
      self.__termite_data = TermiteData(self)
    return self.__termite_data
  
  def generate_viz(self):
    self.termite_data.data_for_client()
  
  @property
  def basedir(self):
    return join(get_models_base_dir(), self.label)
  
  def _ensure_basedir(self):
    #basedir = join(get_models_base_dir(), self.label)
    # http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary
    # to handle potenital race conditions (though probably not applicable in my case).
    # also not sure why this wouldn't be handled more robustly by the python api itself 
    import errno
    try:
      os.makedirs(self.basedir)
    except OSError as exception:
      if exception.errno != errno.EEXIST:
        raise
  
  def save(self):
    # save the classifier
    self._ensure_basedir()
    fname = join(self.basedir, 'classifier.pkl')
    _ = joblib.dump(self.model, fname, compress=9)
    self.model_ctxt.save_corpus(basedir=self.basedir)
  
  @classmethod
  def load(cls, label):
    basedir = join(get_models_base_dir(), label)
    fname = join(basedir, 'classifier.pkl')
    ctxt  = ModelContext.load_corpus(basedir=basedir)
    rslt = cls(label, ctxt, model=joblib.load(fname))
    assert rslt.model.components_.shape[1] == len(rslt.model_ctxt.get_terms()) 
    return rslt
  
  def docs_per_topic(self):
    pass

class LDAResult(ModelResult):
  def __init__(self, label, lda_ctxt, lda=None, ntopics=None, npasses=None, *model_params):
    def init_model():
      corpus = lda_ctxt.corpus 
      dictionary = lda_ctxt.dictionary
      lda = LdaModel(corpus, num_topics=ntopics, 
                     id2word=dictionary.id2token, 
                     passes=npasses,
                     *model_params)
      self.model = lda
    super(LDAResult, self).__init__(label, lda_ctxt, lda, init_model=init_model)
    self._docs_per_topic = {}
    self._topics_per_doc = None
  @property
  def term_topic_matrix(self):
    return self.lda.state.sstats
  @property
  def num_topics(self):
    return self.lda.num_topics
  def save(self):
    self._ensure_basedir()
    self.lda_ctxt.save_corpus(basedir=self.basedir)
    self.lda.save(join(self.basedir, 'run.lda'))
    # should also save some of the state
  @classmethod
  def load(cls, label):
    basedir = join(get_models_base_dir(), label)
    lda_ctxt = LDAContext.load_corpus(basedir=basedir)
    lda_model_loc = join(get_models_base_dir(), label, 'run.lda')
    lda = LdaModel.load(lda_model_loc)
    lda_result = LDAResult(label, lda_ctxt, lda)
    return lda_result
  def as_dataframe(self):
    return pd.DataFrame(self.lda.state.sstats, columns=self.lda_ctxt.get_terms())
  # get index location of term:
  # df.columns.get_loc('jew')
  @property
  def lda_ctxt(self):
    return self.model_ctxt
  @property
  def lda(self):
    return self.model
  @property
  def doc_names(self):
    return self.lda_ctxt.doc_names
  @property
  def corpus(self):
    return self.lda_ctxt.corpus

  def docs_per_topic(self, topic):
    if topic not in self._docs_per_topic:
      from gensim import similarities

      # create a pseudo-doc
      topic_term_mixture = self.lda.expElogbeta[topic]
      
      mask = topic_term_mixture>.0001
      term_scores = topic_term_mixture[mask]
      # normalize the scores
      term_scores = term_scores/term_scores.sum()
      term_indxs = np.where(mask)
      # not sure why this is coming back as a tuple...
      vec_bow = zip(term_indxs[0], term_scores)
      vec_lda = self.lda[vec_bow]
      
      #self.model_ctxt.dictionary.id2token[1661]
      sim_index = similarities.MatrixSimilarity(self.lda[self.corpus])
      sims = sim_index[vec_lda]
      sims = sorted([(self.doc_names[i], score) for i,score in enumerate(sims)], 
                    key=lambda item: -item[1])
#       d = {}
#       #import pydevd;pydevd.settrace()
#       #eps = .0001
#       #corpus_scores = self.lda[self.corpus, eps]
#       for doc_nm in self.doc_names:
#         #print 'j:', doc_nm, 'i:', i
#         # scores will look like (idx, score):
#         # (2, 0.13756479927604343),
#         # (3, 0.081981422061639414)
#         for score in scores:
#           topic = score[0]
#           d.setdefault(topic, []).append((doc_nm, score[1]))
      self._docs_per_topic[topic] = sims
    return self._docs_per_topic[topic]

  
  @property
  def topics_per_doc(self):
    """
    The score here is really relevant per document, not topic. 
    The mixture of topics accountable for generating the document.
    So the total score per document will sum to 1.0, and per topic
    being the proportion of the topic responsible for the doc. This 
    becomes evident with a small epsilon -- otherwise the less significant
    topics are not returned 
    """
    if self._topics_per_doc is None:
      d = {}
      # in LDAModel, it is effectively recalculating the doc score
      # by calling inference(), which is called during the expectation
      # phase of the training
      
      #import pydevd;pydevd.settrace()
      eps = .0001
      corpus_scores = self.lda[self.corpus, eps]
      for scores, doc_nm in zip(corpus_scores, self.doc_names):
        #print 'j:', doc_nm, 'i:', i
        # scores will look like (idx, score):
        # (2, 0.13756479927604343),
        # (3, 0.081981422061639414)
        for score in scores:
          topic = score[0]
          d.setdefault(topic, []).append((doc_nm, score[1]))
      self._topics_per_doc = d
    return self._topics_per_doc
  
  def print_lda_results(self):
    doc_results = self.lda[self.corpus]
    for idx, doc_rslt in enumerate(doc_results):
      character = self.doc_names[idx]
      print '*'*80
      print character
      for topic, score in doc_rslt:
        print '\ttopic %d, score: %s' % (topic, score)
        print '\t', self.lda.show_topic(topic)
  def perplexity(self):
    return perplexity_score(join(self.basedir, 'gensim.log'))

def top_terms(df, term):
  term_col = df[term]
  return term_col[term_col > 1e-6]

def perplexity_score(logfile):
  ppx_scores = []
  with open(logfile) as fh:
    pat = re.compile(' ([-.0-9]+) per-word bound, ([-.0-9]+) perplexity')
    for li in fh.readlines():
      m = pat.search(li)
      if m:
        ppx_scores.append(m.group(2))
    # -12.572 per-word bound, 6088.5 perplexity
  return ppx_scores

CACHED_MODEL_RSLTS = {}
def get_lda_rslt(group, label, reload_ctx=False, cls=None):
  global CACHED_MODEL_RSLTS
  
  key = (group, label)
  if key not in CACHED_MODEL_RSLTS or reload_ctx:
    if cls is None:
      cls = LDAResult
    #basedir = join(get_models_base_dir(), group, label)
    #if not os.path.exists(basedir):
    #	pass
    CACHED_MODEL_RSLTS[key] = cls.load(join(group, label))
  return CACHED_MODEL_RSLTS[key]

