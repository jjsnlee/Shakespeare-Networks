import numpy as np
import json, sys, os, re
import logging
import pandas as pd

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
		#return self.get_terms().index(t)
	#def term_cnt_matrix(self):
	#    pass
	
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
			from shakespeare.clusters_termite import TermiteData
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
	
	@property
	def docs_per_topic(self):
		pass

class NMFResult(ModelResult):
	def __init__(self, label, ctxt, model=None, ntopics=10, npasses=200):
		def init_model():
			from sklearn.decomposition import NMF
			self.model = NMF(n_components=ntopics,
			                 init=None, sparseness=None,
			                 beta=1, eta=0.1, tol=0.0001, max_iter=npasses, nls_max_iter=2000,
			                 random_state=None)
			self.H = self.model.fit_transform(ctxt.corpus)
		super(NMFResult, self).__init__(label, ctxt, model, init_model=init_model)
		self._docs_per_topic = None
		self._topics_per_doc = None
	
	def save(self):
		super(NMFResult, self).save()
		fname = join(self.basedir, 'classifier_H.pkl')
		_ = joblib.dump(self.H, fname, compress=9)
	@classmethod
	def load(cls, label):
		rslt = super(NMFResult, cls).load(label)
		basedir = join(get_models_base_dir(), label)
		fname = join(basedir, 'classifier_H.pkl')
		rslt.H = joblib.load(fname)
		return rslt
	
	@property
	def topics_per_doc(self):
		if self._topics_per_doc is None:
			self.docs_per_topic
		return self._topics_per_doc
	@property
	def docs_per_topic(self):
		if self._docs_per_topic is None:
			d1 = {}
			d2 = {}
			doc_nms = self.model_ctxt.doc_names
			corpus_scores = self.H
			for scores, doc_nm in zip(corpus_scores, doc_nms):
				#print 'j:', doc_nm, 'i:', i
				for topic, score in enumerate(scores):
					if score>0:
						d1.setdefault(topic, []).append((doc_nm, score))
						d2.setdefault(doc_nm, []).append((topic, score))
			self._docs_per_topic = d1
			self._topics_per_doc = d2
		return self._docs_per_topic

# class RBMPipelineResult(_ModelResult):
#     def __init__(self, label, ctxt, model=None, ntopics=16, npasses=200):
#         def init_model():
#             from sklearn.neural_network import BernoulliRBM
#             from sklearn import linear_model
#             pass
#         super(RBMPipelineResult, self).__init__(label, ctxt, model, init_model=init_model)
# class RBMResult(ModelResult):
#     def __init__(self, label, ctxt, model=None, ntopics=16, npasses=200, verbose=True):
#         def init_model():
#             from sklearn.neural_network import BernoulliRBM
#             # learning_rate - highly recommended to tune this...
#             self.model = BernoulliRBM(n_components=ntopics,
#                                       random_state=0,
#                                       n_iter=npasses,
#                                       verbose=verbose
#                                       )
#             self.model.fit(ctxt.corpus)
#         super(RBMResult, self).__init__(label, ctxt, model, init_model=init_model)

class GMMResult(ModelResult):
	def __init__(self, label, ctxt, model=None, ntopics=10, npasses=200): 
		def init_model():
			from sklearn.mixture import GMM
			self.model = GMM(n_components=ntopics, 
			                 #init=None, sparseness=None, 
			                 #beta=1, eta=0.1, tol=0.0001, max_iter=npasses, nls_max_iter=2000, 
			                 #random_state=None
			                 )
			self.H = self.model.fit_transform(ctxt.corpus)
		super(GMMResult, self).__init__(label, ctxt, model, init_model=init_model)
#         self._docs_per_topic = None
#         self._topics_per_doc = None

class AffinityPropagationResult(ModelResult):
	def __init__(self, label, ctxt, model=None, ntopics=None, npasses=None):
		def init_model():
			from sklearn.covariance import GraphLassoCV
			from sklearn.cluster import affinity_propagation
			
			mat = ctxt.corpus
			# standardize the time series: using correlations rather than covariance
			# is more efficient for structure recovery
			X = mat.values.copy().T
			X /= X.std(axis=0)
			
			# EDGES - Learn a graphical structure from the correlations
			#edge_model = GraphLassoCV(verbose=True, n_jobs=2)
			edge_model = GraphLassoCV(verbose=True)
			edge_model.fit(X)
			
			self.edge_model = edge_model
			
			# Cluster using affinity propagation - just the labels are clustered
			# purely based on correlations in this case?
			#plays = mat.index
			#_, labels = affinity_propagation(edge_model.covariance_)
			#n_labels = labels.max()
		# can do anything with Affinity Propagation
		super(AffinityPropagationResult, self).__init__(label, ctxt, model, init_model=init_model)

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
		self._docs_per_topic = None
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
	@property
	def docs_per_topic(self):
		if self._docs_per_topic is None:
			d = {}
			# in LDAModel, it is effectively recalculating the doc score
			# by calling inference(), which is called during the expectation
			# phase of the training
			corpus_scores = self.lda[self.corpus]
			for scores, doc_nm in zip(corpus_scores, self.doc_names):
				#print 'j:', doc_nm, 'i:', i
				# scores will look like (idx, score):
				# (2, 0.13756479927604343),
				# (3, 0.081981422061639414)
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
def get_lda_rslt(label, reload_ctx=False, cls=None):
	global CACHED_MODEL_RSLTS
	if label not in CACHED_MODEL_RSLTS or reload_ctx:
		if cls is None:
			cls = LDAResult
		
		basedir = join(get_models_base_dir(), label)
		if not os.path.exists(basedir):
			pass
		
		CACHED_MODEL_RSLTS[label] = cls.load(label)
	return CACHED_MODEL_RSLTS[label]

def runs_affine_prop(mat):
	from sklearn import cluster, covariance, manifold
	
	print '--- A'
	# EDGES - Learn a graphical structure from the correlations
	#edge_model = covariance.GraphLassoCV(verbose=True, n_jobs=2)
	edge_model = covariance.GraphLassoCV(verbose=True)
	# standardize the time series: using correlations rather than covariance
	# is more efficient for structure recovery
	X = mat.values.copy().T
	X /= X.std(axis=0)
	print '--- B'
	edge_model.fit(X)
	print '--- C'
	
	# Cluster using affinity propagation - just the labels are clustered
	# purely based on correlations in this case?
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

def create_lda_corpus_with_mat(mat):
	class MyCorpus(object):
		def __iter__(self):
			for idx in range(len(mat.index)):
				#doc_nm = mat.index[idx]
				yield mat.ix[idx]
	return MyCorpus()

def _pcibook_init():
	rootdir = helper.get_root_dir()
	if join(rootdir, 'py-external') not in sys.path:
		sys.path.append(join(rootdir, 'py-external'))

def runs_multi_nmf(mat, nruns=5, pc=16, iters=100):
	_pcibook_init()
	from pcibook import nmf
	runs = []
	for _n in range(nruns):
		print 'Start:', get_ts() 
		w,h = nmf.factorize(mat.values, pc=pc, iters=iters)
		print 'End:', get_ts()
		print w.shape, h.shape
		runs.append((w,h))
	return runs

def analyze_factors(mat, runs):
	_pcibook_init()
	from pcibook import nmf
	for n in range(len(runs)):
		w,h = runs[n]
		# top patterns, pattern names
		tps, ptns = nmf.showfeatures(w, h, mat.index, mat.columns, out='output/features_%d.txt'%n)
		nmf.showarticles(mat.index, tps, ptns, out='output/articles_%d.txt'%n)
	# it would be good to be able to get the difference between different runs

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

#def runs_lingo(mat):
#    import carrot_alg_lingo as lingo
#    from org.carrot2.core import ControllerFactory
#    controller = ControllerFactory.createSimple();
#    byTopicClusters = controller.process(documents, "data mining", LingoClusteringAlgorithm.class);
#    final List<Cluster> clustersByTopic = byTopicClusters.getClusters();
