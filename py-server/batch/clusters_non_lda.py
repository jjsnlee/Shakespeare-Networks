from batch.clusters import ModelResult, get_models_base_dir
from sklearn.externals import joblib
from os.path import join

from clusters_runner import create_model_ctxt
import numpy as np
from datetime import datetime
import time
import logging
import helper
logger = helper.setup_sysout_handler(__name__)
logging.basicConfig(level=logging.DEBUG)

def doNMF(ntopics=50, npasses=200):
	from batch.clusters import NMFResult
	model_ctxt = create_model_ctxt(by='Char')
	t = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	label = 'nmf-char-%s-%s-%s' % (t, ntopics, npasses)
	print 'Label:', label
	print 'Started', t
	model_rslt = NMFResult(label, model_ctxt, ntopics=ntopics, npasses=npasses)
	ended = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	print 'Completed:', ended
	return model_rslt

def doDBScan(by='Char'):
	from sklearn import metrics, cluster
	model_ctxt = create_model_ctxt(by=by)
	X = model_ctxt.corpus
	for r in np.arange(2, 20, 5):
		#for n in range(5, 21, 15):
		st = time.time()
		db = cluster.DBSCAN(eps=r,
		                    #min_samples=n,
		                    algorithm='kd_tree')
		db.fit(X)
		end = time.time()
		
		labels = db.labels_
		silh_score = metrics.silhouette_score(X, labels, metric='euclidean')
		print 'r: %s, sc: %s, took %s secs'  % (r, silh_score, end-st)

def doRBM(ntopics=50, npasses=200, verbose=True):
	from batch.clusters import RBMResult
	model_ctxt = create_model_ctxt(by='Char')
	t = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	label = 'rbm-char-%s-%s-%s' % (t, ntopics, npasses)
	print 'Label:', label
	print 'Started', t
	model_rslt = RBMResult(label, model_ctxt, 
	                       ntopics=ntopics, 
	                       npasses=npasses, 
	                       verbose=verbose)
	ended = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	print 'Completed:', ended
	return model_rslt

def doAffProp(ctx='shakespeare', by='Char/Scene', ):
	#from clusters import AffinityPropagationResult
	pass

# def doNMF2(prc_ctx):
# 	from pcibook import nmf, clusters
# 	#-- NMF
# 	mat = process_data(prc_ctx, max_df=.8) # ngram data frame
# 	#mat = sc.process_data(prc_ctx, max_df=.8, raw=True) # ngram data frame
# 	clust = clusters.hcluster(mat.values)
# 	# some error here
# 	clusters.drawdendrogram(clust, map(str, mat.index), jpeg='shakespeare.jpg')
# 	#rdata = clusters.rotatematrix(mat)
# 	#wordclust = clusters.hcluster(rdata)
# 	#w,h = nmf.factorize(a*b, pc=3, iters=100)
# 	#-- 
# 	w,h = nmf.factorize(mat.values, pc=16, iters=100)
# 	tps, ptns = nmf.showfeatures(w, h, mat.index, mat.columns)
# 	nmf.showarticles(mat.index, tps, ptns)
# 	#-- 
# 	_runs = sc.runs_multi_nmf(mat=mat, nruns=5)
# 	#runs = sc.runs_lda(mat=mat, nruns=5)

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
			print 'corpus_scores:', corpus_scores
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

