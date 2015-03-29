from clusters_lda import ModelResult, get_models_base_dir, get_ts
from os.path import join

from clusters_runner import create_model_ctxt
import numpy as np
from datetime import datetime
import time, sys
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

from sklearn.externals import joblib

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
