from shakespeare import clusters as sc
from shakespeare import clusters_termite
from shakespeare.clusters import LDAContext, LDAResult, get_lda_rslt
import plays_n_graphs as png
from os.path import join
#import pandas as pd
import numpy as np

from datetime import datetime
import time
import os
import logging
import helper
logger = helper.setup_sysout_handler(__name__)
logging.basicConfig(level=logging.DEBUG)

from shakespeare.clusters_documents import ShakespeareDocumentsCtxt

# 2014-05-13 00:50:36.652535_50_50.lda
# 2014-06-01 12:55:34.874782_20_50.lda

def main(label=None, train_new=False):
	"""
	maybe:
	    - reduce the sample size by removing characters with x # of lines
	"""
	if train_new:
		lda_rslt = doLDA(label, ntopics=50, npasses=50)
	else:
		#lda_key = '../data/dynamic/lda/2014-05-13 00:50:36.652535_50_50.lda'
		# char_scene_2014-06-29 19.49.11.703618_100_50_lda
		lda_key = '2014-05-13 00:50:36.652535_50_50.lda'
		lda_rslt = get_lda_rslt(lda_key)

	td = clusters_termite.TermiteData(lda_rslt)
	#td.saliency
	#td.similarity
	#td.seriation
	td.data_for_client()
	return td

def doLDA(ntopics=50, npasses=50, ctx='shakespeare', by='Char/Scene', as_bow=True):
	"""
	import shakespeare.clusters_runner as scr
	ldar=scr.doLDA(ntopics=10, npasses=20)
	"""
	
	play_ctx = png.get_plays_ctx(ctx)
	prc_ctx = ShakespeareDocumentsCtxt(play_ctx, by=by)
	#prc_ctx.preproc(by=by) # by='Char'
	doc_titles, docs_content = prc_ctx.get_doc_content()
	
	t = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	baselabel = 'char' if by=='Char' else 'char-scene'
	baselabel += '-bow' if as_bow else '-tfidf'
	
	label = 'lda-%s_%s_%s_%s' % (baselabel, t, ntopics, npasses)
	basedir = join(sc.get_models_base_dir(), label)
	os.makedirs(basedir)
	logfile = join(basedir, 'gensim.log')
	
	# Need this to analyze the perplexity
	logger = logging.getLogger('gensim.models.ldamodel')
	try:
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
	finally:
		logger.removeHandler(fh)
		fh.close()
	
	return lda_rslt

	#sc.print_lda_results(lda, lda_ctxt.corpus, doc_titles)
#     doc_results = lda[corpus]
#     from gensim.models.tfidfmodel import TfidfModel
#     tfidf_model = TfidfModel( )

from clusters import ModelContext
def create_model_ctxt(ctx='shakespeare', by='Char'):
	# by='Char', 'Char/Scene'
	play_ctx = png.get_plays_ctx(ctx)
	prc_ctx = ShakespeareDocumentsCtxt(play_ctx, by=by)
	doc_titles, docs_content = prc_ctx.get_doc_content()
	ctxt = ModelContext(doc_titles, docs_content, stopwds=_get_stopwords())
	return ctxt

def doNMF(ntopics=50, npasses=200):
	from clusters import NMFResult
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
	from clusters import RBMResult
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

def doNMF2(prc_ctx):
	from pcibook import nmf, clusters
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
	_runs = sc.runs_multi_nmf(mat=mat, nruns=5)
	#runs = sc.runs_lda(mat=mat, nruns=5)

if (__name__=="__main__"):
	main()

