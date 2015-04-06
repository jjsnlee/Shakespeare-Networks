import clusters_termite
from clusters_lda import LDAContext, LDAResult, ModelContext, \
	get_lda_rslt, get_models_base_dir
from os.path import join

from datetime import datetime
import time, json, os, logging
import helper
logger = helper.setup_sysout_handler(__name__)
logging.basicConfig(level=logging.DEBUG)

from clusters_documents import ShakespeareDocumentsCtxt, EEBODocumentsCtxt

# 2014-05-13 00:50:36.652535_50_50.lda
# 2014-06-01 12:55:34.874782_20_50.lda

def main(label=None, train_new=False):
	"""
	maybe:
	    - reduce the sample size by removing characters with x # of lines
	"""
	if train_new:
		lda_rslt = doLDA_Plays(label, ntopics=50, npasses=50)
	else:
		#lda_key = '../data/dynamic/lda/2014-05-13 00:50:36.652535_50_50.lda'
		# char_scene_2014-06-29 19.49.11.703618_100_50_lda
		lda_key = '2014-05-13 00:50:36.652535_50_50.lda'
		lda_rslt = get_lda_rslt('plays-shakespeare', lda_key)

	td = clusters_termite.TermiteData(lda_rslt)
	#td.saliency
	#td.similarity
	#td.seriation
	td.data_for_client()
	return td

def create_basedir(group, baselabel, ntopics, npasses):
	t = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
	#baselabel = 'char' if by=='Char' else 'char-scene'
	#baselabel += '-bow' if as_bow else '-tfidf'
	label = '%s/lda-%s_%s_%s_%s' % (group, baselabel, t, ntopics, npasses)
	basedir = join(get_models_base_dir(), label)
	os.makedirs(basedir)
	return basedir, label

def doLDA_EEBO(searchterm, ntopics=50, npasses=50, as_bow=True):
	docs_ctx = EEBODocumentsCtxt(searchterm)
	baselabel = searchterm
	return do_LDA(docs_ctx, 'eebo', baselabel, ntopics, npasses, as_bow=True)

def doLDA_Plays(ntopics=50, npasses=50, ctx='shakespeare', by='Char/Scene', as_bow=True):
	"""
	import batch.clusters_runner as scr
	ldar=scr.doLDA(ntopics=10, npasses=20)
	"""
	import shakespeare.plays_n_graphs as png
	play_ctx = png.get_plays_ctx(ctx)
	prc_ctx = ShakespeareDocumentsCtxt(play_ctx, by=by)
	#prc_ctx.preproc(by=by) # by='Char'
	baselabel = 'char' if by=='Char' else 'char-scene'
	baselabel += '-bow' if as_bow else '-tfidf'
	return do_LDA(prc_ctx, 'plays-shakespeare', baselabel, ntopics, npasses, as_bow=True)

def do_LDA(docs_process_context, group, baselabel, ntopics, npasses, as_bow=True):	
	basedir, label = create_basedir(group, baselabel, ntopics, npasses)
	
	# add a metadata file
	metadata = {
		ntopics : ntopics,
		npasses : npasses,
		group : group,
		label : label
	}
	metadata_json = json.dumps(metadata, ensure_ascii=False)
	with open(join(basedir, 'metadata.json'), 'w') as fh:
		fh.write(metadata_json)

	# Need this to analyze the perplexity
	logfile = join(basedir, 'gensim.log')
	logger = logging.getLogger('gensim.models.ldamodel')
	try:
		fh = logging.FileHandler(logfile)
		fh.setLevel(logging.DEBUG)
		fh.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
		logger.addHandler(fh)
		#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		
		doc_titles, docs_content = docs_process_context.get_doc_content()
		
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

def create_model_ctxt(ctx='shakespeare', by='Char'):
	# by='Char', 'Char/Scene'
	import shakespeare.plays_n_graphs as png
	play_ctx = png.get_plays_ctx(ctx)
	prc_ctx = ShakespeareDocumentsCtxt(play_ctx, by=by)
	doc_titles, docs_content = prc_ctx.get_doc_content()
	ctxt = ModelContext(doc_titles, docs_content, stopwds=_get_stopwords())
	return ctxt

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

if (__name__=="__main__"):
	main()

