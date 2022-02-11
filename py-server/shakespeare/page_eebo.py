# import helper
# from os.path import join
# import logging, traceback, os, json
# logging.basicConfig(level=logging.DEBUG)
# logger = helper.setup_sysout_handler(__name__)
#
# try:
# 	from django.http import HttpResponse
# except:
# 	logger.warn("Couldn't find Django...")
#
# class CorpusDataJsonHandler:
#
# 	@classmethod
# 	def dispatch_map(cls):
# 		return {
# 			'characters'   : cls.handle_chardata,
# 			'topicModels'  : cls.handle_topic_models
# 		}
#
# 	@classmethod
# 	def dispatch(cls, req, play_set):
# 		try:
# 			path_elmts = filter(None, req.path.split('/'))
# 			info = None
# 			if len(path_elmts) > 2:
# 				info = path_elmts[2] # expect format '/shakespeare/corpus/lineCounts'
#
# 			logger.debug('info: %s', info)
# 			play_data_ctx = get_plays_ctx(play_set)
# 			handler = cls.dispatch_map().get(info)
# 			if not handler:
# 				raise 'No handler defined for [%s]' % info
#
# 			rslt_json = handler(play_data_ctx, path_elmts)
# 			return HttpResponse(rslt_json, content_type='application/json')
#
# 		except Exception as e:
# 			# Without the explicit error handling the JSON error gets swallowed
# 			st = traceback.format_exc()
# 			#print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)
# 			logger.error('Problem parsing [%s]:\n%s\n%s', req, e, st)
# 			errmsg = json.dumps({'error' : st})
# 			return HttpResponse(errmsg, content_type='application/json', status=500)
#
# 	@classmethod
# 	def handle_topic_models(cls, play_data_ctx, path_elmts):
# 		"""
# 		    Expected format:
# 		    /shakespeare/corpus/topicModels :
# 		        Get All Topic Models
# 			  /shakespeare/corpus/topicModels/[Topic Model Name]/summary :
# 		        Summary
# 		    /shakespeare/corpus/topicModels/[Topic Model Name]/[Topic #] :
# 		        Specific Topic Info
# 		    /shakespeare/corpus/topicModels/[Topic Model Name]/termite/*.json :
# 		        static termite json info
# 		"""
# 		from batch import clusters_lda
#
# 		MODEL_KEYS = {
# 		  'char-scene-bow-LDA-100-50'    : 'lda-char-scene-bow_2014-06-29_19.49.11_100_50',
# 		  'char-scene-bow-LDA-50-200'    : 'lda-char-scene-bow_2014-08-30_14.32.36_50_200',
#
# 		  'eebo-test' : 'eebo-test'
# 		}
#
# 		lda_group = 'eebo'
#
# 		if len(path_elmts)==3:
# 			# list all the topic models
# 			topics = MODEL_KEYS.keys()
# 			print 'TOPICS:', topics
# 			topics.sort()
# 			topic_json = json.dumps(topics, ensure_ascii=False)
#
# 		else:
# 			# get a specific topic model
# 			topic_model = path_elmts[3]
# 			topic_context = path_elmts[4]
# 			model_key = MODEL_KEYS.get(topic_model)
# 			cls = None
# 			if type(model_key)==tuple:
# 				model_key, cls = model_key
#
# 			#logger.debug('which_topic: %s', which_topic)
#
# 			if topic_context=='summary':
# 				#model_rslt = clusters.get_lda_rslt(model_key, cls=cls)
#
# 				#helper.add_bkpt()
# 				#for topic, vals in model_rslt.docs_per_topic.iteritems():
# 				#	zip(*vals)
# 				# apparently can do this faster with numpy but deal with that later
# 				#import pandas as pd
# 				#arr = [zip(*vals) for vals in model_rslt.docs_per_topic.values()]
#
#
# 				summary_info = \
# 				{
# 					'Doc score median' : 1,
# 					'Top scoring doc' : 1
# 				}
# 				topic_json = json.dumps(summary_info, ensure_ascii=False)
#
# 			elif topic_context=='termite':
# 				json_file = path_elmts[5]
# 				fname = join(clusters_lda.get_models_base_dir(),
# 										 lda_group,
# 				             model_key, 'termite', 'public_html', json_file)
# 				topic_json = open(fname, 'r').read()
#
# 			else:
# 				# The documents in the topic
# 				try:
# 					which_topic = topic_context
# 					logger.debug('which_topic: %s', which_topic)
# 					model_rslt = clusters_lda.get_lda_rslt(lda_group, model_key, cls=cls)
# 					topic_info = {
# 						'docs'    : model_rslt.docs_per_topic[int(which_topic)],
# 						'summary' : {
# 							'Doc score median' : 1,
# 							'Top scoring doc' : 1,
# 							'Most similar topics' : 1
# 						}
# 					}
#
# 				except Exception as e:
# 					topic_info = \
# 					{
# 					'error' : e
# 					}
# 				#logger.debug('topic_info: %s', topic_info)
# 				topic_json = json.dumps(topic_info, ensure_ascii=False)
#
# 		return topic_json
#
# 	@classmethod
# 	def handle_chardata(cls, play_data_ctx, path_elmts):
# 		"""
# 		    Expected format:
# 		        /shakespeare/corpus/characters/[charKey]
# 		"""
# 		char_key = path_elmts[3]
#
# 		char_nm, title = char_key.split(' in ')
# 		logger.debug('title: %s, char_nm: %s', title, char_nm)
#
# 		# need to get play alias by the title
# 		alias = play_data_ctx.map_by_title.get(title)
# 		logger.debug('play alias: %s', alias)
#
# 		play = play_data_ctx.get_play(alias)
# 		# then the character
# 		char = play.characters.get(char_nm)
#
# 		# fix this!!!
# 		if not char:
# 			#char_nm, act, scene = char_nm.split(',')
# 			import re
# 			from plays_n_graphs import Character
# 			CHAR_NM_RE = re.compile('^([^,]+), Act (\d+), Sc (\d+)$')
# 			m = CHAR_NM_RE.match(char_nm)
# 			char_nm, act, sc = m.group(1), m.group(2), m.group(3)
#
# 			char = play.characters.get(char_nm)
# 			char_lines = []
# 			for li in char.clean_lines:
# 				if li.act==act and li.scene==sc:
# 					char_lines.append(li)
# 			artif_char = Character(char_nm, play)
# 			artif_char._cleaned_lines = char_lines
# 			char = artif_char
#
# 		char_lines = []
# 		prev = curr = None
# 		for cl in char.clean_lines:
# 			try:
# 				if prev is None \
# 				        or prev.act!=cl.act \
# 				        or prev.scene!=cl.scene \
# 				        or int(prev.lineno)+1!=int(cl.lineno):
# 					curr = []
# 					char_lines.append(curr)
# 				li = str(cl)
# 				curr.append(li)
#
# 			except Exception as _e:
# 				logger.error('Problem parsing [%s] [%s] [%s], [%s]', char_lines, prev, cl, cl.lineno)
# 				li = '[Problem parsing: [%s] [%s]]' % (cl, cl.lineno)
# 				curr.append(li)
# 				#raise e
# 			prev = cl
#
# 		#print 'char_lines: ', char_lines
# 		#char_lines = char_lines.replace('france', '<yellow>france</yellow>')
#
# 		char_data = \
# 		{
# 		 'character'   : char_nm,
# 		 'play'        : title,
# 		 'doc_name'    : char_key,
# 		 'doc_content' : char_lines #[str(li) for li in char.clean_lines]
# 		}
# 		char_json = json.dumps(char_data, ensure_ascii=False)
# 		return char_json
