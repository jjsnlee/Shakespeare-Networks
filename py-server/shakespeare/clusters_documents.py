import json
from os.path import join
import logging
import helper
logger = helper.setup_sysout_handler(__name__)
logging.basicConfig(level=logging.DEBUG)

class DocumentsCtxtI:
	@property
	def documents(self):
		pass

class EEBODocumentsCtxt(object):
	"""
	import shakespeare.clusters_runner as scr
	import shakespeare.clusters_documents as scd
	import shakespeare.clusters as sc
	c=scd.EEBODocumentsCtxt()
	titles,itfn=c.get_doc_content(getmax=10)
	ldac=sc.LDAContext(titles,itfn(),stopwds=scr._get_stopwords())
	ldar=sc.LDAResult('eebo-test', ldac, ntopics=10, npasses=20)
	ldar.termite_data.data_for_client()
	
	#ldar._ModelResult__termite_data=None
	#ldar.save()
	"""
	def __init__(self):
		self.pseudodoc_titles = None
		self.basedir = join(helper.get_root_dir(), 
		                    '../RenaissanceNLP/data/Global Renaissance/raw/')
	@property
	def documents(self):
		fname = join(self.basedir, 'all_entries_summary.json')
		docs_json = json.loads(open(fname, 'r').read())
		return docs_json.values()
	
	def get_doc_content(self, minlines=10, getmax=None):
		docs = self.documents
		docs = docs[:getmax] if getmax else docs 
		
		def include_doc(d):
			return True
		def create_title(d):
			return d['short_title']+''
	
		docs_titles = [create_title(doc) for doc in docs if include_doc(doc)]

		def docs_content_iterator():
			self.pseudodoc_titles = []
			for doc in docs:
				docpath = join(self.basedir, 
				               doc['group_dir'], 
				               doc['short_title']+'_content.json')
				docjson = json.loads(open(docpath, 'r').read())
				if not include_doc(doc):
					continue
				for _i, section in enumerate(docjson):
					section_nm = section['section']
					section_content = ' '.join(section['content'])
					print 'Yielding [%s] - [%s]' % (doc['short_title'], section_nm)
					self.pseudodoc_titles.append('%s - %s'%(doc['short_title'], section_nm))
					yield section_content
	
		return docs_titles, docs_content_iterator

class ShakespeareDocumentsCtxt(object):
	def __init__(self, play_ctx, by='Play'):
		from plays_n_graphs import RootPlayCtx
		assert(isinstance(play_ctx, RootPlayCtx))
		#self.plays = play_ctx.play_details
		self.plays = dict([(p, play_ctx.get_play(p)) for p in play_ctx.map_by_alias.keys()])
		#self.reset()
		self._documents = None # plays, characters, etc
		self.by = by
		# remove documents: scenes/characters with very few lines
		#self.min_lines_per_doc = 10

#     @property
#     def chars_per_play(self):
#         if self._chars_per_play is None:
#             chars_per_play = {}
#             for play_alias in play_ctx.map_by_alias:
#                 p = play_ctx.get_play(play_alias)
#                 chars_per_play[play_alias] = set(p.characters.keys())
#             self._chars_per_play = chars_per_play 
#         return self._chars_per_play

#     def reset(self):
#         self.pruned_characters = {}
#         self.pruned_max_terms = []

	@property
	def documents(self):
		if self._documents is None:
			self._preproc()
		return self._documents
	
	def _preproc(self, plays_to_filter=None):
		"""
		get all the characters, the key should be name and play
		then all their lines, and relationships?
		it could be an interesting game of clustering
		"""
		assert(self.by in ['Play', 'Char', 'Char/Scene'])
		plays = self.plays.values()
		if plays_to_filter:
			plays_to_filter = set(plays_to_filter)
			plays = [k for k in plays if k.title in plays_to_filter]
		#self.reset()
		
		if self.by == 'Play':
			self._documents = plays
		
		elif self.by == 'Char':
			clines = []
			for p in plays:
				clines.extend(p.characters.values())
			self._documents = clines
		
		elif self.by == 'Char/Scene':
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
			self._documents = clines

	def get_doc_content(self, minlines=10):
		doc_titles   = []
		docs_content = []
		for doc in self.documents:
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

# def get_character_names(prc_ctx):
#     #name_d = helper.init_name_dict()
#     all_c_in_play = set()
#     for play_name in prc_ctx.plays.keys():
#         # Only characters in ALL CAPS are considered major, do not 
#         # include minor characters in the list of stopwords.
#         # There may be minor characters in the play
#         # such as "Lord" in Hamlet. Do not want those terms to be removed. 
#         c_in_play = prc_ctx.chars_per_play[play_name]
#         c_in_play = set([c.lower() for c in c_in_play if c.isupper()])
#         for c in c_in_play:
#             v = prc_ctx.pruned_characters.setdefault(c, set())
#             v.add(play_name)
#         all_c_in_play.update(c_in_play)
#     return all_c_in_play

