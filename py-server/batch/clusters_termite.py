from os.path import join

from termite import Model, Tokens, ComputeSaliency, ComputeSimilarity, ComputeSeriation, \
    ClientRWer, SaliencyRWer, SimilarityRWer, SeriationRWer

from batch.clusters import get_models_base_dir

class TermiteData(object):
	def __init__(self, model_rslt, from_cache=False):
		#assert(isinstance(lda_rslt, LDAResult))
		model_ctxt = model_rslt.model_ctxt
		
		self.model_ctxt = model_ctxt
		self.basepath = join(get_models_base_dir(), model_rslt.label, 'termite')
		
		model = Model()
		model.term_topic_matrix = model_rslt.term_topic_matrix.T
		model.topic_index = map(lambda n: 'Topic %d' % (n+1), range(model_rslt.num_topics))
		model.term_index = model_ctxt.get_terms()
		self.model = model
		
		tokens = Tokens()
		tokens.data = dict(self.docs_tokenized_zipped())
		#tokens.data = self.model_ctxt.doc_contents_tokenized
		self.tokens = tokens
		
		self._saliency = None
		self._similarity = None
		self._seriation = None
	
	def docs_tokenized_zipped(self):
		return [(t, c) for t,c in zip(self.model_ctxt.doc_names, 
									  self.model_ctxt.doc_contents_tokenized)]
	
# 		doc_contents_tokenized = [simple_preprocess(doc) for doc in self.model_ctxt.doc_contents]
# 		print 'doc_contents_tokenized:', self.model_ctxt.doc_contents_tokenized
# 		doc_contents_tokenized = []
# 		for doc in self.model_ctxt.doc_contents:
# 			print 'doc:', doc
# 			doc_contents_tokenized.append(simple_preprocess(doc))
# 		return [(t, c) for t,c in zip(self.model_ctxt.doc_names, doc_contents_tokenized)]

	@property
	def saliency(self):
		if self._saliency is None:
			saliency_calc = ComputeSaliency(self.model)
			saliency_calc.execute()
			self._saliency = saliency_calc.saliency
			SaliencyRWer.write(self._saliency, self.basepath) 
		return self._saliency

	@property
	def similarity(self):
		if self._similarity is None:
			similarity_calc = ComputeSimilarity()
			similarity_calc.execute(self.tokens)
			self._similarity = similarity_calc.similarity
			SimilarityRWer.write(self._similarity, self.basepath)
		return self._similarity

	@property
	def seriation(self):
		if self._seriation is None:
			seriation_calc = ComputeSeriation()
			seriation_calc.execute(self.saliency, self.similarity)
			self._seriation = seriation_calc.seriation
			SeriationRWer.write(self._seriation, self.basepath)
		return self._seriation

	def data_for_client(self):
		import helper
		helper.add_bkpt()

		from termite.prepare_data_for_client import Client 
		client = Client()
		client.prepareSeriatedParameters(self.model, self.seriation)
		client.prepareFilteredParameters(self.seriation, self.saliency)
		client.prepareGlobalTermFreqs(self.saliency)
		ClientRWer.write(client, self.basepath)
		return client

	def load(self, which):
		if which=='saliency':
			self._saliency = SaliencyRWer.read(self.basepath)
		elif which=='seriation':
			self._seriation = SeriationRWer.read(self.basepath)
		elif which=='similarity':
			self._similarity = SimilarityRWer.read(self.basepath)
