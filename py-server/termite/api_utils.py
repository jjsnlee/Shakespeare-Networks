#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import re
# import json
from io_utils import CheckAndMakeDirs
from io_utils import ReadAsList, ReadAsVector, ReadAsMatrix, ReadAsSparseVector, ReadAsSparseMatrix, ReadAsJson
from io_utils import WriteAsList, WriteAsVector, WriteAsMatrix, WriteAsSparseVector, WriteAsSparseMatrix, WriteAsJson, WriteAsTabDelimited
from utf8_utils import UnicodeReader, UnicodeWriter

class Documents( object ):
	ACCEPTABLE_FORMATS = frozenset( [ 'file' ] )
	def __init__( self, _format, path ):
		assert format in Documents.ACCEPTABLE_FORMATS
		self.format = _format
		self.data = {}
	
class Tokens(object):
	def __init__(self):
		self.data = {}

class Model(object):
	def __init__(self):
		self.topic_index = []
		self.term_index = []
		#self.topic_count = 0
		#self.term_count = 0
		self.term_topic_matrix = []
	@property
	def topic_count(self):
		return len(self.topic_index)
	@property
	def term_count(self):
		return len(self.term_index)

class Saliency(object):
	def __init__(self):
		self.term_info = {}
		self.topic_info = {}

class Similarity(object):
	def __init__(self):
		self.document_occurrence = {}
		self.document_cooccurrence = {}
		self.window_occurrence = {}
		self.window_cooccurrence = {}
		self.unigram_counts = {}
		self.bigram_counts = {}
		self.document_g2 = {}
		self.window_g2 = {}
		self.collocation_g2 = {}
		self.combined_g2 = {}

class Seriation(object):
	def __init__(self):
		self.term_ordering = []
		self.term_iter_index = []

class DocumentRWer(object):
	@classmethod
	def read(cls, path):
		#self.data = {}
		docs = Documents()
		filename = path
		with open( filename, 'r' ) as f:
			lines = f.read().decode( 'utf-8', 'ignore' ).splitlines()
			for line in lines:
				docID, docContent = line.split( '\t' )
				docs.data[ docID ] = docContent
		return docs

def dumpObjects(path, tokens, model, saliency, similarity, seriation):
	TokensRWer.write(path, tokens)
	ModelRWer.write(path, model)
	SaliencyRWer.write(path, saliency)
	SimilarityRWer.write(path, similarity)
	SeriationRWer.write(path, seriation)

class TokensRWer(object):
	SUBFOLDER = 'tokens'
	TOKENS = 'tokens.txt'
	@classmethod
	def read(cls, path):
		path = '{}/{}/'.format( path, cls.SUBFOLDER)
		tkns = Tokens()
		tkns.data = {}
		filename = tkns.path + cls.TOKENS
		with open( filename, 'r' ) as f:
			lines = UnicodeReader( f )
			for ( docID, docTokens ) in lines:
				tkns.data[ docID ] = docTokens.split( ' ' )
	
	@classmethod
	def write(cls, path, tokens):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		CheckAndMakeDirs( path )
		filename = path + cls.TOKENS
		with open( filename, 'w' ) as f:
			writer = UnicodeWriter( f )
			for ( docID, docTokens ) in tokens.data.iteritems():
				writer.writerow( [ docID, ' '.join(docTokens) ] )

class ModelRWer(object):
	SUBFOLDER = 'model'
	TOPIC_INDEX = 'topic-index.txt'
	TERM_INDEX = 'term-index.txt'
	TERM_TOPIC_MATRIX = 'term-topic-matrix.txt'
	
	@classmethod
	def read(cls, path):
		path = '{}/{}/'.format(path, cls.SUBFOLDER )
		model = Model()
		model.topic_index = ReadAsList( path + cls.TOPIC_INDEX )
		model.term_index = ReadAsList( path + cls.TERM_INDEX )
		model.term_topic_matrix = ReadAsMatrix( path + cls.TERM_TOPIC_MATRIX )
		cls.verify(model)
		return model
	
	@classmethod
	def verify(cls, model):
		#model.topic_count = len( model.topic_index )
		#model.term_count = len( model.term_index )
		assert model.term_count == len( model.term_topic_matrix )
		for row in model.term_topic_matrix:
			assert model.topic_count == len(row)
	
	@classmethod
	def write(cls, model, path):
		cls.verify(model)
		path = '{}/{}/'.format(path, cls.SUBFOLDER )
		CheckAndMakeDirs( path )
		WriteAsList( model.topic_index, path + cls.TOPIC_INDEX )
		WriteAsList( model.term_index, path + cls.TERM_INDEX )
		WriteAsMatrix( model.term_topic_matrix, path + cls.TERM_TOPIC_MATRIX )
		
class SaliencyRWer(object):
	SUBFOLDER = 'saliency'
	TOPIC_WEIGHTS = 'topic-info.json'
	TOPIC_WEIGHTS_TXT = 'topic-info.txt'
	TOPIC_WEIGHTS_FIELDS = [ 'term', 'saliency', 'frequency', 'distinctiveness', 'rank', 'visibility' ]
	TERM_SALIENCY = 'term-info.json'
	TERM_SALIENCY_TXT = 'term-info.txt'
	TERM_SALIENCY_FIELDS = [ 'topic', 'weight' ]
	@classmethod
	def read( cls, path ):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		saliency = Saliency()
		saliency.term_info = ReadAsJson( path + cls.TERM_SALIENCY )
		saliency.topic_info = ReadAsJson( path + cls.TOPIC_WEIGHTS )
		return saliency
	@classmethod
	def write( cls, saliency, path ):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		CheckAndMakeDirs( path )
		WriteAsJson( saliency.term_info, path + cls.TERM_SALIENCY )
		WriteAsTabDelimited( saliency.term_info, path + cls.TERM_SALIENCY_TXT, cls.TOPIC_WEIGHTS_FIELDS )
		WriteAsJson( saliency.topic_info, path + cls.TOPIC_WEIGHTS )
		WriteAsTabDelimited( saliency.topic_info, path + cls.TOPIC_WEIGHTS_TXT, cls.TERM_SALIENCY_FIELDS )

class SimilarityRWer(object):
	SUBFOLDER = 'similarity'
	DOCUMENT_OCCURRENCE = 'document-occurrence.txt'
	DOCUMENT_COOCCURRENCE = 'document-cooccurrence.txt'
	WINDOW_OCCURRENCE = 'window-occurrence.txt'
	WINDOW_COOCCURRENCE = 'window-cooccurrence.txt'
	UNIGRAM_COUNTS = 'unigram-counts.txt'
	BIGRAM_COUNTS = 'bigram-counts.txt'
	DOCUMENT_G2 = 'document-g2.txt'
	WINDOW_G2 = 'window-g2.txt'
	COLLOCATAPIN_G2 = 'collocation-g2.txt'
	COMBINED_G2 = 'combined-g2.txt'
	@classmethod
	def read(cls, path, read_all=False):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		similarity = Similarity()
		if read_all:
			similarity.document_occurrence = ReadAsSparseVector( path + cls.DOCUMENT_OCCURRENCE )
			similarity.document_cooccurrence = ReadAsSparseMatrix( path + cls.DOCUMENT_COOCCURRENCE )
			similarity.window_occurrence = ReadAsSparseVector( path + cls.WINDOW_OCCURRENCE )
			similarity.window_cooccurrence = ReadAsSparseMatrix( path + cls.WINDOW_COOCCURRENCE )
			similarity.unigram_counts = ReadAsSparseVector( path + cls.UNIGRAM_COUNTS )
			similarity.bigram_counts = ReadAsSparseMatrix( path + cls.BIGRAM_COUNTS )
			similarity.document_g2 = ReadAsSparseMatrix( path + cls.DOCUMENT_G2 )
			similarity.window_g2 = ReadAsSparseMatrix( path + cls.WINDOW_G2 )
			similarity.collocation_g2 = ReadAsSparseMatrix( path + cls.COLLOCATAPIN_G2 )
		similarity.combined_g2 = ReadAsSparseMatrix( path + cls.COMBINED_G2 )
		return similarity
	@classmethod
	def write(cls, similarity, path, write_all=False):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		CheckAndMakeDirs( path )
		if write_all:
			WriteAsSparseVector( similarity.document_occurrence, path + cls.DOCUMENT_OCCURRENCE )
			WriteAsSparseMatrix( similarity.document_cooccurrence, path + cls.DOCUMENT_COOCCURRENCE )
			WriteAsSparseVector( similarity.window_occurrence, path + cls.WINDOW_OCCURRENCE )
			WriteAsSparseMatrix( similarity.window_cooccurrence, path + cls.WINDOW_COOCCURRENCE )
			WriteAsSparseVector( similarity.unigram_counts, path + cls.UNIGRAM_COUNTS )
			WriteAsSparseMatrix( similarity.bigram_counts, path + cls.BIGRAM_COUNTS )
			WriteAsSparseMatrix( similarity.document_g2, path + cls.DOCUMENT_G2 )
			WriteAsSparseMatrix( similarity.window_g2, path + cls.WINDOW_G2 )
			WriteAsSparseMatrix( similarity.collocation_g2, path + cls.COLLOCATAPIN_G2 )
		WriteAsSparseMatrix( similarity.combined_g2, path + cls.COMBINED_G2 )

class SeriationRWer(object):
	SUBFOLDER = 'seriation'
	TERM_ORDERING = 'term-ordering.txt'
	TERM_ITER_INDEX = 'term-iter-index.txt'
	@classmethod	
	def read( cls, path ):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		ser = Seriation()
		ser.term_ordering = ReadAsList( cls.path + cls.TERM_ORDERING )
		ser.term_iter_index = ReadAsList( cls.path + cls.TERM_ITER_INDEX )
		return ser
	@classmethod
	def write( cls, ser, path ):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		CheckAndMakeDirs( path )
		WriteAsList( ser.term_ordering, path + cls.TERM_ORDERING )
		WriteAsList( ser.term_iter_index, path + cls.TERM_ITER_INDEX )

class ClientRWer(object):
	SUBFOLDER = 'public_html'
	SERIATED_PARAMETERS = 'seriated-parameters.json'
	FILTERED_PARAMETERS = 'filtered-parameters.json'
	GLOBAL_TERM_FREQS = 'global-term-freqs.json'
	@classmethod
	def read( cls, path ):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		from prepare_data_for_client import Client
		client = Client()
		client.seriated_parameters = ReadAsJson( path + cls.SERIATED_PARAMETERS )
		client.filtered_parameters = ReadAsJson( path + cls.FILTERED_PARAMETERS )
		client.global_term_freqs = ReadAsJson( path + cls.GLOBAL_TERM_FREQS )
		return client
	@classmethod
	def write( cls, client, path ):
		path = '{}/{}/'.format( path, cls.SUBFOLDER )
		CheckAndMakeDirs( path )
		WriteAsJson( client.seriated_parameters, path + cls.SERIATED_PARAMETERS )
		WriteAsJson( client.filtered_parameters, path + cls.FILTERED_PARAMETERS )
		WriteAsJson( client.global_term_freqs, path + cls.GLOBAL_TERM_FREQS )
