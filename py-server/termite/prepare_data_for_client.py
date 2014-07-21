#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import argparse
#import ConfigParser
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('termite')

"""
Reformats data necessary for client to run. 

Extracts a subset of the complete term list and term-topic matrix and writes
the subset to a separate file. Also, generates JSON file that merges/packages term
information with the actual term.

Input is term-topic probability distribution and term information, stored in 4 files:
    'term-topic-matrix.txt' contains the entries of the matrix.
    'term-index.txt' contains the terms corresponding to the rows of the matrix.
    'topic-index.txt' contains the topic labels corresponding to the columns of the matrix.
    'term-info.txt' contains information about individual terms.

Output is a subset of terms and matrix, as well as the term subset's information.
Number of files created or copied: 5
	'submatrix-term-index.txt'
    'submatrix-topic-index.txt'
    'submatrix-term-topic.txt'
    'term-info.json'
    'term-info.txt'
"""

# 	def execute( self, model, saliency, seriation ):
# 		
# 		self.logger.info( '--------------------------------------------------------------------------------' )
# 		self.logger.info( 'Preparing data for client...'                                                     )
# 		#self.logger.info( '    data_path = %s', data_path                                                    )
# 		
# 		self.logger.info( 'Connecting to data...' )
# 		self.model = model
# 		self.saliency = saliency
# 		self.seriation = seriation
# 		self.client = Client()
# 		
# # 		self.logger.info( 'Reading data from disk...' )
# # 		self.model.read()
# # 		self.saliency.read()
# # 		self.seriation.read()
# 
# 		self.logger.info( 'Preparing parameters for seriated matrix...' )
# 		self.prepareSeriatedParameters()
# 		
# 		self.logger.info( 'Preparing parameters for filtered matrix...' )
# 		self.prepareFilteredParameters()
# 		
# 		self.logger.info( 'Preparing global term freqs...' )
# 		self.prepareGlobalTermFreqs()
# 		
# # 		self.logger.info( 'Writing data to disk...' )
# # 		self.client.write()

class Client(object):
	def __init__(self, td=None):
		self.seriated_parameters = None
		self.filtered_parameters = None
		self.global_term_freqs = None

	def prepareSeriatedParameters(self, model, seriation, TERM_THRESHOLD=.01):
		topic_index = model.topic_index
		term_index_set = set(model.term_index)

		# This appears to be just the seriated order documented in
		# the Termite paper
		term_ordering = seriation.term_ordering
		term_subindex = [term for term in term_ordering if term in term_index_set]
		
		# this has the full VxT matrix;
		# V - full vocabulary
		#term_topic_matrix = model.term_topic_matrix
		topic_term_matrix = pd.DataFrame(model.term_topic_matrix.T, columns=model.term_index) #
		
# 		print 'topic_term_matrix:\n', topic_term_matrix
# 		term_topic_submatrix = topic_term_matrix[topic_term_matrix>TERM_THRESHOLD] #.T.to_sparse()
# 		print 'term_topic_submatrix:\n', term_topic_submatrix
# 		term_topic_submatrix = term_topic_submatrix.T

		# the highest scoring terms per topic
		topic_term_submatrix = []
		for i in topic_term_matrix.index:
			ti = topic_term_matrix.ix[i]
			v = ti[ti>TERM_THRESHOLD].to_sparse().to_dict()
			v = dict([(k, round(v, 4)) for k,v in v.iteritems()])
			topic_term_submatrix.append(v)

		#term_topic_submatrix = topic_term_submatrix.T
# 		for term in term_ordering:
# 			if term in term_index:
# 				index = term_index.index( term )
# 				# b/c numpy arrays aren't json serializable
# 				term_topic_submatrix.append( [x for x in term_topic_matrix[ index ]] )
# 				term_subindex.append( term )
# 			else:
# 				logger.info( 'ERROR: Term (%s) does not appear in the list of seriated terms', term )
	
		self.seriated_parameters = {
			'termIndex'  : term_subindex,
			'topicIndex' : topic_index,
			#'matrix'     : term_topic_submatrix
			'matrix'     : topic_term_submatrix
		}
	
	def prepareFilteredParameters(self, seriation, saliency):
		term_rank_map = { term: value for value, term in enumerate( seriation.term_iter_index ) }
		term_order_map = { term: value for value, term in enumerate( seriation.term_ordering ) }
		
		term_saliency_map = \
			{ d['term'] : d['saliency'] for d in saliency.term_info }
		term_distinctiveness_map = \
			{ d['term'] : d['distinctiveness'] for d in saliency.term_info }
	
		self.filtered_parameters = {
			# these will only have a handful
			'termRankMap' : term_rank_map,
			'termOrderMap' : term_order_map,
			# the entire corpus of terms
			'termDistinctivenessMap' : term_distinctiveness_map,
			'termSaliencyMap' : term_saliency_map
		}
	
	def prepareGlobalTermFreqs(self, saliency):
# 		td = self.td
# 		topic_index = td.model.topic_index
# 		term_index = td.model.term_index
# 		term_topic_matrix = td.model.term_topic_matrix
# 		term_ordering = td.seriation.term_ordering
# 		term_topic_submatrix = []
# 		term_subindex = []
# 		for term in term_ordering:
# 			if term in term_index:
# 				index = term_index.index( term )
# 				# b/c numpy arrays aren't json serializable
# 				term_topic_submatrix.append( [x for x in term_topic_matrix[ index ]] )
# 				term_subindex.append( term )
# 			else:
# 				logger.info( 'ERROR: Term (%s) does not appear in the list of seriated terms', term )

		if not self.seriated_parameters:
			self.prepareSeriatedParameters()
		term_freqs = { d['term']:d['frequency'] for d in saliency.term_info }
	
		self.global_term_freqs = {
			'termFreqMap' : term_freqs
		}
		self.global_term_freqs.update(self.seriated_parameters)

# def main():
# 	parser = argparse.ArgumentParser( description = 'Prepare data for client.' )
# 	parser.add_argument( 'config_file', type = str, default = None    , help = 'Path of Termite configuration file.' )
# 	parser.add_argument( '--data-path', type = str, dest = 'data_path', help = 'Override data path.'                 )
# 	parser.add_argument( '--logging'  , type = int, dest = 'logging'  , help = 'Override logging level.'             )
# 	args = parser.parse_args()
# 	
# 	args = parser.parse_args()
# 	
# 	data_path = None
# 	logging_level = 20
# 	
# 	# Read in default values from the configuration file
# 	if args.config_file is not None:
# 		config = ConfigParser.RawConfigParser()
# 		config.read( args.config_file )
# 		if config.has_section( 'Termite' ) and config.has_option( 'Termite', 'path' ):
# 			data_path = config.get( 'Termite', 'path' )
# 		if config.has_section( 'Misc' ) and config.has_option( 'Misc', 'logging' ):
# 			logging_level = config.getint( 'Misc', 'logging' )
# 	
# 	# Read in user-specifiec values from the program arguments
# 	if args.data_path is not None:
# 		data_path = args.data_path
# 	if args.logging is not None:
# 		logging_level = args.logging
	#PrepareDataForClient( logging_level ).execute( data_path )

# if __name__ == '__main__':
# 	main()
