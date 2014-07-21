from unittest import TestCase
#from shakespeare import clusters as sc
#import pandas as pd
#from shakespeare.clusters import TermiteData
from termite import ComputeSaliency, Model, Client, Seriation
import numpy as np

class ComputeSaliencyTest(TestCase):

    def test_computeTopicInfo(self):
        model = Model()
        # for convenience test arr as TxV, transpose below
        orig_arr = [[.11, .12, .13, .14, .15], [.21, .22, .23, .24, .25]]
        
        # simulating a VxT matrix, where |V|=5, |T|=2
        model.term_topic_matrix = np.array(orig_arr).T
        model.topic_index = ['Topic 1', 'Topic 2']
        
        print 'model.term_topic_matrix.shape:', model.term_topic_matrix.shape
        
        saliency = ComputeSaliency(model)
        saliency.computeTopicInfo()
        
        topic_info = saliency.saliency.topic_info
        print 'topic_info:', topic_info
        
        assert(len(topic_info)==2)
        topic1, topic2 = topic_info
        assert(topic1['topic']=='Topic 1')
        assert(topic2['topic']=='Topic 2')
        assert(topic1['weight']==sum(orig_arr[0]))
        assert(topic2['weight']==sum(orig_arr[1]))

class PrepareDataForClientTest(TestCase):
    def test_prepareSeriatedParameters(self):
        
        model = Model()
        # for convenience test arr as TxV, transpose below
        orig_arr = [                              # should retain:
        [.011, .12, .013, .14, .015], # B=.12, D=.14 
        [.021, .022, .23, .24, .251694932]   # C=.23, D=.24, E=.25
        ]
        
        # simulating a VxT matrix, where |V|=5, |T|=2
        model.term_topic_matrix = np.array(orig_arr).T
        model.topic_index = ['Topic 1', 'Topic 2']
        model.term_index = ['A', 'B', 'C', 'D', 'E']
        
        seriation = Seriation()
        seriation.term_ordering = ['C', 'A']
        
        client = Client()
        client.prepareSeriatedParameters(model, seriation, TERM_THRESHOLD=.1)
        p = client.seriated_parameters
        
        #print 'client.seriated_parameters:', p
        self.assertEquals(p['topicIndex'], ['Topic 1', 'Topic 2'])
        self.assertEquals(p['termIndex'], ['C', 'A'])
        
        matrix = p['matrix']
        print 'matrix:\n', matrix
        #print 'matrix.shape:', matrix.shape
        
        d1 = matrix[0]
        self.assertEquals(d1, {'B':.12, 'D':.14})
        
        d2 = matrix[1]
        self.assertEquals(d2, {'C':.23, 'D':.24, 'E':.2517}) 
