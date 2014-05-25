from unittest import TestCase
#from shakespeare import clusters as sc
from shakespeare.clusters_runner import TermiteData
#import pandas as pd

from termite import ComputeSimilarity, Tokens
from gensim.utils import simple_preprocess

class ShakespeareClustersRunnerTest(TestCase):
    def xxxtest_termite_data(self):
        lda = None
        lda_ctxt = None
        td = TermiteData(lda, lda_ctxt)
    
    def test_similarity(self):
        tkns = Tokens()
        tkns.data = \
        {
         'Doc 1' : 'where is lord helicanus? he can resolve you.',
         'Doc 2' : 'where is lord helicanus? he can resolve you.', 
         'Doc 3' : 'how now! how now! do you hear this? content.',
         'Doc 4' : 'how now! how now! do you hear this? content.'         
        }
        
        tkns.data = dict([(k, simple_preprocess(v)) for k,v in tkns.data.iteritems()])
        
        similarity_calc = ComputeSimilarity()
        similarity_calc.execute(tkns)
        print 'document_occurrence: ', similarity_calc.similarity.document_occurrence

        