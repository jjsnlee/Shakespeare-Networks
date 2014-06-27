from unittest import TestCase
#from shakespeare import clusters as sc
from shakespeare.clusters import TermiteData
from shakespeare.clusters_runner import ClustersCtxt, process_data
from shakespeare.plays_n_graphs import Line, Character
import numpy as np
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

    def test_process_data(self):
        from shakespeare.plays_n_graphs import RootPlayCtx
        class TestPlayCtx(RootPlayCtx):
            def _load_play(self, play, play_alias):
                char1 = Character('Test Char', play.title)
                char1.lines = [
                    Line('Test Play', 'Test Char', 1, 1, 1, 'Hello and how are thou?')
                ]
                char1._cleaned_lines = char1.lines
                play.characters = { char1.name : char1 }
        
        datadir = 'somefolder' 
        cfg = {
            'plays' : [('test_play', 'Test Play'), ('test_play2', 'Test Play 2')],
            'classifications' : None,
            'vintage' : None 
        }
        play_ctx = TestPlayCtx(datadir, cfg)
        play_ctx.do_init_graphs = False
        prc_ctx = ClustersCtxt(play_ctx)
        prc_ctx.preproc()
        ngdf = process_data(prc_ctx, minlines=1, stopwords=set(['and']), min_df=1)
        
        expected = ['Test Play', 'Test Play 2']
        self.assertTrue(len(ngdf.index)==len(expected) and np.all(np.equal(ngdf.index, expected)), 
                        msg='Unexpected: [%s]' % ngdf.index)

        expected = ['are', 'hello', 'how', 'thou']
        self.assertTrue(len(ngdf.columns)==len(expected) and np.all(np.equal(ngdf.columns, expected)), 
                        msg='Unexpected: [%s]' % ngdf.columns)
        
        