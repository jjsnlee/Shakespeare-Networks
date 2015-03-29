from unittest import TestCase
import numpy as np
#import pandas as pd

from batch.clusters_lda import ModelContext
from batch.clusters_termite import TermiteData
from batch.clusters_documents import ShakespeareDocumentsCtxt
from shakespeare.plays_n_graphs import Line, Character

from batch.termite import ComputeSimilarity, Tokens
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

    def create_test_play(self, data_cllbk=None):
        from shakespeare.plays_n_graphs import RootPlayCtx
        if not data_cllbk:
            def data_cllbk(play, play_alias):
                char1 = Character('Test Char', play.title)
                char1.lines = [
                    Line('Test Play', 'Test Char', 1, 1, 1, 'Hello and how are thou?')
                ]
                char1._cleaned_lines = char1.lines
                play.characters = { char1.name : char1 } 
        
        class TestPlayCtx(RootPlayCtx):
            def _load_play(self, play, play_alias):
                data_cllbk(play, play_alias)

        datadir = 'somefolder' 
        cfg = {
            'plays' : [('test_play', 'Test Play'), ('test_play2', 'Test Play 2')],
            'classifications' : None,
            'vintage' : None 
        }
        play_ctx = TestPlayCtx(datadir, cfg)
        play_ctx.do_init_graphs = False
        return play_ctx

    def test_process_data(self):
        play_ctx = self.create_test_play()
        prc_ctx = ShakespeareDocumentsCtxt(play_ctx)
        doc_titles, docs_content = prc_ctx.get_doc_content(minlines=1)
        model_ctx = ModelContext(doc_titles, docs_content, stopwds=['and'],
                                 min_df=1)
        ngdf = model_ctx.corpus
        
        expected = ['Test Play', 'Test Play 2']
        self.assertTrue(len(ngdf.index)==len(expected) 
                        and np.all(np.equal(ngdf.index, expected)), 
                        msg='Unexpected: [%s]' % ngdf.index)

        expected = ['are', 'hello', 'how', 'thou']
        self.assertTrue(len(ngdf.columns)==len(expected) 
                        and np.all(np.equal(ngdf.columns, expected)), 
                        msg='Unexpected: [%s]' % ngdf.columns)
    
    def test_preproc(self):
        def data_cllbk(play, play_alias):
            if play_alias != 'test_play':
                return
            print 'play.title:', play.title
            char1 = Character('Char A', play)
            char1.lines = [
                Line(play, 'Char A', 1, 1, 1, 'Hello and how are thou?'),
                Line(play, 'Char A', 1, 1, 2, 'Where doest thou come from...'),
                Line(play, 'Char A', 1, 1, 4, 'Yes, bon chance'),
                Line(play, 'Char A', 1, 2, 10, 'Some more lines from me'),
                Line(play, 'Char A', 3, 1, 100, 'Finally, here\'s what i have to say!'),
            ]
            char1._cleaned_lines = char1.lines
            
            char2 = Character('Char B', play)
            char2.lines = [
                Line(play, 'Char B', 1, 1, 3, 'I am responding to thee now'),
            ]
            char2._cleaned_lines = char2.lines

            play.characters = { 
                char1.name : char1,
                char2.name : char2,
            } 
        
        play_ctx = self.create_test_play(data_cllbk=data_cllbk)
        
        prc_ctx = ShakespeareDocumentsCtxt(play_ctx, by='Char')
        docs = set([repr(d) for d in prc_ctx.documents])
        self.assertEquals(docs, set(['Char A in Test Play', 'Char B in Test Play']))
        
        prc_ctx = ShakespeareDocumentsCtxt(play_ctx, by='Char/Scene')
        docs = set([repr(d) for d in prc_ctx.documents])
        
        expected = set([
            'Char A, Act 1, Sc 1 in Test Play',
            'Char A, Act 1, Sc 2 in Test Play',
            'Char A, Act 3, Sc 1 in Test Play', 
            'Char B, Act 1, Sc 1 in Test Play'
        ])
        
        self.assertEquals(docs, expected)
        
        doc_titles, docs_content = prc_ctx.get_doc_content(minlines=1)
        model_ctx = ModelContext(doc_titles, docs_content,
                                 min_df=1)
        ngdf = model_ctx.corpus
        print ngdf

        actual_docs = set(ngdf.index.tolist())
        self.assertTrue(len(ngdf.index)==len(expected) 
                        and np.all(np.equal(actual_docs, expected)),
                        msg='Unexpected: [%s]' % ngdf.index)
