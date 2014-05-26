from unittest import TestCase
from shakespeare import clusters as sc
from shakespeare.plays_n_graphs import Line, Character
import pandas as pd
import numpy as np

class ShakespeareClustersTest(TestCase):

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
        prc_ctx = sc.ClustersCtxt(play_ctx)
        prc_ctx.preproc()
        ngdf = sc.process_data(prc_ctx, minlines=1, stopwords=set(['and']), min_df=1)
        
        expected = ['Test Play', 'Test Play 2']
        self.assertTrue(len(ngdf.index)==len(expected) and np.all(np.equal(ngdf.index, expected)), 
                        msg='Unexpected: [%s]' % ngdf.index)

        expected = ['are', 'hello', 'how', 'thou']
        self.assertTrue(len(ngdf.columns)==len(expected) and np.all(np.equal(ngdf.columns, expected)), 
                        msg='Unexpected: [%s]' % ngdf.columns)
        
    def test_lda_context(self):
        doc_nms = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4']
        doc_content = \
        [
         'where is lord helicanus? he can resolve you.',
         'where is lord helicanus? he can resolve you.', 
         'how now! how now! do you hear this? content.',
         'how now! how now! do you hear this? content.'
        ]
        lda_ctx = sc.LDAContext(doc_nms, doc_content)
        print 'lda_ctx:', lda_ctx.corpus

    def test_lda_results(self):
        from gensim.models.ldamodel import LdaModel
        doc_nms = ['Doc 1', 'Doc 2'] 
        doc_contents = ['Here are some contents', 'Another document\'s contents']
        lda_ctxt = sc.LDAContext(doc_nms, doc_contents)
        corpus = lda_ctxt.corpus 
        dictionary = lda_ctxt.dictionary
        lda = LdaModel(corpus, num_topics=2, id2word=dictionary.id2token, passes=2)
        lda_rslt = sc.LDAResult(lda, lda_ctxt)
        print 'docs_per_topic:', lda_rslt.docs_per_topic

    def test_make_matrices(self):
        ngrams = [ 
            { 'hello' : 1, 'bye' : 3, 'test' : 1 },
            { 'hello' : 2, 'ok' : 4, 'test' : 1 },
            { 'hello' : 2, 'ok' : 1 }  
        ]
        docs = ['Doc 1', 'Doc 2', 'Doc 3']
        
        ngdf = pd.DataFrame(ngrams, index=docs)
        
        # words in 70% or less of the docs
        mat = sc.make_matricesXXX(ngdf, max_threshold=.7)
        
        self.assertEquals(['ok', 'test'], mat.columns.tolist())
        self.assertEquals({'ok':2, 'test':2}, mat[mat>0].count().to_dict())
        self.assertEquals({'ok':5, 'test':2}, mat.sum().to_dict())
        
        mat = sc.make_matricesXXX(ngdf, min_cnt=1, max_threshold=.5)
        self.assertEquals({'bye':1}, mat[mat>0].count().to_dict())
        self.assertEquals({'bye':3}, mat.sum().to_dict())
        
        mat = sc.make_matricesXXX(ngdf, min_cnt=3, max_threshold=1.)
        self.assertEquals({'hello':3}, mat[mat>0].count().to_dict())
        self.assertEquals({'hello':5}, mat.sum().to_dict())
        
#        print '\n'
#        mat = sc.make_matrices(cnts, ngdf, max_threshold=.7, min_cnt=0)
#        print 'mat 2:\n', mat
