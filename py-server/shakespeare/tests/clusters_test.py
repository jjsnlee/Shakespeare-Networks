from unittest import TestCase
from shakespeare import clusters as sc
import pandas as pd

class ShakespeareClustersTest(TestCase):

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
#        from gensim.models.ldamodel import LdaModel
        doc_nms = ['Doc 1', 'Doc 2'] 
        doc_contents = ['Here are some contents', 'Another document\'s contents']
        lda_ctxt = sc.LDAContext(doc_nms, doc_contents)
#         corpus = lda_ctxt.corpus 
#         dictionary = lda_ctxt.dictionary
        #lda = LdaModel(corpus, num_topics=2, id2word=dictionary.id2token, passes=2)
        lda_rslt = sc.LDAResult('Some label', lda_ctxt, 2, 2)
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
