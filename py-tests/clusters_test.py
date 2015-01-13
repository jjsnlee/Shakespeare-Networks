from unittest import TestCase
import pandas as pd
from shakespeare import clusters as sc

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
        lda_ctx = sc.LDAContext(doc_nms, doc_content, terms_min=2)
        print 'lda_ctx:', lda_ctx.corpus
        self.assertEquals(len(lda_ctx.corpus), 4)
        self.assertEquals(len(lda_ctx.dictionary.token2id), 13)
        # You is removed b/c it is in every sentence in the corpus 
        self.assertTrue('you' not in lda_ctx.dictionary.token2id.keys())
        self.assertEquals(set(lda_ctx.dictionary.token2id.keys()), 
                          set('where is lord helicanus he can resolve how now do hear this content'.split()))

    def test_lda_results(self):
        doc_nms = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5', 'Doc 6'] 
        doc_contents = ['Here are some contents', 'Here are some contents', 
                        'Another document\'s contents', 'Another document\'s contents', 
                        'One more set of words.', 'One more set of words.']
        lda_ctx = sc.LDAContext(doc_nms, doc_contents, terms_min=1)
        self.assertEquals(len(lda_ctx.dictionary.token2id), 10)
        #lda = LdaModel(corpus, num_topics=2, id2word=dictionary.id2token, passes=2)
        lda_rslt = sc.LDAResult('Some label', lda_ctx, ntopics=2, npasses=2)
        print 'docs_per_topic:', lda_rslt.docs_per_topic
        df = lda_rslt.as_dataframe()
        self.assertEquals(df.shape, (2, 10))

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
