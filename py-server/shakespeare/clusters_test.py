from unittest import TestCase
from shakespeare import clusters as sc
from plays_n_graphs import Line, Play, Character
import pandas as pd

class ShakespeareClustersTest(TestCase):

    def test_process_data(self):
        tplay = Play('Test Play')
        char1 = Character('Test Char', 'Test Play')
        char1.lines = [
            Line('Test Play', 'Test Char', 1, 1, 1, 'Hello and how are thou?')
        ]
        char1._cleaned_lines = char1.lines
        tplay.characters = { char1.name : char1 }

        x = { tplay.title : tplay }
        ngdf = sc.process_data(x, minlines=0)
        
        #print cnts

        self.assertEquals(ngdf.index, ['Test Play'])
        #self.assertEquals(cnts.to_dict(), {'hello' : 1})

    def test_make_matrices(self):
        ngrams = [ 
            { 'hello' : 1, 'bye' : 3, 'test' : 1 },
            { 'hello' : 2, 'ok' : 4, 'test' : 1 },
            { 'hello' : 2, 'ok' : 1 }  
        ]
        docs = ['Doc 1', 'Doc 2', 'Doc 3']
        
        ngdf = pd.DataFrame(ngrams, index=docs)
        
        # words in 70% or less of the docs
        mat = sc.make_matrices(ngdf, max_threshold=.7)
        
        self.assertEquals(['ok', 'test'], mat.columns.tolist())
        self.assertEquals({'ok':2, 'test':2}, mat[mat>0].count().to_dict())
        self.assertEquals({'ok':5, 'test':2}, mat.sum().to_dict())
        
        mat = sc.make_matrices(ngdf, min_cnt=1, max_threshold=.5)
        self.assertEquals({'bye':1}, mat[mat>0].count().to_dict())
        self.assertEquals({'bye':3}, mat.sum().to_dict())
        
        mat = sc.make_matrices(ngdf, min_cnt=3, max_threshold=1.)
        self.assertEquals({'hello':3}, mat[mat>0].count().to_dict())
        self.assertEquals({'hello':5}, mat.sum().to_dict())
        
#        print '\n'
#        mat = sc.make_matrices(cnts, ngdf, max_threshold=.7, min_cnt=0)
#        print 'mat 2:\n', mat
