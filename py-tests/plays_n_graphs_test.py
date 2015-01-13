from unittest import TestCase
import networkx as nx

import os
print 'os.path.curdir:', os.path.abspath(os.path.curdir)

from shakespeare.plays_n_graphs import ShakespearePlayCtx, Play, Scene, _init_graphs

nolines = lambda c: ['Line %i' % (i+1) for i in range(c)]

def plot_test_graph(key, graph):
    import matplotlib.pyplot as plt
    from shakespeare.plays_n_graphs import draw_graph
    plt.figure(figsize=(8,5))
    draw_graph(key, graph)
    plt.show()

class PlaysAndGraphsTest(TestCase):

    def test_load_play(self):
        play_ctx = ShakespearePlayCtx()

    def _create_scene1(self, test_play):
        sc = Scene(test_play, 'Act I', 'Scene 1', 'A castle', None)
        sc.add_dialogue('Char 1', nolines(3))
        sc.add_dialogue('Char 2', nolines(50))
        sc.add_dialogue('Char 1', nolines(3))
        for _i in range(20):
            sc.add_dialogue('Char 3', nolines(3))
            sc.add_dialogue('Char 4', nolines(3))
        test_play.add_scene(sc)
        return sc
        
    def _create_scene2(self, test_play):
        sc = Scene(test_play, 'Act II', 'Scene 2', 'Another room', None)
        sc.add_dialogue('Char 2', nolines(20))
        sc.add_dialogue('Char 1', nolines(10))
        sc.add_dialogue('Char 5', nolines(30))
        test_play.add_scene(sc)
        return sc

    def test_init_graphs(self):
        test_play = Play('Test Play')
        scene1  = self._create_scene1(test_play)
        _scene2 = self._create_scene2(test_play)

        # this actually needs to be initialized
        _init_graphs(test_play)
        
        #print 'sc.graph:', sc.graph
        G = test_play.scenes_idx[scene1.act+'_'+scene1.scene].graph
        #G = test_play.scenes_idx[0].G
        self.assertEqual(G, scene1.graph, 'Make sure both graph instances reference the same object')
        #print nx.connected_components(G)
        
        print G.degree(weight='weight')
        chars = G.nodes()
        self.assertEqual(set(['Char 1', 'Char 2', 'Char 3', 'Char 4']), set(chars))
        self.assertEqual(G.node['Char 1']['nlines'], 6)
        self.assertEqual(G.node['Char 2']['nlines'], 50)
        self.assertEqual(G.node['Char 3']['nlines'], 3*20)
        self.assertEqual(G.node['Char 4']['nlines'], 3*20)

        cnxs = nx.degree(G)
        self.assertEqual(cnxs['Char 1'], 2)
        self.assertEqual(cnxs['Char 2'], 1)
        self.assertEqual(cnxs['Char 3'], 2)
        self.assertEqual(cnxs['Char 4'], 1)

        edges = G.edges(data=True)
        print 'edges:', edges
        print 'Char 1 edges:', G.edges('Char 1', data=True)
        
        #plot_test_graph(str(scene), scene.graph)

        tG = test_play.totalG
        print tG
        
        chars = tG.nodes()
        self.assertEqual(set(['Char 1', 'Char 2', 'Char 3', 'Char 4', 'Char 5']), set(chars))
        self.assertEqual(tG.node['Char 1']['nlines'], 6  + 10)
        self.assertEqual(tG.node['Char 2']['nlines'], 50 + 20)
        self.assertEqual(tG.node['Char 3']['nlines'], 3*20)
        self.assertEqual(tG.node['Char 4']['nlines'], 3*20)
        self.assertEqual(tG.node['Char 5']['nlines'], 30)
        #plot_test_graph('totalG', tG)
        
        tCnxs = nx.degree(tG)
        print 'degree:', tCnxs 
        self.assertEqual(tCnxs['Char 1'], 3)
        self.assertEqual(tCnxs['Char 2'], 1)
        self.assertEqual(tCnxs['Char 3'], 2)
        self.assertEqual(tCnxs['Char 4'], 1)
        self.assertEqual(tCnxs['Char 5'], 1)
        
#        # New Scenario
#        test_play = Play('Test Play')
#        scene = self._create_scene1(test_play)
#        test_play.add_scene(scene)
#        scene = self._create_scene2(test_play)
#        test_play.add_scene(scene)
#        png = ShakespearePlayGraph(test_play)
        
        