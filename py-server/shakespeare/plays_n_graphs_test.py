from unittest import TestCase
from plays_n_graphs import Play, Scene, ShakespearePlayGraph

class PlaysAndGraphsTest(TestCase):

    def test_process_data(self):
        tplay = Play('Test Play')
        sc = Scene(tplay, 'Act I', 'Scene I', 'A castle', None)
        
        nolines = lambda c: ['Line %i' % (i+1) for i in range(c)]

        sc.add_dialogue('Char 1', nolines(3))
        sc.add_dialogue('Char 2', nolines(50))
        sc.add_dialogue('Char 1', nolines(3))
        
        for _i in range(20):
            sc.add_dialogue('Char 3', nolines(3))
            sc.add_dialogue('Char 4', nolines(3))
        
        tplay.add_scene(sc)
        
        png = ShakespearePlayGraph(tplay)
        png.create_graph()
        
        #print 'sc.graph:', sc.graph
        G = sc.graph
        #print nx.degree(G)
        #print nx.connected_components(G)
        
        print G.degree(weight='weight')
        
        chars = G.nodes()
        self.assertEquals(set(['Char 1', 'Char 2', 'Char 3', 'Char 4']), set(chars))
        
        edges = G.edges(data=True)
        
        #print 'chars:', chars
        print 'edges:', edges
        
        print 'Char 1 edges:', G.edges('Char 1', data=True)

        import matplotlib.pyplot as plt
        from plays_n_graphs import draw_graph
        plt.figure(figsize=(8,5))
        draw_graph(str(sc), sc.graph)
        plt.show()
        