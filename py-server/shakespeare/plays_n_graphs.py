import re
import networkx as nx
import helper
import matplotlib.pyplot as plt
import plays_cfg
import pandas as pd
import numpy as np
from pprint import pformat
from os.path import join
from collections import OrderedDict

_ALL_PLAYS = None
def get_plays_ctx(reload_ctx=False):
    global _ALL_PLAYS
    if _ALL_PLAYS is None or reload_ctx:
        _ALL_PLAYS = ShakespearePlayCtx() 
    return _ALL_PLAYS

class ShakespearePlayCtx:

    def __init__(self):
        self.basedir = join(helper.get_root_dir(), 'data/shakespeare/')
        # Will be a list of play tuples ('[key]', '[Title]')
        self.plays = list(plays_cfg.plays)
        print 'Plays:\n', pformat(self.plays)
        self.play_details = {}
        
        self.map_by_alias = dict(self.plays)
        self.map_by_title = dict([(v,k) for k,v in self.plays])

    def load_play(self, play_alias):
        if play_alias in self.play_details:
            return self.play_details[play_alias]

        title = self.map_by_alias[play_alias]
        play = Play(title)
        play.type = plays_cfg.classifications[title]
        play.year = plays_cfg.vintage[title]
        
        basedir = self.basedir
        toc_file = join(basedir, play_alias, 'index.html')
        print 'Processing', title
        
        # would be good to combine these
        acts_ptn = 'Act (\d+), Scene (\d+): <a href="([^\"]+)">([^>]+)</a>'
        prologue_ptn  = 'Act (\d+), Prologue: <a href="([^\"]+)">([^>]+)</a>'
        # For Taming of the Shrew
        prologue_ptn2 = 'Induction, Scene (\d+): <a href="([^\"]+)">([^>]+)</a>'
        # For Henry IV, Pt 2
        prologue_ptn3 = 'Induction: <a href="([^\"]+)">([^>]+)</a>'
        
        with open(toc_file) as f:
            x = ''.join(f.readlines())
            rs = re.findall(acts_ptn, x)
            for r in rs:
                act, sc, html, loc = r
                play.add_scene(Scene(play, act, sc, loc, html))

            rs = re.findall(prologue_ptn, x)
            for r in rs:
                act, html, loc = r
                play.add_scene(Scene(play, act, '0', loc, html))

            rs = re.findall(prologue_ptn2, x)
            for r in rs:
                sc, html, loc = r
                play.add_scene(Scene(play, '0', sc, loc, html))

            rs = re.findall(prologue_ptn3, x)
            for r in rs:
                html, loc = r
                play.add_scene(Scene(play, '0', '0', loc, html))    
        
        # Remove trailing : from the character's name's as well
        play_dialogue_ptn = '<A NAME=speech\d+><b>([^<]+?):{0,1}</b></a>(.+?(?:</body>|</blockquote>))'
        full_file = toc_file.replace('index.html', 'full.html')
        with open(full_file) as f:
            x = ''.join(f.readlines())
            rs = re.finditer(play_dialogue_ptn, x, re.S)
            for r in rs:
                speaker, dialogue = r.groups()
                
                # FIXME hack to rename characters... 
                speaker = _repl_speakers.get(speaker, speaker)
                
                dialogue = dialogue.replace('<blockquote>',  '')
                dialogue = dialogue.replace('</blockquote>', '')
                dialogue = dialogue.strip()
                lines = dialogue.split('''\n''')
                #print lines
                a_s = re.search('<A NAME=(\d+)\.(\d+)\.', lines[0])
                # some cases with multiple speakers?
                i = 1
                while not a_s:
                    a_s = re.search('<A NAME=(\d+)\.(\d+)\.', lines[i])
                    i += 1
                act, sc = a_s.groups()
                Sc = play.scenes_idx[act+'_'+sc]
                Sc.add_dialogue(speaker, lines)

        self.play_details[play_alias] = _init_graphs(play)
        return self.play_details[play_alias]

# hack to aliases where they are known
_repl_speakers = { 'LEAR' : 'KING LEAR' }

def _init_graphs(play):
    #print self.play.title, '\n\t', self.play.toc_as_str()
    for sc in play.scenes:
        # probably want to make these MultiGraphs
        G = sc.graph = nx.Graph()

        prev_speaker = None
        for speaker, lines in sc.dialogues:
            if speaker not in G.node:
                G.add_node(speaker, nlines=0)
            
            G.node[speaker]['nlines'] += len(lines)
            if prev_speaker:
                edge = G.get_edge_data(prev_speaker, speaker, {})
                w = edge.get('weight', 0)
                G.add_weighted_edges_from([(prev_speaker, speaker, w+1)])
            prev_speaker = speaker
    return play

#from networkx.readwrite.json_graph import node_link_data    
def draw_graph(scene_title, G):
    """
    Draw the graph, the node size will be based on the number of lines squared 
    the character spoke in that scene, indicating their dominance/activity.
    
    The strength of the edges are based on the 
    """
    
    #print 'json repr:', node_link_data(G)
    print 'nlines:', [(n, G.node[n]['nlines']) for n in G]

    plt.title(str(scene_title))
    plt.xticks([])
    plt.yticks([])
    
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.spectral_layout(G)
    
    node_size = [int(G.node[n]['nlines'])**2 for n in G]
    #node_size = [int(G.node[n]['nlines']) for n in G]

    c = 'b'
    nx.draw_networkx_edges(G, pos, edge_color=c, width=1, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c, alpha=0.5)
    # not sure what this is for...
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='k')

    # would be good to have a little "judder" b/n nodes
    # need to fix the label spacing
    pos_lbls = pos.copy()
    if len(pos_lbls) > 1:
        for k in pos_lbls.keys():
            pos_lbls[k] += [0.01, 0.02]
            #pos_lbls[k] += [0.05, 0.03]

    nx.draw_networkx_labels(G, pos_lbls, alpha=0.5)

class Play:
    def __init__(self, title):
        self.title = title
        self.type = '' # Comedy, Tragedy, History
        self.scenes = []
        self.scenes_idx = {}
        self.characters = OrderedDict()
        self.year = None
        self._totalG = None
        #self.html = ''
    @property
    def clean_lines(self):
        lines = []
        # probably should be based on the sequential order
        for name in self.characters:
            c = self.characters[name]
            lines.extend(c.clean_lines)
        return lines

    @property
    def totalG(self):
        if not self._totalG:
            print 'Calculating total G?'
            totalG = nx.Graph()
            for sc in self.scenes:
                # for some reason without the copy the size of the 
                # nodes would blow up. need to see why this is...
                G_ = sc.graph.copy()
                origTG = totalG
                totalG = nx.compose(origTG, G_)

                for speaker in G_.nodes():
                    # only need to do this if the character spoke before. 
                    # otherwise take the value from the new node
                    if speaker in origTG.node:
                        totalG.node[speaker]['nlines'] = \
                            G_.node[speaker]['nlines'] + origTG.node[speaker]['nlines']
                
            self._totalG = totalG
        return self._totalG.copy()

    def add_scene(self, scene):
        self.scenes.append(scene)
        self.scenes_idx[scene.act+'_'+scene.scene] = scene
    def toc_as_str(self):
        return '\n\t'.join(map(str, self.scenes))
    def __repr__(self):
        return '%s\n\t%s' % (self.title, self.toc_as_str())
    def __str__(self):
        return self.title

LINE_RE   = re.compile('^<A NAME=(\d+)\.(\d+)\.(\d+)>(.+)</A><br>$')
ACTION_RE = re.compile('^<p>(.+)</p>$')
def parse_lines(x):
    m = LINE_RE.search(x)
    if m:
        return (m.group(1), m.group(2), m.group(3), m.group(4))
    m = ACTION_RE.search(x)
    if m: 
        return
    return '[NO MATCH}'+x

class Scene:
    def __init__(self, play, act, scene, location, html):
        self.play = play
        self.act = act
        self.scene = scene
        self.location = location
        self.html = html

        self.dialogues = []
        self._cleaned_lines = []
        
        self._mat = None
        
        self.graph = None
        self.graph_img_f = None

    def add_dialogue(self, speaker, lines):
        self.dialogues.append((speaker, lines))
        char = self.play.characters.setdefault(speaker, Character(speaker, self.play))
        char.lines.append(lines)

    def dialogue_summary(self):
        lines = {}
        for d in self.dialogues:
            speaker, dialogue = d
            sp = lines.setdefault(speaker, dict(nlines=0, ntimes=0))
            sp['nlines'] += len(dialogue)
            sp['ntimes'] += 1
            sp['dialogue'] = dialogue
            #print speaker, len(dialogue)
        return lines
    def __repr__(self):
        return 'ACT %s, SC %s. %s' % (self.act, self.scene, self.location)
    
    @property
    def clean_lines(self):
        if not self._cleaned_lines:
            clean_lines = []
            for character, lines in self.dialogues:
                parsed_dialogue_lines = [parse_lines(line) for line in lines]
                dialogue_lines = [ Line(self.play, character, a[0], a[1], a[2], a[3]) 
                                   for a in parsed_dialogue_lines if a ]
                clean_lines.append((character, dialogue_lines))
            self._cleaned_lines = clean_lines
        return self._cleaned_lines

    @property
    def dialogue_matrix(self):
        if self._mat is not None:
            seq_lines = []
            all_chars = self.play.characters.keys()
            for character, lines in self.dialogues:
                idx = all_chars.index(character)
                z = np.zeros(len(all_chars))
                z[idx] = len(lines)
                seq_lines.append(z)
            all_chars = sorted(self.play.characters.keys())
            m = pd.DataFrame(seq_lines, columns=all_chars)
            # Convert to sparse, and also just keep characters who speak
            m = m.to_sparse(0)
            self._mat = m.T[m.sum() > 0].T 
        return self._mat

class Character:
    def __init__(self, name, play):
        self.name = name
        self.play = play
        self.lines = []
        self._cleaned_lines = []

    def __repr__(self):
        return '%s in %s' % (self.name, self.play.title)

    @property
    def clean_lines(self):
        if not self._cleaned_lines:
            parsed_lines = [parse_lines(line) for lines in self.lines for line in lines]
            self._cleaned_lines = [Line(self.play, self.name, a[0], a[1], a[2], a[3]) 
                                   for a in parsed_lines if a]

        return self._cleaned_lines
    
class Line:
    def __init__(self, play, character, act, scene, lineno, line):
        self.play = play
        self.character = character 
        self.act = act
        self.scene = scene 
        self.lineno = lineno 
        self.line = line
    
    direction_re = re.compile('^\[[^\]]+\]')
    
    @property
    def spoken_line(self):
        return self.direction_re.sub('', self.line)
    
    def __repr__(self):
        return '%s (%s,%s,%s)' % (self.line, self.act, self.scene, self.lineno)

def main(title_to_run=''):
    title_to_run = 'King Lear'
    #title_to_run = 'Pericles, Prince of Tyre'
    ctx = get_plays_ctx()
    plays_list = ctx.plays
    for p in plays_list:
        toc_file, title = p
        if not (title == title_to_run):
            continue
        ctx.load_play( toc_file)
    #print pformat(plays)
    for play_graph in ctx.play_details.values():
        play_graph.create_graph()
        #draw_graph_nscenes(play_graph.play)

#def get_plays():
#    info = get_list_of_works()
#    for toc_file, title in info['plays']:
#        load_play(info, toc_file, title)
#    return info['play_details']
#problematic_plays = \
#set([#'Pericles, Prince of Tyre',
#     ])

if (__name__=="__main__"):
    main()
