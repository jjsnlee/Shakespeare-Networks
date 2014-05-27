import helper
from os.path import join
from json.encoder import JSONEncoder
import networkx as nx

def transform(play_html):
    from BeautifulSoup import BeautifulSoup
    soup = BeautifulSoup(play_html)
    toc_content = soup.findAll('p', {'class':'toc'})
    for toc in toc_content:
        links = toc.findAll('a')
        if links:
            for link in links:
                print link.text, link['href']
    #print toc_content
    
    content = soup.findAll('pre', {'xml:space':'preserve'})
    print content

DYNAMIC_ASSETS_BASEDIR = helper.get_dynamic_rootdir()

def generate_shakespeare_files(gen_imgs=False, gen_md=False, gen_lines=False, limit_plays=[]):
    """
    gen_imgs  : generate png images 
    gen_md    : generate the metadata as json
    gen_lines : generate the content (of the plays) as json
    limit_plays : subset of plays to generate files
    """

    from plays_n_graphs import get_plays_ctx, init_play_imgs
    import json
    data_ctx = get_plays_ctx('shakespeare')
    plays = data_ctx.plays
    #play_set = 'shakespeare'
    
    basedir = DYNAMIC_ASSETS_BASEDIR
    helper.ensure_path(join(basedir, 'json'))
    limit_plays = set(limit_plays)
    for play_alias, _ in plays:
        if limit_plays and play_alias not in limit_plays:
            continue
        print 'Processing play:', play_alias
        play = data_ctx.get_play(play_alias)
        if gen_md or gen_lines:
            #play = init_play(play_set, play_alias, False)
            
            if gen_md:
                json_rslt = json.dumps(play, ensure_ascii=False,
                                       cls=PlayJSONMetadataEncoder, indent=True)
                fname = join(basedir, 'json', play_alias+'_metadata.json') 
                with open(fname, 'w') as fh:
                    fh.write(json_rslt)

            if gen_lines:
                json_rslt = json.dumps(play, ensure_ascii=False, 
                                       cls=PlayJSONContentEncoder, indent=True)
                fname = join(basedir, 'json', play_alias+'_content.json') 
                with open(fname, 'w') as fh:
                    fh.write(json_rslt)
        
        if gen_imgs:
            init_play_imgs(play, play_alias, True)

class PlayEncoderBase(JSONEncoder):
    def from_keys(self, obj, keys):
        return dict([(k,getattr(obj,k)) for k in keys])
    def get_acts_from_play(self, play):
        acts = []
        curr_act = curr_scenes = None
        for scene in play.scenes:
            if curr_act != scene.act:
                curr_act = scene.act
                curr_scenes = []
                acts.append(curr_scenes)
            curr_scenes.append(scene)
        return acts

class PlayJSONContentEncoder(PlayEncoderBase):
    def default(self, obj):
        from plays_n_graphs import Play, Scene, Character, Line
        d = {}        
        if isinstance(obj, Play):
            d['acts'] = self.get_acts_from_play(obj)
        elif isinstance(obj, Scene):
            d = self.from_keys(obj, ('act', 'scene', 'location'))
            d['lines'] = obj.clean_lines
        elif isinstance(obj, Character):
            d = self.from_keys(obj, ('name', 'clean_lines'))
        elif isinstance(obj, Line):
            #d = self.from_keys(obj, ('lineno', 'line'))
            d = repr(obj)
        return d

class PlayJSONMetadataEncoder(PlayEncoderBase):
    def default(self, obj):
        from plays_n_graphs import Play, Scene

        if isinstance(obj, Play):
            d = self.from_keys(obj, ('title', 'year', 'type'))
            d['acts'] = self.get_acts_from_play(obj)
            d['characters'] = characters = obj.characters.keys()
            d['char_data'] = char_data = {}
            
            playG = obj.totalG
            cnxs = nx.degree(playG)
            for idx, c in enumerate(characters):
                nlines = playG.node[c]['nlines']
                ratio = 0
                if cnxs[c] > 0:
                    ratio = round(float(nlines) / cnxs[c], 2)
                
                char_data[c] = \
                {
                 'nlines'    : nlines,
                 'nedges'    : cnxs[c],
                 'ratio'     : ratio,
                 'order_app' : idx
                }
            
        elif isinstance(obj, Scene):
            d = self.from_keys(obj, ('act', 'scene', 'location', 'graph_img_f'))
            d['char_data'] = char_data = {}
            
            #dmat = obj.dialogue_matrix
            #d['dialogue_matrix'] = dmat
            # skip this for now...
            #d['lines'] = obj.clean_lines
            
            # May want to extract this so that we only do this when 
            # necessary for plays
            sceneG = obj.graph
            cnxs = nx.degree(sceneG)
            for c in sceneG.node:
                nlines = sceneG.node[c]['nlines']
                ratio = 0
                if cnxs[c] > 0:
                    ratio = round(float(nlines) / cnxs[c], 2)
                char_data[c] = \
                {
                 'nlines'     : nlines,
                 'nedges'     : cnxs[c],
                 'ratio'      : ratio
                }
            
        else:
            d = obj.__dict__.copy()
        
        return d

def main_shakespeare_batch():
    generate_shakespeare_files()

def main_gutenberg():
    rootdir = helper.get_root_dir()
    fname = join(rootdir, 'data/marlowe', 'tamburlaine_1.html')
    with open(fname, 'r') as fh:
        play_content = fh.read()
    transform(play_content)

if (__name__=="__main__"):
    #main_gutenberg()
    main_shakespeare_batch()
