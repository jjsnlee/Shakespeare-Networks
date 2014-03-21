from plays_n_graphs import get_plays_ctx, init_play
import os, json, traceback
from operator import itemgetter
from json.encoder import JSONEncoder
import networkx as nx

try:
    from django.http import HttpResponse
except:
    print "Couldn't find Django, maybe ok..."

def view_page(req):
    qry_str = req.REQUEST
    print 'REQUEST:\n', qry_str
    sld_play = qry_str.get('play', '')
    
    force_img_regen = False
    if sld_play:
        force_img_regen = qry_str.get('force_regen', False) == '1'
        if force_img_regen:
            _play_data = init_play(sld_play, 1)
    
    html = _create_html(sld_play)
    return HttpResponse(html)

def _create_html(play):
    data_ctx = get_plays_ctx()
    plays = sorted(data_ctx.plays, key=itemgetter(1))
    print 'play:', play
    all_plays = json.dumps([{'value':p[0],'label':p[1]} for p in plays])
    html = open('shakespeare/page.html', 'r').read()
    html = html.replace('__ALL_PLAYS__', all_plays)
    html = html.replace('__PLAY__', play)
    return html

def get_play(req):
    """ JSON representation for the play """
    try:
        path = req.path
        play = path.split('/')[-1]
        #print 'REQUEST:\n', play
        play = init_play(play, False)
        json_rslt = json.dumps(play, ensure_ascii=False, cls=PlayJSONMetadataEncoder)
        return HttpResponse(json_rslt, content_type='application/json')
    except Exception as e:
        # Without the explicit error handling the JSON error gets swallowed
        st = traceback.format_exc()
        print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)

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
            d = self.from_keys(obj, ('title', 'year'))
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

def main():
    play = 'King Lear'
    #play = "A Midsummer Night's Dream"
    html = _create_html(play, basedir='../', absroot=False, force_img_regen=False, incl_hdr=False)
    #print 'HTML:', html
    tmpfile = 'tmp.html'
    with open(tmpfile, 'w') as fh:
        fh.writelines(html)

    import webbrowser
    lcl_url = 'file://'+os.path.abspath(tmpfile)
    print 'Opening', lcl_url
    webbrowser.open_new_tab(lcl_url)

if (__name__=="__main__"):
    main()
