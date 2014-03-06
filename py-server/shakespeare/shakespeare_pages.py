from plays_n_graphs import get_plays_ctx, draw_graph
import matplotlib.pyplot as plt
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
        json_rslt = json.dumps(play, ensure_ascii=False, cls=PlayJSONEncoder)
        return HttpResponse(json_rslt, content_type='application/json')
    except Exception as e:
        # Without the explicit error handling the JSON error gets swallowed
        st = traceback.format_exc()
        print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)

class PlayJSONEncoder(JSONEncoder):

    def from_keys(self, obj, keys):
        return dict([(k,getattr(obj,k)) for k in keys])
    
    def default(self, obj):
        from plays_n_graphs import Play, Scene, Character, Line

        if isinstance(obj, Play):
            d = self.from_keys(obj, ('title', 'year'))
            acts = []
            curr_act = curr_scenes = None
            for scene in obj.scenes:
                if curr_act != scene.act:
                    curr_act = scene.act
                    curr_scenes = []
                    acts.append(curr_scenes)
                curr_scenes.append(scene)
            d['acts'] = acts
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
                 'nlines'     : nlines,
                 'nedges'     : cnxs[c],
                 'ratio'      : ratio,
                 'appearance' : idx
                }
            
        elif isinstance(obj, Scene):
            d = self.from_keys(obj, ('act', 'scene', 'location', 'graph_img_f'))
            d['char_data'] = char_data = {}
            
            dmat = obj.dialogue_matrix
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
            
        elif isinstance(obj, Character):
            d = self.from_keys(obj, ('name', 'clean_lines'))
        
        elif isinstance(obj, Line):
            #d = self.from_keys(obj, ('lineno', 'line'))
            d = repr(obj)

        else:
            d = obj.__dict__.copy()
        
        return d

def init_play(play_name, force_img_regen, basedir=''):
    play_data_ctx = get_plays_ctx()

    if play_name not in play_data_ctx.map_by_alias:
        raise Exception('Can''t find play [%s].' % play_name)
    
    play  = play_data_ctx.load_play(play_name)
    print play.title, '\n\t', play.toc_as_str()
    
    if not os.path.exists('%simgs/' % basedir):
        os.makedirs('%simgs/' % basedir)

    # somewhere in here this message is cropping up occasionally
    #objc[5300]: Object 0x100306b70 of class __NSArrayI autoreleased with no pool in place - just leaking - break on objc_autoreleaseNoPool() to debug

    # /usr/local/lib/python2.7/dist-packages/matplotlib/pyplot.py:412: 
    # RuntimeWarning: More than 20 figures have been opened. Figures created through the 
    # pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed 
    # and may consume too much memory. (To control this warning, see the rcParam 
    # `figure.max_num_figures`).

    full_title = play_data_ctx.map_by_alias.get(play_name)
    for sc in play.scenes:
        sc.graph_img_f = '%simgs/%s_%s_%s.png' % (basedir, full_title, sc.act, sc.scene)
        if not os.path.exists(sc.graph_img_f) or force_img_regen:
            plt.figure(figsize=(8,5))
            draw_graph(str(sc), sc.graph)
            plt.savefig(sc.graph_img_f)
            plt.close()

    return play

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
