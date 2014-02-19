from plays_n_graphs import get_plays_ctx, draw_graph
import matplotlib.pyplot as plt
import os, json, traceback
from operator import itemgetter
from json.encoder import JSONEncoder

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

    html = create_html(sld_play)
    return HttpResponse(html)

def create_html(play):
    data_ctx = get_plays_ctx()
    plays = sorted(data_ctx.plays, key=itemgetter(1))
    print 'play:', play
    all_plays = json.dumps([{'value':p[0],'label':p[1]} for p in plays])
    html = open('shakespeare/page.html', 'r').read()
    html = html.replace('__ALL_PLAYS__', all_plays)
    html = html.replace('__PLAY__', play)
    return html

def get_play(req):
    """Return JSON representation for the play"""
    try:
        path = req.path
        play = path.split('/')[-1]
        #print 'REQUEST:\n', play
        play_data = init_play(play, False)
        json_rslt = json.dumps(play_data['play'], ensure_ascii=False, cls=PlayJSONEncoder)
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
            d['characters'] = obj.characters.keys() 
            
        elif isinstance(obj, Scene):
            d = self.from_keys(obj, ('act', 'scene', 'location', 'graph_img_f'))
            dmat = obj.dialogue_matrix
            #d['dialogue_matrix'] = dmat
            d['lines'] = obj.clean_lines
            
        elif isinstance(obj, Character):
            d = self.from_keys(obj, ('name', 'clean_lines'))
        elif isinstance(obj, Line):
            #d = self.from_keys(obj, ('lineno', 'line'))
            d = repr(obj)
        else:
            d = obj.__dict__.copy()
        
        return d

def init_play(sld_play, force_img_regen, basedir=''):
    play_data_ctx = get_plays_ctx()

    if sld_play not in play_data_ctx.map_by_alias:
        raise Exception('Can''t find play [%s].' % sld_play)
    
    graph_of_play = play_data_ctx.load_play(sld_play)
    title = play_data_ctx.map_by_alias.get(sld_play)

    graph_of_play.create_graph()
    play = graph_of_play.play
    rslt = { 'img' : [], 'scenes' : [], 'play' : play }
    print play.title, '\n\t', play.toc_as_str()
    
    if not os.path.exists('%simgs/' % basedir):
        os.makedirs('%simgs/' % basedir)

    for sc in play.scenes:
        arr = '<option value="%s,%s">Act %s Sc %s %s</option>' \
            % (sc.act, sc.scene, sc.act, sc.scene, sc.location)
        rslt['scenes'].append(arr)
        #print sc
        sc.graph_img_f = '%simgs/%s_%s_%s.png' % (basedir, title, sc.act, sc.scene)
        if not os.path.exists(sc.graph_img_f) or force_img_regen:
            plt.figure(figsize=(8,5))
            draw_graph(str(sc), sc.graph)
            plt.savefig(sc.graph_img_f)

    return rslt

def main():
    play = 'King Lear'
    #play = "A Midsummer Night's Dream"
    html = create_html(play, basedir='../', absroot=False, force_img_regen=False, incl_hdr=False)
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
