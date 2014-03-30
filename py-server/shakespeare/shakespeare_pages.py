import os, json, traceback
from operator import itemgetter
from os.path import join
import helper
from plays_n_graphs import get_plays_ctx, init_play
from plays_transform import PlayJSONMetadataEncoder

try:
    from django.http import HttpResponse
except:
    print "Couldn't find Django, maybe ok..."

def get_page_html(req):
    path = req.path
    info = path.split('/')[-1]
    print 'path:', path
    
    if info == 'otherCharts':
        html = open('shakespeare/page_all_plays.html', 'r').read()
    
    else:
        qry_str = req.REQUEST
        print 'REQUEST:\n', qry_str
        sld_play = qry_str.get('play', '')
        print 'play:', sld_play
        force_img_regen = False
        if sld_play:
            force_img_regen = qry_str.get('force_regen', False) == '1'
            if force_img_regen:
                _play_data = init_play(sld_play, 1)

        data_ctx = get_plays_ctx()
        plays = sorted(data_ctx.plays, key=itemgetter(1))
        all_plays = json.dumps([{'value':p[0],'label':p[1]} for p in plays])
        html = open('shakespeare/page.html', 'r').read()

        # Should get rid of this...
        html = html.replace('__ALL_PLAYS__', all_plays)

    return HttpResponse(html)

def get_corpus_data_json(req):
    try:
        path = req.path
        info = path.split('/')[-1]
        print 'info:', info
        if info == 'lineCounts':
            play_data_ctx = get_plays_ctx()
            plays = play_data_ctx.plays

            all_plays_json = {}
            for play_alias, _ in plays:
                fname = join(DYNAMIC_ASSETS_BASEDIR, 'json', play_alias+'_metadata.json')
                if not os.path.exists(fname):
                    print 'File path [%s] doesn\'t exist!' % fname
                play_json = json.loads(open(fname, 'r').read())
                
                #all_plays_json.append({play_alias : play_json['char_data']})
                all_plays_json[play_alias] = play_json['char_data']

            all_json_rslt = json.dumps(all_plays_json, ensure_ascii=False)
            return HttpResponse(all_json_rslt, content_type='application/json')
        
    except Exception as e:
        # Without the explicit error handling the JSON error gets swallowed
        st = traceback.format_exc()
        print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)

DYNAMIC_ASSETS_BASEDIR = helper.get_dynamic_rootdir()

def get_play_data_json(req):
    """ JSON representation for the play """
    try:
        path = req.path
        play_alias = path.split('/')[-1]
        #print 'REQUEST:\n', play_alias
        
        # Probably want to streamline this, so we take the existing files where possible...
        play = init_play(play_alias, False)
        json_rslt = json.dumps(play, ensure_ascii=False, cls=PlayJSONMetadataEncoder)

#        fname = join(DYNAMIC_ASSETS_BASEDIR, 'json', play_alias+'_metadata.json')
#        print 'Loading json from %s' % fname
#        if not os.path.exists(fname):
#            print 'File path doesn\'t exist!'
#        play_json = open(fname, 'r').read()
#        json_rslt = json.loads(play_json)
        return HttpResponse(json_rslt, content_type='application/json')
    except Exception as e:
        # Without the explicit error handling the JSON error gets swallowed
        st = traceback.format_exc()
        print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)

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
