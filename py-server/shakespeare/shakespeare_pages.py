import os, json, traceback
from operator import itemgetter
from os.path import join
import helper
from plays_n_graphs import get_plays_ctx, init_play
from plays_transform import PlayJSONMetadataEncoder
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('shakespeare.shakespeare_pages')

try:
    from django.http import HttpResponse
except:
    logger.warn("Couldn't find Django, maybe ok...")

def get_page_html(req, play_set):
    path = req.path
    info = path.split('/')[-1]
    print 'path:', path
    
    if play_set not in ['shakespeare', 'chekhov']:
        raise Exception('Invalid content [%s].' % play_set)
    
    if info == 'otherCharts':
        html = open('shakespeare/page_all_plays.html', 'r').read()
    
    else:
        qry_str = req.REQUEST
        logger.debug('REQUEST: %s', qry_str)
        sld_play = qry_str.get('play', '')
        logger.debug('play: %s', sld_play)

        force_img_regen = False
        if sld_play:
            force_img_regen = qry_str.get('force_regen', False) == '1'
            if force_img_regen:
                _play_data = init_play(play_set, sld_play, 1)

        data_ctx = get_plays_ctx(play_set)
        plays = sorted(data_ctx.plays, key=itemgetter(1))
        all_plays = json.dumps([{'value':p[0],'label':p[1]} for p in plays])
        html = open('shakespeare/page.html', 'r').read()

        # Should get rid of this...
        html = html.replace('__ALL_PLAYS__', all_plays)

    return HttpResponse(html)

def get_corpus_data_json(req, play_set):
    try:
        path_elmts = filter(None, req.path.split('/'))
        info = None
        if len(path_elmts) > 2:
            info = path_elmts[2] # expect format '/shakespeare/corpus/lineCounts'

        print 'info:', info

        if info == 'lineCounts':
            play_data_ctx = get_plays_ctx(play_set)
            plays = play_data_ctx.plays

            all_plays_json = {}
            for play_alias, _ in plays:
                fname = join(DYNAMIC_ASSETS_BASEDIR, 'json', play_alias+'_metadata.json')
                if not os.path.exists(fname):
                    logger.warn('File path [%s] doesn\'t exist!', fname)
                play_json = json.loads(open(fname, 'r').read())
                all_plays_json[play_alias] = {
                    'chardata' : play_json['char_data'],
                    'title'    : play_json['title'],
                    'genre'    : play_json['type']
                }

            all_json_rslt = json.dumps(all_plays_json, ensure_ascii=False)
            return HttpResponse(all_json_rslt, content_type='application/json')

        elif info == 'LDA':
            which_json = path_elmts[3]
            if which_json == 'seriated-parameters.json':
                pass
            elif which_json == 'filtered-parameters.json':
                pass
            elif which_json == 'global-term_freqs.json':
                pass
        
    except Exception as e:
        # Without the explicit error handling the JSON error gets swallowed
        st = traceback.format_exc()
        #print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)
        logger.error('Problem parsing [%s]:\n%s\n%s', req, e, st)

DYNAMIC_ASSETS_BASEDIR = helper.get_dynamic_rootdir()

def get_play_data_json(req, play_set):
    """ 
    JSON representation for the play. This is for the initial load of
    the play and its scenes, and I believe the scenes will have the basic 
    
    """
    try:
        path_elmts = filter(None, req.path.split('/'))
        play_alias = path_elmts[2] # expect format '/shakespeare/play/hamlet' 
        
        #play_alias = path.split('/')[-1]
        print 'REQUEST:', play_alias, path_elmts
        
        # Probably want to streamline this, so we take the existing files where possible...
        play = init_play(play_set, play_alias, False)
        json_rslt = json.dumps(play, ensure_ascii=False, cls=PlayJSONMetadataEncoder)
        return HttpResponse(json_rslt, content_type='application/json')

    except Exception as e:
        # Without the explicit error handling the JSON error gets swallowed
        st = traceback.format_exc()
        #print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)
        logger.error('Problem parsing [%s]:\n%s\n%s', req, e, st)

#def main():
#    play = 'King Lear'
#    #play = "A Midsummer Night's Dream"
#    html = _create_html(play, basedir='../', absroot=False, force_img_regen=False, incl_hdr=False)
#    #print 'HTML:', html
#    tmpfile = 'tmp.html'
#    with open(tmpfile, 'w') as fh:
#        fh.writelines(html)
#
#    import webbrowser
#    lcl_url = 'file://'+os.path.abspath(tmpfile)
#    print 'Opening', lcl_url
#    webbrowser.open_new_tab(lcl_url)
#
#if (__name__=="__main__"):
#    main()
