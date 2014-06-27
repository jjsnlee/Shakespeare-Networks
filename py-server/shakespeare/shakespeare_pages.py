import os, json, traceback
from operator import itemgetter
from os.path import join
import helper
from plays_n_graphs import get_plays_ctx, init_play_imgs
from plays_transform import PlayJSONMetadataEncoder
from clusters import get_lda_rslt, get_lda_base_dir
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

        data_ctx = get_plays_ctx(play_set)

        force_img_regen = False
        if sld_play:
            force_img_regen = qry_str.get('force_regen', False) == '1'
            if force_img_regen:
                play = data_ctx.get_play(sld_play)
                _play_data = init_play_imgs(play, sld_play, 1)

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

        logger.debug('info: %s', info)
        play_data_ctx = get_plays_ctx(play_set)

        if info == 'lineCounts':
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
                    'genre'    : play_json['type'],
                    'year'     : play_json['year']
                }

            all_json_rslt = json.dumps(all_plays_json, ensure_ascii=False)
            return HttpResponse(all_json_rslt, content_type='application/json')

        elif info == 'ldatopics':
            #'/shakespeare/corpus/ldatopics'
            which_topic = path_elmts[3]

            logger.debug('which_topic: %s', which_topic)
            #lda_key = '../data/dynamic/lda/2014-05-13 00:50:36.652535_50_50.lda'
            lda_key = '2014-06-01 12:55:34.874782_20_50.lda'
            lda_key = join(get_lda_base_dir(), lda_key)
            lda_rslt = get_lda_rslt(lda_key)
            topic_info = lda_rslt.docs_per_topic[int(which_topic)]
            logger.debug('topic_info: %s', topic_info)
            
            topic_json = json.dumps(topic_info, ensure_ascii=False)
            return HttpResponse(topic_json, content_type='application/json')
            
#             if which_json == 'seriated-parameters.json':
#                 pass
#             elif which_json == 'filtered-parameters.json':
#                 pass
#             elif which_json == 'global-term_freqs.json':
#                 pass

        elif info == 'characters':
            #'/shakespeare/corpus/characters/[charKey]'
            char_key = path_elmts[3]
            
            char_nm, title = char_key.split(' in ')
            logger.debug('title: %s, char_nm: %s', title, char_nm)
            
            # need to get play alias by the title
            alias = play_data_ctx.map_by_title.get(title)
            logger.debug('play alias: %s', alias)
            
            play = play_data_ctx.get_play(alias)
            # then the character
            char = play.characters.get(char_nm)
            
            char_lines = []
            prev = curr = None
            for cl in char.clean_lines:
                if prev is None \
                        or prev.act!=cl.act \
                        or prev.scene!=cl.scene \
                        or int(prev.lineno)+1!=int(cl.lineno):
                    curr = []
                    char_lines.append(curr)
                curr.append(str(cl))
                prev = cl

            #print 'char_lines: ', char_lines
            
            char_data = \
            {
             'character'   : char_nm,
             'play'        : title,
             'doc_name'    : char_key,
             'doc_content' : char_lines #[str(li) for li in char.clean_lines]
            }
            char_json = json.dumps(char_data, ensure_ascii=False)
            return HttpResponse(char_json, content_type='application/json')
        
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
        
        # expect format '/shakespeare/play/{play_alias}/{?act}/{?scene}/?content'
        
        play_alias = path_elmts[2] # i.e, '/shakespeare/play/hamlet' 
        logger.debug('REQUEST: [%s], [%s]', play_alias, path_elmts)
        
        play_data_ctx = get_plays_ctx(play_set)
        play = play_data_ctx.get_play(play_alias)
        
        if path_elmts[-1] == 'content':
            act_num = path_elmts[-3]
            scene_num = path_elmts[-2]
            scene = play.get_scene(act_num, scene_num)
            scene_data = \
            {
             'title'    : play.title,
             'act'     : act_num, 
             'scene'   : scene_num,
             'content' : [{'speaker' : char_lines[0], 
                           'lines'   : [str(line) for line in char_lines[1]]} for char_lines in scene.clean_lines]
            }
            json_rslt = json.dumps(scene_data, ensure_ascii=False)
            return HttpResponse(json_rslt, content_type='application/json')
        
        else:
            # Probably want to streamline this, so we take the existing files where possible...
            init_play_imgs(play, play_alias, False)
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
