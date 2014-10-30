import os, json, traceback
from operator import itemgetter
from os.path import join
import helper
from plays_n_graphs import get_plays_ctx, init_play_imgs
from plays_transform import PlayJSONMetadataEncoder

import logging
logging.basicConfig(level=logging.DEBUG)
logger = helper.setup_sysout_handler(__name__)

try:
    from django.http import HttpResponse
except:
    logger.warn("Couldn't find Django...")

def get_page_html(req, play_set):
    """ Basic HTML with interpolated values """
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

class CorpusDataJsonHandler:
    
    @classmethod
    def dispatch_map(cls):
        return {
           'sceneSummary' : cls.handle_scene_summary,
           #'sceneDetail'  : cls.handle_scene_detail,
           #'lda'          : cls.handle_LDA,
           'characters'   : cls.handle_chardata,
           'topicModels'  : cls.handle_topic_models
        }
    
    @classmethod
    def dispatch(cls, req, play_set):
        try:
            path_elmts = filter(None, req.path.split('/'))
            info = None
            if len(path_elmts) > 2:
                info = path_elmts[2] # expect format '/shakespeare/corpus/lineCounts'
    
            logger.debug('info: %s', info)
            play_data_ctx = get_plays_ctx(play_set)
            handler = cls.dispatch_map().get(info)
            if not handler:
                raise 'No handler defined for [%s]' % info
            
            rslt_json = handler(play_data_ctx, path_elmts)
            return HttpResponse(rslt_json, content_type='application/json')

        except Exception as e:
            # Without the explicit error handling the JSON error gets swallowed
            st = traceback.format_exc()
            #print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)
            logger.error('Problem parsing [%s]:\n%s\n%s', req, e, st)

    @classmethod
    def handle_scene_summary(cls, play_data_ctx, path_elmts):
        import itertools
        plays = play_data_ctx.plays
        
        def copy_d(d):
            new_d = dict([(k,d[k]) for k in [
                'density','location','total_degrees','graph_img_f','total_lines',
                'closeness_vitality','avg_clustering','deg_assort_coeff','avg_shortest_path']])
            new_d['scene'] ='Act %s, Sc %s' % (p['act'], p['scene'])
            return new_d
        
        all_plays_json = {}
        for play_alias, _ in plays:
            fname = join(DYNAMIC_ASSETS_BASEDIR, 'json', play_alias+'_metadata.json')
            if not os.path.exists(fname):
                logger.warn('File path [%s] doesn\'t exist!', fname)
            play_json = json.loads(open(fname, 'r').read())

            scenes = [copy_d(p) for p in itertools.chain(*play_json['acts'])]
            
            all_plays_json[play_alias] = {
                'chardata' : play_json['char_data'],
                'scenes'   : scenes,
                'title'    : play_json['title'],
                'genre'    : play_json['type'],
                'year'     : play_json['year']
            }

        all_json_rslt = json.dumps(all_plays_json, ensure_ascii=False)
        return all_json_rslt

#     @classmethod
#     def handle_scene_detail(cls, play_data_ctx, path_elmts):
#         plays = play_data_ctx.plays
#         all_plays_json = {}
#         for play_alias, _ in plays:
#             fname = join(DYNAMIC_ASSETS_BASEDIR, 'json', play_alias+'_metadata.json')
#             if not os.path.exists(fname):
#                 logger.warn('File path [%s] doesn\'t exist!', fname)
#             play_json = json.loads(open(fname, 'r').read())
#             all_plays_json[play_alias] = {
#                 'acts'   : play_json['acts'],
#                 'title'  : play_json['title'],
#                 'genre'  : play_json['type'],
#                 'year'   : play_json['year']
#             }
#         all_json_rslt = json.dumps(all_plays_json, ensure_ascii=False)
#         return all_json_rslt

    @classmethod
    def handle_topic_models(cls, play_data_ctx, path_elmts):
        """
            Expected format:
            /shakespeare/corpus/topicModels : 
                Get All Topic Models
            /shakespeare/corpus/topicModels/[Topic Model Name]/[Topic #] :
                Specific Topic Info
            /shakespeare/corpus/topicModels/[Topic Model Name]/termite/*.json :
                static termite json info
        """
        import clusters
        MODEL_KEYS = {
          'char-scene-bow-LDA-100-50'    : 'lda-char-scene-bow_2014-06-29_19.49.11_100_50',
          'char-scene-bow-LDA-50-200'    : 'lda-char-scene-bow_2014-08-30_14.32.36_50_200',

          'char-scene-tfidf-LDA-50-50'   : 'lda-char-scene-tfidf_2014-08-24_23.04.15_50_50',
          'char-scene-tfidf-LDA-50-50-v2': 'lda-char-scene-tfidf_2014-08-26_00.43.50_50_50',
          'char-scene-tfidf-LDA-50-100'  : 'lda-char-scene-tfidf_2014-08-26_01.47.56_50_100',
          'char-scene-tfidf-LDA-50-200'  : 'lda-char-scene-tfidf_2014-08-29_23.00.09_50_200',
          
          'char-bow-LDA-50-200'          : 'lda-char-bow_2014-09-21_23.33.07_50_200',
          'char-bow-NMF-50-250'          : ('nmf-char-2014-09-20_02.06.43-50-250', clusters.NMFResult),
          'char-bow-NMF-50-200'          : ('nmf-char-2014-09-20_03.22.31-50-200', clusters.NMFResult),
        }

        if len(path_elmts)==3:
            topics = MODEL_KEYS.keys()
            print 'TOPICS:', topics
            topics.sort()
            topic_json = json.dumps(topics, ensure_ascii=False)
        else:
            topic_model = path_elmts[3]
            topic_context = path_elmts[4]
            model_key = MODEL_KEYS.get(topic_model)
            cls = None
            if type(model_key)==tuple:
                model_key, cls = model_key
            
            #logger.debug('which_topic: %s', which_topic)
            
            if topic_context=='termite':
                json_file = path_elmts[5]
                fname = join(clusters.get_models_base_dir(), 
                             model_key, 'termite', 'public_html', json_file)
                topic_json = open(fname, 'r').read()
                #print 'topic_json:', topic_json
            else:
                which_topic = topic_context
                logger.debug('which_topic: %s', which_topic)
                model_rslt = clusters.get_lda_rslt(model_key, cls=cls)
                topic_info = model_rslt.docs_per_topic[int(which_topic)]
                #logger.debug('topic_info: %s', topic_info)
                topic_json = json.dumps(topic_info, ensure_ascii=False)
        return topic_json

    @classmethod
    def handle_chardata(cls, play_data_ctx, path_elmts):
        """ 
            Expected format:
                /shakespeare/corpus/characters/[charKey]
        """
        char_key = path_elmts[3]
        
        char_nm, title = char_key.split(' in ')
        logger.debug('title: %s, char_nm: %s', title, char_nm)
        
        # need to get play alias by the title
        alias = play_data_ctx.map_by_title.get(title)
        logger.debug('play alias: %s', alias)
        
        play = play_data_ctx.get_play(alias)
        # then the character
        char = play.characters.get(char_nm)
        
        # fix this!!!
        if not char:
            #char_nm, act, scene = char_nm.split(',')
            import re
            from plays_n_graphs import Character
            CHAR_NM_RE = re.compile('^([^,]+), Act (\d+), Sc (\d+)$')
            m = CHAR_NM_RE.match(char_nm)
            char_nm, act, sc = m.group(1), m.group(2), m.group(3)                     
            
            char = play.characters.get(char_nm)
            char_lines = []
            for li in char.clean_lines:
                if li.act==act and li.scene==sc:
                    char_lines.append(li)
            artif_char = Character(char_nm, play)
            artif_char._cleaned_lines = char_lines
            char = artif_char
        
        char_lines = []
        prev = curr = None
        for cl in char.clean_lines:
            if prev is None \
                    or prev.act!=cl.act \
                    or prev.scene!=cl.scene \
                    or int(prev.lineno)+1!=int(cl.lineno):
                curr = []
                char_lines.append(curr)
            li = str(cl)
            curr.append(li)
            prev = cl

        #print 'char_lines: ', char_lines
        #char_lines = char_lines.replace('france', '<yellow>france</yellow>')
        
        char_data = \
        {
         'character'   : char_nm,
         'play'        : title,
         'doc_name'    : char_key,
         'doc_content' : char_lines #[str(li) for li in char.clean_lines]
        }
        char_json = json.dumps(char_data, ensure_ascii=False)
        return char_json

DYNAMIC_ASSETS_BASEDIR = helper.get_dynamic_rootdir()

def get_play_data_json(req, play_set):
    """ 
    JSON representation for the play. This is for the initial load of
    the play and its scenes, and the scenes will have basic metadata.
    
    Expects format:
        /shakespeare/play/{play_alias}/{?act}/{?scene}/?content
    """
    try:
        path_elmts = filter(None, req.path.split('/'))
        
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
