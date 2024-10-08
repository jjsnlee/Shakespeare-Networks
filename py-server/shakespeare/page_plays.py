import json
import logging
import os
import traceback
from operator import itemgetter
from os.path import join

import helper
from shakespeare.plays_n_graphs import get_plays_ctx, init_play_imgs
from shakespeare.plays_transform import PlayJSONMetadataEncoder

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
    logger.debug('path: %s', path)

    if play_set not in ['shakespeare', 'chekhov']:
        raise Exception('Invalid content [%s].' % play_set)
    
    content_dir = join(helper.get_root_dir(), "py-server")

    if info == 'otherCharts':
        html = open(join(content_dir, 'shakespeare/page_all_plays.html'), 'r').read()

    else:
        qry_str = req.GET
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
        all_plays = json.dumps([{'value': p[0], 'label': p[1]} for p in plays])
        html = open(join(content_dir, 'shakespeare/page.html'), 'r').read()

        # Should get rid of this...
        html = html.replace('__ALL_PLAYS__', all_plays)

    return HttpResponse(html)


def do_req(f):
    def handler(req, *a):
        try:
            return f(req, *a)
        except Exception as e:
            # Without the explicit error handling the JSON error gets swallowed
            st = traceback.format_exc()
            # print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)
            logger.error('Problem parsing [%s]:\n%s\n%s', req, e, st)
    return handler


# from django.views.decorators.csrf import csrf_exempt
class BaseHandler:

    # https://docs.djangoproject.com/en/dev/ref/csrf/#ajax
    # http://stackoverflow.com/questions/16458166/how-to-disable-djangos-csrf-validation
    @classmethod
    def dispatch(cls, req, corpus):
        try:
            logger.debug(req)
            rslt_json = cls.dispatch_hook(req, corpus)
            return HttpResponse(rslt_json, content_type='application/json')
        except Exception as e:
            # Without the explicit error handling the JSON error gets swallowed
            st = traceback.format_exc()
            # print 'Problem parsing [%s]:\n%s\n%s' % (req, e, st)
            logger.error('Problem parsing [%s]:\n%s\n%s', req, e, st)
            errmsg = json.dumps({'error': str(e)})
            return HttpResponse(errmsg, content_type='application/json', status=500)

    @classmethod
    def dispatch_hook(cls, req, corpus):
        pass


class CorpusDataJsonHandler(BaseHandler):

    @classmethod
    def dispatch_map(cls):
        return {
            'sceneSummary': cls.handle_scene_summary,
            'documents': cls.handle_chardata,
        }

    @classmethod
    def dispatch_hook(cls, req, corpus):
        path_elmts = [_f for _f in req.path.split('/') if _f]
        info = None
        if len(path_elmts) > 2:
            info = path_elmts[2]  # expect format '/corpus/shakespeare/lineCounts'
        logger.debug('info: %s', info)

        play_data_ctx = get_plays_ctx(corpus)
        handler = cls.dispatch_map().get(info)
        if not handler:
            raise 'No handler defined for [%s]' % info

        rslt_json = handler(play_data_ctx, path_elmts)
        return rslt_json

    @classmethod
    def handle_scene_summary(cls, play_data_ctx, path_elmts):
        import itertools
        plays = play_data_ctx.plays

        def copy_d(d):
            new_d = dict([(k, d[k]) for k in [
                'density', 'location', 'total_degrees', 'graph_img_f', 'total_lines',
                'closeness_vitality', 'avg_clustering', 'deg_assort_coeff', 'avg_shortest_path']])
            new_d['scene'] = 'Act %s, Sc %s' % (d['act'], d['scene'])
            return new_d

        all_plays_json = {}
        for play_alias, _ in plays:
            fname = join(DYNAMIC_ASSETS_BASEDIR, 'json', play_alias + '_metadata.json')
            if not os.path.exists(fname):
                logger.warn('File path [%s] doesn\'t exist!', fname)
            play_json = json.loads(open(fname, 'r').read())

            scenes = [copy_d(p) for p in itertools.chain(*play_json['acts'])]

            all_plays_json[play_alias] = {
                'chardata': play_json['char_data'],
                'scenes': scenes,
                'title': play_json['title'],
                'genre': play_json['type'],
                'year': play_json['year']
            }

        all_json_rslt = json.dumps(all_plays_json, ensure_ascii=False)
        return all_json_rslt

    @classmethod
    def handle_chardata(cls, play_data_ctx, path_elmts):
        """
            Expected format:
                /corpus/shakespeare/characters/[charKey]
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
            # char_nm, act, scene = char_nm.split(',')
            import re
            from shakespeare.plays_n_graphs import Character
            CHAR_NM_RE = re.compile('^([^,]+), Act (\d+), Sc (\d+)$')
            m = CHAR_NM_RE.match(char_nm)
            char_nm, act, sc = m.group(1), m.group(2), m.group(3)

            char = play.characters.get(char_nm)
            char_lines = []
            for li in char.clean_lines:
                if li.act == act and li.scene == sc:
                    char_lines.append(li)
            artif_char = Character(char_nm, play)
            artif_char._cleaned_lines = char_lines
            char = artif_char

        char_lines = []
        prev = curr = None
        for cl in char.clean_lines:
            try:
                if prev is None \
                        or prev.act != cl.act \
                        or prev.scene != cl.scene \
                        or int(prev.lineno) + 1 != int(cl.lineno):
                    curr = []
                    char_lines.append(curr)
                li = str(cl)
                curr.append(li)

            except Exception as _e:
                logger.error('Problem parsing [%s] [%s] [%s], [%s]', char_lines, prev, cl, cl.lineno)
                li = '[Problem parsing: [%s] [%s]]' % (cl, cl.lineno)
                curr.append(li)
                # raise e
            prev = cl

        # print 'char_lines: ', char_lines
        # char_lines = char_lines.replace('france', '<yellow>france</yellow>')

        char_data = \
            {
                'character': char_nm,
                'play': title,
                'doc_name': char_key,
                'doc_content': char_lines  # [str(li) for li in char.clean_lines]
            }
        char_json = json.dumps(char_data, ensure_ascii=False)
        return char_json

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


DYNAMIC_ASSETS_BASEDIR = helper.get_dynamic_rootdir()


def get_play_data_json(req, play_set):
    """
    JSON representation for the play. This is for the initial load of
    the play and its scenes, and the scenes will have basic metadata.

    Expects format:
        /shakespeare/play/{play_alias}/{?act}/{?scene}/?content
    """
    path_elmts = [_f for _f in req.path.split('/') if _f]

    play_alias = path_elmts[2]  # i.e, '/shakespeare/play/hamlet'
    logger.debug('REQUEST: [%s], [%s]', play_alias, path_elmts)

    play_data_ctx = get_plays_ctx(play_set)
    play = play_data_ctx.get_play(play_alias)

    if path_elmts[-1] == 'content':
        act_num = path_elmts[-3]
        scene_num = path_elmts[-2]
        scene = play.get_scene(act_num, scene_num)
        scene_data = \
            {
                'title': play.title,
                'act': act_num,
                'scene': scene_num,
                'content': [{'speaker': char_lines[0],
                             'lines': [str(line) for line in char_lines[1]]} for char_lines in scene.clean_lines]
            }
        json_rslt = json.dumps(scene_data, ensure_ascii=False)
        return HttpResponse(json_rslt, content_type='application/json')

    else:
        # Probably want to streamline this, so we take the existing files where possible...
        init_play_imgs(play, play_alias, False)
        json_rslt = json.dumps(play, ensure_ascii=False, cls=PlayJSONMetadataEncoder)
        return HttpResponse(json_rslt, content_type='application/json')

# def main():
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
# if (__name__=="__main__"):
#    main()
