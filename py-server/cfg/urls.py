from django.conf.urls import re_path
from django.http import HttpResponse
from django.views.static import serve
from cfg import settings

from shakespeare import page_plays

import helper
import os.path as op
import logging


# https://stackoverflow.com/questions/5836674/why-does-debug-false-setting-make-my-django-static-files-access-fail
# from django.views.static import serve
# import settings
from shakespeare.page_plays import do_req

logger = helper.setup_sysout_handler(__name__)
logging.getLogger('boto3').setLevel(logging.INFO)


def view_imgs(req):
    img = req.path
    img = img.replace('/', '', 1)
    img = op.join(helper.get_dynamic_rootdir(), img)
    logger.debug('img: %s', img)
    image_data = open(img, "rb").read()
    return HttpResponse(image_data, content_type="image/png")


def view_static(filepath):
    root_dir = helper.get_root_dir()
    filepath = op.join(root_dir, 'static', filepath)
    data = open(filepath, 'r').read()
    return HttpResponse(data, content_type="text/html")


# def reverse_proxy(req, port, path):
#   pass


urlpatterns = [

    # don't have to move static files into nginx
    # https://stackoverflow.com/a/49722734/14084291

    # but probably want to change it eventually, as it is "grossly inefficient"
    # https://docs.djangoproject.com/en/3.2/ref/contrib/staticfiles/#django.contrib.staticfiles.views.serve
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATICFILES_DIRS[0]}),

    # might want to move these to a different server
    re_path('^(shakespeare|chekhov)/?$', page_plays.get_page_html),
    re_path('^(shakespeare|chekhov)/otherCharts$', page_plays.get_page_html),

    # maybe this should just be included in the dispatch
    re_path('^(shakespeare|chekhov)/play/', do_req(page_plays.get_play_data_json)),
    re_path('^corpus/(shakespeare)', page_plays.CorpusDataJsonHandler.dispatch),
    # re_path('^corpus/search/(.+)', page_dispatcher.CorpusDataJsonHandler.search),

    re_path(r'^imgs/.*\.png$', view_imgs),

    re_path("^$", lambda req: view_static("index.html")),
]
