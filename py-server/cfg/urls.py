from django.conf.urls import patterns #, include, url
from django.http import HttpResponse
#from django.views.generic.simple import direct_to_template
from shakespeare import shakespeare_pages
import helper
from os.path import join

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

def view_imgs(req):
    img = req.path
    img = img.replace('/', '', 1)
    img = join(helper.get_dynamic_rootdir(), img)
    image_data = open(img, "rb").read()
    return HttpResponse(image_data, mimetype="image/png")

def view_static(req):
    filepath = req.path
    filepath = filepath.replace('/', '', 1)
    data = open(filepath, 'r').read()
    return HttpResponse(data)

urlpatterns = patterns('',
    ('^shakespeare/$', shakespeare_pages.get_page_html),
    ('^shakespeare/otherCharts$', shakespeare_pages.get_page_html),
    
    ('^shakespeare/corpus/', shakespeare_pages.get_corpus_data_json),
    ('^shakespeare/play/', shakespeare_pages.get_play_data_json),
    ('^imgs/.*\.png$', view_imgs),
    ('^js/.*$', view_static),

    # Uncomment the admin/doc line below to enable admin documentation:
    #url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    #url(r'^admin/', include(admin.site.urls)),
)
