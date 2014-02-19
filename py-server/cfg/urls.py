from django.conf.urls import patterns #, include, url
from django.http import HttpResponse
#from django.views.generic.simple import direct_to_template
from shakespeare import shakespeare_pages

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

def view_imgs(req):
    img = req.path
    img = img.replace('/', '', 1)
    image_data = open(img, "rb").read()
    return HttpResponse(image_data, mimetype="image/png")

def view_static(req):
    filepath = req.path
    filepath = filepath.replace('/', '', 1)
    data = open(filepath, 'r').read()
    return HttpResponse(data)

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'shakespeare.views.home', name='home'),
    # url(r'^shakespeare/', include('shakespeare.foo.urls')),
    ('^shakespeare/$', shakespeare_pages.view_page),
    ('^shakespeare/play/.*$', shakespeare_pages.get_play),
    
    ('^imgs/.*\.png$', view_imgs),
    ('^html/.*$', view_static),
    
    # Uncomment the admin/doc line below to enable admin documentation:
    #url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    #url(r'^admin/', include(admin.site.urls)),
)
