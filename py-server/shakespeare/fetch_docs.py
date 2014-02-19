import httplib, re, os
#from BeautifulSoup import BeautifulSoup

def get_homepage(fname):
    conn = httplib.HTTPConnection('shakespeare.mit.edu')
    conn.request("GET", "/")
    rsp = conn.getresponse()
    print rsp.status, rsp.reason
    data = rsp.read()
    print data
    conn.close()
    with open(fname, 'w') as f:
        f.writelines(data)

def do_get(conn, url):
    """
    http://localhost:8000/?plays=richardiii%2Findex.html
    """
    try:
        conn.request("GET", "/"+url)
        rsp = conn.getresponse()
        print rsp.status, rsp.reason, url
        data = rsp.read()
        return data
    except Exception as e:
        print e
    
def get_desturl(url):
    desturl = 'shakespeare_files/'+url
    destdir = os.path.dirname(desturl)
    os.makedirs(destdir)
    if not os.path.exists(destdir):
        return desturl

from plays_n_graphs import links_ptn

def parse_hp(fname):
    """ Get the HTML & parse it """
    with open(fname) as f:
        data = f.read()
    #links_ptn = '<a href="([^\"]+)">([^>]+)</a>'
    rslt = re.findall(links_ptn, data)
    conn = httplib.HTTPConnection('shakespeare.mit.edu')
    index = {}
    for url, title in rslt:
        title = title.strip()
        index[title] = url
        desturl = get_desturl(url)
        data = do_get(conn, url)
        with open(desturl, 'w') as f:
            f.writelines(data)
        if data and re.search('<p><a href="full.html">Entire play</a> in one page</p>', data):
            url = url.replace('index.html', 'full.html')
            desturl = get_desturl(url)
            data = do_get(conn, url)
            with open(desturl, 'w') as f:
                f.writelines(data)
    conn.close()

if (__name__=="__main__"):
    if not os.path.exists('shakespeare_files'):
        os.makedirs('shakespeare_files')
    hp_file = 'shakespeare_files/shakespeare_hp.html'
    get_homepage(hp_file)
    parse_hp(hp_file)
    