import os, sys, re

_root_dir = None
def get_root_dir():
    global _root_dir
    if _root_dir is None:
        full_currdir = os.path.abspath(os.path.curdir)
        path_re = re.compile('^(.+SKPN/)')
        m = path_re.search(full_currdir)
        if not m:
            raise Exception('Cannot find the SKPN directory in the path.')
        _root_dir = m.group(1)
        print 'Root Directory:', _root_dir
        #_root_dir = '/Users/jason/Projects/RenaissanceNLP'
    return _root_dir
