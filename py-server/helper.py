import os, re

_root_dir = None
def get_root_dir():
    global _root_dir
    if _root_dir is None:
        full_currdir = os.path.abspath(os.path.curdir)
        path_re = re.compile('^(.+)/py-server')
        print full_currdir
        m = path_re.search(full_currdir)
        if not m:
            raise Exception('Cannot find the SKPN directory in the path.')
        _root_dir = m.group(1)
        print 'Root Directory:', _root_dir
    return _root_dir

if __name__ == '__main__':
    _root_dir = None
    get_root_dir()

