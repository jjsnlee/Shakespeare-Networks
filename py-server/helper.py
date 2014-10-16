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
            raise Exception('Cannot find the py-server directory in the path.')
        _root_dir = m.group(1)
        print 'Root Directory:', _root_dir
    return _root_dir

def get_dynamic_rootdir():
    return os.path.join(get_root_dir(), 'data/dynamic')

def ensure_path(d):
    if not os.path.exists(d):
        os.makedirs(d)

import logging

def setup_sysout_handler(logger_name):
    logger = logging.getLogger(logger_name)
#     formatter = logging.Formatter('[%(levelname)s] [%(name)s] [%(asctime)s] [%(lineno)s]: %(message)s')
#     #stdout_handler = logging.StreamHandler()
#     
#     print 'logger.handlers:', logger.handlers
#     if logger.handlers:
#         hdlr = logger.handlers[0]
#     else:
#         hdlr = logging.StreamHandler()
#         logger.addHandler(hdlr)
# 
#     hdlr.setFormatter(formatter)

    return logger

if __name__ == '__main__':
    _root_dir = None
    get_root_dir()

