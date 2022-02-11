import os
import re
import logging


_root_dir = None


def get_root_dir():
	global _root_dir
	if _root_dir is None:
		full_currdir = os.path.abspath(os.path.curdir)
		path_re = re.compile('^(.+)/(?:py-server|py-tests)')
		print(full_currdir)
		m = path_re.search(full_currdir)
		if m:
			_root_dir = m.group(1)
		else:
			# This is really only for the sake of the unit tests
			if os.path.exists(os.path.join(full_currdir, 'py-server')):
				_root_dir = full_currdir
			else:
				raise Exception('Cannot find the py-server or py-tests directory in the path.')
		print('Root Directory:', _root_dir)
	return _root_dir


def get_dynamic_rootdir():
	return os.path.join(get_root_dir(), 'data/dynamic')


def ensure_path(d):
	if not os.path.exists(d):
		os.makedirs(d)


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
