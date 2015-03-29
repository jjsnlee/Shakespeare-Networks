#!/usr/bin/env python
import os
import sys

# This is to allow remote debugging
# http://pydev.org/manual_adv_remote_debugger.html
sys.path.append(r'/Users/jason/Local/share/eclipse-4.3_kepler/plugins/org.python.pydev_3.9.2.201502050007/pysrc')
#import pydevd
#pydevd.patch_django_autoreload(patch_remote_debugger=True, patch_show_console=True)

if __name__ == "__main__":
  os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cfg.settings")
  from django.core.management import execute_from_command_line
  execute_from_command_line(sys.argv)
