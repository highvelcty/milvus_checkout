#!/usr/bin/env python
import os
import sys

from repo_tools import venv
from repo_tools.paths import VenvPath
venv.make()

os.execvp(VenvPath.PYTHON, [VenvPath.PYTHON, '-m', 'unittest', '-k', 'uts'] + sys.argv[1:])
