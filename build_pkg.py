#!/usr/bin/env python3
import shlex
import subprocess

from repo_tools import venv
from repo_tools.paths import RootPath, VenvPath

if __name__ == '__main__':
    venv.make_for_build()
    cmd = f'{VenvPath.PYTHON} {RootPath.SETUP_PY} ' \
          f'build --build-base {RootPath.BUILD_OUTPUT} ' \
          f'egg_info --egg-base {RootPath.BUILD_OUTPUT} ' \
          f'sdist --dist-dir {RootPath.DIST_OUTPUT} ' \
          f'bdist_wheel --dist-dir {RootPath.DIST_OUTPUT}'
    subprocess.run(shlex.split(cmd), check=True)
