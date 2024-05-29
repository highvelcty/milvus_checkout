from typing import List
import os
import shlex
import subprocess
import sys
import pickle

from .paths import RootPath, VenvPath


class UnderTest(dict):
    class Key(object):
        PKGS = 'pkgs'

    @property
    def pkgs(self) -> List[str]:
        return eval(self[self.Key.PKGS])


def activate():
    if not os.path.samefile(sys.prefix, VenvPath.PATH):
        os.execvp(VenvPath.PYTHON, [VenvPath.PYTHON] + sys.argv)


def make():
    cmd = f'{VenvPath.PIP} install -e {RootPath.PATH}'
    try:
        pip_history = pickle.load(open(VenvPath.PIP_HISTORY, 'rb'))
    except FileNotFoundError:
        pip_history = []

    if not pip_history or cmd != pip_history[-1]:
        _create()
        subprocess.run(shlex.split(f'{VenvPath.PIP} install -e {RootPath.PATH}'))
        pip_history.append(cmd)
        pickle.dump(pip_history, open(VenvPath.PIP_HISTORY, 'wb'))


def make_for_build():
    if not VenvPath.PATH.exists():
        _create()
    subprocess.run(shlex.split(f'{VenvPath.PIP} install -e {RootPath.PATH}[BUILD]'),
                   check=True)


def _create():
    subprocess.run(shlex.split(f'python -m venv {VenvPath.PATH} --clear --system-site-packages '
                               f'--upgrade-deps'), check=True)
