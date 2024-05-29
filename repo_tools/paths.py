from pathlib import Path
import os


class RootPath:
    PATH = Path(os.path.relpath(Path(__file__).parent.parent.resolve(), os.getcwd()))
    BUILD_OUTPUT = PATH / '__build__'
    DIST_OUTPUT = PATH / '__dist__'
    SETUP_PY = PATH / 'setup.py'
    PKG = PATH / 'milvus_client'


class VenvPath:
    PATH = RootPath.PATH / '__venv__'
    PIP_HISTORY = PATH / 'pip_history.pkl'
    BIN = PATH / 'bin'
    PIP = BIN / 'pip'
    PYTHON = BIN / 'python'
