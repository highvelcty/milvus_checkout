from setuptools import setup


def get_version() -> str:
    with open('milvus_client/__init__.py', 'r') as inf:
        for line in inf.readlines():
            if line.startswith('__version__'):
                return line.split('=')[-1].strip().strip('"').strip("'")


setup(version=get_version())
