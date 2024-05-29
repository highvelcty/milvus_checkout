from unittest import TestCase
import shlex
import subprocess
import sys

import time

from repo_tools import paths
from demo_client import DemoClient
import numpy as np


class BaseTest(TestCase):
    @classmethod
    def setUpClass(cls):
        sys.stdout.write('Restarting the local milvus server...')
        sys.stdout.flush()

        cls.tearDownClass()

        cmd = 'sudo rm -rf volumes'
        subprocess.run(shlex.split(cmd), check=True, cwd=paths.LocalServerPath.PATH,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

        cmd = 'sudo docker-compose up -d'
        subprocess.run(shlex.split(cmd), check=True, cwd=paths.LocalServerPath.PATH,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

        sys.stdout.write('done.\n')
        sys.stdout.flush()

    @classmethod
    def tearDownClass(cls):
        cmd = 'sudo docker-compose down'
        proc = subprocess.Popen(shlex.split(cmd), cwd=paths.LocalServerPath.PATH,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        stdout, _ = proc.communicate()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)


class TestDemoClient(BaseTest):
    def test_construction(self):
        demo_client = DemoClient('test_user', 'test_password')
        demo_client.close_and_delete_all()

    def test_context(self):
        test_username = 'test_user'
        with DemoClient(test_username, 'test_password') as client:
            self.assertEqual(test_username, client.username)

    def test_insert_and_search(self):
        test_pk = 'test_pk'
        with DemoClient('test_user', 'test_password') as client:
            test_embeddings = np.random.rand(client.embedding_dimension).tolist()
            client.insert(test_pk, test_embeddings)

    def test_multiple_clients(self):
        test_pk_val = 'test_pk_val'
        clients = list()
        for ii in range(3):
            client = DemoClient('test_user', 'test_password')
            clients.append(client)
            test_embeddings = np.random.rand(client.embedding_dimension).tolist()
            client.insert(test_pk_val, test_embeddings)
        for client in clients:
            client.close_and_delete_all()


class TestDemoClientLoop(BaseTest):
    def test(self):
        iterations = 100
        sys.stdout.write(f'Running {iterations} iterations of DemoClient '
                         f'construction/destruction...\n')
        sys.stdout.flush()

        for iteration in range(iterations):
            startt = time.time()
            demo_client = DemoClient('test_user', 'test_password')
            demo_client.close_and_delete_all()
            sys.stdout.write(f'\titeration: {iteration}, {time.time() - startt}\n')
            iteration += 1


class TestDemoClientOpenCloseLoop(TestCase):

    def test_this(self):
        print(f'emey was here')


class Test(TestDemoClientOpenCloseLoop):
    pass
