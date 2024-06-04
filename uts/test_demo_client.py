from unittest import skipIf, TestCase
import shlex
import subprocess
import sys

import time

from repo_tools import paths
from demo_client import CollectionEntity, DemoClient
import numpy as np


class BaseLocalTest(TestCase):
    """
    Subclass this will cause a local milvus instance to be created/destroyed encompassing the
    class tests.
    """
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


class TestDemoClient(TestCase):
    _test_user = 'test_user'
    _test_password = 'test_password'
    _test_pk_val = 'test_pk_val'

    def test_construction_and_removal(self):
        client = DemoClient(self._test_user, self._test_password)
        client.remove_all()

    def test_context(self):
        with DemoClient(self._test_user, self._test_password):
            pass

    def test_close(self):
        DemoClient(self._test_user, self._test_password).close()

    def test_remove_all_and_close(self):
        client = DemoClient(self._test_user, self._test_password)
        client.remove_all()
        client.close()

    def test_insert_and_search(self):
        client = DemoClient(self._test_user, self._test_password)

        entities = list()
        for ii in range(10):
            entities.append(
                CollectionEntity(pk=f'entity_{ii}',
                                 embeddings=np.random.rand(client.vector_dimension)))

        client.insert(*entities)

        results = client.search(entities[-1].embeddings)

        self.assertEqual({'id': 'entity_9', 'distance': 0.0, 'entity': {'pk': 'entity_9'}},
                         results[0][0])

    def test_multiple_clients(self):
        clients = list()
        for ii in range(3):
            client = DemoClient(self._test_user, self._test_password)
            clients.append(client)
            test_embeddings = np.random.rand(client.vector_dimension)
            client.insert(CollectionEntity(pk=self._test_pk_val, embeddings=test_embeddings))
        for client in clients:
            client.remove_all()

    def test_list_collections(self):
        client = DemoClient(self._test_user, self._test_password)
        print(client.user_client.list_collections())


@skipIf(False, 'Long running test.')
class TestDemoClientLoop(TestCase):
    def test(self):
        iterations = 100
        print(f'Running {iterations} iterations of DemoClient construction and removal...')
        print(f'iteration, elapsed_time_seconds')

        for iteration in range(iterations):
            startt = time.time()
            demo_client = DemoClient('test_user', 'test_password')
            demo_client.remove_all()
            #
            demo_client.close()
            print(f'{iteration}, {time.time() - startt}')
            iteration += 1
