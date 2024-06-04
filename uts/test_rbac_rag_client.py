from unittest import skipIf, TestCase
import time

from demo_client.rbac_rag_client import (MILVUS_HOST, RbacRagClient, UserPassCollection,
                                         MILVUS_ROOT_USERNAME, MILVUS_ROOT_PASSWORD)
from pymilvus import MilvusClient


# Notes from talking with Bijan
# All directories except "code" directory will be documents to be indexed
#
from repo_tools import paths


class Test(TestCase):
    _test_user = 'test_user'
    _test_password = 'test_password'

    def test_get_query_engine(self):
        with RbacRagClient(username=self._test_user, password=self._test_password) as client:
            client.add_documents(paths.RootPath.PKG / 'docs')
            print(client.query_engine.query('Who is the author?'))
            client.remove_all()

    def test_open_from_existing(self):
        client = RbacRagClient(username=self._test_user, password=self._test_password)
        client.add_documents(paths.RootPath.PKG / 'docs')

        client.close()

        client = RbacRagClient(username=self._test_user, password=self._test_password)

        print(client.query_engine.query('Who is the author?'))
        print(client.query_engine.query("What is the author's last name?"))

        client.remove_all()


class TestUserPass(TestCase):
    _test_user = 'test_user'
    _test_password = 'test_password'
    _root_client = None

    def test_create_new_collection_and_new_user(self):
        client = RbacRagClient(username=self._test_user, password=self._test_password)
        client.close()

        root_client = MilvusClient(uri=MILVUS_HOST,
                                   user=MILVUS_ROOT_USERNAME, password=MILVUS_ROOT_PASSWORD)

        self.assertTrue(UserPassCollection._NAME in root_client.list_collections())
        root_client.drop_collection(UserPassCollection._NAME)
        self.assertFalse(UserPassCollection._NAME in root_client.list_collections())

        client = RbacRagClient(username=self._test_user, password=self._test_password)
        client.remove_all()

    def test_invalid_password(self):
        with RbacRagClient(username=self._test_user, password=self._test_password) as client:
            with self.assertRaises(PermissionError):
                RbacRagClient(username=self._test_user, password='invalid_password')
            client.remove_all()


@skipIf(True, 'Long running test.')
class TestDemoClientLoop(TestCase):
    def test(self):
        iterations = 100
        print(f'Running {iterations} iterations of DemoClient construction and removal...')
        print(f'iteration, elapsed_time_seconds')

        for iteration in range(iterations):
            startt = time.time()
            client = RbacRagClient(username='test_user', password='test_password')
            client.remove_all()
            #
            client.close()
            print(f'{iteration}, {time.time() - startt}')
            iteration += 1