from unittest import skipIf, TestCase
import time

from demo_client.base_rbac_rag_client import (UserPassCollection, MILVUS_ROOT_USERNAME,
                                              MILVUS_ROOT_PASSWORD)
from demo_client.llamaidx_rbac_rag_client import LlamaIdxRbacRagClient
from pymilvus import MilvusClient
from repo_tools import paths


class Base(TestCase):
    _milvus_host = 'http://localhost:19530'


class Test(Base):
    _test_user = 'test_user'
    _test_password = 'test_password'

    def test_get_query_engine(self):
        with (LlamaIdxRbacRagClient(uri=self._milvus_host, username=self._test_user,
                                    password=self._test_password) as client):
            client.add_documents(paths.RootPath.PKG / 'docs')
            resp = client.query('Who is the author?')
            from pprint import pprint
            pprint(resp)
            self.assertTrue('Paul Graham' in str(resp))
            client.remove_all()

    def test_open_from_existing(self):
        client = LlamaIdxRbacRagClient(self._milvus_host,
                                       username=self._test_user, password=self._test_password)
        client.add_documents(paths.RootPath.PKG / 'docs')

        client.close()

        client = LlamaIdxRbacRagClient(self._milvus_host,
                                       username=self._test_user, password=self._test_password)
        resp = client.query("What is the author's last name?")
        self.assertTrue('Graham' in str(resp))
        client.remove_all()


class TestUserPass(Base):
    _test_user = 'test_user'
    _test_password = 'test_password'
    _root_client = None

    def test_create_new_collection_and_new_user(self):
        client = LlamaIdxRbacRagClient(uri=self._milvus_host,
                                       username=self._test_user, password=self._test_password)
        client.close()

        root_client = MilvusClient(uri=self._milvus_host,
                                   user=MILVUS_ROOT_USERNAME, password=MILVUS_ROOT_PASSWORD)

        self.assertTrue(UserPassCollection._NAME in root_client.list_collections())
        root_client.drop_collection(UserPassCollection._NAME)
        self.assertFalse(UserPassCollection._NAME in root_client.list_collections())

        client = LlamaIdxRbacRagClient(uri=self._milvus_host,
                                       username=self._test_user, password=self._test_password)
        client.remove_all()

    def test_invalid_password(self):
        with (LlamaIdxRbacRagClient(uri=self._milvus_host, username=self._test_user,
                                    password=self._test_password) as client):
            with self.assertRaises(PermissionError):
                LlamaIdxRbacRagClient(uri=self._milvus_host,
                                      username=self._test_user, password='invalid_password')
            client.remove_all()


@skipIf(True, 'Long running test.')
class TestDemoClientLoop(Base):
    def test(self):
        iterations = 100
        print(f'Running {iterations} iterations of DemoClient construction and removal...')
        print(f'iteration, elapsed_time_seconds')

        for iteration in range(iterations):
            startt = time.time()
            client = LlamaIdxRbacRagClient(username='test_user', password='test_password')
            client.remove_all()
            #
            client.close()
            print(f'{iteration}, {time.time() - startt}')
            iteration += 1
