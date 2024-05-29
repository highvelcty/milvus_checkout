from unittest import TestCase

from milvus_client import DemoClient
import numpy as np


class TestDemoClient(TestCase):
    def test_construction(self):
        demo_client = DemoClient('test_user', 'test_password')
        demo_client.close()

    def test_context(self):
        test_username = 'test_user'
        with DemoClient(test_username, 'test_password') as client:
            self.assertEqual(test_username, client.username)

    def test_insert_and_search(self):
        test_pk = 'test_pk'
        with DemoClient('test_user', 'test_password') as client:
            test_embeddings = np.random.rand(client.embedding_dimension).tolist()
            client.insert(test_pk, test_embeddings)
