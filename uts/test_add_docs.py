from unittest import TestCase

from demo_client.get_query_engine import get_query_engine
from demo_client import DemoClient


class Test(TestCase):
    _test_user = 'test_user'
    _test_password = 'test_password'

    def test_get_query_engine(self):
        with DemoClient(self._test_user, self._test_password) as demo_client:
            query_engine = get_query_engine(demo_client)
            print(query_engine.query('Who is the author?'))
