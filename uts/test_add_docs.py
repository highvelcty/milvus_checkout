from unittest import TestCase

from demo_client.get_query_engine import get_query_engine


class Test(TestCase):
    def test_get_query_engine(self):
        query_engine = get_query_engine()
        print(query_engine.query('Who is the author?'))

