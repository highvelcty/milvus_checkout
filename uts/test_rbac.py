#! /usr/bin/env python
from typing import Any, Dict, List
import contextlib
import io
import random
import string
import time
import unittest

import numpy as np
from packaging.version import Version
import pymilvus.exceptions
from pymilvus import DataType, MilvusClient
from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous


URI = 'http://localhost:19530'

EMBEDDING_DIMENSION = 8
NUM_ENTITIES = 300
VECTOR_TO_SEARCH = np.random.rand(EMBEDDING_DIMENSION).tolist()


class User:
    ROOT = 'root'
    USER1 = 'user1'
    USER2 = 'user2'

    collection = {
        ROOT: f'{ROOT}_collection',
        USER1: f'{USER1}_collection',
        USER2: f'{USER2}_collection'
    }

    password = {
        ROOT: 'Milvus',
        USER1: 'user1_password',
        USER2: 'user2_password'
    }

    role = {
        USER1: f'{USER1}_role',
        USER2: f'{USER2}_role'
    }

    all_ = list(password.keys())


class FieldName:
    PK = 'pk'
    RANDOM = 'random'
    EMBEDDINGS = 'embeddings'


class BaseTest(unittest.TestCase):
    _root_client = None
    _user1_client = None
    _user2_client = None

    @classmethod
    def setUpClass(cls):
        # Create client
        cls._root_client = MilvusClient(URI, user=User.ROOT, password=User.password[User.ROOT])
        cls._user1_client = cls._add_user(User.USER1)
        cls._user2_client = cls._add_user(User.USER2)

    @classmethod
    def tearDownClass(cls):
        for username in (User.USER1, User.USER2):
            cls._rm_user(username)

    @classmethod
    def _add_user(cls, username: str) -> MilvusClient:
        cls._root_client.create_user(username, password=User.password[username])
        return MilvusClient(URI, username, password=User.password[username])

    @classmethod
    def _add_collection(cls, collection_name: str, client: MilvusClient = None):
        if client is None:
            client = cls._root_client
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="A test collection.",
        )

        schema.add_field(field_name=FieldName.PK,
                         datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name=FieldName.RANDOM, datatype=DataType.DOUBLE)
        schema.add_field(field_name=FieldName.EMBEDDINGS,
                         datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            consistency_level="Strong"
        )

        def generate_random_string(length):
            return ''.join(random.choice(string.ascii_letters + string.digits)
                           for _ in range(length))

        def generate_random_entities(num_entities, dim) -> List[Dict[str, Any]]:
            entities_ = []
            for _ in range(num_entities):
                pk = generate_random_string(10)
                random_value = random.random()
                embeddings = np.random.rand(dim).tolist()
                entities_.append({FieldName.PK: pk,
                                  FieldName.RANDOM: random_value,
                                  FieldName.EMBEDDINGS: embeddings})
            return entities_

        entities = generate_random_entities(NUM_ENTITIES, EMBEDDING_DIMENSION)
        entities.append({
            FieldName.PK: 'testing_pk',
            FieldName.RANDOM: 3.141593,
            FieldName.EMBEDDINGS: VECTOR_TO_SEARCH})

        client.insert(collection_name=collection_name, data=entities)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name=FieldName.PK
        )

        index_params.add_index(
            field_name=FieldName.EMBEDDINGS,
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )

    @classmethod
    def _rm_collection(cls, collection_name: str, client: MilvusClient = None):
        if client is None:
            client = cls._root_client
        client.drop_collection(collection_name)

    @classmethod
    def _rm_user(cls, username: str):
        cls._root_client.drop_user(username)


class Test(BaseTest):
    def test_list_users(self):
        self.assertEqual(User.all_, self._root_client.list_users())

    def test_list_collections(self):
        self._rm_collection(User.collection[User.ROOT])
        self.assertFalse(self._root_client.list_collections())
        self.assertEqual([], self._user1_client.list_collections())
        self._add_collection(User.collection[User.ROOT])
        self.assertEqual([User.collection[User.ROOT]], self._root_client.list_collections())
        self.assertEqual([], self._user1_client.list_collections())
        self._rm_collection(User.collection[User.ROOT])
        self.assertEqual([], self._root_client.list_collections())

    def test_create_collection(self):
        self._add_collection(User.collection[User.ROOT])
        self._rm_collection(User.collection[User.ROOT])

        with self.assertRaises(_MultiThreadedRendezvous), \
             contextlib.redirect_stderr(io.StringIO()):
            self._add_collection(User.collection[User.USER1], self._user1_client)


class TestWithPrivileges(BaseTest):
    def test_list_collections(self):
        exp_version = Version('2.4.3')
        if Version(pymilvus.__version__) <= exp_version:
            raise unittest.SkipTest(f'{pymilvus.__version__} is less than or equal '
                                    f'to {exp_version}.\n'
                                    f'  See: https://github.com/milvus-io/milvus/issues/33382')
        error_str = ''
        self._add_privileges(User.USER1, 'Global', 'All', '*')
        self._add_privileges(User.USER1, 'Global', 'CreateCollection', 'CreateCollection')
        self._add_privileges(User.USER1, 'Global', 'ShowCollections', 'ShowCollections')
        try:
            self._add_collection(User.collection[User.USER1], self._user1_client)
        except (_InactiveRpcError, _MultiThreadedRendezvous) as err:
            error_str = f'Permission denied hit while creating user collection:\n{err}'

        self._add_collection(User.collection[User.ROOT])
        user_visible_collections = self._user1_client.list_collections()
        root_visible_collections = self._root_client.list_collections()

        if not error_str:
            try:
                self._rm_collection(User.collection[User.USER1], self._user1_client)
            except (_InactiveRpcError, _MultiThreadedRendezvous) as err:
                error_str = f'Permission denied hit while removing user collection:\n{err}'

        self._rm_collection(User.collection[User.ROOT], self._root_client)
        self._rm_role_and_privileges(User.USER1)

        self.assertEqual(root_visible_collections, user_visible_collections)
        self.assertTrue(user_visible_collections)
        self.assertFalse(error_str)

    def test_load_and_search_collection(self):
        timeout_sec = 30
        delay_sec = 5
        error_str = ''

        # Add collection creation privileges
        self._add_privileges(User.USER1, 'Global', 'All', '*')
        self._add_privileges(User.USER1, 'Global', 'CreateCollection', 'CreateCollection')
        time.sleep(delay_sec)

        # Create a collection
        try:
            self._add_collection(User.collection[User.USER1], self._user1_client)
        except (_InactiveRpcError, _MultiThreadedRendezvous) as err:
            error_str = (f'Permission denied hit while creating collection '
                         f'{User.collection[User.USER1]}.\n{err}')
        time.sleep(delay_sec)

        # Add collection loading privileges
        self._add_privileges(User.USER1, 'Collection', 'Load', '*')
        time.sleep(delay_sec)

        # Load collection
        try:
            self._user1_client.load_collection(User.collection[User.USER1])
        except (_InactiveRpcError, _MultiThreadedRendezvous):
            error_str = f'Permission denied hit when attempting to load {User.USER1} collection.'
        except pymilvus.exceptions.MilvusException as err:
            if 'collection not found' in err:
                error_str = (f'Collection not found error hit when attempting to load '
                             f'collection "{User.collection[User.USER1]}".\n{err}')
            else:
                raise

        # Search collection
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        result = self._user1_client.search(
            collection_name=User.collection[User.USER1],
            data=[VECTOR_TO_SEARCH],
            anns_field="embeddings",
            search_params=search_params,
            limit=3,
            output_fields=["random"]
        )

        print(result)

        # Remove collection
        try:
            self._rm_collection(User.collection[User.USER1], self._user1_client)
        except (_InactiveRpcError, _MultiThreadedRendezvous) as err:
            error_str = (f'Permission denied when attempting to remove collection "'
                         f'{User.collection[User.USER1]}.\n{err}')

        # Remove role and privileges
        self._rm_role_and_privileges(User.USER1)

        # Assertions
        self.assertFalse(error_str)

    @classmethod
    def _add_privileges(cls, username: str, object_type: str, privilege: str, object_name: str):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cls._root_client.create_role(User.role[username])
        except pymilvus.exceptions.MilvusException as err:
            if 'already exists' not in str(err):
                raise err

        cls._root_client.grant_privilege(role_name=User.role[username],
                                         object_type='Global',
                                         privilege='All',
                                         object_name='*')
        cls._root_client.grant_privilege(role_name=User.role[username],
                                         object_type='Global',
                                         privilege='CreateCollection',
                                         object_name='CreateCollection')
        cls._root_client.grant_privilege(role_name=User.role[username],
                                         object_type='Global',
                                         privilege='ShowCollections',
                                         object_name='ShowCollections')
        cls._root_client.grant_role(username, User.role[username])

    @classmethod
    def _rm_role_and_privileges(cls, username: str):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                # .. note:: The typehint is List[Dict], but functionally it is Dict.
                # noinspection PyTypeChecker
                privileges = cls._root_client.describe_role(User.role[username])['privileges']
        except pymilvus.exceptions.MilvusException:
            privileges = []

        # .. note:: describe role is type hinted as List[Dict], but the actual return is Dict.
        for privilege in privileges:
            cls._root_client.revoke_privilege(User.role[username],
                                              privilege['object_type'],
                                              privilege['privilege'],
                                              privilege['object_name'])
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cls._root_client.revoke_role(username, User.role[username])
        except pymilvus.exceptions.MilvusException:
            pass

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cls._root_client.drop_role(User.role[username])
        except pymilvus.exceptions.MilvusException:
            pass

    @classmethod
    def _rm_user(cls, username: str):
        cls._rm_role_and_privileges(username)
        super()._rm_user(username)


if __name__ == '__main__':
    unittest.main()
