#! /usr/bin/env python
from typing import Any, Dict, List
import contextlib
import io
import random
import string
import unittest

import numpy as np
import pymilvus.exceptions
from pymilvus import DataType, MilvusClient


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


class Test(unittest.TestCase):
    _root_client = None
    _user1_client = None
    _user2_client = None

    @classmethod
    def setUpClass(cls):
        # Create client
        cls._root_client = MilvusClient(URI, user=User.ROOT, password=User.password[User.ROOT])

        # Ensure Milvus state
        cls._initialize_milvus_state()

        # Create users
        cls._root_client.create_user(User.USER1, password=User.password[User.USER1])
        cls._root_client.create_user(User.USER2, password=User.password[User.USER2])

        # Create user clients
        cls._user1_client = MilvusClient(URI, user=User.USER1, password=User.password[User.USER1])
        cls._user2_client = MilvusClient(URI, user=User.USER2, password=User.password[User.USER2])

        # Create collections
        cls._create_collection(cls._root_client, User.collection[User.ROOT])

    @classmethod
    def tearDownClass(cls):
        cls._initialize_milvus_state()

    @classmethod
    def _initialize_milvus_state(cls):
        # .. note:: This does raise an exception if the role does not exist.
        # .. note:: stderr has to be redirected
        for role in User.role.values():
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    cls._root_client.drop_role(role)
            except pymilvus.exceptions.MilvusException:
                pass

        # .. note:: This does not raise an exception if the user does not exist.
        cls._root_client.drop_user(user_name=User.USER1)
        cls._root_client.drop_user(user_name=User.USER2)

        # .. note:: this does not raise an exception if the collection does not exist.
        for collection in User.collection.values():
            cls._root_client.drop_collection(collection)

    @staticmethod
    def _create_collection(client: MilvusClient, collection_name: str):
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

    def test_list_users(self):
        self.assertEqual(User.all_, self._root_client.list_users())

    def test_list_collections(self):
        self.assertEqual(sorted(User.collection.values()),
                         sorted(self._root_client.list_collections()))

        # The user's should not have admin-like access
        self.assertFalse(self._user1_client.list_collections())

    def test_failed_login(self):
        # .. note:: Import of a protected exception from a protected module.
        # .. note:: This passes if `security.authorizationEnabled` is false.
        # .. note:: The need to squelch stdout
        from grpc._channel import _InactiveRpcError
        with self.assertRaises(_InactiveRpcError), \
             contextlib.redirect_stderr(io.StringIO()):
            MilvusClient(URI, user=User.USER1, password='something invalid')

    def test_all_admin_privileges(self):
        priv_params = (User.USER1,              # user
                       User.role[User.USER1],   # Role
                       'Global',                # object type
                       'All',                   # privilege
                       'All')                   # object_name
        collections_list_wo_privileges = self._user1_client.list_collections()
        self._add_privileges(*priv_params)
        collections_list_w_privileges = self._user1_client.list_collections()

        self._rm_privileges(*priv_params)

        self.assertFalse(collections_list_wo_privileges)
        self.assertEqual(sorted(User.collection.values()), sorted(collections_list_w_privileges))

    def test_load_collection_privileges(self):

        # grpc._channel._MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated
        # with:
        # status = StatusCode.PERMISSION_DENIED
        # details = "PrivilegeLoad: permission deny to user1 in the `default` database"
        # debug_error_string = "UNKNOWN:Error received from peer  {grpc_message:"PrivilegeLoad:
        # permission deny to user1 in the `default` database", grpc_status:7,
        # created_time:"2024-05-24T15:14:31.03170218-06:00"}"
        #
        # .. note:: the import of the protected class from the protected module
        # .. note:: the redirection of stderr
        from grpc._channel import _MultiThreadedRendezvous
        with self.assertRaises(_MultiThreadedRendezvous), \
             contextlib.redirect_stderr(io.StringIO()):
            self._user1_client.load_collection(User.collection[User.USER1])

        priv_params = (User.USER1, User.role[User.USER1], 'Collection', 'Load',
                       User.collection[User.USER1])
        self._add_privileges(*priv_params)

        exception_hit = None
        try:
            self._user1_client.load_collection(User.collection[User.USER1])
        except _MultiThreadedRendezvous as err:
            exception_hit = err
        finally:
            self._rm_privileges(*priv_params)

        self.assertFalse(exception_hit, f'Hit error "{exception_hit}"')

    def test_create_user_db(self):
        self._add_privileges(User.USER1, object_type='Global', privilege='CreateCollection',
                             object_name='CreateCollection')
        # self._add_privileges(User.USER1, object_type='Global', privilege='*',
        #                      object_name='*')
        self._add_privileges(User.USER1, object_type='Global', privilege='All',
                             object_name='All')
        import time
        time.sleep(5)
        try:
            self._create_collection(self._user1_client, User.collection[User.USER1])
        except Exception as err:
            print(f'failure!')

        import time

        time.sleep(5)

        self._rm_privileges(User.USER1)

    def _add_privileges(self, user, object_type, privilege, object_name):
        # Create a role
        role = User.role[user]

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                self._root_client.create_role(role)
        except pymilvus.exceptions.MilvusException as err:
            if 'already exists' not in str(err):
                raise err
        # Grant privileges to the role
        self._root_client.grant_privilege(role, object_type, privilege, object_name)
        # Grant the role to the user
        self._root_client.grant_role(user, role)

    def _rm_privileges(self, user: str):
        role = User.role[user]

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                # .. note:: The typehint is List[Dict], but functionally it is Dict.
                # noinspection PyTypeChecker
                privileges = self._root_client.describe_role(User.role[User.USER1])['privileges']
        except pymilvus.exceptions.MilvusException:
            privileges = []

        # .. note:: describe role is type hinted as List[Dict], but the actual return is Dict.
        for privilege in privileges:
            self._root_client.revoke_privilege(role, privilege['object_type'],
                                               privilege['privilege'], privilege['object_name'])
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                self._root_client.revoke_role(user, role)
        except pymilvus.exceptions.MilvusException:
            pass

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                self._root_client.drop_role(role)
        except pymilvus.exceptions.MilvusException:
            pass


if __name__ == '__main__':
    unittest.main()
