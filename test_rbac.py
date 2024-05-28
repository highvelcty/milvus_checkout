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

        # Initialize state
        cls.tearDownClass()

        cls._user1_client = cls._add_user(cls._root_client, User.USER1)
        cls._user2_client = cls._add_user(cls._root_client, User.USER2)

        # Create collections
        for user in User.all_:
            cls._add_collection(cls._root_client, User.collection[user])

    @classmethod
    def tearDownClass(cls):
        for username in (User.USER1, User.USER2):
            cls._rm_user(cls._root_client, username)

        for collection_name in cls._root_client.list_collections():
            cls._root_client.drop_collection(collection_name)

    def test_list_users(self):
        self.assertEqual(User.all_, self._root_client.list_users())

    def test_list_collections(self):
        self.assertEqual(sorted(User.collection.values()),
                         sorted(self._root_client.list_collections()))
        print(F'emey: {self._user1_client.list_collections()}')
        #
        # # The user's should not have admin-like access
        # self.assertFalse(self._user1_client.list_collections())

    @staticmethod
    def _add_user(root_client: MilvusClient, username: str) -> MilvusClient:
        root_client.create_user(username, password=User.password[username])

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                root_client.create_role(User.role[username])
        except pymilvus.exceptions.MilvusException as err:
            if 'already exists' not in str(err):
                raise err

        root_client.grant_privilege(role_name=User.role[username],
                                    object_type='Global',
                                    privilege='All',
                                    object_name='*')
        root_client.grant_privilege(role_name=User.role[username],
                                    object_type='Global',
                                    privilege='CreateCollection',
                                    object_name= 'CreateCollection')
        root_client.grant_privilege(role_name=User.role[username],
                                    object_type='Global',
                                    privilege='ShowCollections',
                                    object_name='ShowCollections')
        root_client.grant_role(username, User.role[username])

        return MilvusClient(URI, username, password=User.password[username])

    @staticmethod
    def _add_collection(client: MilvusClient, collection_name: str):
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

    @staticmethod
    def _rm_collection(root_client: MilvusClient, collection_name: str):
        root_client.drop_collection(collection_name)

    @staticmethod
    def _rm_user(root_client: MilvusClient, username: str):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                # .. note:: The typehint is List[Dict], but functionally it is Dict.
                # noinspection PyTypeChecker
                privileges = root_client.describe_role(User.role[username])['privileges']
        except pymilvus.exceptions.MilvusException:
            privileges = []

        # .. note:: describe role is type hinted as List[Dict], but the actual return is Dict.
        for privilege in privileges:
            root_client.revoke_privilege(User.role[username],
                                         privilege['object_type'],
                                         privilege['privilege'],
                                         privilege['object_name'])
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                root_client.revoke_role(username, User.role[username])
        except pymilvus.exceptions.MilvusException:
            pass

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                root_client.drop_role(User.role[username])
        except pymilvus.exceptions.MilvusException:
            pass

        root_client.drop_user(username)


if __name__ == '__main__':
    unittest.main()


    # def _add_privileges(self, user, object_type, privilege, object_name):
    #     # Create a role
    #     role = User.role[user]
    #
    #     try:
    #         with contextlib.redirect_stderr(io.StringIO()):
    #             self._root_client.create_role(role)
    #     except pymilvus.exceptions.MilvusException as err:
    #         if 'already exists' not in str(err):
    #             raise err
    #     # Grant privileges to the role
    #     self._root_client.grant_privilege(role, object_type, privilege, object_name)
    #     # Grant the role to the user
    #     self._root_client.grant_role(user, role)

    # def _rm_privileges(self, user: str):
    #     role = User.role[user]
    #
    #     try:
    #         with contextlib.redirect_stderr(io.StringIO()):
    #             # .. note:: The typehint is List[Dict], but functionally it is Dict.
    #             # noinspection PyTypeChecker
    #             privileges = self._root_client.describe_role(User.role[User.USER1])['privileges']
    #     except pymilvus.exceptions.MilvusException:
    #         privileges = []
    #
    #     # .. note:: describe role is type hinted as List[Dict], but the actual return is Dict.
    #     for privilege in privileges:
    #         self._root_client.revoke_privilege(role, privilege['object_type'],
    #                                            privilege['privilege'], privilege['object_name'])
    #     try:
    #         with contextlib.redirect_stderr(io.StringIO()):
    #             self._root_client.revoke_role(user, role)
    #     except pymilvus.exceptions.MilvusException:
    #         pass
    #
    #     try:
    #         with contextlib.redirect_stderr(io.StringIO()):
    #             self._root_client.drop_role(role)
    #     except pymilvus.exceptions.MilvusException:
    #         pass

    # def test_failed_login(self):
    #     # .. note:: Import of a protected exception from a protected module.
    #     # .. note:: This passes if `security.authorizationEnabled` is false.
    #     # .. note:: The need to squelch stdout
    #     from grpc._channel import _InactiveRpcError
    #     with self.assertRaises(_InactiveRpcError), \
    #          contextlib.redirect_stderr(io.StringIO()):
    #         MilvusClient(URI, user=User.USER1, password='something invalid')
    #
    # def test_all_admin_privileges(self):
    #     priv_params = (User.USER1,              # user
    #                    User.role[User.USER1],   # Role
    #                    'Global',                # object type
    #                    'All',                   # privilege
    #                    'All')                   # object_name
    #     collections_list_wo_privileges = self._user1_client.list_collections()
    #     self._add_privileges(*priv_params)
    #     collections_list_w_privileges = self._user1_client.list_collections()
    #
    #     self._rm_privileges(*priv_params)
    #
    #     self.assertFalse(collections_list_wo_privileges)
    #     self.assertEqual(sorted(User.collection.values()), sorted(collections_list_w_privileges))
    #
    # def test_load_collection_privileges(self):
    #
    #     # grpc._channel._MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated
    #     # with:
    #     # status = StatusCode.PERMISSION_DENIED
    #     # details = "PrivilegeLoad: permission deny to user1 in the `default` database"
    #     # debug_error_string = "UNKNOWN:Error received from peer  {grpc_message:"PrivilegeLoad:
    #     # permission deny to user1 in the `default` database", grpc_status:7,
    #     # created_time:"2024-05-24T15:14:31.03170218-06:00"}"
    #     #
    #     # .. note:: the import of the protected class from the protected module
    #     # .. note:: the redirection of stderr
    #     from grpc._channel import _MultiThreadedRendezvous
    #     with self.assertRaises(_MultiThreadedRendezvous), \
    #          contextlib.redirect_stderr(io.StringIO()):
    #         self._user1_client.load_collection(User.collection[User.USER1])
    #
    #     priv_params = (User.USER1, User.role[User.USER1], 'Collection', 'Load',
    #                    User.collection[User.USER1])
    #     self._add_privileges(*priv_params)
    #
    #     exception_hit = None
    #     try:
    #         self._user1_client.load_collection(User.collection[User.USER1])
    #     except _MultiThreadedRendezvous as err:
    #         exception_hit = err
    #     finally:
    #         self._rm_privileges(*priv_params)
    #
    #     self.assertFalse(exception_hit, f'Hit error "{exception_hit}"')