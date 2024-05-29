import io
from typing import Dict, List
import contextlib
import os
import sys
import time

import pymilvus
from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous


class FieldName:
    PK = 'pk'
    EMBEDDINGS = 'embeddings'


class DemoClient:
    _CMD_TIMEOUT_SEC = 30
    _CMD_POLLING_SEC = 0.5

    def __init__(self, username: str, password: str,
                 uri: str = os.environ.get('MILVUS_DEMO_URI', 'http://localhost:19530'),
                 embedding_dimension: int = os.environ.get('MILVUS_DEMO_EMBEDDING_DIMENSION', 8)):
        self._username = username
        self._collection_name = username + '_collection'
        self._role_name = username + '_role'
        self._uri = uri
        self._embedding_dimension = embedding_dimension

        self._root_client = pymilvus.MilvusClient(uri, user='root', password='Milvus')
        self._root_client.create_user(username, password=password)

        self._user_client = self._add_user(uri, username, password)
        self._add_collection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def role_name(self):
        return self._role_name

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def username(self):
        return self._username

    def add_entities(self, entities: List[Dict]):
        pass

    def insert(self, primary_key_value: str, embeddings: List[float]):
        def callable_():
            self._user_client.insert(collection_name=self._collection_name,
                                     data={FieldName.PK: primary_key_value,
                                           FieldName.EMBEDDINGS: embeddings})
        self._wa_cmd_privileges(callable_, f'insert embeddings with primary key value '
                                           f'{primary_key_value} into collection '
                                           f'{self._collection_name} for user {self._username}.')

    def search(self):
        pass

    def close(self):
        self._rm_collection()
        self._rm_user()

        # .. note:: Workaround to make user removal a blocking operation.
        startt = time.time()
        timeout = startt + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            users = self._root_client.list_users()
            if self._username in users:
                time.sleep(self._CMD_POLLING_SEC)
                continue
            else:
                break
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds removing user'
                               f' {self._username}.')

    def _add_collection(self):
        schema = self._user_client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="A user collection for retrieval augmented generation (RAG).",
        )

        schema.add_field(field_name=FieldName.PK,
                         datatype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name=FieldName.EMBEDDINGS,
                         datatype=pymilvus.DataType.FLOAT_VECTOR, dim=self._embedding_dimension)

        def create_collection():
            self._user_client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                consistency_level="Strong"
            )
        self._wa_cmd_privileges(create_collection, f'create collection {self._collection_name} '
                                                   f'for user {self._username}.')

        index_params = self._user_client.prepare_index_params()

        index_params.add_index(field_name=FieldName.PK )

        index_params.add_index(
            field_name=FieldName.EMBEDDINGS,
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        self._wa_cmd_privileges(lambda: self._user_client.create_index(
            collection_name=self._collection_name,
            index_params=index_params), f'create index for {self._collection_name} for user '
                                        f'{self._username}.')

    def _add_user(self, uri, username, password) -> pymilvus.MilvusClient:
        user_client = pymilvus.MilvusClient(uri, username, password)
        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            if self._username in self._root_client.list_users():
                break
            else:
                time.sleep(self._CMD_POLLING_SEC)
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds waiting to create user '
                               f'{self._username}.')

        self._root_client.create_role(self._role_name)
        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            if self._role_name in self._root_client.list_roles():
                break
            else:
                time.sleep(self._CMD_POLLING_SEC)
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds waiting to create role '
                               f'{self._role_name} for user {self._username}.')

        requested_privileges = (('Global', 'All', '*'),
                                ('Global', 'CreateCollection', 'CreateCollection'),
                                ('Global', 'ShowCollections', 'ShowCollections'),
                                ('Collection', 'Load', '*'),
                                ('Collection', 'Search', '*'),
                                ('Collection', 'Query', '*'))

        exp_privileges = [{'object_type': 'Collection', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'GetLoadState', 'grantor_name': 'root'},
                          {'object_type': 'Collection', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'GetLoadingProgress', 'grantor_name': 'root'},
                          {'object_type': 'Collection', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'Load', 'grantor_name': 'root'},
                          {'object_type': 'Collection', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'Query', 'grantor_name': 'root'},
                          {'object_type': 'Collection', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'Search', 'grantor_name': 'root'},
                          {'object_type': 'Global', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'All', 'grantor_name': 'root'},
                          {'object_type': 'Global', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'CreateCollection', 'grantor_name': 'root'},
                          {'object_type': 'Global', 'object_name': '*',
                           'db_name': 'default', 'role_name': 'test_user_role',
                           'privilege': 'ShowCollections', 'grantor_name': 'root'}]

        for privilege in requested_privileges:
            self._root_client.grant_privilege(role_name=self._role_name,
                                              object_type=privilege[0],
                                              privilege=privilege[1],
                                              object_name=privilege[2])

        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            # .. note:: The typehint is List[Dict], but functionally it is Dict.
            # noinspection PyTypeChecker
            assigned_privileges = self._root_client.describe_role(self._role_name)['privileges']

            if all(exp_privilege in assigned_privileges for exp_privilege in exp_privileges):
                break
            else:
                time.sleep(self._CMD_POLLING_SEC)
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds waiting for privileges '
                               f'to be assigned to role {self._role_name} for user '
                               f'{self._username}.')

        self._root_client.grant_role(username, self._role_name)

        return user_client

    def _rm_collection(self):
        self._root_client.drop_collection(self._collection_name)

        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            if self._collection_name not in self._root_client.list_collections():
                break
            else:
                time.sleep(self._CMD_POLLING_SEC)
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds waiting to remove '
                               f'collection {self._collection_name} for user {self._username}.')

    def _rm_user(self):
        # .. note:: The typehint is List[Dict], but functionally it is Dict.
        # noinspection PyTypeChecker
        privileges = self._root_client.describe_role(self._role_name)['privileges']

        # .. note:: describe role is type hinted as List[Dict], but the actual return is Dict.
        for privilege in privileges:
            self._root_client.revoke_privilege(self._role_name,
                                               privilege['object_type'],
                                               privilege['privilege'],
                                               privilege['object_name'])

        self._root_client.revoke_role(self._username, self._role_name)

        self._root_client.drop_role(self._role_name)

        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            if self._role_name not in self._root_client.list_roles():
                break
            else:
                time.sleep(self._CMD_POLLING_SEC)
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} waiting to drop role '
                               f'{self._role_name} for user {self._username}.')

        self._root_client.drop_user(self._username)
        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            if self._username not in self._root_client.list_users():
                break
            else:
                time.sleep(self._CMD_POLLING_SEC)
        else:
            raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds waiting to drop '
                               f'username {self._username}.')

    def _wa_cmd_privileges(self, callable_, description: str = ''):
        stderr = io.StringIO()
        timeout = time.time() + self._CMD_TIMEOUT_SEC
        while time.time() < timeout:
            try:
                with contextlib.redirect_stderr(stderr):
                    callable_()
                    break
            except (_InactiveRpcError, _MultiThreadedRendezvous) as err:
                if 'PERMISSION_DENIED' in str(err):
                    time.sleep(self._CMD_POLLING_SEC)
                else:
                    stderr.seek(0)
                    sys.stderr.write(stderr.read())
                    raise err
        else:
            if description:
                raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC} seconds waiting to '
                                   f'{description}.')
            else:
                raise TimeoutError(f'Timeout of {self._CMD_TIMEOUT_SEC}.')
