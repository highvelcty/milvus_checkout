import io
from typing import Any, Union
import contextlib
import os
import sys
import time

import numpy as np
import pymilvus
from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous


class CollectionEntity(dict):
    class Key:
        PK = 'pk'
        EMBEDDINGS = 'embeddings'

    def __init__(self, dict_=None, /, **kwargs):
        if dict_ is None:
            super().__init__(**kwargs)
        else:
            super().__init__(dict_, **kwargs)

    @property
    def pk(self) -> str:
        return self[self.Key.PK]

    @pk.setter
    def pk(self, value: str):
        self[self.Key.PK] = value

    @property
    def embeddings(self) -> np.array:
        return self[self.Key.EMBEDDINGS]

    @embeddings.setter
    def embeddings(self, value: np.array):
        self[self.Key.EMBEDDINGS] = value


class DemoClient:
    _LOAD_COLLECTION_TIMEOUT_SEC = 30
    _LIST_ROLES_TIMEOUT_SEC = 3
    _GRANT_PRIVILEGES_TIMEOUT_SEC = 10
    _CMD_POLLING_SEC = 1
    _CMD_RETRIES = 10

    def __init__(self, username: str, password: str,
                 uri: str = os.environ.get('MILVUS_DEMO_URI', 'http://localhost:19530'),
                 embedding_dimension: int = os.environ.get('MILVUS_DEMO_EMBEDDING_DIMENSION', 8)):
        self._username = username
        self._collection_name = username + '_collection'
        self._role_name = username + '_role'
        self._uri = uri
        self._embedding_dimension = embedding_dimension

        self._root_client = pymilvus.MilvusClient(uri, user='root', password='Milvus')

        self._user_client = self._add_user(uri, username, password)
        self._add_collection()

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

    def insert(self, *entities: Union[CollectionEntity, dict]):
        self._user_client.insert(collection_name=self._collection_name,
                                 data=list(entities))

    def search(self, *vectors_to_search: np.array, limit: int = 3):
        return self._user_client.search(
            collection_name=self._collection_name,
            data=list(vectors_to_search),
            anns_field=CollectionEntity.Key.EMBEDDINGS,
            search_params={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=[CollectionEntity.Key.PK]
        )

    def remove_all(self):
        if self._collection_name in self._root_client.list_collections():
            self._rm_collection()
        if self._username in self._root_client.list_users():
            self._rm_user()
        self._root_client.close()

    def _add_collection(self):
        if self._collection_name in self._root_client.list_collections():
            return

        schema = self._user_client.create_schema(
            auto_id=False,
            enable_dynamic_fields=True,
            description="A user collection for retrieval augmented generation (RAG).",
        )

        schema.add_field(field_name=CollectionEntity.Key.PK,
                         datatype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name=CollectionEntity.Key.EMBEDDINGS,
                         datatype=pymilvus.DataType.FLOAT_VECTOR, dim=self._embedding_dimension)

        self._user_client.create_collection(collection_name=self._collection_name,
                                            schema=schema,
                                            consistency_level="Strong")

        index_params = self._user_client.prepare_index_params()

        conn = self._user_client._get_connection()

        index_params.add_index(field_name=CollectionEntity.Key.PK)

        index_params.add_index(
            field_name=CollectionEntity.Key.EMBEDDINGS,
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        self._user_client.create_index(collection_name=self._collection_name,
                                       index_params=index_params)

        for index_name in self._user_client.list_indexes(self._collection_name):
            conn.wait_for_creating_index(self._collection_name, index_name)

        self._wa_cmd_privileges(lambda: self._user_client.load_collection(self._collection_name),
                                self._LOAD_COLLECTION_TIMEOUT_SEC,
                                f'load collection {self._collection_name}')
        conn.wait_for_loading_collection(self._collection_name)

    def _add_user(self, uri, username, password) -> pymilvus.MilvusClient:
        if self._username in self._root_client.list_users():
            return pymilvus.MilvusClient(uri, username, password)

        self._root_client.create_user(username, password=password)

        user_client = pymilvus.MilvusClient(uri, username, password)

        self._root_client.create_role(self._role_name)

        requested_privileges = (('Global', 'All', '*'),
                                ('Global', 'CreateCollection', '*'),
                                ('Global', 'DropCollection', '*'),
                                ('Global', 'DescribeCollection', '*'),
                                ('Global', 'ShowCollections', '*'),
                                ('Collection', 'CreateIndex', '*'),
                                ('Collection', 'DropIndex', '*'),
                                ('Collection', 'IndexDetail', '*'),
                                ('Collection', 'Load', '*'),
                                ('Collection', 'GetLoadingProgress', '*'),
                                ('Collection', 'GetLoadState', '*'),
                                ('Collection', 'Release', '*'),
                                ('Collection', 'Insert', '*'),
                                ('Collection', 'Delete', '*'),
                                ('Collection', 'Upsert', '*'),
                                ('Collection', 'Search', '*'),
                                ('Collection', 'Query', '*'),
                                ('Collection', 'GetStatistics', '*'))

        # This retry mechanism is a workaround for bug
        # https://github.com/milvus-io/milvus/issues/32632
        for retry in range(self._CMD_RETRIES):
            for privilege in requested_privileges:
                self._root_client.grant_privilege(role_name=self._role_name,
                                                  object_type=privilege[0],
                                                  privilege=privilege[1],
                                                  object_name=privilege[2])

            time.sleep(self._CMD_POLLING_SEC)
            timeout = time.time() + self._GRANT_PRIVILEGES_TIMEOUT_SEC
            exp_privilege_count = 18
            while time.time() < timeout:
                # .. note:: The typehint is List[Dict], but functionally it is Dict.
                # noinspection PyTypeChecker
                privilege_count = (
                    len(self._root_client.describe_role(self._role_name)['privileges']))
                if exp_privilege_count <= privilege_count:
                    break
                else:
                    time.sleep(self._CMD_POLLING_SEC)
            else:
                raise TimeoutError(f'Timeout of {self._GRANT_PRIVILEGES_TIMEOUT_SEC} seconds '
                                   f'waiting to grant permission to role {self._role_name} for '
                                   f'user {self._username}.')

            if retry:
                time.sleep(self._CMD_POLLING_SEC)
            self._root_client.grant_role(username, self._role_name)
            if retry:
                time.sleep(self._CMD_POLLING_SEC)

            try:
                roles = self._wa_cmd_privileges(user_client.list_roles,
                                                self._LIST_ROLES_TIMEOUT_SEC,
                                                'list user granted roles')
                if self._role_name in roles:
                    break
                else:
                    continue
            except TimeoutError:
                continue
        else:
            raise Exception('Failed to list user granted role.')

        return user_client

    def _rm_collection(self):
        self._root_client.drop_collection(self._collection_name)

    def _rm_user(self):
        # .. note:: The typehint is List[Dict], but functionally it is Dict.
        # noinspection PyTypeChecker
        privileges = self._root_client.describe_role(self._role_name)['privileges']

        for privilege in privileges:
            self._root_client.revoke_privilege(self._role_name,
                                               privilege['object_type'],
                                               privilege['privilege'],
                                               privilege['object_name'])

        self._root_client.revoke_role(self._username, self._role_name)
        self._root_client.drop_role(self._role_name)
        self._user_client.close()
        self._root_client.drop_user(self._username)

    def _wa_cmd_privileges(self, callable_, timeout_sec: int, description: str = '') -> Any:
        stderr = io.StringIO()
        timeout = time.time() + timeout_sec
        while time.time() < timeout:
            try:
                with contextlib.redirect_stderr(stderr):
                    return callable_()
            except (_InactiveRpcError, _MultiThreadedRendezvous) as err:
                if 'PERMISSION_DENIED' in str(err):
                    time.sleep(self._CMD_POLLING_SEC)
                else:
                    stderr.seek(0)
                    sys.stderr.write(stderr.read())
                    raise err
        else:
            if description:
                raise TimeoutError(f'Timeout of {timeout_sec} seconds waiting to '
                                   f'{description}.')
            else:
                raise TimeoutError(f'Timeout of {timeout_sec}.')
