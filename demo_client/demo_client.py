import io
from typing import Any, Dict, List
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
    _CMD_TIMEOUT_SEC = 3
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_and_delete_all()

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

    def insert_multiple(self, entities: List[Dict]):
        pass

    def insert(self, primary_key_value: str, embeddings: List[float]):
        self._user_client.insert(collection_name=self._collection_name,
                                 data={FieldName.PK: primary_key_value,
                                       FieldName.EMBEDDINGS: embeddings})

    def search(self):
        pass

    def close_and_delete_all(self):
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

        schema.add_field(field_name=FieldName.PK,
                         datatype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name=FieldName.EMBEDDINGS,
                         datatype=pymilvus.DataType.FLOAT_VECTOR, dim=self._embedding_dimension)

        self._user_client.create_collection(collection_name=self._collection_name,
                                            schema=schema,
                                            consistency_level="Strong")

        index_params = self._user_client.prepare_index_params()

        index_params.add_index(field_name=FieldName.PK)

        index_params.add_index(
            field_name=FieldName.EMBEDDINGS,
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        self._user_client.create_index(collection_name=self._collection_name,
                                       index_params=index_params)

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
                                ('Global', 'FlushAll', '*'),
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
                                ('Collection', 'Flush', '*'),
                                ('Collection', 'GetFlushState', '*'),
                                ('Collection', 'Query', '*'),
                                ('Collection', 'GetStatistics', '*'))

        for privilege in requested_privileges:
            self._root_client.grant_privilege(role_name=self._role_name,
                                              object_type=privilege[0],
                                              privilege=privilege[1],
                                              object_name=privilege[2])

        self._root_client.grant_role(username, self._role_name)

        for privilege in requested_privileges:
            self._root_client.grant_privilege(role_name=self._role_name,
                                              object_type=privilege[0],
                                              privilege=privilege[1],
                                              object_name=privilege[2])

        self._root_client.grant_role(username, self._role_name)

        # This retry mechanism is a workaround for bug
        # https://github.com/milvus-io/milvus/issues/32632
        for retry in range(self._CMD_RETRIES):
            if retry:
                for privilege in requested_privileges:
                    self._root_client.grant_privilege(role_name=self._role_name,
                                                      object_type=privilege[0],
                                                      privilege=privilege[1],
                                                      object_name=privilege[2])
                time.sleep(self._CMD_POLLING_SEC)
                self._root_client.grant_role(username, self._role_name)
                time.sleep(self._CMD_POLLING_SEC)
            try:
                roles = self._wa_cmd_privileges(user_client.list_roles, 'list user granted roles')
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

    def _wa_cmd_privileges(self, callable_, description: str = '') -> Any:
        stderr = io.StringIO()
        timeout = time.time() + self._CMD_TIMEOUT_SEC
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
                raise TimeoutError(f'Timeout of {timeout} seconds waiting to '
                                   f'{description}.')
            else:
                raise TimeoutError(f'Timeout of {timeout}.')
