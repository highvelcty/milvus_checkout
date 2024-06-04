import io
from typing import Any, Dict, Tuple, Union
import contextlib
import os
import sys
import time

# meyere, this to be provided by the demo group
from demo_client.utils import NeMoEmbedding, login_aioli, get_api_key

from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient


AIOLI_HOST = os.environ['AIOLI_HOST']
CHAT_API_BASE = f'http://{os.environ["AIOLI_CHAT_HOST"]}/v1'
AIOLI_USERNAME = os.environ['AIOLI_USER']
AIOLI_PASSWORD = os.environ['AIOLI_PW']

LOCAL_MILVUS_TESTING = True
if LOCAL_MILVUS_TESTING:
    MILVUS_HOST = f'http://localhost:19530'
else:
    MILVUS_HOST = f'http://{AIOLI_HOST}:19530'

PR_DIRS = ('/pfs/parse-hpe', '/pfs/parse-nvidia')


class RbacMilvusVectorStore(MilvusVectorStore):
    """
    Customized for RBAC robustness.
    """
    _LIST_COLLECTION_RETRY_POLLING_SEC = 1
    _LIST_COLLECTION_RETRIES = 5
    _LIST_COLLECTION_TIMEOUT_SEC = 5

    def _create_index_if_required(self, force: bool = False) -> None:
        """
        This method was copied from the llama_index codebase so that blocking/robustness RBAC hooks
        can be added post collection creation, post index creation and post collection load.
        """
        # Robustness improvement - Wait for the collection to become ready
        for retry in range(self._LIST_COLLECTION_RETRIES):
            if self._milvusclient.list_collections():
                break
            else:
                time.sleep(self._LIST_COLLECTION_RETRY_POLLING_SEC)
        else:
            raise Exception(f'Exhausted {self._LIST_COLLECTION_RETRIES} retries waiting for '
                            f'a collection to be created.')

        conn = self._milvusclient._get_connection()

        if self.enable_sparse is False:
            if (self._collection.has_index() and self.overwrite) or force:
                self._collection.release()
                self._collection.drop_index()
                base_params: Dict[str, Any] = self.index_config.copy()
                index_type: str = base_params.pop("index_type", "FLAT")
                index_params: Dict[str, Union[str, Dict[str, Any]]] = {
                    "params": base_params,
                    "metric_type": self.similarity_metric,
                    "index_type": index_type,
                }
                self._collection.create_index( self.embedding_field, index_params=index_params)

                # Robustness improvement - Wait for indices to become ready
                for index_name in self._milvusclient.list_indexes(self._collection_name):
                    conn.wait_for_creating_index(self._collection_name, index_name)

                # Robustness improvement - Wait for the collection to load
                self._wa_cmd_privileges(
                    lambda: self._collection.load,
                    self._LOAD_COLLECTION_TIMEOUT_SEC,
                    f'load collection {self._collection_name}')
                conn.wait_for_loading_collection(self._collection_name)
        else:
            if (
                self._collection.has_index(index_name=self.embedding_field)
                and self.overwrite
            ) or force:
                if self._collection.has_index(index_name=self.embedding_field) is True:
                    self._collection.release()
                    self._collection.drop_index(index_name=self.embedding_field)
                if (
                    self._collection.has_index(index_name=self.sparse_embedding_field)
                    is True
                ):
                    self._collection.drop_index(index_name=self.sparse_embedding_field)

                self._create_hybrid_index(self.collection_name)

                # Robustness improvement - Wait for indices to become read
                for index_name in self._milvusclient.list_indexes(self._collection_name):
                    conn.wait_for_creating_index(self._collection_name, index_name)

                # Robustness improvement - Wait for the collection to load
                self._wa_cmd_privileges(
                    lambda: self._collection.load,
                    self._LOAD_COLLECTION_TIMEOUT_SEC,
                    f'load collection {self._collection_name}')
                conn.wait_for_loading_collection(self._collection_name)


class MilvusRbacRagClient:
    _LOAD_COLLECTION_TIMEOUT_SEC = 30
    _LIST_ROLES_TIMEOUT_SEC = 3
    _GRANT_PRIVILEGES_TIMEOUT_SEC = 10
    _CMD_POLLING_SEC = 1
    _CMD_RETRIES = 10

    def __init__(self, username: str, password: str, uri: str = MILVUS_HOST,
                 vector_dimension: int = 1024):
        self._username = username
        self._password = password
        self._collection_name = username + '_collection'
        self._role_name = username + '_role'
        self._uri = uri
        self._vector_dimension = vector_dimension

        self._root_client = MilvusClient(uri, user='root', password='Milvus')
        self._user_client = self._add_user(uri, username, password)
        self._index = self._add_rag_index()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_dimension(self) -> int:
        return self._vector_dimension

    @property
    def role_name(self):
        return self._role_name

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def query_engine(self) -> BaseQueryEngine:
        return self._index.as_query_engine()

    @property
    def root_client(self) -> MilvusClient:
        return self._root_client

    @property
    def username(self) -> str:
        return self._username

    @property
    def user_client(self) -> MilvusClient:
        return self._user_client

    def add_documents(self, *paths_to_doc_dirs):
        for path_to_doc_dir in paths_to_doc_dirs:
            documents = SimpleDirectoryReader(str(path_to_doc_dir)).load_data()
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=self._index.storage_context)

    def close(self):
        self._user_client.close()
        self._root_client.close()

    def remove_all(self):
        if self._collection_name in self._root_client.list_collections():
            self._rm_collection()
        if self._username in self._root_client.list_users():
            self._rm_user()
        self._root_client.close()

    def _add_privileges(self, user_client):
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

        # workaround for https://github.com/milvus-io/milvus/issues/32632
        for retry in range(self._CMD_RETRIES):
            for privilege in requested_privileges:
                self._root_client.grant_privilege(role_name=self._role_name,
                                                  object_type=privilege[0],
                                                  privilege=privilege[1],
                                                  object_name=privilege[2])

            time.sleep(self._CMD_POLLING_SEC)
            timeout = time.time() + self._GRANT_PRIVILEGES_TIMEOUT_SEC

            # 18 was chosen from experimentation, e.g `time.sleep(60); print(len(privileges))`
            # pymilvus.__version__ is 2.4.3
            # server version
            exp_privilege_count = 18
            while time.time() < timeout:
                # .. note:: The typehint is List[Dict], but functionally it is Dict.
                # noinspection PyTypeChecker
                len_privileges = (
                    len(self._root_client.describe_role(self._role_name)['privileges']))
                if exp_privilege_count <= len_privileges:
                    break
                else:
                    time.sleep(self._CMD_POLLING_SEC)
            else:
                raise TimeoutError(f'Timeout of {self._GRANT_PRIVILEGES_TIMEOUT_SEC} seconds '
                                   f'waiting to grant permission to role {self._role_name} for '
                                   f'user {self._username}.')

            if retry:
                time.sleep(self._CMD_POLLING_SEC)
            self._root_client.grant_role(self._username, self._role_name)
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
            raise Exception(f'Exhausted {self._CMD_RETRIES} when attempting to list user granted '
                            f'role.')

    def _add_user(self, uri, username, password) -> MilvusClient:
        if self._username in self._root_client.list_users():
            return MilvusClient(uri, username, password)

        self._root_client.create_user(username, password=password)
        user_client = MilvusClient(uri, username, password)
        self._root_client.create_role(self._role_name)
        self._add_privileges(user_client)
        return user_client

    def _add_rag_index(self) -> VectorStoreIndex:
        emb_host = "nv-embed-qa.default.example.com"
        emb_model = "NV-Embed-QA"
        llm_host = "llama-2-13b-chat-hf.default.example.com"
        llm_model = "llama-2-13b-chat-hf"
        tokenizer = "meta-llama/Llama-2-13b-chat-hf"

        user_token = login_aioli(AIOLI_HOST, AIOLI_USERNAME, AIOLI_PASSWORD)
        api_key = get_api_key(AIOLI_HOST, user_token, llm_model)

        default_headers_chat = {"host": llm_host}
        default_headers_embedding = {"host": emb_host}
        chat = OpenAILike(model=llm_model, api_key=api_key, api_base=CHAT_API_BASE,
                          default_headers=default_headers_chat, is_chat_model=True,
                          tokenizer=tokenizer, temperature=0.1,
                          max_tokens=500)
        embedding = NeMoEmbedding(model=emb_model, api_key=api_key, api_base=CHAT_API_BASE,
                                  default_headers=default_headers_embedding, truncate="END")

        Settings.llm = chat
        Settings.embedding = embedding
        Settings.embed_model = embedding
        Settings.chunk_size = 256
        Settings.chunk_overlap = 20

        vector_store = self._wa_cmd_privileges(
            lambda: RbacMilvusVectorStore(uri=self.uri, user=self._username,
                                          password=self._password,
                                          collection_name=self._collection_name,
                                          dim=self.vector_dimension, overwrite=False),
            self._LOAD_COLLECTION_TIMEOUT_SEC, 'creating, indexing and loading collection')

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

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
