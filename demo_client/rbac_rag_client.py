import io
from typing import Optional, Tuple
import contextlib
import os

# meyere, this to be provided by the demo group
from demo_client.utils import NeMoEmbedding, login_aioli, get_api_key

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import DataType, MilvusClient

# Environment Variables
AIOLI_HOST = os.environ['AIOLI_HOST']
CHAT_API_BASE = f'http://{os.environ["AIOLI_CHAT_HOST"]}/v1'
AIOLI_USERNAME = os.environ['AIOLI_USER']
AIOLI_PASSWORD = os.environ['AIOLI_PW']
MILVUS_ROOT_USERNAME = os.environ.get('MILVUS_ROOT_USERNAME', 'root')
MILVUS_ROOT_PASSWORD = os.environ.get('MILVUS_ROOT_PASSWORD', 'Milvus')

MILVUS_HOST = f'http://{AIOLI_HOST}:19530'

# Constants
PR_DIRS = ('/pfs/parse-hpe', '/pfs/parse-nvidia')


class RbacRagClient:
    """An retrieval augmented generation (RAG) client with simple role based access control (
    RBAC) support."""

    # ask: Add parameters for the chunking/embedding
    _USERPASS_COLLECTION = 'userpass'

    def __init__(self, uri: str = MILVUS_HOST,
                 username: str = AIOLI_USERNAME, password: str = AIOLI_PASSWORD,
                 chunk_overlap: int = 20,
                 chunk_size: int = 256,
                 embedding_host: str = 'nv-embed-qa.default.example.com',
                 embedding_model: str = 'NV-Embed-QA',
                 llm_host: str = 'llama-2-13b-chat-hf.default.example.com',
                 llm_model: str = 'llama-2-13b-chat-hf',
                 max_tokens: int = 500,
                 temperature: float = 0.1,
                 tokenizer: str = 'meta-llama/Llama-2-13b-chat-hf',
                 vector_dimension: int = 1024):

        self._uri = uri
        self._username = username
        self._chunk_overlap = chunk_overlap
        self._chunk_size = chunk_size
        self._embedding_host = embedding_host
        self._embedding_model = embedding_model
        self._llm_host = llm_host
        self._llm_model = llm_model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tokenizer = tokenizer
        self._vector_dimension = vector_dimension

        self._collection_name = username + '_collection'
        self._query_engine = None

        # Check credentials, creating as needed.
        with UserPassCollection(uri, MILVUS_ROOT_USERNAME, MILVUS_ROOT_PASSWORD) as userpass:
            saved_pass = userpass.get(username)
            if saved_pass is None:
                userpass.insert(username, password)
            elif saved_pass != password:
                raise PermissionError(f'Invalid password for username "{username}"')

        # Create vector store for retrieval augmented generation (RAG)
        self._vector_store, self._index = self._add_rag_index()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def embedding_host(self) -> str:
        return self._embedding_host

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def llm_host(self) -> str:
        return self._llm_host

    @property
    def llm_model(self) -> str:
        return self._llm_model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def tokenizer(self) -> str:
        return self._tokenizer

    @property
    def vector_dimension(self) -> int:
        return self._vector_dimension

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def query_engine(self) -> BaseQueryEngine:
        if self._query_engine is None:
            self._query_engine = self._index.as_query_engine()
        return self._query_engine

    @property
    def username(self) -> str:
        return self._username

    def add_documents(self, *paths_to_doc_dirs):
        for path_to_doc_dir in paths_to_doc_dirs:
            documents = SimpleDirectoryReader(str(path_to_doc_dir)).load_data()
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=self._index.storage_context)

    def close(self):
        self._vector_store.client.close()

    def remove_all(self):
        UserPassCollection(self.uri,
                           MILVUS_ROOT_USERNAME, MILVUS_ROOT_PASSWORD).delete(self._username)
        self._rm_collection()
        self.close()

    def _add_rag_index(self) -> Tuple[MilvusVectorStore, VectorStoreIndex]:
        user_token = login_aioli(AIOLI_HOST, AIOLI_USERNAME, AIOLI_PASSWORD)

        with contextlib.redirect_stdout(io.StringIO()):
            api_key = get_api_key(AIOLI_HOST, user_token, self._llm_model)

        default_headers_chat = {'host': self._llm_host}
        default_headers_embedding = {'host': self._embedding_host}
        chat = OpenAILike(model=self._llm_model, api_key=api_key, api_base=CHAT_API_BASE,
                          default_headers=default_headers_chat, is_chat_model=True,
                          tokenizer=self._tokenizer, temperature=self._temperature,
                          max_tokens=self._max_tokens)
        embedding = NeMoEmbedding(model=self._embedding_model, api_key=api_key,
                                  api_base=CHAT_API_BASE,
                                  default_headers=default_headers_embedding, truncate="END")

        Settings.llm = chat
        Settings.embedding = embedding
        Settings.embed_model = embedding
        Settings.chunk_size = self._chunk_size
        Settings.chunk_overlap = self._chunk_overlap

        vector_store = MilvusVectorStore(uri=self.uri, user=MILVUS_ROOT_USERNAME,
                                         password=MILVUS_ROOT_PASSWORD,
                                         collection_name=self._collection_name,
                                         dim=self.vector_dimension, overwrite=False)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return (vector_store,
                VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context))

    def _rm_collection(self):
        self._vector_store.client.drop_collection(self._collection_name)


class UserPassCollection:
    """
    A simple username : password mapping within milvus, accessible via root to provide thin RBAC
    support.
    """
    _NAME = 'userpass'

    class Field:
        USERNAME = 'username'
        PASSWORD = 'password'

    def __init__(self, uri: str, root_username: str, root_password: str):
        self._client = MilvusClient(uri, root_username, root_password)

        # Create collection as needed
        if self._NAME not in self._client.list_collections():
            schema = self._client.create_schema(
                auto_id=False,
                description='Username/password collection.'
            )
            schema.add_field(field_name=self.Field.USERNAME,
                             datatype=DataType.VARCHAR, max_length=256, is_primary=True)
            schema.add_field(field_name=self.Field.PASSWORD,
                             datatype=DataType.VARCHAR, max_length=256)

            self._client.create_collection(collection_name=self._NAME,
                                           consistency_level='Strong',
                                           schema=schema)

            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name=self.Field.USERNAME)
            index_params.add_index(field_name=self.Field.PASSWORD)
            self._client.create_index(collection_name=self._NAME,
                                      index_params=index_params)
            self._client.load_collection(self._NAME)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._client.close()

    def delete(self, username: str):
        self._client.delete(self._NAME, ids=username)

    def get(self, username: str) -> Optional[str]:
        resp = self._client.get(self._NAME, ids=username, output_fields=[self.Field.PASSWORD])
        if not resp:
            return None
        return resp[0][self.Field.PASSWORD]

    def insert(self, username: str, password: str):
        self._client.insert(collection_name=self._NAME,
                            data=[{self.Field.USERNAME: username,
                                   self.Field.PASSWORD: password}])



