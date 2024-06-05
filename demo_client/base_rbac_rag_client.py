from abc import abstractmethod
from typing import Optional
import os

try:
    # Provided by the demo group
    from utils import NeMoEmbedding, login_aioli, get_api_key
except ImportError:
    # Provided by this repo
    from demo_client.utils import NeMoEmbedding, login_aioli, get_api_key

from pymilvus import DataType, MilvusClient

# Environment Variables
AIOLI_HOST = os.environ['AIOLI_HOST']
CHAT_API_BASE = f'http://{os.environ["AIOLI_CHAT_HOST"]}/v1'
AIOLI_USERNAME = os.environ['AIOLI_USER']
AIOLI_PASSWORD = os.environ['AIOLI_PW']
MILVUS_ROOT_USERNAME = os.environ.get('MILVUS_ROOT_USERNAME', 'root')
MILVUS_ROOT_PASSWORD = os.environ.get('MILVUS_ROOT_PASSWORD', 'Milvus')
# MILVUS_HOST = f'http://{AIOLI_HOST}:19530'
MILVUS_HOST = f'http://localhost:19530'


class BaseRbacRagClient:
    """An retrieval augmented generation (RAG) client with simple role based access control
       (RBAC) support."""
    _USERPASS_COLLECTION = 'userpass'

    class DefaultArg:
        uri = MILVUS_HOST
        username = AIOLI_USERNAME
        password = AIOLI_PASSWORD
        collection_name = ''
        chunk_overlap = 20
        chunk_size = 256
        embedding_host = 'nv-embed-qa.default.example.com'
        embedding_model = 'NV-Embed-QA'
        llm_host = 'llama-2-13b-chat-hf.default.example.com'
        llm_model = 'llama-2-13b-chat-hf'
        max_tokens = 500
        temperature = 0.1
        tokenizer = 'meta-llama/Llama-2-13b-chat-hf'
        vector_dimension = 1024

    def __init__(self, uri: str = DefaultArg.uri,
                 username: str = DefaultArg.username,
                 password: str = DefaultArg.password,
                 collection_name: str = DefaultArg.collection_name,
                 chunk_overlap: int = DefaultArg.chunk_overlap,
                 chunk_size: int = DefaultArg.chunk_size,
                 embedding_host: str = DefaultArg.embedding_host,
                 embedding_model: str = DefaultArg.embedding_model,
                 llm_host: str = DefaultArg.llm_host,
                 llm_model: str = DefaultArg.llm_model,
                 max_tokens: int = DefaultArg.max_tokens,
                 temperature: float = DefaultArg.temperature,
                 tokenizer: str = DefaultArg.tokenizer,
                 vector_dimension: int = DefaultArg.vector_dimension):

        self._uri = uri
        self._username = username
        if collection_name:
            self._collection_name = f'{username}_{collection_name}'
        else:
            self._collection_name = f'{username}_collection'
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

        # Lazy initialized on first access.
        self._query_engine = None

        # Check credentials, creating as needed.
        with UserPassCollection(uri, MILVUS_ROOT_USERNAME, MILVUS_ROOT_PASSWORD) as userpass:
            saved_pass = userpass.get(username)
            if saved_pass is None:
                userpass.insert(username, password)
            elif saved_pass != password:
                raise PermissionError(f'Invalid password for username "{username}"')

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
    @abstractmethod
    def query_engine(self):
        ...

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
    def username(self) -> str:
        return self._username

    @abstractmethod
    def add_documents(self, *paths_to_doc_dirs):
        ...

    def close(self):
        ...

    @abstractmethod
    def query(self, query_str):
        ...

    def remove_all(self):
        UserPassCollection(self.uri,
                           MILVUS_ROOT_USERNAME, MILVUS_ROOT_PASSWORD).delete(self._username)
        self._rm_collection()
        self.close()

    @abstractmethod
    def _add_rag_index(self):
        ...

    @abstractmethod
    def _rm_collection(self):
        ...


class UserPassCollection:
    """
    A simple username : password mapping within milvus, accessible via root to provide thin RBAC
    support.
    """
    _NAME = 'userpass'

    class Field:
        USERNAME = 'username'
        PASSWORD = 'password'
        VECTOR = 'vector'

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
            schema.add_field(field_name=self.Field.VECTOR, datatype=DataType.FLOAT_VECTOR, dim=2)

            self._client.create_collection(collection_name=self._NAME,
                                           consistency_level='Strong',
                                           schema=schema)

            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name=self.Field.USERNAME)
            index_params.add_index(field_name=self.Field.PASSWORD)
            index_params.add_index(field_name=self.Field.VECTOR)
            self._client.create_index(collection_name=self._NAME, index_params=index_params)

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
                                   self.Field.PASSWORD: password,
                                   self.Field.VECTOR: [1.0, 1.0]}])
