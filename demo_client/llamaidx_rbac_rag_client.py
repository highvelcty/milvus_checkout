import io
from typing import Tuple
import contextlib

try:
    # Provided by the demo group
    from utils import NeMoEmbedding, login_aioli, get_api_key
except ImportError:
    # Provided by this repo
    from demo_client.utils import NeMoEmbedding, login_aioli, get_api_key

import base_rbac_rag_client

from llama_index.core.base.base_query_engine import BaseQueryEngine, RESPONSE_TYPE
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore


class LlamaIdxRbacRagClient(base_rbac_rag_client.BaseRbacRagClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create vector store for retrieval augmented generation (RAG)
        self._vector_store, self._index = self._add_rag_index()

    def add_documents(self, *paths_to_doc_dirs):
        for path_to_doc_dir in paths_to_doc_dirs:
            documents = SimpleDirectoryReader(str(path_to_doc_dir)).load_data()
            self._index = VectorStoreIndex.from_documents(
                documents, strage_context=self._index.storage_context)

    def close(self):
        self._vector_store.client.close()

    def query(self, query_str) -> RESPONSE_TYPE:
        return self.query_engine.query(query_str)

    @property
    def query_engine(self) -> BaseQueryEngine:
        if self._query_engine is None:
            self._query_engine = self._index.as_query_engine()
        return self._query_engine

    def _add_rag_index(self) -> Tuple[MilvusVectorStore, VectorStoreIndex]:
        user_token = login_aioli(base_rbac_rag_client.AIOLI_HOST,
                                 base_rbac_rag_client.AIOLI_USERNAME,
                                 base_rbac_rag_client.AIOLI_PASSWORD)

        with contextlib.redirect_stdout(io.StringIO()):
            api_key = get_api_key(base_rbac_rag_client.AIOLI_HOST, user_token, self._llm_model)

        default_headers_chat = {'host': self._llm_host}
        default_headers_embedding = {'host': self._embedding_host}
        chat = OpenAILike(model=self._llm_model, api_key=api_key,
                          api_base=base_rbac_rag_client.CHAT_API_BASE,
                          default_headers=default_headers_chat, is_chat_model=True,
                          tokenizer=self._tokenizer, temperature=self._temperature,
                          max_tokens=self._max_tokens)
        embedding = NeMoEmbedding(model=self._embedding_model, api_key=api_key,
                                  api_base=base_rbac_rag_client.CHAT_API_BASE,
                                  default_headers=default_headers_embedding, truncate="END")

        Settings.llm = chat
        Settings.embedding = embedding
        Settings.embed_model = embedding
        Settings.chunk_size = self._chunk_size
        Settings.chunk_overlap = self._chunk_overlap

        vector_store = MilvusVectorStore(uri=self.uri,
                                         user=base_rbac_rag_client.MILVUS_ROOT_USERNAME,
                                         password=base_rbac_rag_client.MILVUS_ROOT_PASSWORD,
                                         collection_name=self._collection_name,
                                         dim=self.vector_dimension, overwrite=False)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return (vector_store,
                VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context))

    def _rm_collection(self):
        self._vector_store.client.drop_collection(self._collection_name)
