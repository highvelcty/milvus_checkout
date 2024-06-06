import os

# from dotenv import load_dotenv
#
# load_dotenv()
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext

# meyere, this to be provided by the demo group
from demo_client.utils import NeMoEmbedding, login_aioli, get_api_key


from repo_tools import paths
from demo_client import DemoClient


def get_query_engine(demo_client: DemoClient):
    aioli_host = os.environ['AIOLI_HOST']
    chat_api_base = f'http://{os.environ["AIOLI_CHAT_HOST"]}/v1'

    hpe_input_path = "/pfs/ingest_hpe_prs/ingested_docs"
    nvidia_input_path = "/pfs/ingest_nvidia_prs/ingested_docs"
    vector_store_dir = "/pfs/out"

    # preexisting_milvus_db = path_to_milvus_db.exists()

    llm_host = "llama-2-13b-chat-hf.default.example.com"
    llm_model = "llama-2-13b-chat-hf"
    tokenizer = "meta-llama/Llama-2-13b-chat-hf"
    emb_host = "nv-embed-qa.default.example.com"
    emb_model = "NV-Embed-QA"

    user_token = login_aioli(aioli_host, os.environ['AIOLI_USER'], os.environ['AIOLI_PW'])
    api_key = get_api_key(aioli_host, user_token, llm_model)

    default_headers_chat = {"host": llm_host}
    default_headers_embedding = {"host": emb_host}
    chat = OpenAILike(model=llm_model, api_key=api_key, api_base=chat_api_base,
                      default_headers=default_headers_chat, is_chat_model=True,
                      tokenizer=tokenizer, temperature=0.1,
                      max_tokens=500)
    embedding = NeMoEmbedding(model=emb_model, api_key=api_key, api_base=chat_api_base,
                              default_headers=default_headers_embedding, truncate="END")

    Settings.llm = chat
    Settings.embedding = embedding
    Settings.embed_model = embedding
    Settings.chunk_size = 256
    Settings.chunk_overlap = 20

    path_to_docs = paths.RootPath.PKG / 'docs'

    # vector_store = MilvusVectorStore(
    #     uri="./milvus_demo.db", dim=1536, overwrite=True
    # )
    # meyere, I'm not sure what to set the dimension to.
    # 1024 seems to work
    #
    # Other values tried:
    #   512: code=2000, message=vector dimension mismatch, expected vector size(byte) 2048,
    #        actual 4096.: segcore error)>
    #   1536: MilvusException: (code=65535, message=the length(83968) of float data should
    #         divide the dim(1536)
    #   2048: MilvusException: (code=2000, message=vector dimension mismatch, expected vector
    #         size(byte) 8192, actual 4096.: segcore error)>
    # vector_store = MilvusVectorStore(uri=MILVUS_HOST,
    #                                  dim=1024, overwrite=not preexisting_milvus_db)

    vector_store = MilvusVectorStore(uri=demo_client.uri,
                                     user=demo_client.username, password=demo_client.password,
                                     collection_name=demo_client.collection_name,
                                     dim=demo_client.vector_dimension, overwrite=False)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(str(path_to_docs)).load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return index.as_query_engine()
