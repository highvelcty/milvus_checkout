from dataclasses import dataclass
from typing import Optional
import logging
import os
import sys
import textwrap
import threading

import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

from utils import (NeMoEmbedding, remove_prefix, login_aioli, get_api_key_by_id, get_deployments,
                   get_model_hostname, is_chat_model)

from llamaidx_rbac_rag_client import LlamaIdxRbacRagClient


@dataclass
class AppArgs:
    api_key: str = "testing"
    llm_host: str = "llama-2-13b-chat-hf.default.example.com"
    model_id: Optional[str] = None
    emb_host: str = "nv-embed-qa.default.example.com"
    llm_model: str = "llama-2-13b-chat-hf"
    emb_model: str = "NV-Embed-QA"
    api_base: str = "http://35.247.117.24/v1"
    aioli_host: str = "34.168.33.192"
    vector_dir: str = "/pfs/vector-store"


args = AppArgs(
    api_key=os.environ.get("API_KEY", "testing"),
    llm_host=os.environ.get("LLM_HOST", "llama-3-8b-instruct.default.example.com"),
    emb_host=os.environ.get("EMB_HOST", "nv-embed-qa.default.example.com"),
    llm_model=os.environ.get("LLM_MODEL", "llama-3-8b-instruct"),
    emb_model=os.environ.get("EMB_MODEL", "NV-Embed-QA"),
    api_base=os.environ.get("API_BASE", "http://35.247.117.24/v1"),
    aioli_host=os.environ.get("AIOLI_HOST", "34.168.33.192"),
    vector_dir=os.environ.get("VECTOR_DIR", "/pfs/create_vector_index")
)


def get_llm_client():
    api_base = args.api_base
    default_headers_chat = {"host": args.llm_host}
    print(args)
    is_chat = is_chat_model(args.aioli_host, st.session_state["token"], args.model_id)
    chat = OpenAILike(model=args.llm_model, api_key=args.api_key, api_base=api_base,
                      default_headers=default_headers_chat, is_chat_model=is_chat,
                      temperature=0.1, max_tokens=500)
    return chat


@st.cache_resource
def get_embedding_client():
    api_base = args.api_base
    default_headers_embedding = {"host": args.emb_host}
    embedding = NeMoEmbedding(model=args.emb_model, api_key=args.api_key, api_base=api_base,
                              default_headers=default_headers_embedding)
    return embedding


def get_hpe_query_engine():
    client = LlamaIdxRbacRagClient(collection_name='parse_hpe')
    return client.query_engine


def get_nvidia_query_engine():
    client = LlamaIdxRbacRagClient(collection_name='parse_nvidia')
    return client.query_engine


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            st.session_state["token"] = login_aioli(args.aioli_host, st.session_state["username"],
                                                    st.session_state["password"])
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        except Exception as e:
            print(e)
            print("Could not login to aioli.")
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Username", type="default", key="username"
        )
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Username", type="default", key="username"
        )
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    title = "HPE NVidia Press Release RAG Demo"
    st.set_page_config(layout="wide", page_title=title)

    with open("style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    ######
    # CSS for formatting top bar
    st.markdown(
        """
        <style>
        .top-bar {
            background-color: #FFFFF;
            padding-bottom: 15px;
            color: white;
            margin-top: -82px;
            border-width: 2px;
            border-bottom-width: 2px;
            border-bottom-color: Black;
            border-bottom-style: solid;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create top bar
    st.markdown(
        """
        <div class="top-bar">
             <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Hewlett_Packard_Enterprise_logo.svg/2560px-Hewlett_Packard_Enterprise_logo.svg.png" alt="HPE Logo" height="60">
        </div>
        """,
        unsafe_allow_html=True,
    )

    ######

    st.title(title)

    if check_password():
        deployments = get_deployments(args.aioli_host, st.session_state["token"])
        _ = st.selectbox(
            "Available Models", [m["name"] for m in deployments if "embed" not in m["name"]],
            key="models_selectbox")
        args.llm_model = st.session_state.models_selectbox
        args.model_id = deployments[[m["name"] for m in deployments].index(args.llm_model)]["model"]
        args.api_key = get_api_key_by_id(args.aioli_host, st.session_state["token"], args.model_id)
        args.llm_host = get_model_hostname(args.aioli_host, st.session_state["token"],
                                           args.llm_model)

        example_questions = [
            "What collaborations does HPE have with Evil Geniuses?",
            "How has Inworld AI used NVidia's generative AI software for game characters?",
            "What are some recent collaborations between HPE and NVIDIA in generative AI?",
            "What is HPE's Machine Learning Inference Service and how does it work with NVIDIA's "
            "NeMo microservices?",
            "What are some joint initiatives for HPE and NVIDIA around the NeMo suite of software "
            "services?",
            "What are key strengths of HPE and NVIDIA in high performance computing?"
        ]

        _ = st.selectbox(
            "Sample Questions", example_questions, key="sample_questions_selectbox"
        )
        user_input = st.text_input(
            label="What's your question?",
            key="input",
            value=st.session_state.sample_questions_selectbox,
        )

        with st.spinner("Setting up..."):
            Settings.llm = get_llm_client()
            Settings.embedding = get_embedding_client()
            Settings.embed_model = get_embedding_client()
            Settings.chunk_size = 256
            Settings.chunk_overlap = 20

        def ask_question_no_rag(question, llm, resp):
            prompt = f"""\
    Answer the "{question}" using knowledge you have from the press releases of HPE and NVIDIA. \
    Provide a concise and informative response.
    """

            model_outputs = llm.complete(prompt)
            resp["no_rag"] = remove_prefix(model_outputs.text,
                                           "<|start_header_id|>assistant<|end_header_id|>").strip()

        def ask_question_company_pr(question: str, query_engine, responses_, key):
            responses_[key] = query_engine.query(question)

        def ask_question_with_rag(question: str, llm, responses_):
            hpe_sources = '-'.join([ii.node.text+"\n" for ii in responses_["hpe"].source_nodes])
            nvidia_sources = '-'.join([ii.node.text+"\n" for ii
                                       in responses_["nvidia"].source_nodes])
            prompt = f"""\
    Compose an answer to the question: "{question}"
    
    Synthesize the answer using the responses from HPE and NVidia press releases.  \
    Include information from retrieved documents that are relevant to the question. \
    
    Remember that company specific information may only show up in the press releases for one \
    of the companies and not the other.  In that case, use the more relevant source, \
    to construct the answer.
    
    Do not confuse HP with HPE, they are different companies.  Exclude any information about HP.
    
    Answer from HPE press releases:
    {remove_prefix(responses["hpe"].response, 
                   "<|start_header_id|>assistant<|end_header_id|>").strip()}
    
    Retrieved passages from HPE press releases:
    {hpe_sources}
    
    Answer from NVidia press releases:
    {remove_prefix(responses["nvidia"].response, 
                   "<|start_header_id|>assistant<|end_header_id|>").strip()}
    
    Retrieved passages from NVIDIA press releases:
    {nvidia_sources}
    
    Format the answer in an easy to read way with list when appropriate. \
    """
            print(prompt)
            model_outputs = llm.complete(prompt)
            return remove_prefix(model_outputs.text,
                                 "<|start_header_id|>assistant<|end_header_id|>").strip()

        if user_input:
            responses = {}
            threads = []

            no_rag_query = threading.Thread(
                target=ask_question_no_rag,
                args=(user_input, Settings.llm, responses)
            )

            hpe_query_thread = threading.Thread(
                target=ask_question_company_pr,
                args=(user_input, get_hpe_query_engine(), responses, "hpe")
            )
            nvidia_query_thread = threading.Thread(
                target=ask_question_company_pr,
                args=(user_input, get_nvidia_query_engine(), responses, "nvidia")
            )

            threads.append(no_rag_query)
            threads.append(hpe_query_thread)
            threads.append(nvidia_query_thread)

            with st.spinner("Processing..."):
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                col1, col2 = st.columns(2)

            with col1:
                output = responses["no_rag"]
                st.markdown("### LLM (Out of the box) Output :smile:")
                st.markdown(output)
                st.divider()
            with col2:
                with st.spinner("Processing..."):
                    print(responses["hpe"].get_formatted_sources())
                    print(responses["nvidia"].get_formatted_sources())
                    output = ask_question_with_rag(user_input, Settings.llm, responses)
                    st.markdown(
                        "###  RAG Output :smile: :smile: :smile:"
                    )
                    st.markdown(output)
                    st.divider()
                    with col2:
                        st.markdown(
                            "#### Related Documents from HPE PRs"
                        )

                        related_docs = responses["hpe"].source_nodes
                        for i, x in enumerate(related_docs):
                            title = x.node.metadata["file_name"]
                            url = x.node.metadata["file_path"]
                            st.markdown(f'#{i+1}: from "{title}"\n\n[{url}]({url})\n')
                            st.text("\n".join(textwrap.wrap(x.node.text, 80)))
                            if i < len(related_docs) - 1:
                                st.divider()
                    with col2:
                        st.markdown(
                            "#### Related Documents from NVIDIA PRs"
                        )

                        related_docs = responses["nvidia"].source_nodes
                        for i, x in enumerate(related_docs):
                            title = x.node.metadata["file_name"]
                            url = x.node.metadata["file_path"]
                            st.markdown(f'#{i+1}: from "{title}"\n\n[{url}]({url})\n')
                            st.text("\n".join(textwrap.wrap(x.node.text, 80)))
                            if i < len(related_docs) - 1:
                                st.divider()


if __name__ == '__main__':
    main()
