import os
import queue
import re
import tempfile
import threading

import streamlit as st

from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import StreamingStdOutCallbackHandlerYield, generate

# ---- Setup and Configuration ---- #
# Set the necessary environmental variables for LangChain.
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'ls__18b33c33a232426186a23e0a5a073b54'  # Use your actual LangChain API key.
os.environ['LANGCHAIN_PROJECT'] = 'SearchMe'  # Your LangChain project name.

API_KEY = "sk-aa"  # Use your actual secondary API key if needed.
show_citations = False

# ---- Function Definitions ---- #
def initialize_app(api_key):
    """
    Initialize the EmbedChain app with the necessary configuration.
    """
    # db_path = tempfile.mkdtemp()
    return App.from_config(
        config={
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4-turbo-preview",
                    "temperature": 0.5,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "stream": True,
                    "api_key": api_key,
                },
            },
            "vectordb": {
                "provider": "chroma",
            },
            "embedder": {"provider": "openai", "config": {"api_key": api_key}},
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )


def get_or_create_app():
    if "app" not in st.session_state:
        st.session_state.app = initialize_app(API_KEY)
    return st.session_state.app


def setup_ui():
    """
    Set up the Streamlit UI components.
    """
    st.title("📄 Chat with your PDF")
    styled_caption = '<p style="font-size: 17px; color: #aaa;">🚀 An LLM powered app to talk to your PDFs</p>'
    st.markdown(styled_caption, unsafe_allow_html=True)


def display_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi! I'm an AI chatbot, which can answer all of your questions!",
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_input(app):
    prompt = st.chat_input("Ask me anything!")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)
        # Placeholder for bot's response
        msg_placeholder = st.empty()
        msg_placeholder.markdown("Thinking...")

        full_response = handle_query(app, prompt)

        msg_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def handle_query(app, prompt):
    """
    Handles the query submission and interaction with the EmbedChain App.
    """
    q = queue.Queue()
    results = {}

    def app_response(result):
        llm_config = app.llm.config.as_dict()
        llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
        config = BaseLlmConfig(**llm_config)
        answer, citations = app.chat(prompt, config=config, citations=True)
        result["answer"] = answer
        result["citations"] = citations

    thread = threading.Thread(target=app_response, args=(results,))
    thread.start()
    thread.join()

    full_response = format_response(results)

    return full_response


def format_response(results):
    """
    Formats the response and citations for display.
    """
    full_response = results["answer"]
    if show_citations:
        if "citations" in results and results["citations"]:
            full_response += "\n\n**Sources**:\n"
            sources = set()
            for citation in results["citations"]:
                source = citation[1]["url"]
                pattern = re.compile(r"([^/]+)\.[^\.]+\.pdf$")
                match = pattern.search(source)
                if match:
                    source = match.group(1) + ".pdf"
                sources.add(source)
            for source in sources:
                full_response += f"- {source}\n"

    return full_response

setup_ui()
app = get_or_create_app()
app.add("Use Case_ Appliance Company FAQs.pdf", data_type="pdf_file")

# ---- Main App Execution ---- #
if __name__ == '__main__':
    display_messages()
    handle_input(app)
