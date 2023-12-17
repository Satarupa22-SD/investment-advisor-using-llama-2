# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-70b-chat-hf"
# Set auth token variable from hugging face
auth_token = "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"

# @st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16,
                            rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
You are an intelligent and ethical AI assistant designed to deliver helpful, respectful, and honest responses. Strive to provide assistance in the most effective and constructive manner while prioritizing safety and ethical considerations. Avoid any content that could be harmful, unethical, biased, or illegal.
Your responses should be socially unbiased and positive, contributing positively to the user experience. If confronted with a question that lacks coherence or factual accuracy, provide a clear explanation rather than furnishing inaccurate information. If uncertain about the answer to a question, prioritize refraining from sharing false information.
Your main objective is to offer insightful and accurate responses related to the financial performance of a company, utilizing your advanced capabilities to enhance user trust and satisfaction..<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Download PDF Loader
PyMuPDFReader = download_loader("PyMuPDFReader")
# Create PDF Loader
loader = PyMuPDFReader()
# Load documents
documents = loader.load(file_path=Path('./data/annualreport.pdf'), metadata=True)

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)
# Setup index query engine using LLM
query_engine = index.as_query_engine()

# Create centered main title
st.title('🦙 Llama Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    # ...and write it out to the screen
    st.write(response)

    # Display raw response object
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())

