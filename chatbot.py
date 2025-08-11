import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set your GEMINI_API_KEY in the .env file.")
    st.stop()

# Streamlit 
st.set_page_config(page_title="My Chatbot", layout="wide")
st.title("My Chatbot")

# Session state for models and vector store
if "llm" not in st.session_state:
    st.session_state.llm = None
if "emb_model" not in st.session_state:
    st.session_state.emb_model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize models lazily (fix for event loop issue)
def init_models():
    # Ensure an event loop exists (needed for Google Generative AI Embeddings)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if st.session_state.llm is None:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )

    if st.session_state.emb_model is None:
        st.session_state.emb_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

# Load context.txt and build vector DB
def load_context():
    with open("context.txt", "r", encoding="utf-8") as f:
        data = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([data])

    st.session_state.vector_store = FAISS.from_documents(
        docs,
        st.session_state.emb_model
    )

# Retrieval using cosine
def retrieve_context(query, k=3):
    results = st.session_state.vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""
)

# Generate answer using LLM
def generate_answer(query):
    relevant_context = retrieve_context(query, k=3)
    final_prompt = prompt_template.format(context=relevant_context, question=query)
    response = st.session_state.llm.invoke(final_prompt)
    return response.content

# User input
user_input = st.text_input("Write your Question:")

if st.button("Ask"):
    if user_input.strip():
        init_models()
        if st.session_state.vector_store is None:
            load_context()

        answer = generate_answer(user_input)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a question first.")

