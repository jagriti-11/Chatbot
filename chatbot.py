import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import initialize_agent,AgentType

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load text file
file_path = "context.txt"
loader = TextLoader(file_path)
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in vector DB
vectorstore = FAISS.from_documents(all_splits, embeddings)

# Streamlit UI
st.set_page_config(page_title="My Chatbot")
st.title("My Chatbot")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Input box
user_input = st.text_input("Write your question here")
def rag_chatbot_tool(user_input):
# Retrieve top-k chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_input)

# Merge context
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

# Create prompt
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context_text}

    Question: {user_input}
    """

# Get answer from LLM
    response = llm.invoke(prompt)
    return response.content
rag_tool=Tool(name="RAG_TOOL",func=rag_chatbot_tool,description="Use this tool to answer questions based on the given text file")

if st.button("Ask") and user_input:
    answer=rag_tool.run(user_input)
    st.markdown(f" **Answer:** {answer}")