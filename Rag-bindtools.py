import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load text file for RAG
file_path = "context.txt"
loader = TextLoader(file_path)
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)
all_splits = text_splitter.split_documents(docs)

# Create embeddings and store in vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_splits, embeddings)

# --- RAG Tool ---
def rag_chatbot_tool(query: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    return f"Answer based on the context:\n\n{context_text}\n\nQuestion: {query}"

rag_tool = Tool(
    name="RAG_QA",
    func=rag_chatbot_tool,
    description="Answer questions based on the provided text file."
)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Bind the tool directly to LLM
llm_with_tools = llm.bind_tools([rag_tool])

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the RAG_QA tool when answering questions."),
    ("user", "{input}")
])

# Create the chain
chain = prompt | llm_with_tools

# Streamlit UI
st.set_page_config(page_title="Chatbot with RAG Tool")
st.title("Chatbot with RAG Tool Binding")

user_input = st.text_input("Ask something:")

if st.button("Ask") and user_input:
    response = chain.invoke({"input": user_input})
    st.markdown(f"**Answer:** {response.content}")
