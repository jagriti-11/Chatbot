import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

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

# Create embeddings and vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_splits, embeddings)

# --- RAG Tool ---
def rag_chatbot_tool(query: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    return f"Answer from context:\n{context_text}"

rag_tool = Tool(
    name="RAG_QA",
    func=rag_chatbot_tool,
    description="Use this tool to answer questions based on the provided text file."
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


prompt = hub.pull("hwchase17/react")   

agent = create_react_agent(
    llm=llm,
    tools=[rag_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[rag_tool],
    verbose=True,                
    handle_parsing_errors=True  
)

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot with ReAct Agent")
st.title("Chatbot with ReAct Agent (RAG)")

user_input = st.text_input("Ask something:")

if st.button("Ask") and user_input:
    response = agent_executor.invoke({"input": user_input})
    st.markdown(f"**Answer:** {response['output']}")
