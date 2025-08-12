import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

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

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# --- Tool 1: RAG Tool ---
def rag_chatbot_tool(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Answer based on the context:\n\n{context_text}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content

# --- Tool 2: Calculator Tool ---
def calculator_tool(expression):
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"The result is {result}"
    except Exception:
        return "Invalid mathematical expression."

# --- Tool 3: Static Weather Tool ---
weather_data = {
    "delhi": "Sunny, 35°C",
    "mumbai": "Rainy, 28°C",
    "kolkata": "Humid, 33°C",
    "bangalore": "Cloudy, 25°C"
}

def weather_tool(state):
    state = state.lower()
    return weather_data.get(state, "Weather data not available for this state.")

# Create tools
tools = [
    Tool(name="RAG_QA", func=rag_chatbot_tool, description="Answer questions based on the provided text file."),
    Tool(name="Calculator", func=calculator_tool, description="Perform basic math calculations."),
    Tool(name="Weather_Info", func=weather_tool, description="Get weather info for Indian states (static).")
]

# Create Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="Chatbot with")
st.title("Chatbot with Agent")

user_input = st.text_input("Ask something:")

if st.button("Ask") and user_input:
    answer = agent.run(user_input)
    st.markdown(f"**Answer:** {answer}")
