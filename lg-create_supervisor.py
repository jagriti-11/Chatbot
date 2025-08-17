import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph_supervisor import create_supervisor

# --------------------
# Load environment variables
# --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------
# Load text file for RAG
# --------------------
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

# --------------------
# LLM
# --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# --------------------
# TOOL 1: RAG
# --------------------
def rag_chatbot_tool(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Answer based on the context:\n\n{context_text}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content

rag_tool = Tool(
    name="RAG_QA",
    func=rag_chatbot_tool,
    description="Answer questions based on the provided text file."
)

# --------------------
# TOOL 2: Calculator
# --------------------
def calculator_tool(expression):
    """It calculates numerical mathematical expressions"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"The result is {result}"
    except Exception:
        return "Invalid mathematical expression."

calc_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Perform basic math calculations."
)

# --------------------
# TOOL 3: Weather (Static Data)
# --------------------
weather_data = {
    "delhi": "Sunny, 35°C",
    "mumbai": "Rainy, 28°C",
    "kolkata": "Humid, 33°C",
    "bangalore": "Cloudy, 25°C",
    "srinagar": "Chilly, 2°C"
}

def weather_tool(state):
    state = state.lower()
    return weather_data.get(state, "Weather data not available for this state.")

weather = Tool(
    name="Weather_Info",
    func=weather_tool,
    description="Get weather info for Indian states."
)

# --------------------
# TOOL 4: Capital Lookup
# --------------------
capital_data = {
    "india": "New Delhi",
    "france": "Paris",
    "germany": "Berlin",
    "japan": "Tokyo",
    "usa": "Washington, D.C."
}

def capital_tool(country):
    country = country.lower()
    return capital_data.get(country, "Capital not available for this country.")

capital = Tool(
    name="Capital_Info",
    func=capital_tool,
    description="Get the capital of a country."
)

# --------------------
# Wrap Tools into ToolNodes
# --------------------
rag_node = ToolNode([rag_tool])
calc_node = ToolNode([calc_tool])
weather_node = ToolNode([weather])
capital_node = ToolNode([capital])

# --------------------
# Supervisor (automatic router using LLM)
# --------------------
supervisor = create_supervisor(
    mode=llm,
    agents=[rag_tool, calc_tool, weather, capital],  # ✅ FIXED
    system_prompt=(
        "You are a supervisor. Route the user’s query "
        "to the most appropriate tool:\n"
        "- Use RAG_QA for context-based Q&A\n"
        "- Use Calculator for math\n"
        "- Use Weather_Info for weather queries\n"
        "- Use Capital_Info for country capital lookups"
    )
)

# --------------------
# Build Graph
# --------------------
graph = StateGraph(MessagesState)

graph.add_node("rag_agent", rag_node)
graph.add_node("calc_agent", calc_node)
graph.add_node("weather_agent", weather_node)
graph.add_node("capital_agent", capital_node)
graph.add_node("supervisor", supervisor)

graph.set_entry_point("supervisor")

# Edges from supervisor to tools
graph.add_edge("supervisor", "rag_agent")
graph.add_edge("supervisor", "calc_agent")
graph.add_edge("supervisor", "weather_agent")
graph.add_edge("supervisor", "capital_agent")

# Each tool ends
graph.add_edge("rag_agent", END)
graph.add_edge("calc_agent", END)
graph.add_edge("weather_agent", END)
graph.add_edge("capital_agent", END)

# --------------------
# Compile Graph
# --------------------
app_graph = graph.compile()

# --------------------
# Streamlit UI (simplified)
# --------------------
st.title("LangGraph Multi-Agent Implementation")

query = st.text_input("Write your question:")

if query:
    final_state = app_graph.invoke({"query": query, "answer": "", "next": ""})
    st.write("Answer:")
    st.success(final_state["answer"])
