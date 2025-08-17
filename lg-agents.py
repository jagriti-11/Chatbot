import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

file_path = "context.txt"
loader = TextLoader(file_path)
docs = loader.load()

# Split docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
all_splits = text_splitter.split_documents(docs)

# Embeddings and Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_splits, embeddings)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# Agent1
def run_agent1(query: str) -> str:
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])
    response = llm.invoke(f"Answer based only on this context:\n{context}\n\nQuestion: {query}")
    return response.content

# Agent2: Capitals of country using dictionary
capitals = {
    "australia": "Canberra",
    "india": "New Delhi",
    "usa": "Washington, D.C.",
    "france": "Paris",
    "germany": "Berlin",
    "italy": "Rome",
    "spain": "Madrid",
    "portugal": "Lisbon",
    "japan": "Tokyo",
    "china": "Beijing",
    "south korea": "Seoul",
    "canada": "Ottawa",
    "mexico": "Mexico City",
    "brazil": "Brasília",
    "argentina": "Buenos Aires",
    "russia": "Moscow",
    "uk": "London",
    "switzerland": "Bern",
    "netherlands": "Amsterdam",
    "belgium": "Brussels",
    "egypt": "Cairo",
    "south africa": "Pretoria (administrative), Cape Town (legislative), Bloemfontein (judicial)",
    "uae": "Abu Dhabi",
    "saudi arabia": "Riyadh",
    "thailand": "Bangkok",
    "vietnam": "Hanoi",
    "indonesia": "Jakarta",
    "singapore": "Singapore",
    "nepal": "Kathmandu",
    "pakistan": "Islamabad",
    "sri lanka": "Sri Jayawardenepura Kotte (official), Colombo (commercial)"
}

def run_agent2(query: str) -> str:
    for country, capital in capitals.items():
        if country in query.lower():
            return f"The capital of {country.capitalize()} is {capital}."
    return "I don’t know the capital of that country."

#State definition

class GraphState(TypedDict):
    query: str
    answer: str

# Supervisor node
def supervisor(state: GraphState):
    query = state["query"]
    if "capital" in query.lower():
        return {"next": "agent2"}
    else:
        return {"next": "agent1"}


def agent1_node(state: GraphState):
    result = run_agent1(state["query"])
    return {"answer": result}

def agent2_node(state: GraphState):
    result = run_agent2(state["query"])
    return {"answer": result}


#Langgraph
graph = StateGraph(GraphState)

graph.add_node("supervisor", supervisor)
graph.add_node("agent1", agent1_node)
graph.add_node("agent2", agent2_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"agent1": "agent1", "agent2": "agent2"}
)
graph.add_edge("agent1", END)
graph.add_edge("agent2", END)

app_graph = graph.compile()


# Streamlit UI
st.title("LangGraph Multi-Agent Implementation")

query = st.text_input("Write your question:")

if query:
    # Run the graph
    final_state = app_graph.invoke(
    {"query": query, "answer": "", "next": ""}
)

    st.write("Answer:")
    st.success(final_state["answer"])
