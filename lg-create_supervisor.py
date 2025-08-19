import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


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


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_splits, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def retrieve_info(query: str):
    """Retrieve relevant context from local text file."""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

retrieval_tool = Tool(
    name="RAG_QA",
    func=retrieve_info,
    description="Answer questions based on the provided text file."
)

# Calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception:
        return "Invalid mathematical expression."

# Weather tool 
weather_data = {
    "delhi": "Sunny, 35°C",
    "mumbai": "Rainy, 28°C",
    "kolkata": "Humid, 33°C",
    "bangalore": "Cloudy, 25°C",
    "srinagar": "Chilly, 2°C"
}

@tool
def get_weather(state: str) -> str:
    """Fetch static weather info for Indian states."""
    state = state.lower()
    return weather_data.get(state, "Weather data not available for this state.")

# Capital lookup tool
capital_data = {
    "india": "New Delhi",
    "france": "Paris",
    "germany": "Berlin",
    "japan": "Tokyo",
    "usa": "Washington, D.C."
}

@tool
def get_capital(country: str) -> str:
    """Lookup the capital of a given country."""
    country = country.lower()
    return capital_data.get(country, "Capital not available for this country.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)


rag_agent = create_react_agent(
    model=llm,
    tools=[retrieval_tool],
    prompt="You are a RAG expert. Always use RAG_QA to answer from context.",
    name="rag_agent"
)

calc_agent = create_react_agent(
    model=llm,
    tools=[calculator],
    prompt="You are a math agent. Solve numerical problems.",
    name="calc_agent"
)

weather_agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a weather assistant. Provide weather details for Indian states.",
    name="weather_agent"
)

capital_agent = create_react_agent(
    model=llm,
    tools=[get_capital],
    prompt="You are a capital lookup agent. Provide country capitals.",
    name="capital_agent"
)


workflow = create_supervisor(
    model=llm,
    agents=[rag_agent, calc_agent, weather_agent, capital_agent],
    prompt=(
        "You are a supervisor managing four agents:\n"
        "- RAG agent: for context-based Q&A\n"
        "- Calculator agent: for math\n"
        "- Weather agent: for weather info\n"
        "- Capital agent: for country capitals\n"
        "Route queries ONLY to the correct agent. Never answer directly."
    )
)

supervisor = workflow.compile(name="supervisor")

st.set_page_config(page_title="Supervisor Chatbot")
st.title("Supervisor Chatbot")

user_input = st.text_input("Ask your question:")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        final_state = supervisor.invoke({"messages": [{"role": "user", "content": user_input}]})

    messages = final_state.get("messages", [])
    if messages:
        bot_reply = messages[-1].content
    else:
        bot_reply = "Sorry, I didn’t get a reply."

    st.markdown(f"**Answer:** {bot_reply}")
