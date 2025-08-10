import streamlit as st
import google.generativeai as genai
import os

# Setting up the API key with env and saving key in terminal
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit app
st.title("My Chatbot")

user_input = st.text_input("Write your question:")

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Loading.."):
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_input)
        st.write("**Response:**", response.text)
    else:
        st.warning("Please enter a question.")


