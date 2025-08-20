import streamlit as st
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 

def agent1_interpreter(user_story: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are an assistant that extracts clear programming requirements from user stories.
    User story: {user_story}

    Return the requirements as a clear checklist.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def agent2_codegen(requirements: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are a Python developer.
    Based on these requirements, write a runnable Python program.

    Requirements:
    {requirements}

    Only output the Python code. No explanation.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

st.title("My Code Generator")

user_story = st.text_area("Enter the user story here:")

if st.button("Generate Code"):
    if not user_story.strip():
        st.warning("Please enter a user story first.")
    else:
        with st.spinner("Interpreting your user story"):
            requirements = agent1_interpreter(user_story)
        with st.spinner("Now generating the code..."):
            code = agent2_codegen(requirements)
        st.subheader("Generated Code")
        st.code(code, language="python")

        filename = "code.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        with open(filename, "rb") as f:
            st.download_button(
                label="Download Code File",
                data=f,
                file_name=filename,
                mime="text/x-python"
            )
