import streamlit as st
from streamlit_modal import Modal

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def display_how_to():
    model = Modal(key = "how_to", title = "🚀 How to Use The App")

    with model.container():
        st.text("▶️ Insert your Open AI API key.")
        st.text("▶️ Upload your PDF document that you want to ask.")
        st.text("▶️ Ask the question and LLM will answer all of your questions.")

