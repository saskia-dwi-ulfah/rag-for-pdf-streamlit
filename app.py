import os
import shutil
import tempfile
import streamlit as st

from utils.function import format_docs, display_how_to
from utils.template import template

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


with st.sidebar:
   st.button("How to Use The App", on_click = display_how_to)

   api_key = st.text_input("Input your Open AI API key")
   os.environ["OPENAI_API_KEY"] = api_key

   # 1. Load file
   uploaded_file = st.file_uploader("Upload your PDF file", type = ["pdf"])

st.title("Ask The PDF 📑🔮🤔")
st.text("Powered by GPT 3.5 Turbo")

if uploaded_file is not None: 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        # Copy the uploaded file to the temporary file
        shutil.copyfileobj(uploaded_file, tmpfile)
        tmpfile_path = tmpfile.name

    loader = PyPDFLoader(tmpfile_path, extract_images = False) # error when load rapidocr-onnxruntime
    docs = loader.load()

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs)

    # 3. Save
    vectorstore = Chroma.from_documents(documents = splits, 
                                embedding = OpenAIEmbeddings())

    # 4. Retrieve and generate 
    retriever = vectorstore.as_retriever()
    custom_rag_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    ) 

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask the PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = rag_chain.invoke(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


