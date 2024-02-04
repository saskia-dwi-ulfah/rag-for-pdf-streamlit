import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------- ## ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------- ## ----------------------------

with st.sidebar:
   api_key = st.text_input("Input your Open AI API key")

   # 1. Load file
   uploaded_file = st.file_uploader("Upload your PDF file", type = ["pdf"])

# google_api_key = "AIzaSyDMTHFK7bGAmifKOXj1iXpfRMCRhjdzKiY"
# openai_api_key = "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"
# os.environ["OPENAI_API_KEY"] = openai_api_key
hf_api_key = "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"

st.title("Ask The PDF 📑")

if uploaded_file is not None: 
    temp_file = "./temporary.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    loader = PyPDFLoader(temp_file, extract_images = True)
    docs = loader.load()

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs)

    # 3. Save
    # vectorstore = Chroma.from_documents(documents = splits, 
    #                                   embedding = GoogleGenerativeAIEmbeddings(
    #                                        model = "models/embedding-001",
    #                                        google_api_key = api_key))

    # vectorstore = Chroma.from_documents(documents = splits, 
    #                                embedding = OpenAIEmbeddings())

    vectorstore = Chroma.from_documents(documents = splits, 
                                    embedding = HuggingFaceHubEmbeddings(model = "core42/jais-13b-chat", huggingfacehub_api_token = hf_api_key))
    # 4. Retrieve and generate 
    retriever = vectorstore.as_retriever()

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences minimum and give the answer in the complete way.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    # llm = GoogleGenerativeAI(model = "gemini-pro", google_api_key = api_key)
    # llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature=0.9)
    llm = HuggingFaceHub(repo_id="core42/jais-13b-chat", api_key = hf_api_key)

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
            # stream = client.chat.completions.create(
            #    model=st.session_state["openai_model"],
            #    messages=[
            #        {"role": m["role"], "content": m["content"]}
            #        for m in st.session_state.messages
            #    ],
            #    stream=True,
            #)
            # queries = [
            #        {"role": m["role"], "content": m["content"]}
            #        for m in st.session_state.messages
            #]

            response = rag_chain.invoke(prompt)

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})



    # query = st.text_area("Ask your question")
    # if query:
    #    resp = rag_chain.invoke(query)
    #    st.text(resp)
