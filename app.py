import os
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "past" not in st.session_state:
        st.session_state.past = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # PDF upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        text = []
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        
        full_text = "\n".join(text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)

        huggingface_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = FAISS.from_texts(chunks, huggingface_embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history")

        llm = ChatGroq(
            model="Llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        # Handle user input
        user_input = st.text_input("Ask a question about the PDF:")
        if user_input:
            response = chain.invoke({"question": user_input})
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response['answer'])
            st.session_state.chat_history.append((user_input, response['answer']))

        # Display chat history using streamlit-chat
        if st.session_state.chat_history:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                message(user_msg, is_user=True, key=str(i) + '_user')
                message(bot_msg, key=str(i))
    else:
        st.write("Please upload a PDF file to start chatting with it.")

if __name__ == "__main__":
    main()
