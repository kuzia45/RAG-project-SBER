import streamlit as st
import sys
import os
import uuid
from langchain_community.vectorstores import FAISS


sys.path.append('./src/utils')
from llm_and_embeddings import create_conversational_rag_chain, get_embeddings
from get_retriever import extract_from_download
CREDENTIALS = os.getenv('CREDENTIALS')  
retriever = FAISS.load_local(folder_path=os.path.abspath('data/db/'),
                                            embeddings=get_embeddings(), 
                                            allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 3})
st.session_state.retriever = retriever
st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(st.session_state.retriever, credentials=CREDENTIALS)
def main_screen():
    st.title("📚 RAG-бот")
    st.write("Привет, я чат-бот. Загрузи в меня документы")

    uploaded_file = st.file_uploader("Загрузить PDF документ", type="pdf", accept_multiple_files=True)
    if uploaded_file:
            with st.spinner("Идет обработка"):
                session_id=str(uuid.uuid4())
                retriever = extract_from_download(uploaded_file, session_id=session_id)
                if retriever:
                    st.session_state.retriever = retriever
                    st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(st.session_state.retriever, credentials=CREDENTIALS)
                    if st.session_state.conversational_rag_chain:
                        st.success("Документ успешно загружен")
                        st.session_state.page = "chat"
                        st.session_state.session_id = session_id
                        st.rerun()
                    else:
                        st.error("Ошбика при создании RAG цепочки")

def chat_screen():
    st.title("💬 RAG-бот")
    st.sidebar.title('Меню')
    if retriever:
        session_id=str(uuid.uuid4())
        st.session_state.session_id = session_id
    if st.sidebar.button("Загрузить новый документ"):
        st.session_state.page = "main"
        st.session_state.chat_history = []
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()
    if st.sidebar.button("Очистить историю чата"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Задайте вопрос")

    if user_input and st.session_state.conversational_rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = st.session_state.conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                )
            if full_response:
                response_placeholder.markdown(full_response['answer'])
                print (full_response)
            else:
                response_placeholder.markdown("Не удалось сгенерировать ответ")
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

def main():
    st.set_page_config(page_title="RAG-бот", page_icon="📚", layout="wide")

    # Initialize session state variables
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""
    if "conversational_rag_chain" not in st.session_state:
        st.session_state.conversational_rag_chain = None
    if "history_store" not in st.session_state:
        st.session_state.history_store = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if st.session_state.page == "main":
        main_screen()
    elif st.session_state.page == "chat":
        chat_screen()

if __name__ == "__main__":
    main()