import streamlit as st
import os
import uuid
from langchain_community.vectorstores import FAISS
from llm_and_embeddings import create_conversational_rag_chain, get_embeddings
from get_retriever import extract_from_download

CREDENTIALS = os.getenv('CREDENTIALS')
retriever = FAISS.load_local(folder_path='''C:/Users/mi/Documents/Kostya's-RAG-project/data/db2''', 
                                            embeddings=get_embeddings(), 
                                            allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 3})
st.session_state.retriever = retriever

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
                    for title in uploaded_file:
                        st.session_state.pdf_name += "\n\n"
                        st.session_state.pdf_name += title.name
                        st.session_state.pdf_name += "\n\n"
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
    if retriever:
        st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(st.session_state.retriever, credentials=CREDENTIALS)
        session_id=str(uuid.uuid4())
        st.session_state.session_id = session_id
    st.sidebar.title("Список загруженных документов")
    st.sidebar.info(f"PDF: {st.session_state.pdf_name}")
    if st.sidebar.button("Загрузить новый документ"):
        st.session_state.page = "main"
        st.session_state.chat_history = []
        st.session_state.history_store = None
        st.session_state.session_id = None
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
            full_response = ""
            try:
                for chunk in st.session_state.conversational_rag_chain.stream(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                ):
                    if isinstance(chunk, dict):
                        content = chunk.get('answer') or chunk.get('text') or chunk.get('content') or ''
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                if full_response:
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown("Не удалось сгенерировать ответ")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
                full_response = "При попытке генерации ответа произошла ошибка. Попробуйте еще раз."
                response_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.button("Очистить историю чата"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()

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