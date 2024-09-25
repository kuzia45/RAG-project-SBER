import streamlit as st
import os
import uuid
from apps.llm_and_embeddings import create_conversational_rag_chain
from apps.get_retriever import extract_from_download

CREDENTIALS='OGMwNTUyMzktMjM0Ny00MDIxLThiZWQtNDlkY2E3ODkxOTk5OmIyNTI4MDQ3LTEyNzQtNGIzMy1iZGNkLTNkNzg4MDEyZWY4Mg=='
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def main_screen():
    st.title("📚 PDF Chat App")
    st.write("Welcome! Please upload your PDF file to begin.")

    uploaded_file = st.file_uploader("Select a PDF file with minimal size for a faster response.", type="pdf", accept_multiple_files=True)
    
    if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                session_id=str(uuid.uuid4())
                retriever = extract_from_download(uploaded_file,session_id)
                if retriever:
                    st.session_state.retriever = retriever
                    for title in uploaded_file:
                        st.session_state.pdf_name += "\n\n"
                        st.session_state.pdf_name += title.name
                        st.session_state.pdf_name += "\n\n"
                    st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(retriever, credentials=CREDENTIALS)
                    if st.session_state.conversational_rag_chain:
                        st.success("PDF uploaded and processed successfully!")
                        st.session_state.page = "chat"
                        st.session_state.session_id = session_id
                        st.rerun()
                    else:
                        st.error("Failed to create conversation chain. Please try again.")

def chat_screen():
    st.title("💬 Talk to your PDF")
    
    st.sidebar.title("PDF Info")
    st.sidebar.info(f"PDF: {st.session_state.pdf_name}")
    if st.sidebar.button("Upload New PDF"):
        st.session_state.page = "main"
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversational_rag_chain = None
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input and st.session_state.conversational_rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            st.write(st.session_state.conversational_rag_chain.invoke({'input': user_input}, config={
                        "configurable": {"session_id": st.session_state.session_id}
                    }))
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
                            response_placeholder.markdown(chunk)
                    elif isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(chunk)
                
                if full_response:
                    response_placeholder.markdown(full_response)
                    response_placeholder.markdown(chunk)
                else:
                    response_placeholder.markdown("I'm sorry, I couldn't generate a response.")
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
                full_response = "I encountered an error while trying to respond. Please try again."
                response_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()

def main():
    st.set_page_config(page_title="PDF Chat App", page_icon="📚", layout="wide")

    # Initialize session state variables
    if "page" not in st.session_state:
        st.session_state.page = "main"
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