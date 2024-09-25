import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from apps.llm_and_embeddings import get_embeddings

os.environ['CURL_CA_BUNDLE'] = ''

def extract_from_download(pdf_files, session_id):
    all=[]
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        text_splitter.split_documents(docs)
        all.extend(text_splitter.split_documents(docs))
        os.unlink(tmp_file_path)  # Delete the temporary file
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(all, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever