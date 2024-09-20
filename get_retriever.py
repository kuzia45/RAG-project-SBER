import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from llm_and_embeddings import get_embeddings

os.environ['CURL_CA_BUNDLE'] = ''

def extract_text_from_dir():
    documents=[]
    for file in os.listdir('files'):
        if file.endswith('.pdf'):
            pdf_path = './files/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = './files/' + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = './files/' + file
            loader = TextLoader(text_path, encoding='utf-8')
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,)
    splits = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    if not embeddings:
        raise ValueError("Failed to initialize embeddings.")
    return retriever

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