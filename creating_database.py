from langchain_chroma import Chroma
from llm_and_embeddings import get_embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile

vector_store = Chroma(
        collection_name="initial_rag_collection",
        embedding_function=get_embeddings(),
        persist_directory="db/",  # Where to save data locally, remove if not neccesary
    )

def creation():
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
    vector_store.add_documents(documents=splits, persist_directory='db/')
    retriever = vector_store.as_retriever()
    return retriever
    #print (retriever)

def extract_from_download(pdf_files, session_id):
    documents=[]
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        documents.extend(text_splitter.split_documents(docs))
        os.unlink(tmp_file_path)  # Delete the temporary file
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents)
    retriever = vectorstore.as_retriever()
    return retriever

def add_doc(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,)
    splits = text_splitter.split_documents([document])
    vector_store.add_documents(documents=splits)
    retriever = vector_store.as_retriever()
    return retriever

creation()