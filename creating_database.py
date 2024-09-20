from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm_and_embeddings import get_embeddings
import os

documents = []
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
embedding = get_embeddings()
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
vectorstore.save_local(folder_path='db/')