import telebot
import tempfile
import os
import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models.gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import logging 
import PyPDF2
import numpy as np
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from telebot import types
os.environ['CURL_CA_BUNDLE'] = ''

API_TOKEN = '7397672553:AAFt26n5bhBl6sOL0zwtgzOgauAVS_NgWZI'

bot = telebot.TeleBot(API_TOKEN)
llm = GigaChat(credentials='YzYxYTQwYTgtZGE4ZC00NWEyLWFiOGEtZmQzNzExZDg1ZWQzOmZkMTg1OWE1LWM2YTAtNDNiNy05YzAwLTg0NjE3YWE0YmE4Mw==', 
               verify_ssl_certs=False)

#Функция для получения эмбединга
def get_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    # model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
        )

#Загружаем файлы
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

#Делаем векторную базу данных и получаем ретривер
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,)
splits = text_splitter.split_documents(documents)
embedding = get_embeddings()
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

#Настраиваем промпты и делаем историю
contextualize_q_system_prompt = (
       "Запоминай историю чата и последний вопрос пользвоателя"
    "Сначала пытайся ответить на вопрос, основываясь на истории чата"
    "сформулируй отдельный вопрос без истории чата"
    "Не отвечай на этот вопрос, просто перефразируй его, если это необходимо, в противном случае верни как есть"
    )
contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
system_prompt = (
       ''' 
    Ты подробно и структурировоно отвечаешь на вопрос на основе загруженных статей об искуственном интелекте.
    Предлагай пользователю задать дополнительные похожие вопросы. Предлагайте контекстную и справочную информацию,
    основываясь на метаданнтых каждого документа из контекста.
    Если ответа на вопрос нет в контексте, предложи вопрос, ответ на который есть в контексте
    Контекст: {context}
    Ответ:'''
    )
qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
def get_file_names():
    file_names = []
    for file in os.listdir('files'):
        path = './files/' + file
        file_names.append(os.path.basename(path))
    return file_names
# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'], content_types=['text'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = types.KeyboardButton('Очистить историю')
    button2 = types.KeyboardButton('Информация о боте')
    markup.add(button1, button2)
    bot.send_message(message.from_user.id, """Привет. Я RAG-бот. В меня загружены документы и я могу по ним ответить на твой вопрос""", reply_markup=markup)


@bot.message_handler(content_types=['text'])
def bot_info(message):
    if message.text == 'Информация о боте':
        bot.send_message(message.from_user.id, f'Я RAG-бот. Я могу ответить на любой вопрос, связанный со следующими документами \n\n {get_file_names}')
# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True,  content_types = ['text'])
def echo_message(message):
    print (message.text)
    answer = conversational_rag_chain.invoke({'input': message.text}, config={
                        "configurable": {"session_id": 'abc123'}
                    })['answer']
    print (answer)
    bot.send_message(message.from_user.id, answer)

bot.polling(none_stop=True, interval=3)

