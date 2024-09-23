import telebot
from telebot import types
import os
from time import sleep
from llm_and_embeddings import create_conversational_rag_chain, get_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
os.environ['CURL_CA_BUNDLE'] = ''

API_TOKEN = '7617861148:AAFKb0TIj5CVRcpHh4QcMgbNVeJpfodqtVI'
CREDENTIALS = 'YzYxYTQwYTgtZGE4ZC00NWEyLWFiOGEtZmQzNzExZDg1ZWQzOmZkMTg1OWE1LWM2YTAtNDNiNy05YzAwLTg0NjE3YWE0YmE4Mw=='

bot = telebot.TeleBot(API_TOKEN)
vectore_store = FAISS.load_local(folder_path='db/', embeddings=get_embeddings(), allow_dangerous_deserialization=True)
retriever = vectore_store.as_retriever(search_kwargs={"k":3})
conversational_rag_chain, store = create_conversational_rag_chain(retriever=retriever, credentials=CREDENTIALS)
#Получаем список загруженных файлов
def get_file_names():
    file_names = []
    for file in os.listdir('files'):
        path = './files/' + file
        file_names.append(os.path.basename(path))
    return file_names

# Функция для автоматического ответа в случае нетекстового сообщения
@bot.message_handler(content_types=['audio',
                                    'video',
                                    'photo',
                                    'sticker',
                                    'voice',
                                    'location',
                                    'contact'])
def not_text(message):
    user_id = message.chat.id
    bot.send_message(message.from_user.id, 'Я работаю только с текстовыми сообщениями!')
#Добавление документа
@bot.message_handler(content_types=['document'])
def add_documnet(document):
    try:
        file_info = bot.get_file(document.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = 'files/' + document.document.file_name;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        loader = PyPDFLoader(src)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        bot.send_message (document.from_user.id, f'Вы загрузили следующий файл {document.document.file_name}')
        vectore_store.add_documents(text_splitter.split_documents(doc))
        #print (new_vectore_store)
        new_retriever = vectore_store.as_retriever(search_kwargs={"k":3})
        conversational_rag_chain, store = create_conversational_rag_chain(retriever=new_retriever, credentials=CREDENTIALS)
    except Exception as e:
        bot.send_message(document.from_user.id, e)
        print (document)

#Кнопка старт
markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
button1 = types.KeyboardButton('Очистить историю')
button2 = types.KeyboardButton('Информация о боте')
button3 = types.KeyboardButton('Добавить документ')
markup.add(button1, button2, button3)
@bot.message_handler(commands=['help', 'start'], content_types=['text'])
def send_welcome(message):
    bot.send_message(message.from_user.id, """Привет. Я RAG-бот. В меня загружены документы и я могу по ним ответить на твой вопрос""", reply_markup=markup)

# Функция, обрабатывающая текстовые сообщения
@bot.message_handler(content_types=['text'])
def answer_the_question(message):
    user_id = message.chat.id
    #Обработка кнопок
    if message.text == 'Информация о боте':
        bot.send_message(message.from_user.id, f'Я RAG-бот. Я могу ответить на любой вопрос, связанный со следующими документами \n\n {get_file_names()}')
    elif message.text == 'Добавить документ':
        bot.send_message(message.from_user.id, 'Прикрепите документ и отправьте мне. Документ должен быть формата PDF')
    else:# Получение и отправка ответа через GigaChat
        response = conversational_rag_chain.invoke({'input': message.text}, config={
                            "configurable": {"session_id": user_id}
                        })
        print (response)
        bot.send_message(user_id, response['answer'])
# Запуск бота
bot.polling(none_stop=True)