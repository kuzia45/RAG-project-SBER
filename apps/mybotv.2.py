import telebot
from telebot import types
import os
import sys
from dotenv import load_dotenv
from llm_and_embeddings import create_conversational_rag_chain, get_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Загружаем переменные окружения из файла .env
load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')  # Получаем значение токена
CREDENTIALS = os.getenv('CREDENTIALS')  # Получаем значение учетных данных

os.environ['CURL_CA_BUNDLE'] = ''

class RAGBot:
    def __init__(self, token, credentials):
        self.bot = telebot.TeleBot(token)
        self.credentials = credentials
        self.vector_store = FAISS.load_local(folder_path='db/', embeddings=get_embeddings(), allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.conversational_rag_chain, _ = create_conversational_rag_chain(retriever=self.retriever, credentials=self.credentials)

        self.file_names = self.get_file_names()  # Получаем список загруженных файлов

        self.setup_message_handlers()

    def restart_bot(self):
        """Перезапускает бот."""
        os.execv(sys.executable, ['python'] + sys.argv)

    def setup_message_handlers(self):
        @self.bot.message_handler(content_types=['audio', 'video', 'photo', 'sticker', 'voice', 'location', 'contact'])
        def not_text(message):
            self.bot.send_message(message.from_user.id, 'Я работаю только с текстовыми сообщениями!')

        @self.bot.message_handler(content_types=['document'])
        def add_document(document):
            self.handle_add_document(document)

        @self.bot.message_handler(commands=['restart'], content_types=['text'])
        def restarting(message):
            #"""Обрабатывает команду на перезапуск бота."""
            self.bot.send_message(message.chat.id, "Бот перезапущен")
            self.restart_bot()
            
        @self.bot.message_handler(commands=['help', 'start'], content_types=['text'])
        def send_welcome(message):
            self.bot.send_message(message.from_user.id, """Привет. Я RAG-бот. В меня загружены документы и я могу по ним ответить на твой вопрос""",
                                  reply_markup=self.get_markup())

        @self.bot.message_handler(content_types=['text'])
        def answer_the_question(message):
            self.handle_answer_question(message)

    def get_markup(self):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton('🗑️ Перезапустить бота')
        button2 = types.KeyboardButton('ℹ️ Информация о боте')
        button3 = types.KeyboardButton('📄 Добавить документ')
        markup.add(button2, button3, button1)
        return markup

    def get_file_names(self):
        return [f for f in os.listdir('files')]

    def handle_add_document(self, document):
        try:
            if not document.document.file_name.endswith('.pdf'):
                self.bot.send_message(document.from_user.id, "Пожалуйста, загрузите файл в формате PDF.")
                return

            # Сохраняем файл локально
            file_info = self.bot.get_file(document.document.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            src = 'files/' + document.document.file_name
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            # Загружаем документ в векторную БД
            loader = PyPDFLoader(src)
            doc = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            self.bot.send_message(document.from_user.id, f'Вы загрузили следующий файл: {document.document.file_name}')
            self.vector_store.add_documents(text_splitter.split_documents(doc))
            self.update_retriever()
            self.file_names = self.get_file_names()
            # Удаляем загруженный файл с локальной машины
            os.remove(src)
        except Exception as e:
            self.bot.send_message(document.from_user.id, str(e))
            print(str(e))

    def update_retriever(self):
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.conversational_rag_chain, _ = create_conversational_rag_chain(retriever=self.retriever, credentials=self.credentials)
    
    def handle_answer_question(self, message):
        if message.text == 'ℹ️ Информация о боте':
            self.bot.send_message(message.from_user.id, f'Я RAG-бот. Я могу ответить на любой вопрос, связанный со следующими документами:\n\n {self.file_names}')
        elif message.text == '📄 Добавить документ':
            self.bot.send_message(message.from_user.id, 'Прикрепите документ и отправьте мне. Документ должен быть формата PDF')
        elif message.text == '🗑️ Перезапустить бота':
            self.bot.send_message(message.chat.id, "Бот перезапущен")
            self.restart_bot()
        else:
            response = self.conversational_rag_chain.invoke({'input': message.text}, config={
                "configurable": {"session_id": message.chat.id}
            })
            self.bot.send_message(message.chat.id, response['answer'])
    

    def run(self):
        self.bot.polling(none_stop=True)

if __name__ == '__main__':
    rag_bot = RAGBot(API_TOKEN, CREDENTIALS)
    rag_bot.run()