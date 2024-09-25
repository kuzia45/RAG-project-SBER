import os
import sys
import logging
import telebot
from telebot import types
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from llm_and_embeddings import get_embeddings, create_conversational_rag_chain 

# Настройка логирования
logging.basicConfig(level=logging.INFO)

API_TOKEN = os.getenv('API_TOKEN')
CREDENTIALS = os.getenv('CREDENTIALS')

class RAGBot:
    def __init__(self, token, credentials):
        self.bot = telebot.TeleBot(token)
        self.credentials = credentials
        self.vector_store = FAISS.load_local(folder_path='db2/', embeddings=get_embeddings(), allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.conversational_rag_chain, _ = create_conversational_rag_chain(retriever=self.retriever, credentials=self.credentials)
        self.file_names = self.get_file_names()
        self.setup_message_handlers()

    def restart_bot(self):
        """Перезапускает бот."""
        os.execv(sys.executable, ['python'] + sys.argv)

    def setup_message_handlers(self):
        @self.bot.message_handler(content_types=['audio', 'video', 'photo', 'sticker', 'voice', 'location', 'contact'])
        def not_text(message):
            self.send_message(message.from_user.id, 'Я работаю только с текстовыми сообщениями!')
        
        @self.bot.message_handler(content_types=['document'])
        def add_document(document):
            self.handle_add_document(document)

        @self.bot.message_handler(commands=['restart'], content_types=['text'])
        def restarting(message):
            self.send_message(message.chat.id, "Бот перезапущен")
            self.restart_bot()

        @self.bot.message_handler(commands=['help', 'start'], content_types=['text'])
        def send_welcome(message):
            self.send_message(message.from_user.id, "Привет. Я RAG-бот. В меня загружены документы и я могу по ним ответить на твой вопрос", reply_markup=self.get_markup())

        @self.bot.message_handler(content_types=['text'])
        def answer_the_question(message):
            self.handle_answer_question(message)

    def send_message(self, user_id, text, reply_markup=None):
        """Отправляет сообщение пользователю."""
        self.bot.send_message(user_id, text, reply_markup=reply_markup)

    def get_markup(self):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton('🗑️ Перезапустить бота')
        button2 = types.KeyboardButton('ℹ️ Информация о боте')
        button3 = types.KeyboardButton('📄 Добавить документ')
        markup.add(button2, button3, button1)
        return markup

    def get_file_names(self):
        """Возвращает имена файлов в папке files, кэшируя их."""
        if not hasattr(self, '_file_names_cache'):
            self._file_names_cache = [f for f in os.listdir('files') if f.endswith('.pdf')]
        return self._file_names_cache

    def handle_add_document(self, document):
        try:
            if not document.document.file_name.endswith('.pdf'):
                self.send_message(document.from_user.id, "Пожалуйста, загрузите файл в формате PDF.")
                return
            
            file_info = self.bot.get_file(document.document.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            src = f'files/{document.document.file_name}'
            
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)
            
            loader = PyPDFLoader(src)
            doc = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            self.send_message(document.from_user.id, f'Вы загрузили следующий файл: {document.document.file_name}')
            self.vector_store.add_documents(text_splitter.split_documents(doc))
            self.update_retriever()
            self.file_names = self.get_file_names()
            os.remove(src)
            logging.info(f'Файл {document.document.file_name} успешно загружен и обработан.')

        except FileNotFoundError:
            self.send_message(document.from_user.id, "Файл не найден.")
            logging.error(f'Ошибка: Файл не найден - {document.document.file_name}')
        except telebot.apihelper.ApiException:
            self.send_message(document.from_user.id, "Ошибка API.")
            logging.error("Ошибка API при загрузке документа.")
        except Exception as e:
            self.send_message(document.from_user.id, str(e))
            logging.error(f'Ошибка при обработке документа: {str(e)}')

    def update_retriever(self):
        """Обновляет retriever на основе текущего векторного хранилища."""
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.conversational_rag_chain, _ = create_conversational_rag_chain(retriever=self.retriever, credentials=self.credentials)

    def handle_answer_question(self, message):
        if message.text == 'ℹ️ Информация о боте':
            self.send_message(message.from_user.id, f'Я RAG-бот. Я могу ответить на любой вопрос, связанный со следующими документами:\n\n {self.file_names}')
        elif message.text == '📄 Добавить документ':
            self.send_message(message.from_user.id, 'Прикрепите документ и отправьте мне. Документ должен быть формата PDF')
        elif message.text == '🗑️ Перезапустить бота':
            self.send_message(message.chat.id, "Бот перезапущен")
            self.restart_bot()
        else:
            response = self.conversational_rag_chain.invoke({'input': message.text}, config={
                "configurable": {"session_id": message.chat.id}
            })
            self.send_message(message.chat.id, response['answer'])

    def run(self):
        """Запускает бота в режиме ожидания новых сообщений."""
        self.bot.polling(none_stop=True)

if __name__ == '__main__':
    rag_bot = RAGBot(API_TOKEN, CREDENTIALS)
    rag_bot.run()