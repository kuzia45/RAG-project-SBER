import telebot
from telebot import types
import os
from time import sleep
from llm_and_embeddings import create_conversational_rag_chain
from get_retriever import extract_text_from_dir
os.environ['CURL_CA_BUNDLE'] = ''

API_TOKEN = '7617861148:AAFKb0TIj5CVRcpHh4QcMgbNVeJpfodqtVI'
CREDENTIALS = 'YzYxYTQwYTgtZGE4ZC00NWEyLWFiOGEtZmQzNzExZDg1ZWQzOmZkMTg1OWE1LWM2YTAtNDNiNy05YzAwLTg0NjE3YWE0YmE4Mw=='

bot = telebot.TeleBot(API_TOKEN)

retriever = extract_text_from_dir()
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
                                    'document',
                                    'photo',
                                    'sticker',
                                    'voice',
                                    'location',
                                    'contact'])
def not_text(message):
  user_id = message.chat.id
  bot.send_message(user_id, 'Я работаю только с текстовыми сообщениями!')
#Кнопка старт
@bot.message_handler(commands=['help', 'start'], content_types=['text'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = types.KeyboardButton('Очистить историю')
    button2 = types.KeyboardButton('Информация о боте')
    markup.add(button1, button2)
    bot.send_message(message.from_user.id, """Привет. Я RAG-бот. В меня загружены документы и я могу по ним ответить на твой вопрос""", reply_markup=markup)
#Кнопка инфо~
@bot.message_handler(commands=['Информация о боте'])
def bot_info(message: types.Message):
    bot.send_message(message.from_user.id, f'Я RAG-бот. Я могу ответить на любой вопрос, связанный со следующими документами \n\n {get_file_names()}')
    print(get_file_names())
# Функция, обрабатывающая текстовые сообщения
@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_id = message.chat.id
    # Получение и отправка ответа через GigaChat
    response = conversational_rag_chain.invoke({'input': message.text}, config={
                        "configurable": {"session_id": user_id}
                    })
    print (response)
    bot.send_message(user_id, response['answer'])
# Запуск бота
bot.polling(none_stop=True)