import os
import logging
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models.gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Конфигурация логирования
logging.basicConfig(level=logging.INFO)

# Используем переменные окружения
CREDENTIALS = os.getenv('CREDENTIALS')

def get_llm(credentials):
    """Возвращает экземпляр GigaChat с предоставленными учетными данными."""
    return GigaChat(credentials=credentials, verify_ssl_certs=False)

def get_embeddings():
    """Возвращает embeddings модель."""
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def create_conversational_rag_chain(retriever, credentials):
    """Создает цепочку для RAG с учетом истории переписки."""
    llm = get_llm(credentials)
    if not llm:
        logging.error("Не удалось инициализировать LLM.")
        return None
        
    # Промпт для контекстуализации вопроса
    contextualize_q_system_prompt = (
        "Запоминай историю чата и последний вопрос пользователя. "
        "Сначала пытайся ответить на вопрос, основываясь на контексте и истории чата, "
        "затем сформулируй отдельный вопрос без истории чата. "
        "Не отвечай на этот вопрос, просто перефразируй его, если это необходимо, в противном случае верни как есть."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Промпт для ответа на вопрос
    system_prompt = (
        '''Ты подробно и структурированно отвечаешь на вопрос только на основе контекста.
        Отвечай максимально подробно
        Твоя роль: отвечать на вопросы пользователя на основе контекста. Ответ должен быть максимально объемным.
        Если вопрос не связан с контекстом, сообщи пользователю об этом.
        Для ответа на вопрос нельзя пользоваться собственными знаниями, можно пользоваться только контекстом
        Предлагай пользователю задать дополнительные похожие вопросы. 
        Найди информацию только в контексте. Если в контексте нет ответа на вопрос, сообщи об этом. 
        Контекст: {context}
        Ответ:'''
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Управление историей чата
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
    
    logging.info("Цепочка для RAG успешно создана.")
    return conversational_rag_chain, store