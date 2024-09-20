import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models.gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

os.environ['CURL_CA_BUNDLE'] = ''
CREDENTIALS = 'YzYxYTQwYTgtZGE4ZC00NWEyLWFiOGEtZmQzNzExZDg1ZWQzOmZkMTg1OWE1LWM2YTAtNDNiNy05YzAwLTg0NjE3YWE0YmE4Mw=='

def get_llm(credentials):
    return GigaChat(credentials=credentials, verify_ssl_certs=False)

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

def create_conversational_rag_chain(retriever, credentials):
    llm = get_llm(credentials)
    if not llm:
        return None
    ### Contextualize question ###
    contextualize_q_system_prompt = (
       "Запоминай историю чата и последний вопрос пользвоателя"
    "Сначала пытайся ответить на вопрос, основываясь контексте и на истории чата"
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
    Ты подробно и структурировоно отвечаешь на вопрос только на основе контекста.
    Предлагай пользователю задать дополнительные похожие вопросы. Предлагайте контекстную и справочную информацию,
    основываясь на метаданнтых каждого документа из контекста.
    Ищи информацию только в контексте и в истории чата
    Если ответа на вопрос нет в контексте, предложи вопрос, ответ на который есть в контексте. Если в контексте нет ответа на вопрос, сообщи об этом 
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
    return conversational_rag_chain, store
