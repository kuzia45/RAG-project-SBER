FROM python:3.12

WORKDIR /RAG-project-SBER

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["bash", "-c", "streamlit run src/apps/streamlit_app.py & python src/apps/botv3.py"]

