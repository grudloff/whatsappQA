""" Streamlit app to ask questions from an specific chat in WhatsApp """
from datetime import datetime, timedelta
import streamlit as st
from whatsoup.whatsoup import WhatsappClient
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import StrOutputParser
from ollama import Client as OllamaClient


@st.cache_resource
def load_client():
    client = WhatsappClient(headless=False)
    client.login()
    client.headless = True
    return client

@st.cache_data
def load_names(_client: WhatsappClient):
    names = _client.get_chat_names()
    return names

@st.cache_data(show_spinner=False)
def load_messages(_client: WhatsappClient, query: str, max_messages: int, start_date: datetime,
                  end_date: datetime) -> pd.DataFrame:
    messages: pd.DataFrame = _client.get_chat(query, max_messages)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    messages = messages.drop(columns=["data-id"])
    messages = messages.drop(columns=["has_emoji_text"])

    max_date = messages["datetime"].max()
    min_date = messages["datetime"].min()
    if start_date < min_date:
        messages = messages[messages["datetime"] <= end_date]
    elif end_date > max_date:
        messages = messages[messages["datetime"] >= start_date]
    return messages

@st.cache_resource
def get_model(model: str, temperature: int) -> ChatOllama:
    model = ChatOllama(model=model, temperature=temperature)
    return model

@st.cache_data
def get_available_models():
    ollama_client = OllamaClient()
    available_models = ollama_client.list()["models"]
    available_models = [model["model"] for model in available_models]
    if len(available_models) == 0:
        raise ValueError("""No models available", please pull one from ollama using the cli. For example:
                         ```bash
                         ollama pull llama3.2:1b
                         ```
                         """)
    return available_models

whatsapp_client = load_client()



# Sidebar options
st.sidebar.header("Options")
st.sidebar.subheader("Chat Options")
number_messages = st.sidebar.number_input("Maximum number of messages", min_value=1, value=100, step=100)
start = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=365))
end = st.sidebar.date_input("End date", datetime.now())

st.sidebar.subheader("Model Options")
model_name = st.sidebar.selectbox("Select a model", get_available_models())
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.8)
ollama_model = get_model(model_name, temp)

# Main interface
with st.form(key='my_form'):
    chat_name = st.selectbox("Select a chat", load_names(whatsapp_client), index=None)
    question = st.text_input("Enter a question")
    submit_button = st.form_submit_button(label='Ask', use_container_width=True)

if submit_button and chat_name and question:
    with st.spinner("Loading chat..."):
        chat = load_messages(whatsapp_client, chat_name, number_messages, start, end)

    prompt = """
    Use the following context to answer the question, you should respond in the question's
    language. The context corresponds to a table with the messages of a WhatsApp chat.
    The columns are: sender (Sender name or number),
    datetime (in %Y-%m-%d %I:%M:%S format), message and has_emoji_text (That indicates wether the
    message has emojis)
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(prompt)
    chain: Runnable = prompt | ollama_model | StrOutputParser()
    with st.spinner("Generating response..."):
        response = chain.invoke({"context": chat, "question": question})

    st.write(response)
    # add chat in collapsible container
    with st.expander("Show context"):
        st.write(chat)
