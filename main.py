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

whatsapp_client = load_client()



# Sidebar options
st.sidebar.header("Options")
st.sidebar.subheader("Chat Options")
number_messages = st.sidebar.number_input("Maximum number of messages", min_value=1, value=100, step=100)
start = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=365))
end = st.sidebar.date_input("End date", datetime.now())

st.sidebar.subheader("Model Options")
model_name = st.sidebar.selectbox("Select a model", ["llama3.2:1b", "deepseek-r1:1.5b"])
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.8)
ollama_model = get_model(model_name, temp)

# Main interface
chat_name = st.selectbox("Select a chat", load_names(whatsapp_client))
question = st.text_input("Enter a question")
if st.button("Ask", use_container_width=True):
    with st.spinner("Loading chat..."):
        chat = load_messages(whatsapp_client, chat_name, number_messages, start, end)

#     prompt_template = ChatPromptTemplate(
#         [
#             SystemMessage(
#                 '''
#                 You are a helpful AI assistant that can answer factual questions about the contents
#                 of a chat attached below. You can recieve a question in any language and should respond in that language.
#                 Do not think about the problem, just provide an answer with the information you have.
#                 '''),
# #            SystemMessage("{chat}"),
#             HumanMessage("{user_input}"),
#             SystemMessage("""
#                           chat:
#                           {chat}
#                           Answer:
#                           """)
#         ]
#     )
    # convert chat to string

    prompt = """
    Use the following context to answer the question. The context corresponds to a pandas dataframe for a chat.
    The columns are: sender (Sender name or number), datetime (in %Y-%m-%d %I:%M:%S format), message and has_emoji_text (That indicates wether the message has emojis)
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(prompt)
    chain: Runnable = prompt | ollama_model | StrOutputParser()
    with st.spinner("Generating response..."):
        response = chain.invoke({"context": chat, "question": question})

    st.write(response)