# WhatsApp QA Streamlit App

This Streamlit app retrieves recent WhatsApp chat messages using the whatsoup library. It provides a simple interface to:
1. Log into a WhatsApp web session.
2. Select a chat, a date range, and a maximum number of messages to load.
3. Ask questions about the retrieved messages using a chosen language model.

## How It Works
• The whatsoup library implements a WhatsApp web scraper using Selenium and BeautifulSoup. It logs into a WhatsApp web session and retrieves chat messages from a specified chat.

• The Streamlit app provides a simple interface to use this library to fetch the messages and ask questions about them.

• The app uses ollama to locally host the language models. The models are loaded into memory when the app starts and are used to answer questions about the chat messages.

• The app uses the LangChain library to interact with the language models.

## Current approach and limitations

The current approach uses a simple language model to answer questions about the chat messages. The chat context is passed to the model as part of the prompt, and the model generates an answer based on the context and the question.

The model is not fine-tuned on a specific task or dataset, so its performance is limited. It may struggle with complex questions or questions that require a deep understanding of the chat context.

Moreover, the local models are more limited than the cloud models, which can be more powerful and have more parameters. One of the main limitations is in the empirical context window, therefore the model may not be able to recall some information from the chat messages if the context is too long.

Most models are not fine-tuned on a specific task or dataset, so their performance is limited. Moreover, most models are highly skewed towards the english language, so they may not perform well on other languages, specially with local dialects.

## Future improvements

1. Use a retriever model to extract relevant chat messages based on the question before passing them to the language model. This can help improve the model's performance by providing more relevant context.

2. Use a reasoning agent workflow to answer questions about the chat messages. This can help the model reason about the chat context and provide more accurate answers.

## Usage
1. Clone the repository.
    ```bash
    git clone github.com/grudloff/whatsappQA
    ```
1. Create a virtual environment. (Optional but recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
1. Install dependencies (including Streamlit and whatsoup).
    ```bash
    pip install -r requirements.txt
    ```
1. Install ollama
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
1. Download the language models using ollama.
    ```bash
    ollama pull ollama3.2:1b
    ollama pull deepseek-r1:1.5b
    ```
1. Start the Streamlit app.
    ```bash
    streamlit run main.py  
    ```
