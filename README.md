# Chat with Multiple PDFs
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/GarvGangwani/Chat-with-multi-pdfs)

This project is a Streamlit web application that allows you to chat with multiple PDF documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to extract relevant information from your documents and generate conversational responses. The backend is powered by the high-speed Groq API for language model inference, FAISS for efficient similarity search, and Hugging Face for sentence embeddings.

## Features

- **Multiple PDF Upload:** Upload and process several PDF files at once.
- **Text Extraction:** Automatically extracts text content from all pages of the uploaded PDFs.
- **Conversational AI:** Ask questions in natural language and receive context-aware answers based on the document content.
- **Fast Inference:** Utilizes the Groq API with models like Llama 3.1 and Mixtral for near-instant responses.
- **Conversation Memory:** Remembers the context of the current conversation.
- **Simple UI:** An intuitive web interface built with Streamlit.

## How It Works

The application follows a standard RAG pipeline:

1.  **PDF Text Extraction:** When you upload PDFs, the application uses `PyPDF2` to read and extract the raw text from each document.
2.  **Text Chunking:** The extracted text is split into smaller, manageable chunks using `langchain.text_splitter.CharacterTextSplitter`. This is crucial for fitting the context into the language model's limits.
3.  **Embedding Generation:** Each text chunk is converted into a numerical vector (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face.
4.  **Vector Store Creation:** The embeddings are indexed and stored in a `FAISS` vector store. This allows for extremely fast and efficient semantic searching.
5.  **Conversational Retrieval:** When you ask a question:
    - The question is embedded.
    - FAISS retrieves the most relevant text chunks from the vector store based on semantic similarity.
    - The question and the retrieved chunks are passed to a Groq language model via `langchain.chains.ConversationalRetrievalChain`.
    - The LLM generates a response based on the provided context, which is then displayed in the chat.

## Setup and Installation

Follow these steps to run the application locally. It is recommended to use a virtual environment. The application requires **Python 3.9**.

**1. Clone the Repository**

```bash
git clone https://github.com/GarvGangwani/Chat-with-multi-pdfs.git
cd Chat-with-multi-pdfs
```

**2. Install Dependencies**

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

**3. Configure API Keys**

This project requires a Groq API key for language model access.

-   Sign up and get your free API key from the [Groq Console](https://console.groq.com/keys).
-   Create a file named `.env` in the root directory of the project.
-   Add your API key to the `.env` file as follows:

```
GROQ_API_KEY="your_actual_groq_api_key"
```

**4. Run the Application**

Launch the Streamlit app using the following command:

```bash
streamlit run app.py
```

The application will open in a new tab in your web browser.

## Usage

1.  **Upload Documents:** Use the sidebar to upload one or more PDF files you wish to chat with.
2.  **Process Documents:** Click the "Process" button. The application will extract text, create embeddings, and build the vector store. A success message will appear when complete.
3.  **Ask a Question:** Type your question into the input box at the bottom of the main page and press Enter.
4.  **View Response:** The model's response, based on the content of your documents, will appear in the chat window.

## Technologies Used

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **LLM Orchestration:** [LangChain](https://www.langchain.com/)
-   **LLM Inference:** [Groq](https://groq.com/) (Llama 3.1, Mixtral, etc.)
-   **Embeddings:** [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
-   **Vector Store:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
-   **PDF Parsing:** [PyPDF2](https://pypi.org/project/PyPDF2/)
