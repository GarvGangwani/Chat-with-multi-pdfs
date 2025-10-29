import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq
import os

# ========================== Helper Functions ==========================

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split large text into overlapping chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def get_vectorstore(text_chunks):
    """Create or load FAISS vector store from text chunks"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Build the conversational retrieval chain with updated Groq models"""
    # Updated model list based on current Groq availability (as of Oct 2024)
    preferred_models = [
        "llama-3.1-8b-instant",      # Fast and efficient
        "llama-3.2-3b-preview",      # Lightweight alternative
        "mixtral-8x7b-32768",        # High quality MoE
        "gemma2-9b-it",              # Google's Gemma
    ]

    llm = None
    working_model = None
    
    for model in preferred_models:
        try:
            # Create the LLM instance
            test_llm = ChatGroq(model=model, temperature=0, max_tokens=2048)
            
            # Test the model with a simple query to ensure it works
            test_llm.invoke("test")
            
            # If we get here, the model works
            llm = test_llm
            working_model = model
            print(f"✅ Successfully using model: {model}")
            break
            
        except Exception as e:
            print(f"⚠️ Model {model} failed: {str(e)}")
            continue

    if llm is None:
        raise RuntimeError(
            "❌ No available Groq models found. Please:\n"
            "1. Check your Groq API key in .env file\n"
            "2. Verify your API key at https://console.groq.com\n"
            "3. Check model availability at https://console.groq.com/docs/models"
        )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        memory=memory
    )
    
    # Store the working model name for user info
    st.session_state.current_model = working_model
    
    return conversation_chain


def handle_userinput(user_question):
    """Process user input and display chat responses"""
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        st.info("💡 Try reprocessing your documents or check your Groq API key.")


# ========================== Streamlit App ==========================

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    st.header("📚 Chat with Multiple PDFs")
    st.markdown("Upload PDFs, process them, and ask questions about their content!")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None

    # Display current model if available
    if st.session_state.current_model:
        st.info(f"🤖 Using model: {st.session_state.current_model}")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
    elif user_question:
        st.warning("⚠️ Please upload and process documents first.")

    with st.sidebar:
        st.subheader("📄 Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )

        if st.button("🚀 Process"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("❌ No readable text found in the PDFs.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success(f"✅ Documents processed successfully! You can now chat below.")
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
                        st.info("💡 Check the console for more details.")

if __name__ == "__main__":
    main()