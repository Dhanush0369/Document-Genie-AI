import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import numpy as np


load_dotenv()
api_key =os.getenv("GOOGLE_API_KEY")


st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

# Initialize session state variables
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

if 'document_index' not in st.session_state:
    st.session_state.document_index = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'question_embeddings' not in st.session_state:
    st.session_state.question_embeddings = []

if 'question_answers' not in st.session_state:
    st.session_state.question_answers = []

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                st.warning(f"Warning: No text found on page {pdf_reader.pages.index(page)}")
    return text

def get_text_chunks(text):
    if not text.strip():
        st.warning("Warning: No valid text found to split into chunks.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    st.info(f"Number of chunks created: {len(chunks)}")
    return chunks

def process_documents(text_chunks):
    if not text_chunks:
        st.error("No text chunks found. Skipping document processing.")
        return False
    
    with st.spinner("Creating embeddings - this may take a few minutes for large documents..."):
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        
        # Create FAISS index from text chunks
        st.session_state.document_index = FAISS.from_texts(
            text_chunks,
            embedding=st.session_state.embedding_model
        )
        
        st.session_state.document_processed = True
        return True


def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context.
If the answer is not in the context, just say, "Answer is not available in the context." Do not make up an answer.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def check_semantic_cache(user_question, cache_status_container, similarity_threshold=0.80):
    if st.session_state.embedding_model is None or len(st.session_state.question_embeddings) == 0:
        return None
    

    try:
        question_embedding = st.session_state.embedding_model.embed_query(user_question)
        
        # Check for similar questions in cache
        max_similarity = 0
        best_match_index = -1
        
        for i, cached_embedding in enumerate(st.session_state.question_embeddings):
            similarity = cosine_similarity(question_embedding, cached_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_index = i
        

        if max_similarity >= similarity_threshold and best_match_index != -1:
            similar_question =st.session_state.question_answers[best_match_index][0]
            cached_answer =st.session_state.question_answers[best_match_index][1]
            cache_status_container.success(f"[CACHE HIT] Similarity: {max_similarity:.2f} with question: '{similar_question}'")
            return {"cached_question": similar_question, "cached_answer": cached_answer}
        else:
            cache_status_container.warning(f"[CACHE MISS] Best similarity: {max_similarity:.2f} (below threshold of {similarity_threshold})")
            return None
        
    except Exception as e:
        cache_status_container.error(f"Error checking cache: {str(e)}")
        return None

def add_to_semantic_cache(user_question, answer):
    try:
        question_embedding = st.session_state.embedding_model.embed_query(user_question)
        
        st.session_state.question_embeddings.append(question_embedding)
        st.session_state.question_answers.append((user_question, answer))
        return True
    except Exception as e:
        st.error(f"Error adding to cache: {str(e)}")
        return False

# Function to handle user input and generate responses
def process_user_question(user_question, cache_status_container):
    if not st.session_state.document_processed:
        return "Please upload and process documents first before asking questions."
    
    cached_context = check_semantic_cache(user_question, cache_status_container)
    
    try:
        docs = st.session_state.document_index.similarity_search(user_question, k=4)
        
        chain = get_conversational_chain()

        if cached_context:
            enhanced_question = f"{user_question}\n\nSimilar question previously asked: {cached_context['cached_question']}\nPrevious answer: {cached_context['cached_answer']}"
        else:
            enhanced_question = user_question
            
        response = chain(
            {"input_documents": docs, "question": enhanced_question},
            return_only_outputs=True
        )
        
        answer = response["output_text"]
        
        add_to_semantic_cache(user_question, answer)
        
        return answer
        
    except Exception as e:
        return f"An error occurred while processing your question: {str(e)}"

def main():
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if st.button("Submit & Process", key="process_button"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Extracting text from PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error("No text could be extracted from the uploaded PDF files. Please upload valid documents.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("No valid text chunks could be created from the uploaded documents.")
                        else:
                            success = process_documents(text_chunks)
                            if success:
                                st.success("Documents processed successfully! You can now ask questions.")
        
        if st.button("Clear Cache", key="clear_cache"):
            st.session_state.question_embeddings =[]
            st.session_state.question_answers =[]
            st.success("Cache cleared successfully!")
        
        if st.session_state.question_answers:
            with st.expander("View Cached Questions"):
                for q, _ in st.session_state.question_answers:
                    st.write(f"- {q}")
    
    st.header("Chat with your Documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.write(user_question)
        
        cache_status_container =st.empty()
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_user_question(user_question, cache_status_container)
                st.write(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
