# app.py
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import tempfile
import os

# Set up the environment
MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def create_vector_store(pdf_path):
    """
    Process PDF and create FAISS vector store
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    """
    Create the Retrieval QA chain
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0,
        repetition_penalty=1.2
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def main():
    st.title("PDF Question Answering System")
    st.write("Upload a PDF and ask questions about its content!")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Process PDF and create vector store
        with st.spinner("Processing PDF..."):
            vectorstore = create_vector_store(tmp_path)
            os.remove(tmp_path)  # Clean up temp file

        # Create QA chain
        qa_chain = create_qa_chain(vectorstore)

        # Store QA chain in session state
        st.session_state['qa_chain'] = qa_chain

        # Question input
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Searching for answer..."):
                result = st.session_state['qa_chain']({"query": question})
                
                st.subheader("Answer:")
                st.write(result["result"])

                # Display source documents
                st.subheader("Source Documents:")
                for doc in result["source_documents"]:
                    st.write(f"Page {doc.metadata['page'] + 1}:")
                    st.write(doc.page_content)
                    st.write("---")

if __name__ == "__main__":
    main()
