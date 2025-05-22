import asyncio
import tempfile
import os
import fitz  # PyMuPDF
import io
import streamlit as st
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# Fix for event loop issues
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def initialize_general_model():
    """Initialize the model for general knowledge questions"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0,
        repetition_penalty=1.2
    )

def create_vector_store(pdf_path):
    """Process PDF and create FAISS vector store"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(texts, embeddings)

def create_qa_chain(vectorstore):
    """Create the Retrieval QA chain for PDF content"""
    pipe = initialize_general_model()
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def render_pdf_page(pdf_bytes, page_number):
    """Render specific PDF page as image"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes()
    return Image.open(io.BytesIO(img_bytes))

def main():
    st.title("VectorAsk")
    st.write("Get answers with source page images!")
    
    # Initialize session states
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None

    mode = st.radio("Select answer source:",
                    ("PDF Content", "Text input"),
                    horizontal=True)

    if mode == "PDF Content":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file is not None:
            st.session_state.pdf_bytes = uploaded_file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(st.session_state.pdf_bytes)
                tmp_path = tmp_file.name

            with st.spinner("Processing PDF..."):
                vectorstore = create_vector_store(tmp_path)
                os.remove(tmp_path)
                st.session_state['qa_chain'] = create_qa_chain(vectorstore)

    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner("Generating answer..."):
            if mode == "General Knowledge":
                if 'general_pipe' not in st.session_state:
                    st.session_state.general_pipe = initialize_general_model()
                
                result = st.session_state.general_pipe(
                    question,
                    max_length=256,
                    temperature=0
                )[0]['generated_text']
                
                st.subheader("Answer:")
                st.write(result)
                st.info("This answer is generated from the model's general knowledge")
                
            elif mode == "PDF Content":
                if 'qa_chain' not in st.session_state:
                    st.warning("Please upload a PDF file first!")
                    return
                
                result = st.session_state['qa_chain']({"query": question})
                
                # Display answer
                st.subheader("Answer:")
                st.write(result["result"])
                
                # Display source documents with images
                st.subheader("Source Evidence:")
                for doc in result["source_documents"]:
                    page_num = doc.metadata['page']
                    
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        try:
                            img = render_pdf_page(st.session_state.pdf_bytes, page_num)
                            st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error rendering page: {str(e)}")
                    
                    with col2:
                        st.write(f"**Page {page_num + 1} Content:**")
                        st.write(doc.page_content)
                    
                    st.write("---")

if __name__ == "__main__":
    main()
