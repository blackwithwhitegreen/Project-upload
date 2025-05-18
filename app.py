# app.py
import streamlit as st
import fitz  # PyMuPDF
from tqdm.auto import tqdm
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import textwrap
from secret_api_keys import HUGGINGFACE_TOKEN  # Import from separate file

# Configuration
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
LLM_MODEL_NAME = "google/gemma-2b-it"
NUM_CONTEXT_CHUNKS = 5
SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30

# Initialize session state
if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm_model():
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
    return tokenizer, model

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = re.sub(r'\.([A-Z])', r'. \1', cleaned_text)
    return cleaned_text

def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages_and_texts = []
    
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number,
            "text": text
        })
    
    # Split into sentence chunks
    nlp = English()
    nlp.add_pipe("sentencizer")
    
    for item in pages_and_texts:
        doc = nlp(item["text"])
        item["sentences"] = [str(sent) for sent in doc.sents]
        item["sentence_chunks"] = [item["sentences"][i:i+SENTENCE_CHUNK_SIZE] 
                                  for i in range(0, len(item["sentences"]), SENTENCE_CHUNK_SIZE)]
    
    # Create final chunks
    pages_and_chunks = []
    for item in pages_and_texts:
        for chunk in item["sentence_chunks"]:
            chunk_text = " ".join(chunk).replace("  ", " ").strip()
            pages_and_chunks.append({
                "page_number": item["page_number"],
                "sentence_chunk": chunk_text
            })
    
    return pages_and_chunks

def get_embeddings(chunks, embedding_model):
    texts = [chunk["sentence_chunk"] for chunk in chunks]
    return torch.tensor(embedding_model.encode(texts))

def retrieve_relevant_chunks(query, embeddings, chunks, k=5):
    query_embedding = torch.tensor(embedding_model.encode(query))
    scores = util.dot_score(query_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(scores, k=k)
    return [chunks[i] for i in top_indices]

def format_prompt(query, context_chunks):
    context = "\n".join([f"Page {chunk['page_number']+1}: {chunk['sentence_chunk']}" 
                        for chunk in context_chunks])
    return f"""Answer the query based on the following context:
{context}

Query: {query}
Answer:"""

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main app
st.title("PDF Q&A with RAG")

# File upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
embedding_model = load_embedding_model()
tokenizer, llm_model = load_llm_model()

if uploaded_file and not st.session_state.processed_pdf:
    with st.spinner("Processing PDF..."):
        chunks = process_pdf(uploaded_file)
        embeddings = get_embeddings(chunks, embedding_model)
        st.session_state.processed_pdf = {
            "chunks": chunks,
            "embeddings": embeddings
        }

# Query input
query = st.text_input("Enter your question:")
if query and st.session_state.processed_pdf:
    chunks = st.session_state.processed_pdf["chunks"]
    embeddings = st.session_state.processed_pdf["embeddings"]
    
    # Retrieve relevant chunks
    context_chunks = retrieve_relevant_chunks(
        query, embeddings, chunks, NUM_CONTEXT_CHUNKS
    )
    
    # Generate answer
    prompt = format_prompt(query, context_chunks)
    answer = generate_answer(prompt, tokenizer, llm_model)
    
    # Display results
    st.subheader("Answer")
    st.write(answer)
    
    st.subheader("Relevant Context")
    for chunk in context_chunks:
        st.write(f"Page {chunk['page_number']+1}: {chunk['sentence_chunk']}")
