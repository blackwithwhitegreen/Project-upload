# QnA




This app uses the google/flan-t5-base model via Hugging Face for text generation and question answering, paired with sentence-transformers/all-MiniLM-L6-v2 for embedding-based semantic search. PDF documents are processed with LangChain's PyPDFLoader and split using RecursiveCharacterTextSplitter. Embeddings are stored in a FAISS vector store for efficient retrieval, and answers are generated using a RetrievalQA chain. Page-level source evidence is rendered using PyMuPDF.
