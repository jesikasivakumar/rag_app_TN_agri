import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_complete_pipeline():
    # 1. Load website data
    print("--- Step 2: Loading Document ---")
    url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} webpage document(s).")
    
    # 2. Split into chunks
    print("\n--- Step 3: Splitting Documents ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split text into {len(chunks)} chunks.")
    
    # 3. Create Embeddings and Save FAISS
    print("\n--- Step 4 & 5: Creating Embeddings and FAISS DB ---")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Building FAISS index and saving locally...")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    
    # Explicitly saving using index_name="index"
    vector_db.save_local(folder_path="faiss_index", index_name="index")
    print("FAISS DB successfully created and files written to 'faiss_index/' folder!")

if __name__ == "__main__":
    run_complete_pipeline()