from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def search_faiss_db(query):
    print(f"\n--- Searching DB for: '{query}' ---")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Explicitly look for the index file named 'index'
    vector_db = FAISS.load_local(
        folder_path="faiss_index", 
        embeddings=embedding_model, 
        index_name="index",
        allow_dangerous_deserialization=True
    )
    
    results = vector_db.similarity_search(query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\n[Match #{i+1}]")
        print(doc.page_content)

if __name__ == "__main__":
    search_faiss_db("Tell me about training schemes and credit guarantee")