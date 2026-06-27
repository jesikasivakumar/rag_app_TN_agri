import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables from the .env file
load_dotenv()

def generate_rag_answer(user_question):
    print(f"\nUser Question: '{user_question}'")
    
    # 2. Setup the embedding model (Must match what we used to build the database)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 3. Load the FAISS vector database
    vector_db = FAISS.load_local(
        folder_path="faiss_index", 
        embeddings=embedding_model, 
        index_name="index",
        allow_dangerous_deserialization=True
    )
    
    # 4. Search for the top 2 matching chunks
    print("Retrieving context from local FAISS database...")
    retrieved_docs = vector_db.similarity_search(user_question, k=2)
    
    # Combine the content of the retrieved chunks into one block of text
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 5. Define the Prompt Template (Giving instructions to the LLM)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant specialized in Tamil Nadu Welfare Schemes.\n"
                   "Answer the user's question strictly using the provided context below. "
                   "If you do not know the answer based on the context, say that you cannot find it.\n\n"
                   "Context:\n{context}"),
        ("user", "{question}")
    ])
    
    # 6. Initialize the OpenAI Language Model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 7. Create the RAG chain and execute it
    # Format the prompt with our context and question, then send it to the LLM
    chain = prompt_template | llm
    
    print("Generating answer from OpenAI...")
    response = chain.invoke({"context": context_text, "question": user_question})
    
    print("\n--- AI Answer ---")
    print(response.content)

if __name__ == "__main__":
    # Test your completed RAG pipeline!
    generate_rag_answer("What training schemes and credit guarantees are available for farmers?")