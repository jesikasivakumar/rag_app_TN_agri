import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# 1. Dictionary to hold chat sessions (keys will be session IDs)
chat_histories = {}

def get_session_history(session_id: str):
    """Retrieves or creates a chat history tracker for a specific session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

def setup_rag_chain():
    # 2. Setup Embedding and Vector DB
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_db = FAISS.load_local(
        folder_path="faiss_index", 
        embeddings=embedding_model, 
        index_name="index",
        allow_dangerous_deserialization=True
    )
    
    # 3. Create a Retriever out of our database
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    # 4. Contextual Prompt containing a MessagesPlaceholder for history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant specialized in Tamil Nadu Welfare Schemes.\n"
                   "Answer the user's question strictly using the provided context below. "
                   "If you do not know the answer, say that you cannot find it.\n\n"
                   "Context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),  # This injects the chat history dynamically
        ("user", "{question}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5. Build a custom chain that pulls context automatically before sending to LLM
    def contextualized_chain(inputs):
        # Retrieve context text based on the user's incoming question
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join([d.page_content for d in docs])
        # Pass everything into the prompt and run through the LLM
        return prompt | llm

    # 6. Wrap our chain with message history tracking capabilities
    # It expects to inject the history into the "history" variable we created in the prompt
    base_chain = (lambda inputs: {"context": "\n\n".join([d.page_content for d in retriever.invoke(inputs["question"])]), "question": inputs["question"], "history": inputs.get("history", [])}) | prompt | llm
    
    conversational_rag_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    return conversational_rag_chain

if __name__ == "__main__":
    print("Initializing conversational RAG engine...")
    chat_chain = setup_rag_chain()
    
    # Define a static configuration session ID
    config = {"configurable": {"session_id": "tn_session_1"}}
    
    print("\n--- Chat Started! Type 'exit' or 'quit' to end. ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Invoke the chain; it handles context retrieval and updates memory automatically
        response = chat_chain.invoke({"question": user_input}, config=config)
        print(f"\nAI: {response.content}")