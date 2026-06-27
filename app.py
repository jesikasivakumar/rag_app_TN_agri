import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="TN Welfare Schemes Advanced Assistant", 
    page_icon="🏛️", 
    layout="wide"  # Wide layout for an advanced dashboard feel
)

# Custom CSS styling (Forced dark charcoal text #1e293b for perfect readability in Dark Mode)
st.markdown("""
    <style>
    .metric-card {
        background-color: #f1f5f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0f766e;
        margin-bottom: 12px;
        color: #1e293b; /* Explicit dark text color */
    }
    .metric-card strong {
        color: #0f766e; /* Dark teal accent for titles */
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Component ---
with st.sidebar:
    st.markdown("## 🏛️ TN GOVT")
    st.title("Control Panel")
    st.markdown("---")
    st.markdown("### 📋 App Metrics")
    
    # Visual layout indicators with forced high-contrast text
    st.markdown('<div class="metric-card"><strong>Knowledge Base</strong><br>TN Welfare Schemes</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><strong>Vector Store</strong><br>FAISS Local Index</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><strong>Model Engine</strong><br>gpt-4o-mini</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    # Quick utility to wipe out chat memory
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Core Layout Split ---
st.title("🏛️ Tamil Nadu Government Welfare Schemes AI")
st.caption("An advanced Retrieval-Augmented Generation (RAG) assistant connected directly to the Agriculture & Farmers Welfare Department datasets.")
st.markdown("---")

# --- Initialize RAG Core (Cached) ---
@st.cache_resource
def initialize_rag():
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
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant specialized in Tamil Nadu Welfare Schemes.\n"
                   "Answer the user's question strictly using the provided context below. "
                   "If you do not know the answer, say that you cannot find it.\n\n"
                   "Context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}")
    ])
    
    return retriever, llm, prompt

try:
    retriever, llm, prompt = initialize_rag()
except Exception as e:
    st.error(f"Failed to connect backend: {e}")
    st.stop()

# --- Initialize Session Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🏛️"):
            st.write(msg.content)

# --- Handle New User Interaction ---
if user_query := st.chat_input("Ask about schemes (e.g., credit guarantees, micro nutrient spray)..."):
    
    # 1. Print User Query
    with st.chat_message("user", avatar="👤"):
        st.write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))
    
    # 2. Get AI Response
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Analyzing document indices & thinking..."):
            
            # Fetch context chunks
            docs = retriever.invoke(user_query)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # Generate Answer
            chain = prompt | llm
            response = chain.invoke({
                "context": context_text,
                "history": st.session_state.messages[:-1],
                "question": user_query
            })
            
            # Render beautifully formatted response
            st.markdown(response.content)
            
            # Collapsible Source Citations window
            with st.expander("🔍 View Retrieved Sources (Ground Truth Context)"):
                for idx, doc in enumerate(docs):
                    st.markdown(f"**Source Chunk #{idx+1}:**")
                    st.caption(doc.page_content)
                    st.markdown("---")
            
    # Save Response to history
    st.session_state.messages.append(AIMessage(content=response.content))