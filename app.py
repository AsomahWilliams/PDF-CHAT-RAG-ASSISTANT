# ----------------------------
# PRODUCTION-READY APP.PY
# ----------------------------
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
import os
from typing import List

# Load env
load_dotenv()

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def get_pdf_text(pdfs) -> str:
    text = ""
    for pdf in pdfs:
        with pdfplumber.open(pdf) as plumber:
            for page in plumber.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text

# ----------------------------
# CHUNKING (Better: Recursive + Overlap)
# ----------------------------
def get_text_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

# ----------------------------
# VECTOR STORE WITH METADATA
# ----------------------------
def create_vector_store(text_chunks, embeddings):
    return FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=[{"source": "pdf"} for _ in text_chunks]
    )

# ----------------------------
# LLM SETUP
# ----------------------------
def get_llm():
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = openrouter_key or openai_key

    if not api_key:
        raise ValueError(
            "Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file."
        )

    # OpenRouter uses a custom base URL and model namespace.
    if openrouter_key:
        return ChatOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
        )

    return ChatOpenAI(
        api_key=openai_key,
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
    )

# ----------------------------
# QA CHAIN WITH CHAT HISTORY (PRODUCTION)
# ----------------------------
def create_qa_chain(vector_store):
    llm = get_llm()
    
    # Get more relevant docs
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )
    
    def get_response(inputs):
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        
        # Get relevant documents
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs[:5]])
        
        # Format chat history as conversation
        history_str = "\n".join([
            f"Human: {msg['content']}\nAssistant: {msg['response']}"
            for msg in chat_history[-5:]  # Last 5 messages
        ])
        
        # Create prompt with history
        if history_str:
            prompt = f"""You are a helpful assistant answering questions about the uploaded PDF document.
Use the context from the document and previous conversation to answer the question.

Previous Conversation:
{history_str}

Document Context:
{context}

Current Question: {question}

Answer:"""
        else:
            prompt = f"""You are a helpful assistant answering questions about the uploaded PDF document.
Use only the document context below to answer the question.

Document Context:
{context}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        return {
            "result": response.content,
            "source_docs": docs,
            "used_context": bool(docs)
        }
    
    return RunnableLambda(get_response)

# ----------------------------
# MAIN APP (PRODUCTION)
# ----------------------------
def main():
    st.set_page_config(
        page_title="PDF Chat RAG",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stChatMessage {padding: 10px;}
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📚 Intelligent PDF Reader")
    st.caption("Upload PDFs → Chat with AI")

    # Initialize session state properly
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar
    with st.sidebar:
        st.subheader("📄 Upload PDFs")
        pdfs = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents to chat with"
        )
        st.session_state.files = pdfs

        # Show chat history count
        if st.session_state.chat_history:
            st.divider()
            st.caption(f"💬 {len(st.session_state.chat_history)} messages in memory")
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.divider()
        
        if st.button("⚡ Process PDFs", type="primary"):
            if not pdfs:
                st.warning("Please upload PDFs first!")
            else:
                with st.spinner("📖 Reading PDFs..."):
                    # Extract text
                    raw_text = get_pdf_text(pdfs)
                    
                    if not raw_text.strip():
                        st.error("Could not read PDF. Is it scanned?")
                    else:
                        # Create embeddings
                        with st.spinner("🔢 Creating embeddings..."):
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={'device': 'cpu'}
                            )
                            
                            chunks = get_text_chunks(raw_text)
                            vector_store = create_vector_store(chunks, embeddings)
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = create_qa_chain(vector_store)
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ Ready! {len(chunks)} chunks")

    # Main chat area
    # Show welcome message
    if not st.session_state.chat_history:
        st.info("👆 Upload a PDF and click **Process PDFs** to start!")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["content"])
        with st.chat_message("assistant"):
            st.markdown(msg["response"])

    # Chat input
    if prompt := st.chat_input("Ask about your PDF..."):
        # Handle question
        if not st.session_state.qa_chain:
            st.error("Please process PDFs first!")
        else:
            # Add user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.qa_chain.invoke({
                        "question": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    result = response["result"]
                    
                    # Show sources
                    if response.get("source_docs"):
                        with st.expander("📚 Sources"):
                            for doc in response["source_docs"][:2]:
                                st.caption(doc.page_content[:300] + "...")
                    
                    st.markdown(result)
            
            # Save to history
            st.session_state.chat_history.append({
                "content": prompt,
                "response": result
            })

if __name__ == "__main__":
    main()
