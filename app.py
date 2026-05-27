import streamlit as st
import ollama
import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer

# ---- SETUP ----
client = chromadb.Client()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create or get a collection to store document chunks
collection = client.get_or_create_collection("docs")

def load_pdf(file):
    """Extract text from uploaded PDF"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def add_to_db(chunks):
    """Embed chunks and store in ChromaDB"""
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

def search_db(query, n=3):
    """Find the most relevant chunks for a query"""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n)
    return results["documents"][0]

# ---- UI ----
st.set_page_config(page_title="Local AI Assistant", page_icon="🤖")
st.title("🤖 My Local AI Chatbot")
st.caption("Powered by Ollama — 100% Private & Local")

with st.sidebar:
    st.header("⚙️ Settings")

    models = [m.model for m in ollama.list().models]
    selected_model = st.selectbox("Choose a model", models)

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
    )

    st.divider()
    st.header("📄 Upload a Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Reading and indexing PDF..."):
            text = load_pdf(uploaded_file)
            chunks = chunk_text(text)
            add_to_db(chunks)
        st.success(f"✅ Indexed {len(chunks)} chunks!")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---- CHAT ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # If a PDF was uploaded, find relevant chunks and add to prompt
        if uploaded_file:
            relevant_chunks = search_db(user_input)
            context = "\n\n".join(relevant_chunks)
            rag_prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {user_input}"""
            messages_to_send = [{"role": "system", "content": system_prompt}] + \
                               st.session_state.messages[:-1] + \
                               [{"role": "user", "content": rag_prompt}]
        else:
            messages_to_send = [{"role": "system", "content": system_prompt}] + \
                               st.session_state.messages

        for chunk in ollama.chat(model=selected_model, messages=messages_to_send, stream=True):
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})