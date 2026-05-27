import streamlit as st
from groq import Groq
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def search_chunks(query, chunks, embeddings, n=3):
    query_emb = embedder.encode([query])
    scores = np.dot(embeddings, query_emb.T).flatten()
    top_indices = scores.argsort()[-n:][::-1]
    return [chunks[i] for i in top_indices]

st.set_page_config(page_title="Local AI Assistant", page_icon="🤖")
st.title("🤖 My Local AI Chatbot")
st.caption("Powered by Groq — Fast & Free")

with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")
    model_options = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    selected_model = st.selectbox("Choose a model", model_options)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.")
    st.divider()
    st.header("📄 Upload a Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and "chunks" not in st.session_state:
        with st.spinner("Indexing PDF..."):
            text = load_pdf(uploaded_file)
            st.session_state.chunks = chunk_text(text)
            st.session_state.embeddings = embedder.encode(st.session_state.chunks)
        st.success(f"✅ Indexed {len(st.session_state.chunks)} chunks!")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if "chunks" in st.session_state:
            del st.session_state.chunks
            del st.session_state.embeddings
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything..."):
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar!")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        groq_client = Groq(api_key=groq_api_key)

        if "chunks" in st.session_state:
            relevant = search_chunks(user_input, st.session_state.chunks, st.session_state.embeddings)
            context = "\n\n".join(relevant)
            rag_prompt = f"Use this context to answer:\n\n{context}\n\nQuestion: {user_input}"
            messages_to_send = [{"role": "system", "content": system_prompt}] + \
                               st.session_state.messages[:-1] + \
                               [{"role": "user", "content": rag_prompt}]
        else:
            messages_to_send = [{"role": "system", "content": system_prompt}] + \
                               st.session_state.messages

        stream = groq_client.chat.completions.create(
            model=selected_model,
            messages=messages_to_send,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
