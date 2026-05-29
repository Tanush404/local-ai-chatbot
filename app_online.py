import streamlit as st
from groq import Groq
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
import uuid

# ---- SETUP ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# ---- SUPABASE FUNCTIONS ----
def save_message(user_id, role, content):
    supabase.table("messages").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()

def load_messages(user_id):
    result = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id").execute()
    return [{"role": r["role"], "content": r["content"]} for r in result.data]

def clear_messages(user_id):
    supabase.table("messages").delete().eq("user_id", user_id).execute()

# ---- PDF UTILS ----
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

# ---- UI ----
st.set_page_config(page_title="Local AI Assistant", page_icon="🤖")
st.title("🤖 My Local AI Chatbot")
st.caption("Powered by Groq — Fast & Free")

# User ID from URL
params = st.query_params
if "user_id" not in params:
    new_id = str(uuid.uuid4())
    st.query_params["user_id"] = new_id
    st.session_state.user_id = new_id
else:
    st.session_state.user_id = params["user_id"]

if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.user_id)

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = {}

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Your ID: `{st.session_state.user_id[:8]}...`")

    model_options = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    selected_model = st.selectbox("Choose a model", model_options)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.")

    st.divider()
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.pdf_store:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    text = load_pdf(uploaded_file)
                    chunks = chunk_text(text)
                    embeddings = embedder.encode(chunks)
                    st.session_state.pdf_store[uploaded_file.name] = {
                        "chunks": chunks,
                        "embeddings": embeddings
                    }
                st.success(f"✅ {uploaded_file.name} indexed!")

    if st.session_state.pdf_store:
        st.subheader("Active Documents")
        selected_pdf = st.selectbox("Search in:", list(st.session_state.pdf_store.keys()))
    else:
        selected_pdf = None

    st.divider()
    st.header("💾 Export Chat")
    if st.session_state.messages:
        chat_text = "\n\n".join([f"{'You' if m['role'] == 'user' else 'AI'}:\n{m['content']}" for m in st.session_state.messages])
        st.download_button("⬇️ Download as .txt", data=chat_text, file_name="chat_export.txt", mime="text/plain")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        clear_messages(st.session_state.user_id)
        st.session_state.messages = []
        st.rerun()

# ---- CHAT ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(st.session_state.user_id, "user", user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        if selected_pdf and selected_pdf in st.session_state.pdf_store:
            store = st.session_state.pdf_store[selected_pdf]
            relevant = search_chunks(user_input, store["chunks"], store["embeddings"])
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
    save_message(st.session_state.user_id, "assistant", full_response)
