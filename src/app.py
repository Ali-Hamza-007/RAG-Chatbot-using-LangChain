import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Document Chatbot")
st.markdown("Upload documents (PDF, TXT) and ask questions. Answers are fully traceable to the source.")

# Sidebar for file upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"Successfully processed {response.json().get('chunks')} chunks.")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend API is not running. Please start FastAPI.")
        else:
            st.warning("Please upload a file first.")
            
# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("Source Metadata & JSON"):
                st.json(message["metadata"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    
                    answer = data.get("answer", "No answer provided.")
                    confidence = data.get("confidence", "low")
                    sources = data.get("sources", [])
                    
                    # Construct markdown response
                    st.markdown(answer)
                    
                    if confidence == "low" and not sources:
                        st.warning("⚠️ Low confidence. Answer not found in documents.")
                    
                    # Render expandable source metadata
                    with st.expander("View Source Metadata & Raw JSON"):
                        st.json(data)
                        
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": data
                    })
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Backend API is not running. Please start FastAPI.")
