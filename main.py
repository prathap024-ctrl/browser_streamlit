import streamlit as st
import uuid
from services.search import search_and_summarize
from services.search_embeddings import search_embeddings
import os
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
)
from lib.db import DBNAME, HOST, PASSWORD, PORT, USER, connection_string
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Multi-Session AI Chat", layout="wide")

st.title("AI Browser")

# Initialize session state
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  

if "current_session" not in st.session_state:
    new_id = str(uuid.uuid4())[:8]  
    st.session_state.current_session = new_id
    st.session_state.sessions[new_id] = []

# Sidebar session manager
st.sidebar.header("Chat History")

# Start new session
if st.sidebar.button("New Session"):
    new_id = str(uuid.uuid4())[:8]
    st.session_state.current_session = new_id
    st.session_state.sessions[new_id] = []

st.sidebar.markdown(f"**Current:** `{st.session_state.current_session}`")

# Show all session IDs
for session_id in st.session_state.sessions.keys():
    if st.sidebar.button(f"Load Session: {session_id}"):
        st.session_state.current_session = session_id

# Load messages from current session
session_id = st.session_state.current_session
messages = st.session_state.sessions[session_id]

# Chat UI
user_input = st.chat_input("Type your message...")
with st.spinner("🔎 Searching..."):
    if user_input:
        # Save user message
        messages.append({"role": "user", "text": user_input})
        
        #Search query
        #ai_summary, source_url = search_and_summarize(user_input)
        #formatted_ai_response = f"{ai_summary}\n\n🔗 Source: {source_url}"
        
        response = search_embeddings(user_input)
        if response == "I don't know the answer!":
            ai_summary, source_url = search_and_summarize(user_input)
            formatted_ai_response = f"{ai_summary}\n\n🔗 Source: {source_url}"
        else:
            formatted_ai_response = response

        history = PostgresChatMessageHistory(
            session_id=session_id,
            connection_string=connection_string
        )
        history.add_user_message(user_input)
        history.add_ai_message(formatted_ai_response)

        messages.append({"role": "ai", "text": formatted_ai_response})
    

# Display chat
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
