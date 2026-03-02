import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid

# Set page config
st.set_page_config(page_title="Abdul Rehman's AI Assistant", layout="wide")

# **************************************** utility functions *************************

def generate_user_id():
    """Generates a unique persistent ID for a user session."""
    if 'user_id' not in st.session_state:
        # Check if they have a cookie/localstorage would be better, 
        # but for this demo session_state works.
        st.session_state['user_id'] = str(uuid.uuid4())[:8]
    return st.session_state['user_id']

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    config = {'configurable': {'thread_id': thread_id}}
    state = chatbot.get_state(config)
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************
user_id = generate_user_id()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    # Filter threads or handle them locally so users only see their own
    # In a real app, we'd query by metadata. 
    # For now, we store them in session_state to ensure individual experience.
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    reset_chat()

# **************************************** Sidebar UI *********************************

st.sidebar.markdown(f"### 👤 User ID: **{user_id}**")
st.sidebar.title('LangGraph Assistant')
st.sidebar.info("Built by AI Engineer Abdul Rehman")

if st.sidebar.button('🆕 Start New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.header('📂 Your History')

# Only show threads started in this session for privacy
for t_id in st.session_state['chat_threads'][::-1]:
    # Label threads by their first 8 chars for cleaner UI
    label = f"💬 Chat {t_id[:8]}"
    if st.sidebar.button(label, key=t_id, use_container_width=True):
        st.session_state['thread_id'] = t_id
        messages = load_conversation(t_id)
        
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            else:
                continue # Skip system/tool messages in UI
            
            if msg.content: # Only add if has text
                temp_messages.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_messages
        st.rerun()

# **************************************** Main UI ************************************

st.title("🤖 Intelligent Assistant")
st.caption(f"Connected to conversation: {st.session_state['thread_id'][:8]}")

# Display message history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('What can I help you with today?')

if user_input:
    # 1. Update UI with User message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 2. Config for LangGraph (Individual thread per user)
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # 3. Get AI Response
    with st.chat_message('assistant'):
        full_response = ""
        placeholder = st.empty()
        
        # We use st.spinner for web search status
        with st.spinner("Thinking & Searching..."):
            # Stream the response
            # Note: We filter for content to avoid showing raw tool calls in UI
            for chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response)

    # 4. Save to history
    if full_response:
        st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})