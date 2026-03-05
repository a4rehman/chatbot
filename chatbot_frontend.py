import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid
import pandas as pd
from pypdf import PdfReader
from io import BytesIO
from PIL import Image
import base64

# Set page config
st.set_page_config(page_title="Abdul Rehman's AI Assistant", layout="wide", page_icon="🤖")

# **************************************** utility functions *************************

def extract_file_content(uploaded_file):
    """Extracts content from different file types using pypdf (Stable)."""
    try:
        filename = uploaded_file.name
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                return f"\n[File: {filename}] - (Warning: This PDF seems to be an image or scanned document.)\n"
            return f"\n--- Start of File: {filename} ---\n{text}\n--- End of File: {filename} ---\n"
        
        elif uploaded_file.type in ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return f"\n--- Start of Spreadsheet: {filename} ---\n{df.to_string()}\n--- End of Spreadsheet: {filename} ---\n"
        
        elif uploaded_file.type.startswith("image/"):
            image = Image.open(uploaded_file)
            mime_type = uploaded_file.type
            buffered = BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
            image.save(buffered, format=fmt)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"IMAGE_DATA_BASE64|{mime_type}|{img_str}|{filename}"
            
        return ""
    except Exception as e:
        return f"\nError processing file {uploaded_file.name}: {str(e)}\n"

def parse_metadata(content):
    """Helper to clean up markers from response for clean UI display."""
    reasoning = "N/A"
    confidence = "N/A"
    clean_text = content
    
    if "[REASONING]" in content:
        parts = content.split("[REASONING]")
        clean_text = parts[0].replace("[RESPONSE]", "").strip()
        if "[CONFIDENCE]" in parts[1]:
            meta_parts = parts[1].split("[CONFIDENCE]")
            reasoning = meta_parts[0].strip()
            confidence = "".join(filter(str.isdigit, meta_parts[1])).strip()
            
    return clean_text, reasoning, confidence

def generate_user_id():
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())[:8]
    return st.session_state['user_id']

def reset_chat():
    st.session_state['thread_id'] = str(uuid.uuid4())
    st.session_state['message_history'] = []
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    st.session_state['chat_threads'].append(st.session_state['thread_id'])

# **************************************** Session Setup ******************************
user_id = generate_user_id()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    reset_chat()

# **************************************** Sidebar UI *********************************

st.sidebar.markdown(f"### 👤 User ID: **{user_id}**")
st.sidebar.title('Advanced LangGraph Bot')
st.sidebar.info("Engineering Edition by Abdul Rehman")

uploaded_files = st.sidebar.file_uploader(
    "📎 Upload Data (PDF, CSV, XLS, Image)", 
    type=["pdf", "csv", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.sidebar.button('🆕 Start New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.header('📂 Your History')

for t_id in st.session_state['chat_threads'][::-1]:
    label = f"💬 Chat {t_id[:8]}"
    if st.sidebar.button(label, key=t_id, use_container_width=True):
        st.session_state['thread_id'] = t_id
        st.rerun()

st.sidebar.divider()
if st.sidebar.checkbox('🔍 Show Backend Process'):
    st.sidebar.markdown("### 🏗️ Technical Workflow")
    backend_graph = """
    digraph G {
        rankdir=TB;
        node [shape=box, style=filled, color="#E1E1E1", fontname="Verdana", fontsize=10];
        
        user [label="User Query", shape=ellipse, color="#A1D490"];
        files [label="PDF/Excel/Image", shape=note, color="#A1C4FD"];
        extract [label="Data Extraction\\n(pypdf/pandas)"];
        agent [label="LangGraph Agent\\n(Deep Analysis)"];
        retry [label="Confidence Check\\n(<60%? Retry)", color="#FFD0D0"];
        output [label="Final Output\\n+ Reasoning", shape=ellipse, color="#A1D490"];

        user -> agent;
        files -> extract;
        extract -> agent [label="Context"];
        agent -> retry;
        retry -> agent [label="Retry if low"];
        retry -> output [label="Success"];
    }
    """
    st.sidebar.graphviz_chart(backend_graph)

# **************************************** Main UI ************************************

st.title("🤖 Intelligent Engineering Assistant")
st.caption(f"Session: {st.session_state['thread_id'][:8]} | Context-Aware Mode")

# Display message history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if 'reasoning' in message:
            st.caption(f"🎯 **Reasoning:** {message['reasoning']} | **Confidence:** {message['confidence']}%")

user_input = st.chat_input('Analyze files or ask a question...')

if user_input:
    # 1. Process files for the CURRENT request
    text_context = ""
    images_to_send = [] # Store tuple of (mime_type, base64, filename)
    
    if uploaded_files:
        with st.spinner("Extracting data from attachments..."):
            for f in uploaded_files:
                result = extract_file_content(f)
                if result.startswith("IMAGE_DATA_BASE64|"):
                    _, m_type, b64, fname = result.split("|")
                    images_to_send.append((m_type, b64, fname))
                else:
                    text_context += result

    # 2. Update UI with User message (Show only user text, hide the raw context)
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 3. Build the LangGraph input (Multimodal + Context)
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    
    # We use a special instruction to the model about the context
    context_instruction = f"\n\n[ATTACHED_FILES_CONTEXT]\n{text_context}\n" if text_context else ""
    final_query = user_input + context_instruction

    if images_to_send:
        msg_content = [{"type": "text", "text": final_query}]
        for m_type, b64, fname in images_to_send:
            msg_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:{m_type};base64,{b64}"}
            })
    else:
        msg_content = final_query

    # 4. Get AI Response
    with st.chat_message('assistant'):
        full_response = ""
        placeholder = st.empty()
        
        with st.spinner("Deep Analysis..."):
            for chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=msg_content)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    full_response += chunk.content
                    display_text = full_response.split("[REASONING]")[0].replace("[RESPONSE]", "").strip()
                    placeholder.markdown(display_text)

        # Final Metadata Parsing
        clean_ans, reason, conf = parse_metadata(full_response)
        placeholder.markdown(clean_ans)
        st.caption(f"🎯 **Reasoning:** {reason} | **Confidence:** {conf}%")

    # 5. Save to history
    st.session_state['message_history'].append({
        'role': 'assistant', 
        'content': clean_ans, 
        'reasoning': reason, 
        'confidence': conf
    })
