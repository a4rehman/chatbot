import streamlit as st
import os
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import collect_runs
from langsmith import Client as LangSmithClient, tracing_context
import uuid
import pandas as pd
from pypdf import PdfReader
from io import BytesIO
from PIL import Image
import base64
from openai import RateLimitError

# Set page config
st.set_page_config(page_title="100Solutionz AI Assistant", layout="wide", page_icon="🤖")

# Custom CSS for ChatGPT/Copilot Style UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp { 
        background: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.12), transparent 45%), #090d16;
        color: #f1f5f9;
    }
    
    [data-testid="stSidebar"] { 
        background: #0d1527; 
        border-right: 1px solid rgba(255, 255, 255, 0.08); 
    }
    
    .centered-header {
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .brand-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        border: 1px solid rgba(96, 165, 250, 0.3);
        margin-bottom: 1rem;
    }
    
    .centered-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .centered-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Streamlit Chat elements */
    [data-testid="stChatMessage"] { 
        border: 1px solid rgba(255, 255, 255, 0.07); 
        border-radius: 16px; 
        padding: 0.8rem 1.1rem; 
        margin-bottom: 1rem; 
        background: rgba(15, 23, 42, 0.55); 
    }
    
    [data-testid="stChatInput"] { 
        border-radius: 24px; 
        border: 1px solid rgba(255, 255, 255, 0.15); 
        background: rgba(15, 23, 42, 0.85); 
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35); 
        max-width: 820px;
        margin: 0 auto;
    }
    
    [data-testid="stChatInput"]:focus-within { 
        border-color: #3b82f6; 
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2); 
    }
    
    .stButton > button { 
        border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.12); 
        background: rgba(30, 41, 59, 0.6); 
        color: #e2e8f0; 
        transition: 0.2s ease; 
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover { 
        border-color: #3b82f6; 
        color: #ffffff; 
        background: rgba(59, 130, 246, 0.25); 
    }
</style>
""", unsafe_allow_html=True)

langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "solutionz-chatbot"
langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"

def is_true(value, default=True):
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

hide_trace_data = is_true(os.getenv("LANGSMITH_HIDE_INPUTS"), default=True)
hide_trace_metadata = is_true(os.getenv("LANGSMITH_HIDE_METADATA"), default=True)

ls_client = None
if langsmith_api_key:
    try:
        client_options = {
            "api_key": langsmith_api_key,
            "api_url": langsmith_endpoint,
        }
        if hide_trace_data:
            client_options["hide_inputs"] = lambda _: {}
            client_options["hide_outputs"] = lambda _: {}
        if hide_trace_metadata:
            client_options["hide_metadata"] = True
        ls_client = LangSmithClient(**client_options)
    except Exception as err:
        print(f"LangSmith client initialization failed: {type(err).__name__}")

# Upload limits protect the app from oversized or malformed user-supplied files.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB limit
MAX_PDF_PAGES = 50
MAX_SPREADSHEET_ROWS = 1_000
MAX_CONTEXT_CHARS = 30_000
Image.MAX_IMAGE_PIXELS = 20_000_000

# **************************************** Utility Functions *************************

def extract_file_content(uploaded_file):
    """Extracts bounded text/image content from an allowed uploaded file."""
    try:
        filename = uploaded_file.name
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            return f"\n[File: {filename}] was skipped because it exceeds the 30 MB limit.\n"

        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            if len(reader.pages) > MAX_PDF_PAGES:
                return f"\n[File: {filename}] was skipped because it exceeds the {MAX_PDF_PAGES}-page safety limit.\n"
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                if len(text) >= MAX_CONTEXT_CHARS:
                    break
            if not text.strip():
                return f"\n[File: {filename}] could not provide readable text.\n"
            return f"\n--- Start of File: {filename} ---\n{text[:MAX_CONTEXT_CHARS]}\n--- End of File: {filename} ---\n"

        if uploaded_file.type in ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file, nrows=MAX_SPREADSHEET_ROWS)
            else:
                df = pd.read_excel(uploaded_file, nrows=MAX_SPREADSHEET_ROWS)
            table_text = df.to_string(index=False)[:MAX_CONTEXT_CHARS]
            return f"\n--- Start of Spreadsheet: {filename} ---\n{table_text}\n--- End of Spreadsheet: {filename} ---\n"

        if uploaded_file.type.startswith("image/"):
            image = Image.open(uploaded_file)
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                return f"\n[File: {filename}] was skipped because its image dimensions exceed limits.\n"
            image.load()
            mime_type = uploaded_file.type
            buffered = BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
            image.save(buffered, format=fmt)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"IMAGE_DATA_BASE64|{mime_type}|{img_str}|{filename}"

        return f"\n[File: {filename}] was skipped because its type is not allowed.\n"
    except Exception:
        return f"\n[File: {uploaded_file.name}] could not be processed safely.\n"

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

def submit_feedback(run_id, score, msg_idx):
    """Sends thumbs up/down user feedback to LangSmith."""
    if not ls_client or not run_id:
        st.info("Feedback captured locally.")
        return
    try:
        ls_client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment="Submitted via Solutionz Chatbot UI"
        )
        st.toast("Feedback sent to LangSmith! 🎯", icon="✅")
    except Exception as e:
        st.warning(f"Could not submit feedback: {e}")

# **************************************** Session Setup ******************************
user_id = generate_user_id()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    reset_chat()

# **************************************** Sidebar UI *********************************

st.sidebar.markdown(f"""
<div style="padding: 1rem; border-radius: 14px; background: rgba(30,41,59,0.5); border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
  <div style="color: #60a5fa; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">100Solutionz AI</div>
  <div style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-top: 0.2rem;">Workspace</div>
  <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;">Session ID: {user_id}</div>
</div>
""", unsafe_allow_html=True)

if ls_client:
    st.sidebar.success(f"📊 LangSmith Tracing Active\n\nProject: **{langsmith_project}**")
else:
    st.sidebar.warning("📊 LangSmith Tracing Inactive")

uploaded_files = st.sidebar.file_uploader(
    "📎 Add photos & files", 
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
if st.sidebar.checkbox('🔍 Show Technical Graph'):
    st.sidebar.markdown("### 🏗️ Backend Graph")
    backend_graph = """
    digraph G {
        rankdir=TB;
        node [shape=box, style=filled, color="#1e293b", fontcolor="#ffffff", fontname="Inter", fontsize=10];
        user [label="User Query", shape=ellipse, color="#3b82f6"];
        extract [label="File Extraction (pypdf/pandas)"];
        agent [label="LangGraph Agent"];
        output [label="Response + Reasoning", shape=ellipse, color="#10b981"];
        user -> agent;
        extract -> agent;
        agent -> output;
    }
    """
    st.sidebar.graphviz_chart(backend_graph)

# **************************************** Main UI ************************************

# Centered Header
st.markdown("""
<div class="centered-header">
  <div class="brand-tag">100SOLUTIONZ AI STUDIO</div>
  <h1>What’s on your mind today?</h1>
  <p>Ask about custom AI software, web & mobile apps, company portfolio—or analyze an attached document.</p>
</div>
""", unsafe_allow_html=True)

# Interactive Action Menu Cards (Only show when starting chat or top level)
if not st.session_state['message_history']:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📎  Add photos & files — Upload from computer", use_container_width=True):
            st.info("👈 Upload your files (PDF, CSV, XLS, Images) using the sidebar uploader.")
        if st.button("🌐  Web search — Find real-time news and info", use_container_width=True):
            st.session_state['pending_prompt'] = "Perform a web search to check current software development trends."
            st.rerun()
    with col2:
        if st.button("📣  Deep research — Get a detailed report", use_container_width=True):
            st.session_state['pending_prompt'] = "Provide a comprehensive report on 100Solutionz core services, tech stack, and portfolio."
            st.rerun()
        if st.button("🖼️  Create image & visualize — Visualize anything", use_container_width=True):
            st.session_state['pending_prompt'] = "What software products and AI architecture solutions does 100Solutionz build?"
            st.rerun()

# Handle preset prompt click
preset_prompt = st.session_state.pop('pending_prompt', None)

# Display message history
for idx, message in enumerate(st.session_state['message_history']):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if 'reasoning' in message:
            st.caption(f"🎯 **Reasoning:** {message['reasoning']} | **Confidence:** {message['confidence']}%")
            
            # Interactive Feedback Section
            run_id = message.get('run_id')
            col_f1, col_f2, _ = st.columns([1, 1, 10])
            with col_f1:
                if st.button("👍", key=f"up_{idx}"):
                    submit_feedback(run_id, 1.0, idx)
            with col_f2:
                if st.button("👎", key=f"down_{idx}"):
                    submit_feedback(run_id, 0.0, idx)

# Chat Input
user_input = st.chat_input('Ask anything...') or preset_prompt

if user_input:
    # 1. Process files for the CURRENT request
    text_context = ""
    images_to_send = []
    
    if uploaded_files:
        with st.spinner("Extracting data from attachments..."):
            for f in uploaded_files:
                result = extract_file_content(f)
                if result.startswith("IMAGE_DATA_BASE64|"):
                    _, m_type, b64, fname = result.split("|")
                    images_to_send.append((m_type, b64, fname))
                else:
                    text_context += result

    if len(text_context) > MAX_CONTEXT_CHARS:
        text_context = text_context[:MAX_CONTEXT_CHARS] + f"\n\n[Attachment context truncated for safety. Original size: {len(text_context)} characters.]"

    # 2. Update UI with User message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 3. Build the LangGraph input with metadata tags for LangSmith
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "tags": [f"user:{user_id}", f"thread:{st.session_state['thread_id'][:8]}"],
        "metadata": {"user_id": user_id, "session_id": st.session_state["thread_id"]}
    }
    
    context_instruction = f"\n\n[UNTRUSTED_ATTACHMENT_DATA]\n{text_context}\n[/UNTRUSTED_ATTACHMENT_DATA]\n" if text_context else ""
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

    # 4. Get AI Response with LangSmith Run Capture
    run_id = None
    with st.chat_message('assistant'):
        full_response = ""
        placeholder = st.empty()
        
        with st.spinner("Deep Analysis..."):
            try:
                with tracing_context(client=ls_client, project_name=project_name, enabled=bool(ls_client)):
                    with collect_runs() as cb:
                        for chunk, metadata in chatbot.stream(
                            {'messages': [HumanMessage(content=msg_content)]},
                            config=CONFIG,
                            stream_mode='messages'
                        ):
                            if isinstance(chunk, AIMessage) and chunk.content:
                                full_response += chunk.content
                                display_text = full_response.split("[REASONING]")[0].replace("[RESPONSE]", "").strip()
                                placeholder.markdown(display_text)

                        if cb.traced_runs:
                            run_id = str(cb.traced_runs[0].id)
            except RateLimitError:
                full_response = (
                    "[RESPONSE]\nThe AI service is temporarily unavailable because its usage limit "
                    "has been reached. Please try again shortly, or contact the app owner.\n\n"
                    "[REASONING]\nThe language-model provider rejected this request due to a rate or quota limit.\n\n"
                    "[CONFIDENCE]\n100"
                )
            except Exception as stream_err:
                print(f"Chat request failed: {type(stream_err).__name__}: {stream_err}")
                full_response = (
                    "[RESPONSE]\nI could not process your request right now. Please try again shortly.\n\n"
                    "[REASONING]\nAn unexpected server-side error occurred while processing the request.\n\n"
                    "[CONFIDENCE]\n0"
                )

        # Final Metadata Parsing
        clean_ans, reason, conf = parse_metadata(full_response)
        placeholder.markdown(clean_ans)
        st.caption(f"🎯 **Reasoning:** {reason} | **Confidence:** {conf}%")

    # 5. Save to history with run_id for feedback tracking
    st.session_state['message_history'].append({
        'role': 'assistant', 
        'content': clean_ans, 
        'reasoning': reason, 
        'confidence': conf,
        'run_id': run_id
    })
