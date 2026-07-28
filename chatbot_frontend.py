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

langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "solutionz-chatbot"
langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
ls_client = None
if langsmith_api_key:
    try:
        ls_client = LangSmithClient(
            api_key=langsmith_api_key,
            api_url=langsmith_endpoint,
            hide_inputs=lambda _: {},
            hide_outputs=lambda _: {},
            hide_metadata=True,
        )
    except Exception as err:
        print(f"LangSmith client initialization failed: {type(err).__name__}")

# Upload limits protect the app from oversized or malformed user-supplied files.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
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
            return f"\n[File: {filename}] was skipped because it exceeds the 10 MB safety limit.\n"

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
                return f"\n[File: {filename}] was skipped because its image dimensions exceed the safety limit.\n"
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
        st.info("Feedback captured locally (LangSmith API key needed for cloud sync).")
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

st.sidebar.markdown(f"### 👤 User ID: **{user_id}**")
st.sidebar.title('Advanced LangGraph Bot')
st.sidebar.info("100Solutionz Engineering Edition")

# LangSmith Status Badge
project_name = langsmith_project
if ls_client:
    st.sidebar.success(f"📊 LangSmith Tracing Active\n\nProject: **{project_name}**")
else:
    st.sidebar.warning("📊 LangSmith Tracing Inactive")

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

st.title("🤖 100Solutionz Intelligent AI Assistant")
st.caption(f"Session: {st.session_state['thread_id'][:8]} | Traced via LangSmith: {project_name}")

# Display message history
for idx, message in enumerate(st.session_state['message_history']):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if 'reasoning' in message:
            st.caption(f"🎯 **Reasoning:** {message['reasoning']} | **Confidence:** {message['confidence']}%")
            
            # Interactive Feedback Section for Assistant Responses
            run_id = message.get('run_id')
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{idx}"):
                    submit_feedback(run_id, 1.0, idx)
            with col2:
                if st.button("👎", key=f"down_{idx}"):
                    submit_feedback(run_id, 0.0, idx)

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

    # Truncation safety check to prevent context length error
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
