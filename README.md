# 🤖 100Solutionz AI Assistant

A production-ready **AI-powered customer assistant** for [100Solutionz](https://100solutionz.vercel.app) — a leading software engineering company. Built with **Streamlit**, **LangGraph**, **LangSmith**, and **OpenRouter/OpenAI**.

The assistant answers questions about 100Solutionz's services, portfolio, and tech stack, and can analyze uploaded documents (PDF, CSV, XLSX) and images (vision) in real time.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.6-1C3C3C?style=flat&logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-Boost%201.0-yellow)

---

## ✨ Features

- 🧠 **LangGraph agent** with a multi-turn conversation checkpointer (SQLite persistence)
- 🔎 **Web search tool** (Tavily) for real-time information
- 🖼️ **Vision support** — upload and analyze images (PNG, JPG, JPEG)
- 📎 **Document analysis** — PDF (up to 50 pages), CSV & Excel (up to 1,000 rows)
- 🪄 **ChatGPT/Copilot-style UI** — dark theme, streaming responses, typing indicator, quick-action cards
- 📊 **LangSmith observability** — run tracing, per-message 👍/👎 feedback, privacy redaction
- 💬 **Chat history** — start new threads, switch between past sessions
- 🏗️ **Interactive backend graph** rendered in the sidebar
- 🔒 **Security hardened** — untrusted attachment data is isolated, system prompts & keys protected

## 🏗️ Architecture

```
User Query ──► [LangGraph Agent]
                    │  ├─ Tools: Tavily Web Search
                    │  ├─ Attachments: PDF / CSV / XLSX / Images
                    │  └─ LLM: OpenRouter (OpenAI-compatible) or OpenAI
                    ▼
        Response + [RESPONSE] / [REASONING] / [CONFIDENCE]
                    ▼
        Streamlit UI (streaming, history, feedback)
```

- **Frontend:** `chatbot_frontend.py` — Streamlit UI, file extraction, streaming, feedback
- **Backend:** `chatbot_backend.py` — LangGraph state machine, LLM + tools, SQLite checkpointer

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/a4rehman/chatbot.git
cd chatbot

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### 3. Run the app

```bash
streamlit run chatbot_frontend.py
```

Open **http://localhost:8501** and start chatting.

## 🔑 Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | *one of* | OpenRouter API key (primary provider) |
| `OPENROUTER_MODEL` | No | Model ID, default `openrouter/free` |
| `OPENROUTER_BASE_URL` | No | Default `https://openrouter.ai/api/v1` |
| `OPENAI_API_KEY` | *one of* | OpenAI API key (fallback provider) |
| `OPENAI_MODEL` | No | Model ID, default `gpt-4o-mini` |
| `TAVILY_API_KEY` | No | Enables the web-search tool |
| `LANGSMITH_API_KEY` | No | LangSmith tracing & feedback |
| `LANGSMITH_PROJECT` | No | Trace project, default `solutionz-chatbot` |
| `LANGSMITH_ENDPOINT` | No | Default `https://api.smith.langchain.com` |
| `LANGSMITH_HIDE_INPUTS/OUTPUTS/METADATA` | No | Privacy redaction flags (`true`/`false`) |

> **Note:** `.env` values also work in **Streamlit Cloud / Railway** via Streamlit Secrets (all env keys are read with a `st.secrets` fallback).

## ☁️ Deployment

### Railway

```bash
railway up
```

A `Dockerfile` and `Procfile`-free setup are included; add your env variables in the Railway dashboard.

### Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud → **New app** → select the repo.
3. Main file: `chatbot_frontend.py`.
4. Add the env vars above under **Secrets**.

## 📊 LangSmith Observability

When `LANGSMITH_API_KEY` is set, every run is traced to your LangSmith project:

- Full run metadata tagged with `user:{id}` and `thread:{id}`
- Thumbs up/down feedback sent from the UI (`key: user_rating`)
- Input/output redaction when `LANGSMITH_HIDE_*` flags are enabled

## 📁 Project Structure

```
├── chatbot_frontend.py      # Streamlit UI, uploads, streaming, feedback
├── chatbot_backend.py       # LangGraph agent, tools, SQLite checkpointer
├── response_utils.py        # Shared [RESPONSE]/[REASONING]/[CONFIDENCE] parser
├── evals.py                 # Evaluation harness
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── Dockerfile               # Containerized deployment
└── .railwayignore           # Railway deployment exclusions
```

## 🛡️ Safety

- Uploads are size/type capped (30 MB, 50 PDF pages, 1,000 spreadsheet rows, 20 MP images)
- Attachment text is wrapped in `[UNTRUSTED_ATTACHMENT_DATA]` and treated as reference-only
- System prompts, API keys, and internal reasoning are never exposed

## 📄 License

Distributed under the [Boost Software License 1.0](LICENSE).
