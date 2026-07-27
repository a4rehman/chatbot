from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

# Explicitly ensure LangSmith tracing environment defaults
if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = "solutionz-chatbot"

# System Instructions
SYSTEM_PROMPT = """You are the official AI Assistant for 100Solutionz, a leading software engineering company.

COMPANY PROFILE & KNOWLEDGE:
- Company Name: 100Solutionz
- Built By: AI Engineer Abdul Rehman
- Core Offerings: Custom Software Development, Web Application Development, Mobile App Development (iOS & Android), Database Management & Optimization, Cloud Solutions, and Enterprise Software Products.
- Objective: Answer customer and client inquiries professionally about 100Solutionz's services, software products, pricing, and technical expertise.

Your responses MUST follow this exact structure at the end of every message:

[RESPONSE]
(Your actual helpful answer here)

[REASONING]
(Briefly explain why you gave this answer in 1-2 short sentences)

[CONFIDENCE]
(A number between 0 and 100 representing how sure you are)

CRITICAL RULES:
1. If anyone asks who built you: 'I was built by AI Engineer Abdul Rehman.'
2. No sensitive data sharing.
3. Use 'web_search' for unknown facts.
4. You have VISION capabilities for images.
5. If text from files is provided in [ATTACHED_FILES_CONTEXT], analyze it CAREFULLY. If the user asks about specific data (like results, fees, or names), look through the ENTIRE context provided.
6. Always provide the [REASONING] and [CONFIDENCE] blocks."""


# Tools setup
tools = []
try:
    if os.getenv("TAVILY_API_KEY"):
        search_tool = TavilySearchResults(max_results=3)
        tools = [search_tool]
except Exception as e:
    print(f"Web search tool could not be initialized: {e}")

tool_node = ToolNode(tools)

# LLM setup with conditional tools binding
api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openrouter/free")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not configured. Add it to Streamlit Secrets or .env.")

base_llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

llm = base_llm.bind_tools(tools) if tools else base_llm


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reasoning: str
    confidence: int
    retry_count: int

def call_model(state: ChatState):
    messages = state['messages']
    retries = state.get('retry_count', 0)
    
    # Prepend system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm.invoke(messages)
    
    # Simple parsing logic for Reasoning and Confidence
    content = response.content
    reasoning = "Calculated based on available context."
    confidence = 100
    
    try:
        if "[REASONING]" in content and "[CONFIDENCE]" in content:
            parts = content.split("[REASONING]")
            ans_part = parts[0].replace("[RESPONSE]", "").strip()
            rest = parts[1].split("[CONFIDENCE]")
            reasoning = rest[0].strip()
            conf_str = "".join(filter(str.isdigit, rest[1]))
            confidence = int(conf_str) if conf_str else 80
            
            # If confidence is low and we haven't retried too much, we trigger a "retry"
            # In LangGraph we can do this via edges, but for speed, let's just flag it
            if confidence < 60 and retries < 2:
                # Add a nudge to the model to be better
                nudge = HumanMessage(content=f"Your previous response had low confidence ({confidence}%). Please provide a more accurate and confident response.")
                return {"messages": [response, nudge], "retry_count": retries + 1}
    except:
        pass

    return {"messages": [response], "reasoning": reasoning, "confidence": confidence, "retry_count": 0}

def should_continue(state: ChatState):
    messages = state['messages']
    last_message = messages[-1]
    
    # Check for retries first
    if state.get('retry_count', 0) > 0 and isinstance(last_message, HumanMessage):
        return "agent" # Go back to agent for retry
        
    if last_message.tool_calls:
        return "tools"
    return END

# Database connection
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph definition
workflow = StateGraph(ChatState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "agent": "agent", END: END})
workflow.add_edge("tools", "agent")

chatbot = workflow.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    """Returns all unique thread IDs from the DB."""
    all_threads = set()
    try:
        # Note: In newer versions of langgraph, .list might behave differently depending on the schema
        # This is a safe way to grab thread IDs if the checkpointer supports it
        for checkpoint in checkpointer.list(None):
            tid = checkpoint.config.get('configurable', {}).get('thread_id')
            if tid:
                all_threads.add(tid)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        
    return list(all_threads)
