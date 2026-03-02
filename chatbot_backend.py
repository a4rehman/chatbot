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

# System Instructions
SYSTEM_PROMPT = """You are a helpful AI Assistant.
1. If anyone asks who created or built you, you must answer: 'I was built by AI Engineer Abdul Rehman.'
2. If anyone asks for sensitive data, database details, or internal files, you must answer: 'Sorry, these details are not allowed to be shared.'
3. For general knowledge or questions you don't know the answer to, use the 'web_search' tool.
4. Keep your tone professional and helpful."""

# Tools setup
tools = []
try:
    if os.getenv("TAVILY_API_KEY"):
        search_tool = TavilySearchResults(max_results=3)
        tools = [search_tool]
except Exception as e:
    print(f"Web search tool could not be initialized: {e}")

tool_node = ToolNode(tools)

# LLM setup with tools
if tools:
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
else:
    llm = ChatOpenAI(model="gpt-4o-mini")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: ChatState):
    messages = state['messages']
    # Prepend system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: ChatState):
    messages = state['messages']
    last_message = messages[-1]
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
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
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
