import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

print("=" * 60)
print("SOLUTIONZ CHATBOT - API KEY & LANGSMITH VERIFICATION")
print("=" * 60)

openai_key = os.getenv("OPENAI_API_KEY")
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "solutionz-chatbot")

print(f"1. OpenAI API Key: {'[OK] Present (' + openai_key[:12] + '...)' if openai_key else '[FAIL] Missing'}")
print(f"2. LangSmith API Key: {'[OK] Present (' + langsmith_key[:12] + '...)' if langsmith_key else '[FAIL] Missing'}")
print(f"3. LangSmith Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"4. LangSmith Project: {langsmith_project}")

# Test OpenAI Connection
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    res = llm.invoke("Say 'OpenAI connection successful!'")
    print(f"\n[OpenAI Test Output]: {res.content.strip()}")
    print("[OK] OpenAI API Test Passed!")
except Exception as e:
    print(f"[FAIL] OpenAI API Test Failed: {e}")

# Test LangSmith Connection
try:
    from langsmith import Client
    client = Client(api_key=langsmith_key)
    projects = [p.name for p in client.list_projects()]
    print(f"\n[LangSmith Projects Found]: {projects[:5]}")
    print("[OK] LangSmith API Test Passed!")
except Exception as e:
    print(f"[FAIL] LangSmith API Test Failed: {e}")

print("=" * 60)
