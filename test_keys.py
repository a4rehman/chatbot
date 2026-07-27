import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

print("=" * 60)
print("SOLUTIONZ CHATBOT - API KEY & LANGSMITH VERIFICATION")
print("=" * 60)

openrouter_key = os.getenv("OPENROUTER_API_KEY")
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "solutionz-chatbot")

print(f"1. OpenRouter API Key: {'[OK] Present (' + openrouter_key[:12] + '...)' if openrouter_key else '[FAIL] Missing'}")
print(f"2. LangSmith API Key: {'[OK] Present (' + langsmith_key[:12] + '...)' if langsmith_key else '[FAIL] Missing'}")
print(f"3. LangSmith Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"4. LangSmith Project: {langsmith_project}")

# Test OpenRouter Connection
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "openrouter/free"), base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), api_key=openrouter_key)
    res = llm.invoke("Say 'OpenRouter connection successful!'")
    print(f"\n[OpenRouter Test Output]: {res.content.strip()}")
    print("[OK] OpenRouter API Test Passed!")
except Exception as e:
    print(f"[FAIL] OpenRouter API Test Failed: {e}")

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
