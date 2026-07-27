import os
import sys
from langchain_openai import ChatOpenAI

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

key = os.getenv("OPENAI_API_KEY")

print(f"Testing provided key (first 20 chars): {key[:20]}...")
try:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=key)
    res = llm.invoke("Hello, respond with 'Key works'.")
    print(f"[OK] Response: {res.content.strip()}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
