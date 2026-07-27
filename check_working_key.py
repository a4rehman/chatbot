import os
import sys
from langchain_openai import ChatOpenAI

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

keys = [
    os.getenv("OPENROUTER_API_KEY"),
    os.getenv("OPENROUTER_API_KEY")
]

print("Testing OpenRouter API keys...")
working_key = None

for idx, key in enumerate(keys, 1):
    print(f"\n--- Testing Key {idx}: {key[:20]}... ---")
    try:
        llm = ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "openrouter/free"), base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), api_key=key)
        res = llm.invoke("Hello, answer with 'API Key Working'")
        print(f"[OK] Response: {res.content.strip()}")
        working_key = key
        break
    except Exception as e:
        print(f"[FAIL] Error: {e}")

if working_key:
    print(f"\nSUCCESS: Working key found!")
else:
    print(f"\nFAIL: None of the provided keys worked.")
