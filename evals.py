import os
from dotenv import load_dotenv
from chatbot_backend import chatbot, SYSTEM_PROMPT
from langchain_core.messages import HumanMessage

load_dotenv()

print("=" * 60)
print("🧪 RUNNING LLM EVALUATIONS FOR SOLUTIONZ CHATBOT")
print("=" * 60)

test_cases = [
    {
        "id": "creator_check",
        "input": "Who built you?",
        "expected_substring": "Abdul Rehman",
        "description": "System Rule 1: Must state built by AI Engineer Abdul Rehman."
    },
    {
        "id": "output_format_check",
        "input": "What is Python?",
        "expected_tags": ["[RESPONSE]", "[REASONING]", "[CONFIDENCE]"],
        "description": "System Rule 6: Format adherence for reasoning & confidence tags."
    },
    {
        "id": "math_reasoning_check",
        "input": "Calculate 15 * 14 and explain step by step.",
        "expected_substring": "210",
        "description": "Basic reasoning & math accuracy test."
    }
]

passed = 0
failed = 0

for case in test_cases:
    print(f"\nEvaluating Test Case [{case['id']}]: {case['description']}")
    config = {"configurable": {"thread_id": f"eval_{case['id']}"}}
    
    try:
        result = chatbot.invoke({"messages": [HumanMessage(content=case["input"])]}, config=config)
        last_msg = result["messages"][-1].content
        print(f"Output Snippet:\n{last_msg[:180]}...")
        
        # Check assertions
        case_passed = True
        if "expected_substring" in case and case["expected_substring"].lower() not in last_msg.lower():
            print(f"❌ Substring Check Failed: '{case['expected_substring']}' not found.")
            case_passed = False
            
        if "expected_tags" in case:
            for tag in case["expected_tags"]:
                if tag not in last_msg:
                    print(f"❌ Format Check Failed: Tag '{tag}' missing in output.")
                    case_passed = False
                    
        if case_passed:
            print("✅ TEST PASSED")
            passed += 1
        else:
            failed += 1
    except Exception as err:
        print(f"❌ TEST ERRORED: {err}")
        failed += 1

print("\n" + "=" * 60)
print(f"📊 EVALUATION RESULTS: {passed} PASSED | {failed} FAILED")
print("=" * 60)
