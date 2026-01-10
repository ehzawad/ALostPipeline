
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def ask_question(question: str, messages: list) -> Dict[str, Any]:
    url = f"{BASE_URL}/ask"
    payload = {
        "question": question,
        "messages": json.dumps(messages, ensure_ascii=False)
    }

    print(f"\n{'='*60}")
    print(f"Question: {question}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect. Start server: uvicorn api.app:app")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def print_response(response: Dict[str, Any]):
    if not response:
        return
    print(f"Tag: {response['response_tag']}")
    print(f"Confidence: {response['probability']:.3f}")
    print(f"Relevant: {'Yes' if response['is_relevant'] else 'No'}")
    print(f"Latency: {response['time_taken']*1000:.0f}ms")

def main():
    print("=" * 60)
    print("pipelineNLP API Test Client")
    print("=" * 60)

    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5).json()
        print(f"[OK] API Status: {health['status']}")
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return

    messages = []

    test_questions = [
        "test question 1",
        "test question 2", 
        "test question 3",
    ]

    for q in test_questions:
        response = ask_question(q, messages)
        if response:
            print_response(response)
            messages = json.loads(response['messages'])

    print(f"\n{'='*60}")
    print("[SUCCESS] Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
