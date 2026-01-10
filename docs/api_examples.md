# API Examples

This folder contains examples for testing the pipelineNLP API.

## Files

- `curl_examples.md` - curl commands for API testing
- `test_client.py` - Python test client

## Quick Start

```bash
# Start server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test health
curl http://localhost:8000/health

# Ask question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is my balance", "messages": "[]"}'
```
