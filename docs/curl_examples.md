# Curl Examples for pipelineNLP API

## Prerequisites

Start the API server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Ask Questions

**Balance inquiry:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is my balance", "messages": "[]"}'
```

**Transfer money:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "transfer money to my savings", "messages": "[]"}'
```

**Weather:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is the weather today", "messages": "[]"}'
```

**Set alarm:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "set an alarm for 7am", "messages": "[]"}'
```

## With Conversation History

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what about tomorrow",
    "messages": "[{\"role\":\"user\",\"content\":\"what is the weather today\"},{\"role\":\"assistant\",\"content\":\"It looks sunny today.\",\"tag\":\"weather\"}]"
  }'
```

## Pretty Print with jq

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is my balance", "messages": "[]"}' | jq '.'
```

## Swagger UI

Interactive docs: http://localhost:8000/docs
