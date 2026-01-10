from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from loguru import logger
import json
import time
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlpcomponents import NLPPipeline
from nlpcomponents.config import NLPPipelineConfig
from nlpcomponents.build.orchestrator import BuildOrchestrator
from .schemas import RequestBody
from .pipeline_holder import PipelineHolder

logger.add(
    "logs/api_requests_{time:YYYY-MM-DD}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    rotation="00:00",
    retention="30 days",
    level="INFO"
)

pipeline_holder = PipelineHolder[NLPPipeline]()

CONFIDENCE_THRESHOLD = 0.3
TOP_K = 10
MAX_HISTORY_MESSAGES = 20

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 80)
    logger.info("Starting pipelineNLP FastAPI Service")
    logger.info("=" * 80)

    try:
        config = NLPPipelineConfig()

        logger.info("Phase 1: Checking and rebuilding artifacts...")
        orchestrator = BuildOrchestrator(
            config=config,
            verbose=True,
            inference_only=True
        )

        rebuild_list = orchestrator.calculate_rebuild_set(force=False)

        if rebuild_list:
            logger.info(f"Rebuilding {len(rebuild_list)} stale/missing artifacts...")
            for artifact in rebuild_list:
                logger.info(f"  - {artifact}")

            import asyncio
            build_timeout = int(os.environ.get("BUILD_TIMEOUT_SECONDS", 600 * len(rebuild_list)))
            logger.info(f"Build timeout: {build_timeout}s for {len(rebuild_list)} artifacts")

            try:
                result = await asyncio.wait_for(
                    run_in_threadpool(orchestrator.build_all),
                    timeout=build_timeout
                )
            except asyncio.TimeoutError:
                error_msg = f"Artifact build timed out after {build_timeout}s. Set BUILD_TIMEOUT_SECONDS env var to increase."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if not result.success:
                error_msg = f"Artifact build failed: {result.failed_artifacts}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Successfully rebuilt {len(result.rebuilt_artifacts)} artifacts")
        else:
            logger.info("All inference artifacts up-to-date")

        logger.info("Phase 2: Initializing NLPPipeline...")
        pipeline = NLPPipeline(config=config)
        pipeline.initialize()

        pipeline_holder.set(pipeline)

        logger.success("Pipeline initialized successfully!")
        logger.info(f"Configuration: top_k={TOP_K}, threshold={CONFIDENCE_THRESHOLD}")
        base_url = f"http://{os.environ.get('UVICORN_HOST', 'localhost')}:{os.environ.get('UVICORN_PORT', '8000')}"
        logger.info(f"Serving endpoints: {base_url}/ (root), {base_url}/health, {base_url}/docs")
        logger.info("Service ready to accept requests")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Failed to initialize service: {e}")
        raise

    yield

    logger.info("Shutting down pipelineNLP FastAPI Service")
    try:
        pipeline_holder.clear(drain_timeout=30.0)
        logger.info("Pipeline holder cleared")

        from nlpcomponents.cache.model_cache import clear_model_cache
        clear_model_cache()
        logger.info("Model cache cleared")
    except Exception as e:
        logger.warning(f"Error during shutdown cleanup: {e}")

app = FastAPI(
    title="pipelineNLP Q&A API",
    description="NID/Voter Registration Question Answering System",
    version="1.0.0",
    lifespan=lifespan
)

ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/")
async def root():
    return {
        "service": "pipelineNLP Q&A API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if pipeline_holder.is_ready else "initializing",
        "pipeline_loaded": pipeline_holder.is_ready,
        "pipeline_version": pipeline_holder.current_version,
        "active_requests": pipeline_holder.active_references,
        "threshold": CONFIDENCE_THRESHOLD,
        "top_k": TOP_K
    }

@app.post("/ask")
async def ask_question(body: RequestBody):
    start_time = time.time()

    try:
        question = (body.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if len(question) > 1000:
            raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")

        try:
            msg_value = (body.messages or "").strip()
            if not msg_value or msg_value == "[]":
                messages = []
            else:
                messages = json.loads(msg_value)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in messages field: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in messages: {e}")

        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a JSON array")

        messages.append({
            "role": "user",
            "content": question
        })

        logger.info(f"Question: {question}")
        logger.debug(f"Conversation history: {len(messages)} messages (not yet used)")

        with pipeline_holder.acquire() as current_pipeline:
            if current_pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized yet")

            result = await run_in_threadpool(current_pipeline.run, question, fusion_top_k=TOP_K)

        response = result["answer"]
        response_tag = result["primary_tag"]

        probability = 0.0
        if result["candidates"]:
            probability = float(result["candidates"][0]["final_score"])

        logger.info(f"Tag: {response_tag}, Score: {probability:.3f}")

        messages.append({
            "role": "assistant",
            "content": response,
            "tag": response_tag
        })

        if len(messages) > MAX_HISTORY_MESSAGES:
            messages = messages[-MAX_HISTORY_MESSAGES:]
            logger.debug(f"Truncated history to {MAX_HISTORY_MESSAGES} messages")

        end_time = time.time()
        time_taken = end_time - start_time

        is_relevant = probability > CONFIDENCE_THRESHOLD

        logger.info(
            f"Response: tag={response_tag}, score={probability:.3f}, "
            f"relevant={is_relevant}, latency={time_taken*1000:.0f}ms"
        )

        response_data = {
            "response": response,
            "response_tag": response_tag,
            "time_taken": time_taken,
            "messages": json.dumps(messages, ensure_ascii=False),
            "probability": probability,
            "user_input": f"User Input: {body.question}",
            "is_relevant": is_relevant
        }

        return response_data

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
