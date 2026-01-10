
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlpcomponents import NLPPipeline
from nlpcomponents.build.orchestrator import BuildOrchestrator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive pipelineNLP probe.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many fused candidates to keep (set 2 to mirror production top-2).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip artifact rebuild check (faster startup).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.skip_build:
        print("[BUILD] Checking artifacts...")
        orchestrator = BuildOrchestrator(verbose=False)
        result = orchestrator.build_all(force=False)
        if not result.success:
            print(f"[BUILD] FAILED: {result.failed_artifacts}")
            print("Cannot start interactive mode without valid artifacts.")
            return 1
        if result.rebuilt_artifacts:
            print(f"[BUILD] Built {len(result.rebuilt_artifacts)} artifacts")
        else:
            print("[BUILD] All artifacts up-to-date")
    
    pipe = NLPPipeline()
    pipe.initialize()
    fusion_top_k = max(1, args.top_k)
    print(f"Interactive pipelineNLP (top-{fusion_top_k}; type 'quit' to exit)")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        result = pipe.run(question, fusion_top_k=fusion_top_k)
        tag = result.get("primary_tag")
        print(f"Top tag: {tag}")
        print("Candidates:")
        for idx, cand in enumerate(result.get("candidates", []), 1):
            print(f"  {idx}. {cand['tag']} (score={cand['final_score']:.3f})")
        print()

if __name__ == "__main__":
    sys.exit(main() or 0)
