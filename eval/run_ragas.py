"""RAGAS evaluation runner for InsightRAG cached answers.

Requires: ragas==0.2.14, langchain-google-genai, sentence-transformers
Usage: python eval/run_ragas.py --tag dev_t05 [--delay 3]

Monkey-patches missing langchain_community.chat_models.vertexai before
importing ragas (ragas==0.2.14 / langchain-community==0.4.x incompatibility).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── patch broken ragas dep before importing ragas ──────────────────────────
from unittest.mock import MagicMock
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from dotenv import load_dotenv

from ragas.metrics import Faithfulness, AnswerRelevancy       # ragas==0.2.14
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run RAGAS scoring on InsightRAG eval cache.")
    p.add_argument("--tag",   required=True, help="Eval tag (reads eval/cache/<tag>/)")
    p.add_argument("--delay", type=float, default=3.0, help="Seconds between API calls")
    return p.parse_args()


# ── score with retry on 429 ─────────────────────────────────────────────────
async def _score_sample(metric, sample, timeout: float = 60.0):
    return await metric.single_turn_ascore(sample, timeout=timeout)


def score_with_retry(metric, sample, delay: float, max_retries: int = 4):
    backoff = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            return asyncio.run(_score_sample(metric, sample))
        except Exception as e:
            last_err = e
            if "429" in str(e) or "quota" in str(e).lower():
                wait = backoff * (2 ** attempt)
                print(f"    429 rate-limit, retrying in {wait:.0f}s …", flush=True)
                time.sleep(wait)
            else:
                raise
    raise last_err


# ── main ────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    args = parse_args()

    api_key    = os.environ["GEMINI_API_KEY"]
    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

    # LLM (Gemini via langchain-google-genai)
    gemini     = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, timeout=60)
    llm_wrap   = LangchainLLMWrapper(gemini)

    # Embeddings (local sentence-transformers)
    hf_embed   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embed_wrap = LangchainEmbeddingsWrapper(hf_embed)

    # Metrics
    faith_metric = Faithfulness(llm=llm_wrap)
    rel_metric   = AnswerRelevancy(llm=llm_wrap, embeddings=embed_wrap)

    # Paths
    cache_dir      = Path("eval/cache")      / args.tag
    ragas_cache_dir = Path("eval/ragas_cache") / args.tag
    results_dir    = Path("eval/results")
    ragas_cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        sys.exit(f"Cache dir not found: {cache_dir}")

    cache_files = sorted(cache_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not cache_files:
        sys.exit(f"No cached files found in {cache_dir}")

    scored, skipped_idk, failed = [], [], []

    for cf in cache_files:
        q_id      = int(cf.stem)
        score_path = ragas_cache_dir / f"{q_id}.json"

        data   = json.loads(cf.read_text(encoding="utf-8"))
        answer = data.get("answer", "")

        # Skip "I don't know" answers
        if answer.strip().lower().startswith("i don't know"):
            print(f"  [{q_id}] skipping (IDK answer)")
            skipped_idk.append(q_id)
            continue

        # Resume from cache
        if score_path.exists():
            score_data = json.loads(score_path.read_text(encoding="utf-8"))
            print(f"  [{q_id}] cached  faith={score_data['faithfulness']}  rel={score_data['answer_relevancy']}")
            scored.append(score_data)
            continue

        question = data.get("query", "")
        contexts = [
            c.get("text") or c.get("snippet", "")
            for c in data.get("citations", [])
            if (c.get("text") or c.get("snippet", ""))
        ]
        if not contexts:
            print(f"  [{q_id}] SKIP — no context texts in citations")
            failed.append({"id": q_id, "error": "no contexts"})
            continue

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        print(f"  [{q_id}] scoring …", flush=True)
        try:
            faith = score_with_retry(faith_metric, sample, delay=args.delay)
            time.sleep(args.delay)
            rel   = score_with_retry(rel_metric,   sample, delay=args.delay)
        except Exception as e:
            print(f"  [{q_id}] FAILED: {e}")
            failed.append({"id": q_id, "error": str(e)})
            continue

        score_data = {
            "id": q_id,
            "faithfulness":      round(float(faith), 4),
            "answer_relevancy":  round(float(rel),   4),
        }
        score_path.write_text(json.dumps(score_data, indent=2), encoding="utf-8")
        print(f"  [{q_id}] done   faith={score_data['faithfulness']}  rel={score_data['answer_relevancy']}")
        scored.append(score_data)

        time.sleep(args.delay)

    # ── summary ──────────────────────────────────────────────────────────────
    n         = len(scored)
    avg_faith = sum(s["faithfulness"]     for s in scored) / n if n else 0.0
    avg_rel   = sum(s["answer_relevancy"] for s in scored) / n if n else 0.0

    summary = {
        "tag":                 args.tag,
        "ragas_version":       "0.2.14",
        "model":               model_name,
        "scored":              n,
        "skipped_idk":         len(skipped_idk),
        "failed":              len(failed),
        "avg_faithfulness":    round(avg_faith, 4),
        "avg_answer_relevancy": round(avg_rel,  4),
        "failed_ids":          failed,
    }

    result_file = results_dir / f"{args.tag}_ragas.json"
    result_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n### RAGAS Results — {args.tag}")
    print("| Metric | Value |")
    print("| :--- | ---: |")
    print(f"| Scored | {n} |")
    print(f"| Skipped (I don't know) | {len(skipped_idk)} |")
    print(f"| Failed | {len(failed)} |")
    print(f"| Avg Faithfulness | {avg_faith:.4f} |")
    print(f"| Avg Answer Relevancy | {avg_rel:.4f} |")
    print(f"\nSaved → {result_file}")


if __name__ == "__main__":
    main()

