"""Evaluation runner for InsightRAG using golden_set.json.

Requests + stdlib only. Resume-safe with per-tag response caching.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsightRAG Golden Set Evaluation Runner")
    parser.add_argument("--tag", type=str, required=True, help="Run tag name for caching and results")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "all"], help="Dataset split")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="InsightRAG base URL")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between requests")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup requests to discard")
    return parser.parse_args()


def calculate_p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * 0.95
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)


def send_ask_request(url: str, query: str, timeout: float = 90.0, max_retries: int = 3) -> Dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/ask"
    payload = {"query": query}
    backoff = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Request failed after {max_retries} attempts: {e}") from e
            time.sleep(backoff)
            backoff *= 2.0
    return {}


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    golden_path = base_dir / "golden_set.json"

    if not golden_path.exists():
        print(f"Error: {golden_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data: List[Dict[str, Any]] = json.load(f)

    # Filter by split
    if args.split != "all":
        questions = [q for q in golden_data if q.get("split") == args.split]
    else:
        questions = golden_data

    cache_dir = base_dir / "cache" / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if live calls are needed
    needs_live = any(not (cache_dir / f"{q['id']}.json").exists() for q in questions)

    # Run discarded warmup calls if live queries needed
    if needs_live and args.warmup > 0:
        print(f"Executing {args.warmup} discarded warmup request(s)...")
        for w_idx in range(args.warmup):
            try:
                send_ask_request(args.url, "Warmup query: ping system", timeout=args.timeout if hasattr(args, 'timeout') else 90.0)
            except Exception as e:
                print(f"Warmup {w_idx + 1} warning: {e}", file=sys.stderr)
            time.sleep(1.0)

    processed_items: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions, start=1):
        q_id = q["id"]
        q_text = q["question"]
        expected_src = q.get("source", "")
        kb_file = q.get("kb_file", "")
        cache_file = cache_dir / f"{q_id}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as cf:
                data = json.load(cf)
        else:
            print(f"[{idx}/{len(questions)}] Querying id={q_id}: {q_text[:60]}...")
            data = send_ask_request(args.url, q_text, timeout=90.0, max_retries=3)

            # Ensure citation texts are saved in cache
            citations = data.get("citations", [])
            for c in citations:
                if "text" not in c or not c["text"]:
                    c["text"] = c.get("snippet", "")

            with open(cache_file, "w", encoding="utf-8") as cf:
                json.dump(data, cf, indent=2)

            if args.delay > 0 and idx < len(questions):
                time.sleep(args.delay)

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        citation_source_types = [c.get("source_type") for c in citations if c.get("source_type")]
        citation_files = [c.get("source") for c in citations if c.get("source")]

        # Ensure text is present on citations
        for c in citations:
            if "text" not in c or not c["text"]:
                c["text"] = c.get("snippet", "")

        conf = data.get("confidence", {})
        conf_score = conf.get("score", 0.0) if isinstance(conf, dict) else float(conf)
        token_usage = data.get("token_usage", {})
        tokens = token_usage.get("total_tokens", 0) if isinstance(token_usage, dict) else int(token_usage or 0)
        cost = float(data.get("estimated_cost_usd", 0.0))
        lat_ms = float(data.get("total_latency_ms", 0.0))
        sources_queried = data.get("sources_queried", [])
        router_decision = data.get("router_decision", "")
        # Count fallback only when router was actually invoked and failed (not kb_sufficient)
        router_fallback = bool(data.get("router_fallback", False)) and (router_decision != "kb_sufficient")
        is_idk = answer.strip().startswith("I don't know")

        processed_items.append({
            "id": q_id,
            "question": q_text,
            "expected_source": expected_src,
            "split": q.get("split", ""),
            "kb_file": kb_file,
            "answer": answer,
            "routing_mode": data.get("routing_mode", ""),
            "sources_queried": sources_queried,
            "router_decision": router_decision,
            "router_fallback": router_fallback,
            "confidence_score": conf_score,
            "citation_source_types": citation_source_types,
            "citation_files": citation_files,
            "citations": citations,
            "total_latency_ms": lat_ms,
            "tokens": tokens,
            "estimated_cost_usd": cost,
            "is_idk": is_idk,
        })

    # Separate router_fallback cases
    fallback_items = [item for item in processed_items if item.get("router_fallback") is True]
    valid_items = [item for item in processed_items if not item.get("router_fallback")]

    n = len(valid_items)
    if n > 0:
        latencies = [item["total_latency_ms"] for item in valid_items]
        avg_lat = sum(latencies) / n
        p95_lat = calculate_p95(latencies)
        avg_tokens = sum(item["tokens"] for item in valid_items) / n
        avg_cost = sum(item["estimated_cost_usd"] for item in valid_items) / n
        idk_rate = sum(1 for item in valid_items if item["is_idk"]) / n

        # Routing accuracy: expected source type appears in citations
        routing_correct = sum(
            1 for item in valid_items
            if item["expected_source"] in item["citation_source_types"]
        )
        routing_acc = routing_correct / n

        # KB hit rate: kb_file appears in citations for kb questions
        kb_items = [item for item in valid_items if item["expected_source"] == "kb" and item["kb_file"]]
        if kb_items:
            kb_hits = sum(
                1 for item in kb_items
                if any(item["kb_file"] in str(f) for f in item["citation_files"])
            )
            kb_hit_rate = kb_hits / len(kb_items)
        else:
            kb_hit_rate = 0.0

        # % of questions that skipped external sources
        skipped_ext_count = sum(
            1 for item in valid_items
            if set(item["sources_queried"]).issubset({"kb"})
        )
        pct_skipped_ext = (skipped_ext_count / n) * 100.0
        mean_conf = sum(item["confidence_score"] for item in valid_items) / n
    else:
        avg_lat = p95_lat = avg_tokens = avg_cost = idk_rate = routing_acc = kb_hit_rate = pct_skipped_ext = mean_conf = 0.0

    summary = {
        "n_evaluated": n,
        "router_fallbacks_excluded": len(fallback_items),
        "total_questions": len(processed_items),
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "avg_tokens": round(avg_tokens, 1),
        "avg_cost_usd": round(avg_cost, 6),
        "idk_rate": round(idk_rate, 4),
        "routing_accuracy": round(routing_acc, 4),
        "kb_hit_rate": round(kb_hit_rate, 4),
        "pct_skipped_external": round(pct_skipped_ext, 2),
        "mean_confidence": round(mean_conf, 4),
    }

    result_payload = {
        "tag": args.tag,
        "split": args.split,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "questions": processed_items,
    }

    result_file = results_dir / f"{args.tag}.json"
    with open(result_file, "w", encoding="utf-8") as rf:
        json.dump(result_payload, rf, indent=2)

    # Print markdown table summary
    print("\n### Evaluation Summary: " + args.tag)
    print("| Metric | Value |")
    print("| :--- | :--- |")
    print(f"| Questions Evaluated (n) | {n} |")
    print(f"| Router Fallbacks Excluded | {len(fallback_items)} |")
    print(f"| Avg Latency (ms) | {avg_lat:.2f} |")
    print(f"| P95 Latency (ms) | {p95_lat:.2f} |")
    print(f"| Avg Tokens | {avg_tokens:.1f} |")
    print(f"| Avg Cost (USD) | ${avg_cost:.6f} |")
    print(f"| 'I don't know' Rate | {idk_rate * 100:.1f}% |")
    print(f"| Routing Accuracy | {routing_acc * 100:.1f}% |")
    print(f"| KB Hit Rate | {kb_hit_rate * 100:.1f}% |")
    print(f"| Skipped External Sources | {pct_skipped_ext:.1f}% |")
    print(f"| Mean Confidence | {mean_conf:.4f} |")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()

