"""Command-line interface for the SentiPulse sentiment classification engine."""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
from sentipulse.client import SentiPulse


def cmd_analyze(args):
    """Analyze a single customer feedback string."""
    client = SentiPulse(api_url=args.api_url)
    result = client.analyze(args.text)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    sentiment = result.get("sentiment", "Neutral")
    confidence = result.get("confidence", 0.0)
    escalate = result.get("escalation_recommended", False)
    latency = result.get("response_time_ms", 0.0)
    word_count = result.get("word_count")
    truncated = result.get("truncated", False)

    print("\n--- SentiPulse Analysis ---")
    print(f"Sentiment:   {sentiment}")
    print(f"Confidence:  {confidence * 100:.1f}%")
    if word_count is not None:
        trunc_note = " (Truncated: Head+Tail 128 tokens)" if truncated else ""
        print(f"Length:      {word_count} words{trunc_note}")
    print(f"Escalate:    {'YES [High Churn Risk]' if escalate else 'No'}")
    print(f"Latency:     {latency:.1f} ms")
    print("Probabilities:")
    for k, v in result.get("probabilities", {}).items():
        print(f"  • {k:8s}: {v * 100:5.1f}%")
    print("---------------------------\n")


def cmd_batch(args):
    """Process a batch of reviews from a CSV file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)
    text_col = args.text_col
    if text_col not in df.columns:
        candidates = ["reviewText", "text", "Text", "review", "content"]
        found = next((c for c in candidates if c in df.columns), None)
        if not found:
            print(f"Error: Column '{text_col}' not found in CSV. Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        text_col = found

    print(f"Analyzing {len(df)} records from '{args.input}' using column '{text_col}'...")
    client = SentiPulse(api_url=args.api_url)

    sentiments = []
    confidences = []
    escalates = []

    for idx, text in enumerate(df[text_col]):
        res = client.analyze(str(text) if pd.notna(text) else "")
        sentiments.append(res.get("sentiment", "Neutral"))
        confidences.append(res.get("confidence", 0.0))
        escalates.append(res.get("escalation_recommended", False))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            print(f"Progress: {idx + 1}/{len(df)} ({(idx + 1) / len(df) * 100:.1f}%)")

    df["sentiment"] = sentiments
    df["confidence"] = confidences
    df["escalate"] = escalates

    output_path = args.output or f"{input_path.stem}_analyzed.csv"
    df.to_csv(output_path, index=False)
    print(f"\nBatch analysis complete! Results written to: {output_path}")


def cmd_serve(args):
    """Start the SentiPulse FastAPI server."""
    import uvicorn
    print(f"Starting SentiPulse API Server on http://{args.host}:{args.port}...")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_health(args):
    """Check health of SentiPulse inference engine."""
    client = SentiPulse(api_url=args.api_url)
    try:
        health_info = client.health()
        print(json.dumps(health_info, indent=2))
    except Exception as e:
        print(f"Error checking health: {e}", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="sentipulse",
        description="SentiPulse: Production-Grade Sentiment Classification & Real-Time MLOps Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze a single feedback string")
    p_analyze.add_argument("text", type=str, help="Customer feedback text")
    p_analyze.add_argument("--api-url", type=str, default=None, help="Remote SentiPulse server URL")
    p_analyze.add_argument("--json", action="store_true", help="Output raw JSON response")
    p_analyze.set_defaults(func=cmd_analyze)

    # 2. Batch
    p_batch = subparsers.add_parser("batch", help="Classify sentiments for an entire CSV dataset")
    p_batch.add_argument("input", type=str, help="Path to input CSV file")
    p_batch.add_argument("--text-col", type=str, default="reviewText", help="Name of column containing review text")
    p_batch.add_argument("--output", "-o", type=str, default=None, help="Destination CSV path")
    p_batch.add_argument("--api-url", type=str, default=None, help="Remote SentiPulse server URL")
    p_batch.set_defaults(func=cmd_batch)

    # 3. Serve
    p_serve = subparsers.add_parser("serve", help="Launch the SentiPulse FastAPI server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    p_serve.set_defaults(func=cmd_serve)

    # 4. Health
    p_health = subparsers.add_parser("health", help="Check engine health")
    p_health.add_argument("--api-url", type=str, default=None, help="Remote SentiPulse server URL")
    p_health.set_defaults(func=cmd_health)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

