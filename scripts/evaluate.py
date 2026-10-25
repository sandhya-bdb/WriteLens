"""
evaluate.py: measure how well WriteLens separates human and ChatGPT answers
on the HC3 dataset (Hello-SimpleAI/HC3), and how often it wrongly flags
human writing.

Usage:
    python scripts/evaluate.py                 # 200 human + 200 ChatGPT answers
    python scripts/evaluate.py --n 500
    python scripts/evaluate.py --data-file all.jsonl   # use a local copy of HC3

Outputs:
    eval/results.json  full metrics
    eval/RESULTS.md    a small table you can paste into the README

The most important number for an education tool is the false positive rate:
how often genuinely human writing gets flagged.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.detector import MAX_CHARS, split_sentences  # noqa: E402
from logic.explainer import signal_level  # noqa: E402

# A text counts as "flagged" at each of these ai_score thresholds
THRESHOLDS = {"strong (>=70)": 70, "mixed or strong (>=40)": 40}


def load_hc3(data_file: str | None) -> list[dict]:
    if data_file:
        path = Path(data_file)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset"))
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def usable(text: str) -> bool:
    text = text.strip()
    return 0 < len(text) <= MAX_CHARS and len(split_sentences(text)) >= 3


def build_samples(rows: list[dict], n: int, seed: int) -> list[tuple[str, int]]:
    """Return up to n human (label 0) and n ChatGPT (label 1) answers."""
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    human, ai = [], []
    for row in rows:
        h = next((a for a in row.get("human_answers", []) if usable(a)), None)
        c = next((a for a in row.get("chatgpt_answers", []) if usable(a)), None)
        if h and len(human) < n:
            human.append((h.strip(), 0))
        if c and len(ai) < n:
            ai.append((c.strip(), 1))
        if len(human) >= n and len(ai) >= n:
            break
    return human + ai


def metrics(labels: list[int], scores: list[int], threshold: int) -> dict:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    total = len(labels)
    return {
        "accuracy": round((tp + tn) / total, 3) if total else 0.0,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def to_markdown(result: dict) -> str:
    lines = [
        f"Evaluated on {result['n_human']} human and {result['n_ai']} ChatGPT answers "
        f"from HC3 (seed {result['seed']}).",
        "",
        "| Flag when | Accuracy | Precision | Recall (AI caught) | False positive rate (humans flagged) |",
        "|---|---|---|---|---|",
    ]
    for name, m in result["thresholds"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
            f"{m['recall']:.1%} | {m['false_positive_rate']:.1%} ({m['fp']} of {m['fp'] + m['tn']}) |"
        )
    lines.append("")
    lines.append("Signal level distribution:")
    for label, dist in result["signal_levels"].items():
        lines.append(f"- {label}: " + ", ".join(f"{k} {v}" for k, v in dist.items()))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="answers per class (default 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-file", help="local path to HC3 all.jsonl")
    parser.add_argument("--out-dir", default=str(ROOT / "eval"))
    args = parser.parse_args()

    from logic.detector import AIDetector  # imported late: loads GPT-2

    samples = build_samples(load_hc3(args.data_file), args.n, args.seed)
    print(f"Scoring {len(samples)} texts with GPT-2 (this can take a while on CPU)…")

    detector = AIDetector()
    labels, scores = [], []
    levels = {"human": {}, "chatgpt": {}}
    for i, (text, label) in enumerate(samples, 1):
        score = detector.analyze(text)["ai_score"]
        labels.append(label)
        scores.append(score)
        bucket = levels["chatgpt" if label else "human"]
        lvl = signal_level(score)
        bucket[lvl] = bucket.get(lvl, 0) + 1
        if i % 25 == 0:
            print(f"  {i}/{len(samples)}")

    result = {
        "dataset": "Hello-SimpleAI/HC3 (all.jsonl)",
        "seed": args.seed,
        "n_human": labels.count(0),
        "n_ai": labels.count(1),
        "thresholds": {name: metrics(labels, scores, t) for name, t in THRESHOLDS.items()},
        "signal_levels": levels,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    (out / "RESULTS.md").write_text(to_markdown(result))
    print(to_markdown(result))
    print(f"Saved to {out / 'results.json'} and {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
