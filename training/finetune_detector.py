"""
finetune_detector.py: fine-tune a small LLM with QLoRA to classify text as
human-written (0) or AI-generated (1), using the HC3 dataset.

    base model : Qwen/Qwen2.5-0.5B (sequence classification head)
    method     : QLoRA, 4-bit NF4 base weights + LoRA adapters (r=16)
    data       : HC3 human vs ChatGPT answers
    held out   : every HC3 question used by scripts/evaluate.py (seed 42) is
                 removed from training, so the comparison with the GPT-2
                 heuristic is fair

Runs on a free Colab T4 GPU. Usage:
    python training/finetune_detector.py                     # defaults
    python training/finetune_detector.py --per-class 3000 --epochs 1

Output: a LoRA adapter folder (default models/writelens-qwen-lora) plus
training_metrics.json inside it.
"""
import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import build_samples, load_hc3, usable  # noqa: E402

EVAL_N = 200      # must match the evaluation run in eval/results.json
EVAL_SEED = 42
MAX_LENGTH = 256  # must match logic/finetuned_detector.py


# ── Data split (pure Python, unit-tested) ─────────────────────────────────────

def heldout_texts(rows: list[dict], n: int = EVAL_N, seed: int = EVAL_SEED) -> set[str]:
    """Exact texts the evaluation script scores."""
    return {text for text, _ in build_samples(rows, n, seed)}


def training_rows(rows: list[dict], heldout: set[str]) -> list[dict]:
    """Drop every question that has any answer in the evaluation set."""
    def touches_eval(row):
        answers = row.get("human_answers", []) + row.get("chatgpt_answers", [])
        return any(a.strip() in heldout for a in answers)
    return [r for r in rows if not touches_eval(r)]


def build_training_set(rows: list[dict], per_class: int, seed: int, val_fraction: float = 0.1):
    """Balanced (text, label) pairs, split into train and validation by question."""
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n_val_rows = max(1, int(len(rows) * val_fraction))
    val_rows, train_rows = rows[:n_val_rows], rows[n_val_rows:]

    def collect(source, limit):
        human, ai = [], []
        for row in source:
            for a in row.get("human_answers", []):
                if usable(a) and len(human) < limit:
                    human.append((a.strip(), 0))
            for a in row.get("chatgpt_answers", []):
                if usable(a) and len(ai) < limit:
                    ai.append((a.strip(), 1))
            if len(human) >= limit and len(ai) >= limit:
                break
        out = human + ai
        rng.shuffle(out)
        return out

    val_limit = max(1, int(per_class * val_fraction))
    return collect(train_rows, per_class), collect(val_rows, val_limit)


# ── Training (needs GPU libraries) ────────────────────────────────────────────

def make_training_args(cls, **kwargs):
    """
    Build TrainingArguments across transformers versions: older releases call
    eval_strategy "evaluation_strategy", and unsupported keys are dropped
    with a warning instead of crashing the run.
    """
    params = inspect.signature(cls.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in params]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"Note: this transformers version ignores {dropped}")
    return cls(**kwargs)

def train(args) -> None:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    rows = load_hc3(args.data_file)
    heldout = heldout_texts(rows)
    clean_rows = training_rows(rows, heldout)
    train_set, val_set = build_training_set(clean_rows, args.per_class, args.seed)
    leak = sum(1 for t, _ in train_set + val_set if t in heldout)
    print(f"HC3 questions: {len(rows)} | after removing eval questions: {len(clean_rows)}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | overlap with eval set: {leak}")
    assert leak == 0, "Training data overlaps the evaluation set"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, quantization_config=bnb, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS",
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, pairs):
            self.enc = tokenizer([t for t, _ in pairs], truncation=True, max_length=MAX_LENGTH)
            self.labels = [y for _, y in pairs]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        return {
            "accuracy": (tp + tn) / max(1, len(labels)),
            "recall": tp / max(1, tp + fn),
            "false_positive_rate": fp / max(1, fp + tn),
        }

    out_dir = Path(args.output_dir)
    grad_accum = 2
    steps_per_epoch = math.ceil(len(train_set) / (args.batch_size * grad_accum))
    warmup_steps = max(1, int(0.05 * steps_per_epoch * args.epochs))
    trainer = Trainer(
        model=model,
        args=make_training_args(
            TrainingArguments,
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=args.lr,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            fp16=torch.cuda.is_available(),
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=TextDataset(train_set),
        eval_dataset=TextDataset(val_set),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("Validation:", metrics)

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "training_metrics.json").write_text(json.dumps({
        "base_model": args.base_model,
        "method": f"QLoRA (4-bit NF4, LoRA r={args.lora_r})",
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_length": MAX_LENGTH,
        "heldout_eval_texts": len(heldout),
        "validation": {k: round(float(v), 4) for k, v in metrics.items()},
    }, indent=2))
    print(f"Adapter saved to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--per-class", type=int, default=2000, help="training answers per class")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-file", help="local path to HC3 all.jsonl")
    p.add_argument("--output-dir", default=str(ROOT / "models" / "writelens-qwen-lora"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
