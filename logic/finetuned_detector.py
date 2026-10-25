"""
finetuned_detector.py: a small language model fine-tuned with QLoRA to classify
text as human-written (label 0) or AI-generated (label 1).

Trained by training/finetune_detector.py on HC3, with the evaluation answers
held out. Loads the base model plus a LoRA adapter folder.

This is a second opinion next to the GPT-2 heuristic in detector.py. Its
output is still a probability, not proof.
"""
import logging

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 256  # tokens; must match training


def ai_probability_to_score(prob: float) -> int:
    """Map P(AI) in [0, 1] to the same 0-100 scale the heuristic uses."""
    prob = min(max(float(prob), 0.0), 1.0)
    return int(round(prob * 100))


class FineTunedDetector:
    def __init__(self, adapter_path: str, base_model: str = DEFAULT_BASE_MODEL):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading fine-tuned detector %s + %s (device=%s)…", base_model, adapter_path, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base, adapter_path).to(self.device).eval()
        logger.info("Fine-tuned detector loaded.")

    def ai_probability(self, text: str) -> float:
        """Probability (0 to 1) that the text is AI-generated, according to the model."""
        enc = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits.float()
        return float(self.torch.softmax(logits, dim=-1)[0, 1].item())

    def analyze(self, text: str) -> dict:
        prob = self.ai_probability(text)
        return {"ai_probability": round(prob, 4), "ai_score": ai_probability_to_score(prob)}
