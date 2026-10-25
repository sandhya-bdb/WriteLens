import logging
import math
import re
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Module-level logger — all messages flow to the root logger configured
# by the application entry point (uvicorn / pytest), so no basicConfig here.
logger = logging.getLogger(__name__)

# ── Input validation constants ───────────────────────────────────────────────
MIN_TOKENS = 5       # GPT-2 needs at least a few tokens to compute a loss
MAX_CHARS  = 8000    # ~2 048 GPT-2 tokens; avoids OOM on long inputs
MIN_SENTENCE_CHARS = 10  # shorter fragments give unstable perplexity


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? and drop very short fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > MIN_SENTENCE_CHARS]


class AIDetector:
    def __init__(self, model_id: str = "gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading GPT-2 detector model (device=%s)…", self.device)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        logger.info("GPT-2 detector model loaded.")

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self, text: str) -> None:
        """
        Raise ValueError for inputs that would produce meaningless or
        crashy results from the GPT-2 forward pass.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text must not be empty.")
        if len(stripped) > MAX_CHARS:
            raise ValueError(
                f"Input text is too long ({len(stripped)} chars). "
                f"Maximum allowed is {MAX_CHARS} characters."
            )
        token_count = len(self.tokenizer.encode(stripped))
        if token_count < MIN_TOKENS:
            raise ValueError(
                f"Input text is too short ({token_count} tokens). "
                f"Provide at least {MIN_TOKENS} tokens for a reliable score."
            )

    # ── Core scoring ─────────────────────────────────────────────────────────

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculates the perplexity of the text using GPT-2.
        Lower perplexity = more likely AI.
        Higher perplexity = more unpredictable / human.
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    def score_sentences(self, text: str) -> list[dict]:
        """
        Perplexity for each sentence, in order.
        Sentences that can't be scored (too short for GPT-2) are skipped.

        Returns: [{"text": str, "perplexity": float}, ...]
        """
        scored = []
        for sentence in split_sentences(text):
            try:
                ppl = self.calculate_perplexity(sentence)
                scored.append({"text": sentence, "perplexity": round(ppl, 2)})
            except (ValueError, RuntimeError) as exc:
                # Short sentence can't be scored — skip, don't crash
                logger.debug("Skipping sentence (%s): %r", exc, sentence[:40])
        return scored

    def calculate_burstiness(self, text: str, sentence_scores: list[dict] | None = None) -> float:
        """
        Calculates burstiness (std-dev of sentence-level perplexities).
        Higher burstiness = more varied sentence complexity = more human.
        Pass sentence_scores to reuse an earlier score_sentences() result.
        """
        if sentence_scores is None:
            sentence_scores = self.score_sentences(text)
        if not sentence_scores:
            logger.debug("No sentences could be scored; burstiness = 0.")
            return 0.0
        return float(np.std([s["perplexity"] for s in sentence_scores]))

    def analyze(self, text: str) -> dict:
        """
        Full analysis: validate input, compute perplexity + burstiness,
        derive an ai_score heuristic in [0, 100].

        Returns:
            {
                "perplexity": float,
                "burstiness": float,
                "ai_score":   int,   # 0 = few AI-like signals, 100 = strong signals
                "sentences":  [{"text": str, "perplexity": float}, ...],
            }

        Raises:
            ValueError: if text is empty, too short, or too long.
            TypeError:  if text is not a str.
        """
        self._validate(text)

        ppl        = self.calculate_perplexity(text)
        sentences  = self.score_sentences(text)
        burstiness = self.calculate_burstiness(text, sentence_scores=sentences)

        logger.info(
            "Analysis complete — perplexity=%.2f  burstiness=%.2f",
            ppl, burstiness,
        )

        # Heuristic scoring
        # GPT-2 PPL for AI text typically 10–50; human text 50–200+
        ai_score = 0
        if ppl < 30:
            ai_score += 60
        elif ppl < 50:
            ai_score += 40
        else:
            ai_score += 10

        if burstiness < 10:
            ai_score += 30

        return {
            "perplexity": round(ppl, 2),
            "burstiness": round(burstiness, 2),
            "ai_score":   min(ai_score, 100),
            "sentences":  sentences,
        }
