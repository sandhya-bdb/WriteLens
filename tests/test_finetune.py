"""
test_finetune.py: tests for the pure parts of the fine-tuning pipeline
(data split and score mapping). No GPU, no model download.
"""
from logic.finetuned_detector import ai_probability_to_score
from training.finetune_detector import build_training_set, heldout_texts, training_rows


def _row(i):
    return {
        "human_answers": [f"Human answer {i} part one. It has a second part. And a third part too."],
        "chatgpt_answers": [f"Model answer {i} part one. It has a second part. And a third part too."],
    }


ROWS = [_row(i) for i in range(60)]


class TestHeldOut:
    def test_heldout_matches_eval_sample_size(self):
        assert len(heldout_texts(ROWS, n=5, seed=42)) == 10

    def test_training_rows_exclude_eval_questions(self):
        heldout = heldout_texts(ROWS, n=5, seed=42)
        clean = training_rows(ROWS, heldout)
        assert len(clean) == len(ROWS) - 5
        for row in clean:
            for a in row["human_answers"] + row["chatgpt_answers"]:
                assert a not in heldout


class TestTrainingSet:
    def test_balanced_and_no_overlap_with_eval(self):
        heldout = heldout_texts(ROWS, n=5, seed=42)
        train, val = build_training_set(training_rows(ROWS, heldout), per_class=20, seed=7)
        labels = [y for _, y in train]
        assert labels.count(0) == labels.count(1) == 20
        assert not any(t in heldout for t, _ in train + val)

    def test_train_and_val_do_not_share_texts(self):
        train, val = build_training_set(ROWS, per_class=20, seed=7)
        assert not set(t for t, _ in train) & set(t for t, _ in val)
        assert len(val) > 0


class TestScoreMapping:
    def test_bounds(self):
        assert ai_probability_to_score(0.0) == 0
        assert ai_probability_to_score(1.0) == 100

    def test_clamps_out_of_range(self):
        assert ai_probability_to_score(1.7) == 100
        assert ai_probability_to_score(-0.2) == 0

    def test_rounding(self):
        assert ai_probability_to_score(0.704) == 70


class TestMakeTrainingArgs:
    def test_drops_unknown_and_maps_old_eval_name(self):
        from training.finetune_detector import make_training_args

        class OldArgs:
            def __init__(self, output_dir, evaluation_strategy="no", learning_rate=1e-4):
                self.output_dir, self.evaluation_strategy, self.learning_rate = output_dir, evaluation_strategy, learning_rate

        a = make_training_args(OldArgs, output_dir="x", eval_strategy="epoch",
                               learning_rate=2e-4, warmup_ratio=0.05)
        assert a.evaluation_strategy == "epoch"
        assert a.learning_rate == 2e-4
