"""
Hugging Face Audio Course - Unit 4: fine-tune a music genre classifier on GTZAN.

ONE module, TWO modes (env var UNIT4_MODE, or the --full flag), default = smoke:

  smoke (default)  Runs on CPU in a couple of minutes. Tiny SYNTHETIC dataset, a few
                   steps, fp16 OFF, push_to_hub OFF. It proves the Trainer pipeline runs
                   end to end (data -> features -> model -> loss -> eval -> accuracy).
                   It does NOT train a real model and does NOT download GTZAN.

  full             The course recipe for a GPU/Colab box: real GTZAN, fp16 ON, 10 epochs,
                   push_to_hub ON with the certificate metadata. ~1 hour on a T4 GPU;
                   many hours on CPU (don't). This is what you run to do the hands-on.

    uv run python units/unit4_music_genre_classifier/finetune.py                  # smoke (CPU)
    UNIT4_MODE=full uv run python units/unit4_music_genre_classifier/finetune.py  # full (Colab/GPU)

The full run needs the training extra:  uv sync --extra training
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
MODEL_ID = "ntu-spml/distilhubert"
SAMPLING_RATE = 16_000
MAX_DURATION = 30.0  # GTZAN clips are 30 s


def get_mode() -> str:
    if "--full" in sys.argv:
        return "full"
    return os.environ.get("UNIT4_MODE", "smoke").lower()


def load_gtzan(trust: bool = True):
    """marsyas/gtzan is a SCRIPT dataset (gtzan.py); datasets 3.6.0 needs
    trust_remote_code=True. There is NO "all" config (the course's
    load_dataset("marsyas/gtzan", "all") fails on this stack). First load
    downloads ~1.2 GB from a flaky academic host; jazz.00054.wav is auto-skipped
    inside the loading script."""
    from datasets import load_dataset

    try:
        return load_dataset("marsyas/gtzan", trust_remote_code=trust)
    except Exception:
        return load_dataset("marsyas/gtzan", trust_remote_code=not trust)


def smoke_dataset():
    """Tiny synthetic dataset (no download) purely to exercise the training loop."""
    from datasets import Audio, ClassLabel, Dataset, Features

    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        secs = 2.0  # short clips keep the smoke test fast
        y = (0.1 * rng.standard_normal(int(secs * SAMPLING_RATE))).astype("float32")
        rows.append({"audio": {"array": y, "sampling_rate": SAMPLING_RATE}, "label": i % len(GENRES)})
    feats = Features({"audio": Audio(sampling_rate=SAMPLING_RATE), "label": ClassLabel(names=GENRES)})
    ds = Dataset.from_list(rows, features=feats)
    split = ds.train_test_split(seed=42, test_size=0.25)
    return split["train"], split["test"]


def build_compute_metrics():
    """evaluate.load('accuracy') if installed, else a scikit-learn fallback so the
    smoke test runs without `uv sync --extra training`."""
    try:
        import evaluate

        metric = evaluate.load("accuracy")

        def compute(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(predictions=preds, references=eval_pred.label_ids)
    except Exception:
        from sklearn.metrics import accuracy_score

        def compute(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}

    return compute


def main() -> None:
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    mode = get_mode()
    on_gpu = torch.cuda.is_available()
    print(f"Unit 4 fine-tune | mode={mode} | cuda={on_gpu} | torch={torch.__version__}")
    if mode == "full" and not on_gpu:
        print("   WARNING: full mode on CPU is many hours. Run it on Colab/GPU instead.")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, do_normalize=True, return_attention_mask=True
    )

    if mode == "full":
        from datasets import Audio

        gtzan = load_gtzan()
        base = gtzan["train"].rename_column("genre", "label")
        split = base.train_test_split(seed=42, shuffle=True, test_size=0.1)
        train_ds = split["train"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        eval_ds = split["test"].cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    else:
        train_ds, eval_ds = smoke_dataset()

    id2label = {str(i): lab for i, lab in enumerate(GENRES)}
    label2id = {lab: str(i) for i, lab in enumerate(GENRES)}

    def preprocess(batch):
        arrays = [x["array"] for x in batch["audio"]]
        return feature_extractor(
            arrays,
            sampling_rate=SAMPLING_RATE,
            max_length=int(SAMPLING_RATE * MAX_DURATION),
            truncation=True,
        )

    train_ds = train_ds.map(preprocess, batched=True, batch_size=8, remove_columns=["audio"])
    eval_ds = eval_ds.map(preprocess, batched=True, batch_size=8, remove_columns=["audio"])

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=len(GENRES), label2id=label2id, id2label=id2label
    )

    if mode == "full":
        args = TrainingArguments(
            output_dir="distilhubert-finetuned-gtzan",
            eval_strategy="epoch",          # transformers 4.57: NOT evaluation_strategy
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            warmup_steps=100,
            logging_steps=5,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,                      # GPU only
            push_to_hub=True,
            report_to=["none"],
        )
    else:  # smoke
        args = TrainingArguments(
            output_dir=tempfile.mkdtemp(prefix="unit4_smoke_"),
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            max_steps=5,                    # a handful of steps is enough on CPU
            logging_steps=1,
            fp16=False,                     # MUST be False on CPU
            use_cpu=not on_gpu,
            push_to_hub=False,
            report_to=["none"],
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=feature_extractor,  # transformers 4.57: NOT tokenizer=
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    print("   eval metrics:", trainer.evaluate())

    if mode == "full":
        # Required for the hands-on certificate: push the trained model with metadata tags.
        push_kwargs = {
            "dataset_tags": "marsyas/gtzan",
            "dataset": "GTZAN",
            "model_name": "distilhubert-finetuned-gtzan",
            "finetuned_from": MODEL_ID,
            "tasks": "audio-classification",
        }
        trainer.push_to_hub(**push_kwargs)
        print("   pushed to the Hub with certificate tags:", push_kwargs)
        print("   hands-on target: >= 87% accuracy (course baseline is 83%).")
    else:
        print("   SMOKE TEST OK — the Trainer pipeline runs. For a real model, run full mode on GPU/Colab.")


if __name__ == "__main__":
    main()
