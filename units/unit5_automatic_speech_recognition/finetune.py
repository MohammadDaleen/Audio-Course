"""
Hugging Face Audio Course - Unit 5: fine-tune Whisper for speech recognition.

ONE module, TWO modes (env var UNIT5_MODE, or the --full flag), default = smoke:

  smoke (default)  Runs on CPU in a couple of minutes. Tiny SYNTHETIC dataset, a few
                   steps, fp16 OFF, push_to_hub OFF. It proves the whole Seq2SeqTrainer
                   pipeline runs end to end (audio -> log-mel -> labels -> collator ->
                   loss -> generate -> WER). It does NOT train a usable model and does
                   NOT download MINDS-14. The WER it prints is meaningless (expect a
                   number above 1.0, which is section 6 of the walkthrough in action).

  full             The hands-on recipe for a GPU/Colab box: openai/whisper-tiny on the
                   American-English subset of PolyAI/minds14, first 450 examples for
                   training and the rest for evaluation, fp16 ON, push_to_hub ON with
                   the certificate metadata. ~25 minutes on a T4; many hours on CPU.

    uv run python units/unit5_automatic_speech_recognition/finetune.py                  # smoke (CPU)
    UNIT5_MODE=full uv run python units/unit5_automatic_speech_recognition/finetune.py  # full (Colab/GPU)

The full run needs the training extra:  uv sync --extra training

The course's own chapter fine-tunes openai/whisper-small on Common Voice 13 Dhivehi and
reaches 14.1% WER in 500 steps:

    MODEL_ID = "openai/whisper-small"
    DATASET_ID, DATASET_CONFIG = "mozilla-foundation/common_voice_13_0", "dv"
    LANGUAGE = "sinhalese"   # Dhivehi is unsupported; Sinhalese is the closest proxy

That dataset is GATED - you must accept its terms on the Hub and be logged in - so this
module uses the hands-on's dataset instead, which is ungated and is what you are graded on.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

MODEL_ID = "openai/whisper-tiny"
DATASET_ID = "PolyAI/minds14"
DATASET_CONFIG = "en-US"       # the hands-on says the American English subset
LANGUAGE = "english"
TASK = "transcribe"
SAMPLING_RATE = 16_000
MAX_DURATION = 30.0            # Whisper's encoder window is exactly 30 s
N_TRAIN = 450                  # the hands-on: first 450 examples train, rest evaluate
PASS_WER = 0.37                # the hands-on passes if normalised WER is below this


def get_mode() -> str:
    if "--full" in sys.argv:
        return "full"
    return os.environ.get("UNIT5_MODE", "smoke").lower()


def build_processor():
    """Load the processor and - importantly - force the language/task prefix tokens.

    On transformers 4.57 the FAST tokenizer (the default) ignores the language and
    task given to from_pretrained() when it encodes text: labels come out as
    <|startoftranscript|><|notimestamps|>, silently missing <|en|><|transcribe|>.
    Generation still forces those tokens, so you would train on one prompt format
    and decode with another. set_prefix_tokens() puts them back.
    """
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    return processor


def load_minds14_split():
    """MINDS-14 en-US, first N_TRAIN rows for training and the remainder for eval.

    `min(N_TRAIN, 80%)` guards the split: if the config has fewer rows than the
    course assumes, select(range(450)) would raise IndexError after you have already
    paid for the download.
    """
    from datasets import Audio, DatasetDict, load_dataset

    kwargs = dict(path=DATASET_ID, name=DATASET_CONFIG, split="train")
    try:
        ds = load_dataset(**kwargs)
    except Exception:
        ds = load_dataset(**kwargs, trust_remote_code=True)

    # MINDS-14 calls the text column "transcription", not "sentence".
    ds = ds.select_columns(["audio", "transcription"]).rename_column("transcription", "sentence")
    # MINDS-14 is 8 kHz telephone audio; Whisper needs 16 kHz. Cast BEFORE the map, or
    # transformers raises ImportError("torchaudio is required to resample").
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    n_train = min(N_TRAIN, int(len(ds) * 0.8))
    print(f"   {DATASET_ID} [{DATASET_CONFIG}]: {len(ds)} rows "
          f"-> {n_train} train / {len(ds) - n_train} eval")
    return DatasetDict(
        train=ds.select(range(n_train)),
        test=ds.select(range(n_train, len(ds))),
    )


def smoke_dataset():
    """Tiny synthetic dataset (no download) purely to exercise the training loop."""
    from datasets import Audio, Dataset, DatasetDict, Features, Value

    rng = np.random.default_rng(0)
    words = ["hello", "world", "please", "check", "my", "account", "balance", "today"]
    rows = []
    for i in range(8):
        y = (0.05 * rng.standard_normal(int(2.0 * SAMPLING_RATE))).astype("float32")
        rows.append({
            "audio": {"array": y, "sampling_rate": SAMPLING_RATE},
            "sentence": " ".join(words[i % len(words):] + words[: i % len(words)][:3]),
        })
    feats = Features({"audio": Audio(sampling_rate=SAMPLING_RATE), "sentence": Value("string")})
    ds = Dataset.from_list(rows, features=feats)
    return DatasetDict(train=ds, test=ds)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Batch a list of examples. Audio and labels need DIFFERENT padding treatment.

    Whisper pads every clip to exactly 30 s, so `input_features` are already uniform
    and only need stacking. Labels are variable-length, so they get padded and the
    padding is then marked with -100 (torch's cross_entropy ignore_index) so it
    contributes no loss.
    """

    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"][0]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # The model prepends the start token itself when it shifts labels right, so
        # drop it here if the tokenizer already added one.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_compute_metrics(processor):
    """Return the hands-on's two metrics as FRACTIONS (the course multiplies by 100;
    the hands-on explicitly asks you not to).

      wer_ortho - orthographic: compare the strings exactly as decoded
      wer       - normalised: lowercase, strip punctuation, then compare

    Uses evaluate.load("wer") when the training extra is installed, and falls back to
    jiwer (a core dependency) otherwise - the same idiom unit4's finetune.py uses.
    """
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    try:
        import evaluate

        metric = evaluate.load("wer")

        def _wer(refs, hyps):
            return metric.compute(references=refs, predictions=hyps)
    except Exception:
        import jiwer

        def _wer(refs, hyps):
            return jiwer.wer(refs, hyps)

    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids

        # -100 was our "ignore this" marker; put the real pad token back so the
        # tokenizer can decode the sequence.
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer_ortho = _wer(label_str, pred_str)

        pred_norm = [normalizer(p) for p in pred_str]
        label_norm = [normalizer(l) for l in label_str]
        # WER divides by the number of reference words, so drop any reference that
        # normalised away to an empty string - it would divide by zero.
        keep = [i for i in range(len(label_norm)) if len(label_norm[i]) > 0]
        if not keep:
            return {"wer_ortho": wer_ortho, "wer": wer_ortho}
        wer = _wer([label_norm[i] for i in keep], [pred_norm[i] for i in keep])

        return {"wer_ortho": wer_ortho, "wer": wer}

    return compute_metrics


def main() -> None:
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
    )

    mode = get_mode()
    on_gpu = torch.cuda.is_available()
    print(f"Unit 5 fine-tune | mode={mode} | cuda={on_gpu} | torch={torch.__version__}")
    if mode == "full" and not on_gpu:
        print("   WARNING: full mode on CPU is many hours. Run it on Colab/GPU instead.")

    processor = build_processor()
    print(f"   label prefix: "
          f"{processor.tokenizer.convert_ids_to_tokens(processor.tokenizer.prefix_tokens)}")

    dataset = load_minds14_split() if mode == "full" else smoke_dataset()

    def prepare_dataset(example):
        audio = example["audio"]
        out = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["sentence"],
        )
        out["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return out

    # num_proc=1 is required by the hands-on (and is the only safe setting on Windows).
    dataset = dataset.map(
        prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1
    )
    dataset = dataset.filter(lambda length: length < MAX_DURATION, input_columns=["input_length"])
    dataset = dataset.remove_columns(["input_length"])
    print(f"   prepared: {len(dataset['train'])} train / {len(dataset['test'])} eval")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    # Gradient checkpointing recomputes activations instead of storing them, which is
    # incompatible with the KV cache during training...
    model.config.use_cache = False
    # ...but generation-based evaluation still wants the cache, or every eval pass runs
    # roughly twice as slow.
    model.generation_config.use_cache = True
    # This is the modern replacement for the course's
    # `model.generate = partial(model.generate, language=..., task=...)`. A monkey-patched
    # generate() is not saved by push_to_hub, so the uploaded model would forget its language.
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.num_beams = 1
    # A no-op on transformers 4.57 (setting language/task already skips the legacy
    # branch) but harmless, and it neutralises a stale generation_config.json.
    model.generation_config.forced_decoder_ids = None

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    if mode == "full":
        args = Seq2SeqTrainingArguments(
            output_dir="whisper-tiny-finetuned-minds14-en",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=50,
            max_steps=600,
            gradient_checkpointing=True,
            fp16=True,                      # GPU only
            fp16_full_eval=True,
            eval_strategy="steps",          # transformers 4.57: NOT evaluation_strategy
            save_strategy="steps",          # must match eval_strategy for load_best_model_at_end
            # Evaluate every 50 rather than every 100 steps: WER on a 113-clip test set
            # is noisy and does not fall monotonically, so more checkpoints means a
            # better chance that load_best_model_at_end catches a good minimum.
            eval_steps=50,
            save_steps=50,
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,        # lower WER is better
            push_to_hub=True,
            report_to=["tensorboard"],
        )
    else:  # smoke
        args = Seq2SeqTrainingArguments(
            output_dir=tempfile.mkdtemp(prefix="unit5_smoke_"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-5,
            max_steps=4,                    # a handful of steps is enough on CPU
            eval_strategy="steps",
            eval_steps=4,
            save_strategy="no",
            logging_steps=1,
            gradient_checkpointing=False,
            fp16=False,                     # MUST be False on CPU
            use_cpu=not on_gpu,
            predict_with_generate=True,
            generation_max_length=32,
            push_to_hub=False,
            report_to=["none"],
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,         # transformers 4.57: NOT tokenizer=
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n   wer_ortho : {metrics.get('eval_wer_ortho', float('nan')):.4f}")
    print(f"   wer       : {metrics.get('eval_wer', float('nan')):.4f}")

    if mode == "full":
        passed = metrics.get("eval_wer", 1.0) < PASS_WER
        print(f"   hands-on threshold: wer < {PASS_WER}  ->  "
              f"{'PASS' if passed else 'NOT YET'}")
        if not passed:
            print("   Try max_steps=1000, or a larger base model (openai/whisper-base).")

        # Required for the hands-on certificate: push the trained model with metadata tags.
        push_kwargs = {
            "dataset_tags": DATASET_ID,
            "dataset": "MINDS-14 (en-US)",
            "language": "en",
            "model_name": "whisper-tiny-finetuned-minds14-en",
            "finetuned_from": MODEL_ID,
            "tasks": "automatic-speech-recognition",
        }
        trainer.push_to_hub(**push_kwargs)
        print("   pushed to the Hub with certificate tags:", push_kwargs)
    else:
        print("\n   SMOKE TEST OK — the Seq2SeqTrainer pipeline runs end to end.")
        print("   The WER above is meaningless (random audio, 4 steps) and will likely")
        print("   exceed 1.0 — which is exactly the walkthrough's point that WER is")
        print("   unbounded above. For a real model, run the hands-on on GPU/Colab.")


if __name__ == "__main__":
    main()
