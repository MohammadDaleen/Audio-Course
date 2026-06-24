"""
Hugging Face Audio Course - Unit 4: pre-trained models for audio classification
================================================================================

Unit 4's "classification_models" page surveys ready-to-use audio classifiers via the
🤗 Transformers `pipeline()`. This script runs all three on CPU:

 1. Keyword spotting          - spot a command word (Audio Spectrogram Transformer)
 2. Zero-shot classification  - classify against arbitrary text labels (CLAP)
 3. Language identification   - identify the spoken language (Whisper-medium / FLEURS)

The music-genre *fine-tuning* lives in `finetune.py`; the genre demo in `gradio_demo.py`.

Run with:

    uv run python units/unit4_music_genre_classifier/walkthrough.py

First run downloads ~2.85 GB of models to the Hugging Face cache (not the repo):
AST speech-commands (~350 MB), CLAP (~1 GB), and whisper-medium LID (~1.5 GB).
Everything runs on CPU; the language-ID step (whisper-medium) is slow (tens of seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   saved {path.relative_to(Path(__file__).parent)}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _cached_speech_clip():
    """A cached English speech clip (no extra download)."""
    from datasets import load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    a = ds[0]["audio"]
    return a["array"], a["sampling_rate"]


# ===========================================================================
# 1. KEYWORD SPOTTING
# ===========================================================================
def demo_keyword_spotting():
    from transformers import pipeline

    banner("1. Keyword spotting (Audio Spectrogram Transformer)")
    classifier = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")

    # Prefer a real single-word command from Speech Commands (streamed, no full download);
    # fall back to a cached speech clip if streaming is unavailable.
    array, sr, note = None, None, ""
    try:
        from datasets import load_dataset

        sc = load_dataset(
            "google/speech_commands", "v0.02", split="validation",
            streaming=True, trust_remote_code=True,
        )
        ex = next(iter(sc))
        array, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
        true = sc.features["label"].int2str(ex["label"]) if hasattr(sc, "features") else "?"
        note = f" (true command: {true!r})"
    except Exception as exc:
        print(f"   (Speech Commands unavailable: {type(exc).__name__}; using a cached speech clip.)")
        array, sr = _cached_speech_clip()
        note = " (a full sentence, not a single command - expect '_unknown_'/low scores)"

    print(f"   classifying a clip{note}:")
    for p in classifier({"array": array, "sampling_rate": sr}, top_k=5):
        print(f"     {p['score']:.3f}  {p['label']}")


# ===========================================================================
# 2. ZERO-SHOT AUDIO CLASSIFICATION
# ===========================================================================
def demo_zero_shot():
    import librosa
    from transformers import pipeline

    banner("2. Zero-shot audio classification (CLAP)")
    classifier = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")

    # CLAP compares the audio against free-text labels you provide - no fixed class list.
    candidate_labels = [
        "a trumpet playing",
        "a person speaking",
        "a dog barking",
        "ocean waves",
        "rain falling",
    ]
    y, _ = librosa.load(librosa.ex("trumpet"), sr=48_000)  # CLAP expects 48 kHz
    # the zero-shot pipeline wants a raw numpy array (already at the model's 48 kHz), not a dict
    preds = classifier(y, candidate_labels=candidate_labels)
    print("   candidate labels scored against a trumpet clip:")
    labels, scores = [], []
    for p in preds:
        print(f"     {p['score']:.3f}  {p['label']}")
        labels.append(p["label"])
        scores.append(p["score"])

    plt.figure(figsize=(10, 4))
    plt.barh(labels[::-1], scores[::-1], color="tab:cyan")
    plt.xlim(0, 1)
    plt.xlabel("similarity score")
    plt.title("Zero-shot (CLAP): text labels vs a trumpet clip")
    save_fig("01_zero_shot_clap.png")


# ===========================================================================
# 3. LANGUAGE IDENTIFICATION
# ===========================================================================
def demo_language_id():
    from transformers import pipeline

    banner("3. Language identification (Whisper-medium, FLEURS) - slow on CPU")
    print("   loading whisper-medium LID (~1.5 GB); inference takes tens of seconds on CPU…")
    lid = pipeline("audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id")

    # This model is trained on FLEURS (clean read speech), so it is most reliable on clean
    # audio. A clean English clip identifies as English; noisy phone audio (e.g. MINDS-14)
    # is out-of-distribution and can be misclassified.
    array, sr = _cached_speech_clip()
    print("   classifying a clean English clip (expect 'English'):")
    for p in lid({"array": array, "sampling_rate": sr}, top_k=5):
        print(f"     {p['score']:.3f}  {p['label']}")


def main() -> None:
    print("Hugging Face Audio Course — Unit 4: pre-trained audio classification survey")
    print(f"Figures will be written to: {FIG_DIR}")
    demo_keyword_spotting()
    demo_zero_shot()
    demo_language_id()
    banner("Done!  Fine-tuning lives in finetune.py; the genre demo in gradio_demo.py.")


if __name__ == "__main__":
    main()
