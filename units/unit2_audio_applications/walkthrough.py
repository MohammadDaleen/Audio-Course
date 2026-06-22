"""
Hugging Face Audio Course - Unit 2: A gentle introduction to audio applications
===============================================================================

A complete, runnable walkthrough of Unit 2's core concepts using the 🤗 Transformers
`pipeline()` function:

 1. Audio classification  - predict the intent of a spoken request (MINDS-14)
 2. Automatic speech recognition (ASR) - transcribe speech to text
 3. Hands-on - transcribe a streamed VoxPopuli example and compare to the reference

Audio *generation* (text-to-speech with Bark + music with MusicGen) is the heavier,
optional part of Unit 2 and lives in `generation_demo.py` (needs `uv sync --extra generation`).

Run with:

    uv run python units/unit2_audio_applications/walkthrough.py

First run downloads two models to the Hugging Face cache (not the repo):
the audio-classification model (~1.2 GB) and the default ASR model (~360 MB).
Everything runs on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters like "→".
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save figures, never block
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure into figures/ and close it."""
    path = FIG_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   saved {path.relative_to(Path(__file__).parent)}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_minds14(name: str = "en-AU", split: str = "train"):
    """Load MINDS-14 robustly whether it resolves as parquet or a script."""
    from datasets import load_dataset

    kwargs = dict(path="PolyAI/minds14", name=name, split=split)
    try:
        return load_dataset(**kwargs)
    except Exception:
        return load_dataset(**kwargs, trust_remote_code=True)


# ===========================================================================
# 1. AUDIO CLASSIFICATION
# ===========================================================================
def section_classification(minds):
    from transformers import pipeline

    banner("1. Audio classification with a pipeline")

    # The pipeline bundles preprocessing + model + post-processing. This model was
    # fine-tuned on MINDS-14 to predict the caller's intent. It expects 16 kHz audio,
    # which is why we resampled the dataset before calling it.
    print("\n[1] Loading the audio-classification pipeline (model ~1.2 GB on first run)…")
    classifier = pipeline(
        "audio-classification",
        model="anton-l/xtreme_s_xlsr_300m_minds14",
    )

    example = minds[0]
    id2label = minds.features["intent_class"].int2str
    true_label = id2label(example["intent_class"])

    preds = classifier(example["audio"]["array"], top_k=5)
    print(f"     true intent : {true_label}")
    print(f"     predicted   : {preds[0]['label']}  (score {preds[0]['score']:.3f})")
    print("     top-5:")
    for p in preds:
        print(f"        {p['score']:.3f}  {p['label']}")

    # bar chart of the top-5 scores
    labels = [p["label"] for p in preds][::-1]
    scores = [p["score"] for p in preds][::-1]
    plt.figure(figsize=(10, 4))
    colors = ["tab:green" if lab == true_label else "tab:blue" for lab in labels]
    plt.barh(labels, scores, color=colors)
    plt.xlabel("score")
    plt.title(f"Audio classification - top 5 (true intent: {true_label})")
    plt.xlim(0, 1)
    save_fig("01_classification_scores.png")


# ===========================================================================
# 2. AUTOMATIC SPEECH RECOGNITION
# ===========================================================================
def section_asr(asr, minds):
    banner("2. Automatic speech recognition with a pipeline")

    example = minds[0]
    result = asr(example["audio"]["array"])
    transcription = result["text"]

    print("\n[2] Default English ASR model (facebook/wav2vec2-base-960h):")
    print(f"     prediction  : {transcription}")
    # MINDS-14 ships a human transcription we can eyeball against.
    if example.get("english_transcription"):
        print(f"     reference   : {example['english_transcription']}")

    print(
        "\n     Note: wav2vec2-base-960h returns UPPERCASE text with no punctuation\n"
        "     (it is a CTC model trained on LibriSpeech). For another language you would\n"
        "     pass a matching model, e.g.:\n"
        "         pipeline('automatic-speech-recognition',\n"
        "                  model='maxidl/wav2vec2-large-xlsr-german')\n"
        "     (not run here to keep the download small)."
    )

    # waveform with the transcription as the title
    arr = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    t = np.arange(len(arr)) / sr
    plt.figure(figsize=(12, 3))
    plt.plot(t, arr, lw=0.5)
    plt.xlabel("time (s)")
    plt.title(f"ASR: \"{transcription[:80]}\"")
    save_fig("02_asr_waveform.png")


# ===========================================================================
# 3. HANDS-ON: transcribe a streamed VoxPopuli example
# ===========================================================================
def section_handson(asr):
    from datasets import load_dataset

    banner("3. Hands-on: transcribe a streamed VoxPopuli example")

    def load_voxpopuli_stream(lang="en"):
        kwargs = dict(path="facebook/voxpopuli", name=lang, split="train", streaming=True)
        try:
            return load_dataset(**kwargs)
        except Exception:
            return load_dataset(**kwargs, trust_remote_code=True)

    print("\n[3] Streaming facebook/voxpopuli (en) - no full download…")
    try:
        vp = load_voxpopuli_stream("en")
        example = next(iter(vp))  # pull a single example on the fly
    except Exception as exc:
        print("     Could not stream VoxPopuli:")
        print(f"       {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        print("     (It may require accepting terms / `huggingface-cli login`.) Skipping.")
        return

    audio = example["audio"]
    result = asr(audio["array"])
    reference = example.get("normalized_text") or example.get("raw_text") or "(none)"

    print(f"     sampling_rate : {audio['sampling_rate']} Hz")
    print(f"     pipeline text : {result['text']}")
    print(f"     reference     : {reference}")
    print(
        "\n     The pipeline output and the reference should match closely (modulo case\n"
        "     and punctuation). This is exactly the Unit 2 hands-on workflow."
    )


def main() -> None:
    from datasets import Audio

    print("Hugging Face Audio Course — Unit 2: A gentle introduction to audio applications")
    print(f"Figures will be written to: {FIG_DIR}")

    # MINDS-14, resampled to 16 kHz (what the classification and ASR models expect).
    minds = load_minds14()
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

    # Build the ASR pipeline once and reuse it for sections 2 and 3.
    from transformers import pipeline

    section_classification(minds)

    print("\nLoading the ASR pipeline (default model ~360 MB on first run)…")
    asr = pipeline("automatic-speech-recognition")
    section_asr(asr, minds)
    section_handson(asr)

    banner("Done!  See figures/ for the plots. Audio generation lives in generation_demo.py.")


if __name__ == "__main__":
    main()
