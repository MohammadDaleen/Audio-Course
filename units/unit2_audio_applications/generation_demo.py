"""
Unit 2 (OPTIONAL): audio generation with a pipeline - text-to-speech + music.

This is the heavy, optional part of Unit 2. The models are multi-GB and generation
is SLOW on CPU (tens of seconds to minutes per clip). Install the extra first:

    uv sync --extra generation
    uv run python units/unit2_audio_applications/generation_demo.py

Models (downloaded to the Hugging Face cache on first run, not the repo):
  - suno/bark-small        text-to-speech   (~2 GB)
  - facebook/musicgen-small  music generation (~2 GB)

Generated clips are written as .wav into this unit's figures/ folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import soundfile as sf

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def save_wav(name: str, audio, sampling_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    path = OUT_DIR / name
    sf.write(path, audio, sampling_rate)
    print(f"   saved {path.name}  ({len(audio) / sampling_rate:.1f}s @ {sampling_rate} Hz)")


def text_to_speech() -> None:
    from transformers import pipeline

    print("\n[TTS] Loading suno/bark-small (~2 GB on first run)…")
    pipe = pipeline("text-to-speech", model="suno/bark-small")

    text = "Hello! This speech was generated on a CPU with the Bark small model."
    print("   generating speech (this is slow on CPU)…")
    out = pipe(text)
    save_wav("11_tts_bark.wav", out["audio"], out["sampling_rate"])


def text_to_music() -> None:
    from transformers import pipeline

    print("\n[MUSIC] Loading facebook/musicgen-small (~2 GB on first run)…")
    try:
        music = pipeline("text-to-audio", model="facebook/musicgen-small")
    except Exception as exc:
        print(f"   could not load MusicGen: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        print("   MusicGen's tokenizer needs the generation extra: uv sync --extra generation")
        return

    text = "90s rock song with electric guitar and heavy drums"
    # max_new_tokens kept small so a clip finishes in reasonable time on CPU.
    print("   generating music (this is slow on CPU)…")
    out = music(text, forward_params={"max_new_tokens": 256})
    save_wav("12_music_musicgen.wav", out["audio"][0], out["sampling_rate"])


def main() -> None:
    print("Unit 2 (optional): audio generation - text-to-speech + music")
    print(f"Output .wav files go to: {OUT_DIR}")
    print("Reminder: heavy models, slow on CPU. Requires `uv sync --extra generation`.")
    text_to_speech()
    text_to_music()
    print("\nDone! Listen to the .wav files in figures/.")


if __name__ == "__main__":
    main()
