# Hugging Face Audio Course — worked examples

Complete, runnable examples for the [Hugging Face Audio Course](https://huggingface.co/learn/audio-course),
one self-contained folder per unit. Everything runs on CPU (Windows-friendly) and is managed with
[uv](https://docs.astral.sh/uv/).

## Units

| Unit | Folder | Covers |
|------|--------|--------|
| 1 — Working with audio data | [`units/unit1_working_with_audio_data/`](units/unit1_working_with_audio_data/) | Sampling, Nyquist, dB, bit depth, waveform, spectrum, spectrogram, mel; loading/resampling/filtering a 🤗 dataset; feature extraction; streaming |
| 2 — A gentle introduction to audio applications | [`units/unit2_audio_applications/`](units/unit2_audio_applications/) | The `pipeline()` function for audio classification, ASR, and audio generation (TTS + music); VoxPopuli streaming hands-on |
| 3 — Transformer architectures for audio | [`units/unit3_transformer_architectures/`](units/unit3_transformer_architectures/) | Runnable demos of the architecture families: waveform vs log-mel inputs, CTC blank-collapse decoding (Wav2Vec2), seq2seq task tokens + translation (Whisper), and spectrogram-patch classification (AST) |

Each unit folder has the same shape:

```
units/unitN_*/
├── walkthrough.py    # runnable script; saves plots/clips to figures/
├── notebook.ipynb    # the same material with inline plots + audio
├── gradio_demo.py    # optional local demo
├── figures/          # generated outputs (git-ignored)
└── README.md         # unit-specific setup, run steps, and notes
```

## Setup

From the repo root:

```bash
uv sync
```

This creates one shared virtual environment (Python 3.12, CPU-only PyTorch) used by every unit.
No per-unit install is needed.

> **Unit 2 audio generation** (Bark TTS + MusicGen) is multi-GB and slow on CPU, so it is an
> optional extra. Install it only when you want to run the generation demo:
> ```bash
> uv sync --extra generation
> ```

## Running an example

```bash
# scripts
uv run python units/unit1_working_with_audio_data/walkthrough.py
uv run python units/unit2_audio_applications/walkthrough.py

# notebooks
uv run jupyter lab        # then open any units/*/notebook.ipynb
```

See each unit's own `README.md` for details, concept coverage, and model/download notes.

## Why these dependency pins?

`datasets` is pinned to `3.6.0` on purpose: from 4.0 it switched its audio backend to
`torchcodec` + FFmpeg, which is awkward on Windows. `3.6.0` uses the `soundfile`/`librosa`
backend, so audio decodes natively with no system codecs. PyTorch is pulled from the CPU-only
wheel index so no multi-GB CUDA build is ever downloaded. **Do not bump `datasets` past 3.6.0.**
