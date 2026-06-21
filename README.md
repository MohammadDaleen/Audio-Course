# Hugging Face Audio Course — Unit 1: Working with Audio Data

A single, complete, **runnable** example that exercises *every* concept from
[Unit 1 of the HF Audio Course](https://huggingface.co/learn/audio-course/chapter1/introduction):
audio fundamentals, visualising audio, loading a 🤗 audio dataset, resampling,
filtering, feature extraction, and streaming.

It ships in two equivalent forms:

| File | What it is | When to use |
|------|------------|-------------|
| [`unit1_audio_data.ipynb`](unit1_audio_data.ipynb) | Annotated Jupyter notebook with inline plots + audio | **Recommended** — best for learning |
| [`unit1_audio_data.py`](unit1_audio_data.py) | The same code as a script that saves plots to `figures/` | If you don't run Jupyter |
| [`gradio_demo.py`](gradio_demo.py) | Optional mini web app to listen to dataset clips | Optional extra |

---

## 1. Setup (with [uv](https://docs.astral.sh/uv/))

This project is managed with **uv**. From this folder:

```bash
uv sync
```

That single command creates an isolated virtual environment, provisions Python
3.12 if needed, and installs everything (including a **CPU-only** build of
PyTorch — no GPU required). No manual `venv` activation is needed; run things
with `uv run` instead.

> **Why the pinned versions?** `datasets` is pinned to `3.6.0` on purpose. From
> `datasets` 4.0 the audio backend switched to `torchcodec` + FFmpeg, which is
> awkward to install on Windows. `3.6.0` uses the `soundfile`/`librosa` backend,
> so MINDS-14 (WAV) and LibriSpeech (FLAC) decode natively with no system-level
> codec install. **Don't upgrade `datasets` past 3.6.0.**

## 2. Run the example

**As a notebook (recommended):**

```bash
uv run jupyter lab
```

Then open `unit1_audio_data.ipynb` and choose *Run → Run All Cells*. Plots and
audible clips render inline.

**As a script:**

```bash
uv run python unit1_audio_data.py
```

All plots and a few `.wav` demo clips are written to the [`figures/`](figures/)
folder, with explanations printed to the console.

**Optional listening demo (local web app):**

```bash
uv run python gradio_demo.py
```

Opens at `http://127.0.0.1:7860`. (Windows may show a one-time firewall prompt;
the app stays fully local — nothing is uploaded.)

> **First run downloads a few small files:** the librosa "trumpet" example clip,
> the MINDS-14 `en-AU` split, and the Whisper feature-extractor *config* (a few
> KB — no model weights). These are cached for subsequent runs.

---

## 3. Concepts covered

| # | Concept | Where |
|---|---------|-------|
| 1 | Sampling & sampling rate (8 / 16 / 44.1 kHz) | §2 synthetic demo |
| 2 | Nyquist limit & aliasing | §2 (hear the alias) |
| 3 | Amplitude & decibels | §2 |
| 4 | Bit depth & dynamic range | §2 quantization demo |
| 5 | Waveform (time domain) | §3 `librosa.display.waveshow` |
| 6 | Frequency spectrum (DFT/FFT) | §4 `np.fft.rfft` |
| 7 | Spectrogram (STFT) | §5 `librosa.stft` |
| 8 | Mel / log-mel spectrogram | §6 `librosa.feature.melspectrogram` |
| 9 | Loading a 🤗 audio dataset | §7 `load_dataset("PolyAI/minds14", ...)` |
| 10 | The `Audio` feature (`array`, `sampling_rate`, `path`) | §7 |
| 11 | Listening to audio | §8 |
| 12 | Resampling | §9 `cast_column("audio", Audio(sampling_rate=16_000))` |
| 13 | Filtering by duration | §10 `dataset.filter(...)` |
| 14 | Feature-extractor preprocessing | §11 `WhisperFeatureExtractor` + `map` |
| 15 | Streaming large datasets | §12 `load_dataset(..., streaming=True)` |

---

## 4. The streaming section uses GigaSpeech (gated)

§12 follows the course exactly and streams `speechcolab/gigaspeech`. That dataset
is **gated** — it won't load until you request access:

1. Create a free account at <https://huggingface.co>.
2. Open <https://huggingface.co/datasets/speechcolab/gigaspeech>, accept the
   terms, and complete the access form (approval can take a little while).
3. Log in locally so your token is cached:
   ```bash
   uv run huggingface-cli login
   ```
   Paste a token from <https://huggingface.co/settings/tokens>.
4. Re-run the streaming section.

**Until access is granted, the example does not crash.** The streaming step is
wrapped in a `try/except` that prints these instructions and automatically falls
back to a small public dataset (`hf-internal-testing/librispeech_asr_dummy`), so
you can still see streaming in action.

---

## 5. Notes / corrections vs. the course text

- `librosa.get_duration(filename=...)` is deprecated → this example uses the
  current `path=` / `y=` arguments.
- The course's streaming auth (`use_auth_token=True`) is outdated → use
  `huggingface-cli login` instead.
- The course launches a Gradio server to listen; in the notebook we prefer the
  zero-setup `IPython.display.Audio` and keep the Gradio app as an optional extra.
