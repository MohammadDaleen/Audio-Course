# Unit 2 — A gentle introduction to audio applications

Worked example for [Unit 2](https://huggingface.co/learn/audio-course/chapter2/introduction) of the
Hugging Face Audio Course. It shows how to solve audio tasks with pre-trained models using the
🤗 Transformers `pipeline()` function.

| File | What it is |
|------|------------|
| [`notebook.ipynb`](notebook.ipynb) | Annotated notebook with inline audio + plots (recommended) |
| [`walkthrough.py`](walkthrough.py) | Script: classification + ASR + the VoxPopuli hands-on; saves plots to `figures/` |
| [`generation_demo.py`](generation_demo.py) | **Optional** audio generation (Bark TTS + MusicGen) |
| [`gradio_demo.py`](gradio_demo.py) | **Optional** local app: a random MINDS-14 clip → predicted intent + transcription |

## Setup & run

From the repo root (the env is shared across units):

```bash
uv sync
uv run python units/unit2_audio_applications/walkthrough.py
# or:  uv run jupyter lab   then open notebook.ipynb
```

The audio generation parts (Bark + MusicGen) are heavy and slow on CPU, so they are an optional extra:

```bash
uv sync --extra generation
uv run python units/unit2_audio_applications/generation_demo.py
```

## Concepts covered

| Concept | Pipeline task | Model |
|---------|---------------|-------|
| Audio classification | `audio-classification` | `anton-l/xtreme_s_xlsr_300m_minds14` (~1.2 GB) |
| Automatic speech recognition | `automatic-speech-recognition` | default `facebook/wav2vec2-base-960h` (~360 MB); non-English example `maxidl/wav2vec2-large-xlsr-german` |
| Text-to-speech | `text-to-speech` | `suno/bark-small` (~2 GB, optional) |
| Music generation | `text-to-audio` | `facebook/musicgen-small` (~2 GB, optional) |
| Streaming hands-on | `automatic-speech-recognition` on `facebook/voxpopuli` (streaming) | (reuses the ASR model) |

## Notes

- Models download to the Hugging Face cache (`~/.cache/huggingface`), not this repo. First run only.
- The classification and ASR models expect **16 kHz** audio; MINDS-14 is resampled before use.
- `wav2vec2-base-960h` outputs **UPPERCASE text with no punctuation** — that is expected for a
  LibriSpeech-trained CTC model, not a bug.
- Everything runs on CPU. Generation is slow (tens of seconds to minutes per clip); token counts
  are kept small in the demo.
- A Windows "symlinks not supported" cache warning may appear — harmless.
