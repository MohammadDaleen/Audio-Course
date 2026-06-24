# Unit 4 — Build a music genre classifier

Worked example for [Unit 4](https://huggingface.co/learn/audio-course/chapter4/introduction) of the
Hugging Face Audio Course: surveying pre-trained audio classifiers and fine-tuning one (DistilHuBERT) on
the GTZAN music-genre dataset.

| File | What it is |
|------|------------|
| [`walkthrough.py`](walkthrough.py) | Pre-trained classification survey: keyword spotting, zero-shot (CLAP), language ID |
| [`finetune.py`](finetune.py) | Fine-tuning: a **CPU smoke test** (default) and a **full GPU/Colab** run |
| [`gradio_demo.py`](gradio_demo.py) | CPU genre classifier using the published fine-tuned model |
| [`notebook.ipynb`](notebook.ipynb) | The survey + genre demo inline; fine-tuning code shown, gated by a flag |
| [`colab_handson.ipynb`](colab_handson.ipynb) | **Ready-to-run Colab notebook for the hands-on**: fully-explained, fine-tunes to >=87% and pushes to the Hub (open it on a GPU runtime) |

## Setup & run

From the repo root (the env is shared across units):

```bash
uv sync
uv run python units/unit4_music_genre_classifier/walkthrough.py     # pre-trained survey
uv run python units/unit4_music_genre_classifier/finetune.py        # CPU smoke test
uv run python units/unit4_music_genre_classifier/gradio_demo.py     # genre classifier demo
```

> The survey downloads ~2.85 GB of models on first run (AST speech-commands ~350 MB, CLAP ~1 GB,
> whisper-medium LID ~1.5 GB). They cache to `~/.cache/huggingface`, not the repo. The language-ID
> step (whisper-medium) is slow on CPU (tens of seconds).

## Fine-tuning: smoke test (CPU) vs full run (GPU/Colab)

Training DistilHuBERT on GTZAN takes ~1 hour on a T4 GPU and is impractical on CPU, so `finetune.py` has
two modes:

- **smoke (default):** runs on CPU in a couple of minutes on a tiny synthetic dataset with a few steps,
  `fp16=False`, no Hub push, no GTZAN download. It proves the whole Trainer pipeline runs
  (data → features → model → loss → eval → accuracy). It does **not** train a real model.
- **full:** the course recipe — real GTZAN, `fp16=True`, 10 epochs, `push_to_hub=True`. Run it on a GPU
  (e.g. Google Colab):

  ```bash
  uv sync --extra training        # installs evaluate + tensorboard
  huggingface-cli login           # needed for push_to_hub
  UNIT4_MODE=full python units/unit4_music_genre_classifier/finetune.py
  ```

### The hands-on exercise

The Unit 4 hands-on asks you to fine-tune on GTZAN to **≥ 87% accuracy** (the course baseline is 83%) and
push the model to the Hub for your certificate. The easiest path is the fully-explained
[`colab_handson.ipynb`](colab_handson.ipynb): open it in Google Colab on a **GPU** runtime
(Runtime → Change runtime type → T4 GPU) and run all cells (~1 hour). It trains for 20 epochs, keeps the
best checkpoint, and pushes the model with the certificate metadata. Equivalently, `finetune.py` full
mode (`UNIT4_MODE=full`) runs the same recipe. To push past the baseline, try more epochs, a different
base model (Wav2Vec2, AST), or a learning-rate sweep. Full training is not feasible on this CPU machine.

## Concepts covered

| Concept | Task / approach | Model |
|---------|------------------|-------|
| Keyword spotting | `audio-classification` | `MIT/ast-finetuned-speech-commands-v2` |
| Zero-shot classification | `zero-shot-audio-classification` (free-text labels) | `laion/clap-htsat-unfused` |
| Language identification | `audio-classification` | `sanchit-gandhi/whisper-medium-fleurs-lang-id` |
| Fine-tuning a classifier | `Trainer` on GTZAN | `ntu-spml/distilhubert` |
| Using a fine-tuned model | `audio-classification` | `sanchit-gandhi/distilhubert-finetuned-gtzan` |

## Notes (CPU / Windows / current libraries)

- This example targets **transformers 4.57** and **gradio 6**, which changed several APIs the course
  predates: `eval_strategy` (not `evaluation_strategy`), `Trainer(processing_class=...)` (not
  `tokenizer=`), and `gr.Label` / `gr.Audio(type="filepath")` (not `gr.outputs.Label`).
- GTZAN loads via `load_dataset("marsyas/gtzan", trust_remote_code=True)` — there is no `"all"` config,
  and the ~1.2 GB download comes from a flaky academic host. The smoke test avoids it entirely.
- `fp16` is GPU-only; the smoke test forces it off.
- CLAP expects 48 kHz audio; the other models expect 16 kHz.
- A benign "symlinks not supported" cache warning may appear on Windows.
