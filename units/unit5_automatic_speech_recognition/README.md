# Unit 5 — Automatic speech recognition

Worked example for [Unit 5](https://huggingface.co/learn/audio-course/chapter5/introduction) of the
Hugging Face Audio Course: choosing between pre-trained ASR models, choosing a dataset, measuring a
transcription with WER, fine-tuning Whisper, and shipping a demo.

Unit 3 already covered the *mechanics* (CTC blank collapse, Whisper's task tokens). Unit 5 is about
**choosing, measuring and adapting**, so this folder leans on evaluation and fine-tuning rather than
re-explaining architectures.

| File | What it is |
|------|------------|
| [`walkthrough.py`](walkthrough.py) | The complete example: 8 sections from CTC-vs-Whisper through WER by hand to what the data collator builds |
| [`finetune.py`](finetune.py) | Fine-tuning Whisper: a **CPU smoke test** (default) and a **full GPU/Colab** run |
| [`gradio_demo.py`](gradio_demo.py) | Tabbed demo: microphone, file + timestamps, and translate-to-English |
| [`notebook.ipynb`](notebook.ipynb) | The same material inline with playable audio; fine-tuning shown but gated by a flag |
| [`colab_handson.ipynb`](colab_handson.ipynb) | **Ready-to-run Colab notebook for the hands-on**: fine-tunes `whisper-tiny` to WER < 0.37 and pushes to the Hub (open it on a GPU runtime) |

## Setup & run

From the repo root (the env is shared across units):

```bash
uv sync
uv run python units/unit5_automatic_speech_recognition/walkthrough.py   # the complete example
uv run python units/unit5_automatic_speech_recognition/finetune.py      # CPU smoke test
uv run python units/unit5_automatic_speech_recognition/gradio_demo.py   # transcription demo
```

> The walkthrough uses four models totalling ~1.77 GB on a cold machine: `wav2vec2-base-960h`
> (~360 MB), `whisper-tiny` (~150 MB), `whisper-base` (~290 MB) and `whisper-small` (~970 MB). If you
> did Units 2 and 3 they are **already cached**, so a run downloads nothing. They live in
> `~/.cache/huggingface`, not the repo. The whole script takes about 4–6 minutes on CPU; set
> `USE_WHISPER_SMALL = False` at the top to drop the largest model.

## Fine-tuning: smoke test (CPU) vs full run (GPU/Colab)

Fine-tuning Whisper takes ~25 minutes on a T4 and is impractical on CPU, so `finetune.py` has two
modes:

- **smoke (default):** runs on CPU in about a minute on a tiny synthetic dataset with 4 steps,
  `fp16=False`, no Hub push, no MINDS-14 download. It proves the whole `Seq2SeqTrainer` pipeline runs
  (audio → log-mel → labels → collator → loss → generate → WER). It does **not** train a usable
  model, and the WER it prints is meaningless.
- **full:** the hands-on recipe — real MINDS-14 en-US, `fp16=True`, 600 steps, `push_to_hub=True`.
  Run it on a GPU:

  ```bash
  uv sync --extra training        # installs evaluate + tensorboard
  huggingface-cli login           # needed for push_to_hub
  UNIT5_MODE=full python units/unit5_automatic_speech_recognition/finetune.py
  ```

### The hands-on exercise

The Unit 5 hands-on asks you to fine-tune `openai/whisper-tiny` on the American-English subset of
`PolyAI/minds14`, using the **first 450 examples for training** and the rest for evaluation, and to
reach a **normalised WER below 0.37** — reported as a *fraction*, not a percentage. The model must be
pushed to the Hub with the dataset/task tags for it to count toward your certificate.

The easiest path is [`colab_handson.ipynb`](colab_handson.ipynb): open it in Google Colab on a **GPU**
runtime (Runtime → Change runtime type → T4 GPU) and run all cells (~30 minutes). Equivalently,
`finetune.py` full mode runs the same recipe.

For calibration, measured here on 40 eval clips before any fine-tuning:

| Model | orthographic WER | normalised WER |
|---|---|---|
| `whisper-tiny` (the starting point) | 0.62 | **0.44** |
| `whisper-base` (bigger, for reference) | 0.57 | 0.39 |

So the exercise is real — the base model does *not* pass — but the gap to 0.37 is small. If you stall
above the threshold, raise `max_steps` to 1000 before reaching for a bigger model.

### Deploy the demo as a Space

The course's demo page finishes by suggesting you host it. To do that yourself: create a new
**Gradio** Space at [huggingface.co/new-space](https://huggingface.co/new-space), then add two files.

`requirements.txt`:

```
transformers>=4.46,<5
torch
librosa
```

`app.py` — the same code as [`gradio_demo.py`](gradio_demo.py) with your own checkpoint and a launch
line Spaces understands:

```python
MODEL_ID = "<your-username>/whisper-tiny-finetuned-minds14-en"
...
demo.launch()
```

A free CPU Space is enough for `whisper-tiny`.

## Concepts covered

| Concept | Where | Model / data |
|---------|-------|--------------|
| CTC vs seq2seq, phonetic vs fluent errors | `walkthrough.py` §1 | `facebook/wav2vec2-base-960h` vs `openai/whisper-{tiny,base}` |
| Whisper checkpoint family, `.en` variants, RTFx | §2 | tiny / base / small |
| Multilingual transcribe vs translate | §3 | MLS Spanish (streamed), MINDS-14 `de-DE` fallback |
| The 30-second wall, chunking, sequential long-form, timestamps | §4 | `openai/whisper-base` |
| The eight English ASR corpora and the four selection axes; ESB | §5 | LibriSpeech vs MINDS-14 measured |
| WER = (S + I + D) / N, word accuracy, CER, corpus aggregation | §6 | hand-rolled + `jiwer` cross-check |
| Orthographic vs normalised WER, the two normalisers | §7 | 4 models × 8 clips |
| `prepare_dataset`, the data collator, `-100` masking | §8 | `openai/whisper-tiny` |
| Fine-tuning with `Seq2SeqTrainer` | `finetune.py` | `openai/whisper-tiny` on MINDS-14 en-US |
| Building a demo (mic / file / translate, timestamps) | `gradio_demo.py` | `openai/whisper-base` |

## Notes (CPU / Windows / current libraries)

This example targets **transformers 4.57**, **gradio 6** and **datasets 3.6.0**. The course text
predates all three, so the following had to change — each is also commented at the point of use.

**The course code raises on this stack:**

- `evaluation_strategy=` → **`eval_strategy=`**.
- Passing audio whose sampling rate isn't 16 kHz raises
  `ImportError: torchaudio is required to resample`. Always
  `cast_column("audio", Audio(sampling_rate=16_000))` first — `torchaudio` is deliberately not
  installed, and `datasets` resamples with librosa instead.
- Audio longer than 30 s with neither `chunk_length_s` nor `return_timestamps=True` raises
  `ValueError` (§4 demonstrates this deliberately).
- `from transformers import BasicTextNormalizer` → it lives in
  `transformers.models.whisper.english_normalizer`.
- `evaluate.load("wer")` needs `jiwer`, which is why `jiwer` is now a core dependency.
- `gr.Audio(source=…)` → `sources=`; `gr.Interface(allow_flagging=…)` → `flagging_mode=`.
- The course's `prepare_dataset` variants that write `out["labels"][0]` break the collator: for a
  single string the tokenizer returns a *flat* list of ids, so `[0]` is one integer.
- `load_best_model_at_end=True` requires `save_strategy` to match `eval_strategy`.

**Silently wrong rather than loud:**

- **`WhisperProcessor.from_pretrained(..., language=…, task=…)` does not reach the fast tokenizer.**
  Labels come out as `<|startoftranscript|><|notimestamps|>` with `<|en|><|transcribe|>` missing,
  while generation *does* force them — so you would train on one prompt format and decode with
  another. `finetune.py` and the Colab notebook call
  `processor.tokenizer.set_prefix_tokens(language=…, task=…)` to fix it; `walkthrough.py` §8 shows the
  before/after.
- `Seq2SeqTrainer(tokenizer=…)` → **`processing_class=…`**.
- `model.generate = partial(model.generate, language=…, task=…)` → set
  `model.generation_config.language` / `.task`. A monkey-patched `generate` is not saved by
  `push_to_hub`, so the uploaded model would forget its language.
- `generation_config.forced_decoder_ids = None` is now a no-op; setting `language`/`task` is what
  skips the legacy branch.
- The hands-on wants WER as a **fraction** (`0.37`), while the course's `compute_metrics` multiplies
  by 100.
- MINDS-14's text column is `transcription`, not `sentence`.
- The course wraps `gr.TabbedInterface` in `with gr.Blocks() as demo:`; on gradio 6 that renders every
  tab twice unless the sub-interfaces are built outside the block.
- The ASR pipeline defaults to `num_beams=5`, which is ~5× slower on CPU for a marginally better
  transcript. Everything here passes `num_beams=1`.
- `facebook/multilingual_librispeech` has no `validation` split (it publishes `dev`/`test`/`train`).

**Substituted:**

- The course's CTC model `facebook/wav2vec2-base-100h` → `facebook/wav2vec2-base-960h`, which is
  already cached from Unit 2, has the identical 32-symbol uppercase vocabulary, and is the ASR
  pipeline's own pinned default.
- The course's fine-tuning dataset, Common Voice 13 Dhivehi, is **gated** (you must accept its terms
  and be logged in). `finetune.py` uses MINDS-14 instead, which is ungated and is what the hands-on
  actually grades. The Dhivehi recipe is preserved as a comment block in the `finetune.py` docstring.

**Other:**

- A benign "symlinks not supported" cache warning may appear on Windows, along with a
  `return_token_timestamps is deprecated` notice from the Whisper feature extractor.
- `fp16` is GPU-only; the smoke test forces it off.

## Supplemental reading

- [Whisper talk](https://www.youtube.com/live/fZMiD8sDzzg) by Jong Wook Kim — motivation, architecture, training, results
- [ESB benchmark paper](https://arxiv.org/abs/2210.13352) — the argument for orthographic WER
- [Fine-tuning Whisper for multilingual ASR](https://huggingface.co/blog/fine-tune-whisper)
- [Fine-tuning MMS adapter models](https://huggingface.co/blog/mms_adapters)
- [Boosting Wav2Vec2 with n-grams](https://huggingface.co/blog/wav2vec2-with-ngram)
