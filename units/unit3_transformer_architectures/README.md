# Unit 3 — Transformer architectures for audio

Worked example for [Unit 3](https://huggingface.co/learn/audio-course/chapter3/introduction) of the
Hugging Face Audio Course. Unit 3 is **conceptual** in the course (it explains the architecture
families rather than running code), so this example makes each idea **runnable** with small demos on
real models.

| File | What it is |
|------|------------|
| [`notebook.ipynb`](notebook.ipynb) | Annotated notebook with inline plots + audio (recommended) |
| [`walkthrough.py`](walkthrough.py) | Script version of the four demos; saves plots to `figures/` |
| [`gradio_demo.py`](gradio_demo.py) | **Optional** local app: one clip through all three architectures |

## Setup & run

From the repo root (the env is shared across units, no per-unit install):

```bash
uv sync
uv run python units/unit3_transformer_architectures/walkthrough.py
# or:  uv run jupyter lab   then open notebook.ipynb
```

> **First run downloads ~1.3 GB of models** to the Hugging Face cache (not the repo):
> `openai/whisper-small` (~970 MB) and `MIT/ast-finetuned-audioset-10-10-0.4593` (~350 MB).
> `facebook/wav2vec2-base-960h` and MINDS-14 are already cached from earlier units. CPU only;
> Whisper generation is a little slow.

## What each section demonstrates

| # | Concept | What you see | Model |
|---|---------|--------------|-------|
| 1 | Input representations | One clip as a raw waveform (Wav2Vec2 input) vs an 80-bin log-mel (Whisper input), and the large sequence-length difference | librosa |
| 2 | **CTC** | The raw per-frame character predictions (with `<pad>` blanks and repeats) collapsing into text, matching `batch_decode` | `facebook/wav2vec2-base-960h` |
| 3 | **Seq2seq** | The encoder's hidden states, the decoder's task tokens (`<|startoftranscript|><|en|><|transcribe|>`), ASR, and German→English translation | `openai/whisper-small` |
| 4 | **Classification** | The Audio Spectrogram Transformer treating a spectrogram as 16×16 ViT patches; top AudioSet labels | `MIT/ast-finetuned-audioset-10-10-0.4593` |

Section 4 also notes that Unit 2's intent classifier is a `Wav2Vec2ForSequenceClassification` — the same
encoder as the CTC model, but mean-pooled with a linear head — contrasting it with AST.

## Notes

- All three model families need **16 kHz mono** input; the trumpet clip is resampled before AST.
- CTC output is uppercase, no punctuation (32-character vocab); Whisper output is cased and punctuated
  (~50k BPE vocab). That contrast is the point.
- `whisper-small` is used for the seq2seq demo because tiny/base translate this clip poorly; even small is not perfect, but the point is that the *same* model switches between transcribe and translate via the task token.
- HuBERT (`facebook/hubert-large-ls960-ft`) is the same CTC architecture but ~1.2 GB, so it is only
  mentioned, not run.
- Benign first-run warnings (Whisper "attention mask not set", Wav2Vec2 "masked_spec_embed newly
  initialized", "hf_xet not installed") are expected.
