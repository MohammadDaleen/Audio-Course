# Unit 6 — From text to speech

Worked example for [Unit 6](https://huggingface.co/learn/audio-course/chapter6/introduction) of the
Hugging Face Audio Course: TTS datasets, the pre-trained TTS model families, fine-tuning SpeechT5,
and the awkward question of how you evaluate any of it.

Unit 5 turned speech into text. Unit 6 runs the tape backwards, and it is **not symmetric**: ASR has
one right answer and a metric to score it with, TTS has many right answers and, per the course, no
metric at all. So this folder leans on *what the model actually emits* and on measuring the output
rather than trusting it.

| File | What it is |
|------|------------|
| [`walkthrough.py`](walkthrough.py) | The complete example: 8 sections from the acoustic-model/vocoder split through the collator to evaluation |
| [`finetune.py`](finetune.py) | Fine-tuning SpeechT5: a **CPU smoke test** (default) and a **full GPU/Colab** run |
| [`gradio_demo.py`](gradio_demo.py) | Three tabs: synthesise with a chosen voice, compare SpeechT5 against MMS/VITS, and inspect what the tokenizer will do to your text |
| [`notebook.ipynb`](notebook.ipynb) | The same material inline with playable audio; fine-tuning shown but gated by a flag |
| [`colab_handson.ipynb`](colab_handson.ipynb) | **Ready-to-run Colab notebook for the hands-on**: fine-tunes SpeechT5, pushes to the Hub, and sets the `pipeline_tag` the grader queries (open it on a GPU runtime) |

## Setup & run

From the repo root (the env is shared across units):

```bash
uv sync
uv run python units/unit6_text_to_speech/walkthrough.py    # the complete example
uv run python units/unit6_text_to_speech/finetune.py       # CPU smoke test
uv run python units/unit6_text_to_speech/gradio_demo.py    # the synthesis demo
```

> The walkthrough uses `speecht5_tts` (~585 MB), `speecht5_hifigan` (~51 MB), `mms-tts-eng`
> (~145 MB), the CMU ARCTIC x-vectors (~18 MB) and `whisper-tiny` (~151 MB, **already cached** from
> Units 2–5). The dialect corpus is **streamed**, so it costs one ~59 MB parquet row group rather
> than the full 395 MB config. About **0.86 GB** on a cold machine, into `~/.cache/huggingface`, not
> the repo. The whole script takes **6–10 minutes** on CPU; set `RUN_BARK = True` at the top to add
> Bark's ~1.7 GB, though Unit 2's `generation_demo.py` already covers it.

## Fine-tuning: smoke test (CPU) vs full run (GPU/Colab)

Fine-tuning SpeechT5 takes ~35 minutes on a T4 and is impractical on CPU, so `finetune.py` has two
modes:

- **smoke (default):** runs on CPU in about a minute on eight fabricated rows with 4 steps,
  `fp16=False`, no Hub push, no dataset download. It fabricates audio and text and then runs the
  *real* processor, so it exercises the genuine `text=` / `audio_target=` path rather than
  hand-rolled tensors. It proves the whole `Seq2SeqTrainer` pipeline agrees on shapes. It does
  **not** train a usable model, and the loss it prints is meaningless — the targets are noise.
- **full:** the hands-on recipe — real `ylacombe/english_dialects`, `fp16=True`, 1000 steps,
  `push_to_hub=True`, and the `metadata_update` that the certificate actually needs. Run it on a GPU:

  ```bash
  uv sync --extra training        # installs evaluate + tensorboard
  huggingface-cli login           # needed for push_to_hub
  UNIT6_MODE=full python units/unit6_text_to_speech/finetune.py
  ```

### The hands-on exercise

The Unit 6 hands-on asks you to fine-tune SpeechT5 on **a dataset of your choosing** and push it to
the Hub tagged as a `text-to-speech` model. Unlike Units 4 and 5 there is **no metric threshold** —
the course says the exercise "will focus on practicing the skills rather than achieving a certain
metric value". You pass as soon as a `text-to-speech`-tagged model exists under your account, which
makes this the most forgiving of the four hands-ons: nothing to chase, and no way to just miss it.

The easiest path is [`colab_handson.ipynb`](colab_handson.ipynb): open it in Google Colab on a **GPU**
runtime (Runtime → Change runtime type → T4 GPU) and run all cells (~45 minutes).

**The one thing that silently fails.** `trainer.push_to_hub(tasks="text-to-speech")` only populates
the model-index; it writes **no `pipeline_tag`**. With that field absent the Hub infers a tag from the
architecture and picks `text-to-audio`, so the grader's `list_models(filter=["text-to-speech"])` query
returns nothing and your model is invisible no matter how well it trained. Fix it with:

```python
from huggingface_hub import metadata_update
metadata_update(repo, {"pipeline_tag": "text-to-speech"}, overwrite=True)
```

Both the Colab notebook and `finetune.py` full mode do this for you. Note `list_models` reads a search
index that can lag the model page by about a minute, so an empty result immediately after the update
is not a failure.

### Deploy the demo as a Space

Create a new **Gradio** Space at [huggingface.co/new-space](https://huggingface.co/new-space), then
add two files.

`requirements.txt`:

```
transformers>=4.46,<5
torch
sentencepiece
datasets
```

`app.py` — the same code as [`gradio_demo.py`](gradio_demo.py) with your own checkpoint and a launch
line Spaces understands:

```python
TTS_ID = "<your-username>/speecht5_finetuned_english_dialects"
...
demo.launch()
```

A free CPU Space is enough — the vocoder and x-vectors are unchanged by fine-tuning, so only the
acoustic model is yours.

## Concepts covered

| Concept | Where | Model / data |
|---------|-------|--------------|
| Acoustic model vs vocoder; the 256-sample hop contract | `walkthrough.py` §1 | `microsoft/speecht5_tts` + `speecht5_hifigan` |
| An 81-token character vocabulary; auditing by tokenizing, not by vocab keys; `normalize=True` | §2 | `SpeechT5Tokenizer` |
| X-vector speaker conditioning; I-Vectors vs X-Vectors; consent | §3 | `Matthijs/cmu-arctic-xvectors` |
| SpeechT5 vs Bark vs MMS/VITS; one-to-many proved via inference-time dropout | §4 | `facebook/mms-tts-eng` |
| The TTS dataset landscape, and which of the course's four still load | §5 | `ylacombe/english_dialects` (streamed) |
| Spectrogram labels, `-100` masking, reduction-factor truncation | §6 | `SpeechT5Processor.pad` |
| The stop token, and text the vocabulary cannot hold | §7 | `microsoft/speecht5_tts` |
| Round-trip WER, a median-f0 check on the output, a blind MOS sheet | §8 | `openai/whisper-tiny` + `jiwer` + `librosa` |
| Fine-tuning with `Seq2SeqTrainer` | `finetune.py` | SpeechT5 on `english_dialects` |
| Synthesis, engine comparison, tokenizer inspection | `gradio_demo.py` | SpeechT5 + MMS/VITS |

## Notes (CPU / Windows / current libraries)

This example targets **transformers 4.57**, **gradio 6** and **datasets 3.6.0**. The course text
predates all three, so the following had to change — each is also commented at the point of use.

**The course code raises on this stack:**

- `from speechbrain.pretrained import EncoderClassifier` → the module was renamed to
  **`speechbrain.inference`** in SpeechBrain 1.0 and the shim is gone in 1.1. It also pulls in
  `torchaudio`, which this repo deliberately does not install.
- `SpeechT5Processor.from_pretrained(...)` raises `ImportError` without **`sentencepiece`**. SpeechT5's
  tokenizer is sentencepiece-only — there is no fast variant — so this is a hard failure, not a slow
  path. That is why `sentencepiece` is now a **core** dependency rather than a `generation` extra.
- `Seq2SeqTrainer(tokenizer=…)` → **`processing_class=…`**.
- `evaluation_strategy=` → **`eval_strategy=`**.
- `gradient_checkpointing=True` on its own makes the first backward raise `RuntimeError: Trying to
  backward through the graph a second time`. `gradient_checkpointing_kwargs` defaults to `None`, so
  `torch.utils.checkpoint` falls back to the legacy reentrant path. Always pass
  **`{"use_reentrant": False}`**.
- `load_dataset("lj_speech")`, `("vctk")` and `("cdminix/libritts-r-aligned")` all raise: they are
  loading-script datasets, and script execution was removed in `datasets` 3.0. **Three of the four
  corpora the course recommends no longer load.**
- `gr.Interface(allow_flagging=…)` → `flagging_mode=`; wrapping `TabbedInterface` in a `gr.Blocks()`
  renders every tab twice on gradio 6.

**Silently wrong rather than loud:**

- **`set(text) - set(tokenizer.get_vocab())` is the wrong way to audit text, and it fails in the worst
  way.** This is a sentencepiece model, so a space is stored as the word-boundary marker `▁` (U+2581)
  and a literal `" "` looks absent from the vocab keys. That check rejects **every row that contains a
  space**, i.e. every row — and if you then assert the survivors are clean, the assertion passes
  vacuously on the empty result. Ask the tokenizer whether it emits `<unk>` instead. §2 runs the wrong
  audit first so you can watch it reject all 40 clips.
- **`out["labels"] = out["labels"][0]` is *required* here** — the processor wraps the spectrogram in a
  batch dimension of one. This is the exact inverse of Unit 5, where the same line was the bug: there
  the labels are a flat list of ids and `[0]` takes a single integer.
- **`model.eval()` does not make SpeechT5 deterministic.** The speech decoder pre-net applies dropout
  `p=0.5` even at inference (the source comment says so, citing Tacotron 2), so two identical calls
  give different-length audio. `torch.manual_seed()` does. §4 proves it both ways.
- `label_names=["labels"]` must be set, or the `Trainer` never finds the labels inside SpeechT5's
  output object and your eval loss is silently never computed.
- `remove_unused_columns=False` must be set, or `speaker_embeddings` is pruned before the collator
  sees it.
- `predict_with_generate=True` does not work: SpeechT5's `generate()` returns a spectrogram, not token
  ids. `can_generate()` returns `True` only so the `GenerationConfig` plumbing works.
- The collator must build the speaker batch with `torch.tensor`, **not** `torch.stack`: the x-vectors
  round-trip through Arrow and come back as plain Python lists.
- Target lengths must be truncated to a multiple of `reduction_factor` (2), or the loss is misaligned
  against the predictions.
- `trainer.push_to_hub(tasks="text-to-speech")` writes **no `pipeline_tag`** — see the hands-on section
  above. This is the single most likely way Unit 6 fails.
- `tokenizer.vocab_size` reports **79**; `len(tokenizer.get_vocab())` and `config.vocab_size` report
  **81**. The gap is the added `<mask>` and `<ctc_blank>`.
- The dialect corpus is **48 kHz**. Hand SpeechT5 un-cast audio and every duration is three times
  wrong, with nothing raised.

**Substituted:**

- The course's speaker embeddings come from SpeechBrain's `spkrec-xvect-voxceleb`. This unit uses the
  pre-computed **`Matthijs/cmu-arctic-xvectors`** instead, mapping each dataset speaker to one fixed
  vector. That makes the x-vector a consistent speaker *code* rather than a true timbre embedding; the
  model still learns the voices, and with no metric to hit nothing is lost. The SpeechBrain recipe is
  preserved as a comment block in the `finetune.py` docstring.
- The course fine-tunes on VoxPopuli Dutch, whose train split is **10.4 GB**. This unit uses
  `ylacombe/english_dialects` `northern_female` (750 clips, ~395 MB, parquet-native) — the hands-on
  explicitly allows any dataset, and the larger download buys nothing on an unscored unit.
- The course's evaluation page recommends **no metric at all**. §8 adds two cheap proxies anyway:
  round-trip WER through `whisper-tiny`, and a median-f0 check that measures the *generated audio*
  rather than the input embeddings (female 184 Hz vs male 101 Hz on the five CMU voices).

**Other:**

- A benign "symlinks not supported" cache warning appears on Windows, along with a
  `return_token_timestamps is deprecated` notice from the Whisper feature extractor.
- `fp16` is GPU-only; the smoke test forces it off.
- Round-trip WER comes out at **0.0000** on the six demo sentences. That is the lesson, not a
  triumph: they are short, common and in-vocabulary, so the metric has almost no room to
  discriminate. Treat it as a regression alarm, not a quality score.

## Supplemental reading

- [HiFi-GAN: GANs for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646.pdf) — the vocoder used here
- [X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — where the speaker embeddings come from
- [FastSpeech 2](https://arxiv.org/pdf/2006.04558.pdf) — the non-autoregressive alternative to SpeechT5 and Bark
- [MQTTS](https://arxiv.org/pdf/2302.04215v1.pdf) — quantized discrete representations instead of mel spectrograms
