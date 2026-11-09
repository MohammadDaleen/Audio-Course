---
title: Speech to Speech Translation (Arabic)
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Speech in any language, Arabic speech out. Whisper + Marian + MMS.
---

# Speech-to-speech translation: anything to Arabic

The Unit 7 hands-on Space. Audio in, Arabic audio out, on a free CPU tier.

| Stage | Model | Size |
|-------|-------|------|
| 1. Transcribe and normalise to English | `openai/whisper-base` | ~290 MB |
| 2. Translate English to Arabic | `Helsinki-NLP/opus-mt-en-ar` | ~307 MB |
| 3. Speak the Arabic | `facebook/mms-tts-ara` | ~145 MB |

About 740 MB on first boot, 184M parameters, no GPU.

## This Space is prepared, not published

Nothing in this repository creates a Space, writes to a Hub account, or needs a token. These
three files are ready to upload and have not been uploaded.

That is not caution, it is arithmetic: **the Unit 7 hands-on currently cannot be graded.**
`huggingface-course/audio-course-u7-assessment` is in `RUNTIME_ERROR` because it loads
`facebook/mms-lid-126` (3.86 GB) at module import and dies on *"No space left on device"*, and
the separate progress Space 404s on a private dataset whose token has expired. Both are on
Hugging Face's side. A correct Space submitted today reaches no grader at all.

To publish it anyway, or once that Space is fixed:

1. Create a new **Gradio** Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Upload `app.py`, `requirements.txt` and this `README.md` unchanged.
3. Make the Space **public** - the grader instantiates `Client(repo_id)` with no token.

## The contract, which is stricter than the exercise text implies

The assessor does not read your code. It does this:

```python
client = Client(repo_id)
audio_file = client.predict("test_short.wav", api_name="/predict")
# then facebook/mms-lid-126 on the returned audio, failing you if the top label is
# 'eng' or if its score is below 0.5
```

**One positional argument in, one `sf.read()`-able return value out.** There is no language
dropdown here, and that is the constraint rather than an oversight: a second input or a second
output changes the endpoint's signature and the grader dies with a `gradio_client` traceback
instead of producing a grade. The version with a dropdown is `../gradio_demo.py`, which is
local and has nothing to honour.

The check is only *"not English, confidently"*. It never verifies that you reached the language
you claimed, so Arabic, French and Dutch all pass identically, while English at 0.99 and Arabic
at 0.4 both fail.

One thing outside our control: the snippet above predates `gradio_client` 1.0. A current client
wants `handle_file("test_short.wav")` rather than a bare string for a file input. That is the
assessor's code, so do not "fix" `app.py` chasing a client-side error.

## Why three stages and not two

The course's template stops after Whisper with `task="translate"`. **That only ever produces
English**, and English is precisely what this exercise fails you for. Passing `language=`
alongside `task="translate"` does not help either: it is silently ignored, and returns
byte-identical output to plain `translate`. Whisper's translation objective has exactly one
target. The Marian hop is the entire Space.

## Why `sdk_version: 5.49.1`, and why not to bump it

It is the template's known-good pin. The sharp reason to leave it alone is that gradio 6
changed how an `Interface` names its endpoint: `gradio/blocks.py` now says *"If api_name is
None or empty string, use the function name"*, where gradio 5 and earlier always used
`predict`. `app.py` names its function `predict`, so the endpoint is `/predict` under both -
but that is a defence, not a licence to bump, and **renaming that function is a silent way to
fail the hands-on.**

For the same reason the template's `TabbedInterface` is collapsed to a single `Interface` here.
Two Interfaces both want `predict`, so gradio appends a suffix and registers `/predict` and
`/predict_1`, which makes the endpoint the grader reaches depend on tab order.

## Two things the template gets wrong

**`(x * 32767).astype(np.int16)` wraps the sign above 1.0.** Measured: `1.05` becomes `-31131`,
`1.5` becomes `-16386`, `-1.2` becomes `+26216`. `app.py` clips to `[-1, 1]` first, in a named
`to_int16()` so it can be tested directly. `mms-tts` peaks below 1.0 today, so this is
insurance rather than a fix for a live bug - but a louder checkpoint would hand the grader a
burst of sign-flipped noise and raise nothing.

**`import spaces` / `@spaces.GPU` are ZeroGPU-only.** On a free CPU Space they stop the app
starting, and a Space that is not running fails the grader with a connection error rather than
a score. Both lines are deleted here, along with the `zipfile` x-vector block, which has
nothing left to condition once SpeechT5 is gone.

## What Arabic costs you, silently

`facebook/mms-tts-ara` has **38 pronounceable symbols, 35 of them bare Arabic letters, no
digits, and no punctuation at all** - not a full stop, not a comma, not even the Arabic comma
or question mark. `VitsTokenizer` is built with `normalize=True`, and that path ends in

```python
filtered_text = "".join(list(filter(lambda char: char in self.encoder, filtered_text)))
```

so anything outside the vocabulary is **deleted**, not turned into `<unk>`, with nothing
logged. Every comma and full stop Marian produces is gone before the model sees it, and the
audio still sounds fluent and complete.

The grader cannot notice: it only asks whether the audio is confidently not English. So this
Space passes while quietly dropping characters, which is Unit 7's thesis in one line. Section 9
of `../walkthrough.py` measures it for English, French and Arabic - none of the three voices
has a single sentence punctuation mark.

## Licence

`cc-by-nc-4.0`, which is the most restrictive of the three checkpoints: `facebook/mms-tts-*` is
CC-BY-NC-4.0 while `Helsinki-NLP/opus-mt-en-ar` is Apache-2.0. **Non-commercial** therefore
governs the whole thing.
