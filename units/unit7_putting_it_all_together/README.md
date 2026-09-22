# Unit 7 — Putting it all together

Worked example for [Unit 7](https://huggingface.co/learn/audio-course/chapter7/introduction) of the
Hugging Face Audio Course: speech-to-speech translation, a voice assistant, and transcribing a
meeting.

Units 1–6 built the parts. Unit 7 bolts them together, and the lesson is that **composition is where
the errors live**: every seam is a place to lose information, add latency, or be silently wrong. So
this folder measures each seam rather than the end of the pipe. It is also the first unit where the
failure of one component shows up as a confusing failure of a *different* one.

| File | What it is |
|------|------------|
| [`walkthrough.py`](walkthrough.py) | The complete example: 9 sections from the three-stage cascade through the text bottleneck to the vocabulary that deletes your punctuation |
| [`assistant.py`](assistant.py) | The four-stage voice assistant — wake word, transcribe, answer, speak — canned by default, `--live` for a Gradio push-to-talk front end |
| [`meeting.py`](meeting.py) | Meeting transcription: Whisper word timestamps, the `speechbox` merge reimplemented with guards, and a diarization sweep over every speaker pair |
| [`gradio_demo.py`](gradio_demo.py) | Four tabs: translate into one of five languages, compare the 2- and 3-stage cascades, ask the assistant, transcribe a meeting |
| [`notebook.ipynb`](notebook.ipynb) | The same material inline with rendered plots and playable audio |
| [`space/`](space/) | The hands-on Space (`app.py`, `requirements.txt`, `README.md`) — **prepared, not published** |

There is **no `finetune.py` and no `colab_handson.ipynb`: nothing is trained in this unit.** Unit 7
composes checkpoints that Units 2–6 already downloaded.

## Setup & run

From the repo root (the env is shared across units):

```bash
uv sync
uv run python units/unit7_putting_it_all_together/walkthrough.py   # the complete example
uv run python units/unit7_putting_it_all_together/assistant.py     # the four-stage assistant
uv run python units/unit7_putting_it_all_together/meeting.py       # timestamps + the scored track
uv run python units/unit7_putting_it_all_together/meeting.py --sweep   # all 10 speaker pairs
uv run python units/unit7_putting_it_all_together/gradio_demo.py   # the four-tab demo
```

> Five checkpoints are new here: `opus-mt-en-fr` (~301 MB), `opus-mt-en-ar` (~307 MB),
> `mms-tts-fra` and `mms-tts-ara` (~145 MB each) and `LaMini-Flan-T5-248M` (~990 MB). Everything
> else — `whisper-base` and `-tiny`, `mms-tts-eng`, the AST keyword spotter, Unit 6's SpeechT5 stack
> — is **already cached** if you ran Units 2–6. The dialect corpus is **streamed**, so §7 costs one
> ~59 MB parquet row group rather than the full 395 MB config. That is about **1.95 GB** new on a
> cold machine, into `~/.cache/huggingface`, not the repo. The walkthrough takes **6–11 minutes** on
> CPU.
>
> `facebook/mms-lid-126` (3.86 GB), the language classifier the grading Space uses, is deliberately
> **not** downloaded — it is what put that Space into `RUNTIME_ERROR` in the first place.

`assistant.py` and `meeting.py` are this repo's first scripts to use **argparse**, so they have a
real `--help`. Units 4–6 toggle one binary mode with `"--full" in sys.argv`, which is enough for one
switch and stops being enough here: `meeting.py --audio` takes a *value*, and a hand-rolled parser
that silently ignores `--audio corse` is exactly the class of bug this unit is about. The `UNIT7_MODE`
and `UNIT7_AUDIO` env vars still work, and are validated explicitly, because argparse checks what it
is given and never its own default.

## The hands-on exercise

**This hands-on cannot currently be submitted, and it is not your code's fault.** Both of Hugging
Face's grading Spaces are down:

- `huggingface-course/audio-course-u7-assessment` is in **`RUNTIME_ERROR`**. It loads
  `facebook/mms-lid-126` (3.86 GB) at module import and dies on *"No space left on device"* — the
  free CPU tier has no room for it. Nothing you submit reaches a grader.
- The separate progress Space **404s**: it reads a private dataset with a token that has expired.

Both are on Hugging Face's side. There is no workaround from here.

**This matters less than it sounds.** This repo is at **3 of 4 hands-ons, and the completion
certificate is already earned.** Unit 7 is here for repo completeness, not for certificate progress.
So [`space/`](space/) is **prepared and ready to upload, but not published**: the code is written and
exercised locally, and publishing it is a decision for later, if and when the assessment Space comes
back.

### What the grader actually calls

Worth knowing anyway, because the contract is stricter than the exercise text implies. The grader
does not read your code. It does this:

```python
client = Client(repo_id)
audio_file = client.predict("test_short.wav", api_name="/predict")
# then facebook/mms-lid-126 on the returned audio, and it fails you if the top label is
# 'eng' or its score is below 0.5
```

Two consequences most people miss:

- **One positional argument in, one `sf.read()`-able return value out.** Add a language dropdown, a
  second output, or a named parameter and you fail with a `gradio_client` traceback rather than a
  grading message. That single line is the entire interface contract.
- **The check is only "not English, confidently".** It never verifies that you hit the language you
  claimed. Arabic, French and Dutch all pass identically; English at 0.99 and Arabic at 0.4 both fail.

And the trap that fails people while their demo *sounds* right: `task="translate"` only ever produces
**English**. See the notes below.

### Deploy the demo as a Space

[`space/`](space/) already holds the three files a Gradio Space needs, so publishing is a copy rather
than a rewrite:

1. Create a new **Gradio** Space at [huggingface.co/new-space](https://huggingface.co/new-space). A
   free CPU tier is enough — the whole cascade is 184M parameters.
2. Upload `space/app.py`, `space/requirements.txt` and `space/README.md`.
3. Make the Space **public**: the grader instantiates `Client(repo_id)` with no token.

`space/app.py` is the three-stage cascade behind exactly one `gr.Interface` — one `gr.Audio` in, one
`gr.Audio` out, endpoint `/predict`, translating to **Arabic**. It clips before the int16 cast and
offers no language selector, because a second input breaks the grader's call.

## Concepts covered

| Concept | Where | Model / data |
|---------|-------|--------------|
| The three-stage cascade, what each seam costs, and cold vs warm timing | `walkthrough.py` §1 | `whisper-base` + `opus-mt-en-fr` + `mms-tts-fra` |
| `task="translate"` has exactly one target, and it is English | §2 | `openai/whisper-base` + `jiwer` |
| Damage attributed to the seam that caused it, by translating twice | §3 | `librispeech_asr_dummy` + `jiwer` |
| Latency is additive; the cheaper cascade emits the wrong language | §4 | `whisper-tiny` vs `whisper-base` |
| The text bottleneck: two voices in, one voice out | §5 | `speecht5_tts` + `speecht5_hifigan` + `librosa.yin` |
| Wake word, transcribe, answer, speak — and the two dead stages | §6 | `ast-finetuned-speech-commands-v2` + `LaMini-Flan-T5-248M` |
| Word timestamps, the merge, and a chance baseline that is not 50% | §7 | `ylacombe/english_dialects` (streamed) + `sklearn` |
| The grader's contract, and the int16 cast that wraps the sign | §8 | `numpy`; `mms-lid-126` deliberately *not* downloaded |
| A 38-symbol vocabulary that deletes what it cannot say | §9 | `opus-mt-en-ar` + `facebook/mms-tts-{eng,fra,ara}` |
| The assistant end to end, canned or push-to-talk | `assistant.py` | AST + Whisper + LaMini + MMS/VITS |
| The diarization sweep over every speaker pair | `meeting.py` | MFCC + `AgglomerativeClustering` |
| Translate into five languages, compare cascades, ask, transcribe | `gradio_demo.py` | the whole cascade |
| One input, one output, endpoint `/predict` | `space/app.py` | `whisper-base` + `opus-mt-en-ar` + `mms-tts-ara` |

## Notes (CPU / Windows / current libraries)

This example targets **transformers 4.57**, **gradio 6** and **datasets 3.6.0**. The course text
predates all three, and Unit 7 leans on more third-party pieces than any other unit, so more of it
has rotted. Each substitution is also commented at the point of use.

**The course code raises on this stack:**

- `from transformers import HfAgent` → **removed from transformers entirely.** The course's closing
  section uses it to generalise the assistant into a tool-using agent. The import raises; there was
  no deprecation window to catch.
- `api-inference.huggingface.co` → the **DNS record no longer resolves** (`socket.gaierror`), so the
  course's assistant page fails before it is even an HTTP error. `tiiuae/falcon-7b-instruct`'s
  `inferenceProviderMapping` is empty too, so there is no drop-in replacement endpoint either.
- `from speechbox import ASRDiarizationPipeline` → pulls in **`torchaudio`**, which this repo
  deliberately does not install. §7 and `meeting.py` reimplement the merge inline instead.
- `pyannote/speaker-diarization` and `pyannote/segmentation` are **gated**: both need an accepted
  licence and a logged-in token, and pyannote pulls in `torchaudio` as well.
- `for_json()` → gone from `pyannote.core` 5.x, so the course's snippet for serialising a
  segmentation raises `AttributeError` even once you are past the gate.
- `import spaces` / `@spaces.GPU` → ZeroGPU-only. On a free CPU Space they stop the app starting, and
  a Space that is not running fails the grader with a connection error rather than a score.
- `kakao-enterprise/vits-ljs` → raises at load: its config sets `phonemize: true`, so `VitsTokenizer`
  requires **espeak-ng** as a system binary. Every `facebook/mms-tts-*` sets `phonemize: false` and
  needs nothing.

**Silently wrong rather than loud:**

- **`task="translate"` only ever produces English, and `language=` is silently ignored alongside
  it.** `translate, language="spanish"` returns **byte-identical** output to plain `translate`:
  Whisper's translation objective has exactly one target. The hands-on requires non-English, so the
  course's own two-stage design cannot pass its own exercise.
- **The "trick" the course recommends is not a translation.** `task="transcribe",
  language="french"` is a transcription *nudged toward French spelling*: on the demo clip "apostle"
  came back as *le passé* and "gospel" as *gosse-boule*, while "glad" stayed in English. Nothing
  raised, nothing warned, and it loses to the MT hop on every measure in §2.
- **`pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-248M")` echoes the prompt back
  verbatim.** The Hub's `pipeline_tag` on that model is **wrong** — it is a T5, a seq2seq. Pass
  `"text2text-generation"` explicitly or your assistant is a parrot that never raises. `assistant.py`
  asserts the reply differs from the prompt for exactly this reason.
- **The "not supported for `text-generation`" dump is logged at `logging.ERROR`, not `WARNING`**, so
  `logging.disable(logging.WARNING)` sails straight through it. It is also a single
  3,273-character line naming 200+ classes, which is its own failure: an error you have to scroll
  past is an error you stop reading.
- **`AutomaticSpeechRecognitionPipeline.preprocess` MUTATES the dict you hand it**, moving `"array"`
  into `"raw"` in place. Reuse one dict across two calls and the **second** raises
  `ValueError: ... needs to contain a "raw" key`, which reads as though your input were malformed
  rather than already consumed. Build a fresh dict per call.
- **Every `facebook/mms-tts-*` voice silently DELETES characters it has no symbol for.**
  `VitsTokenizer` is built with `normalize=True`, and that path ends in
  `filtered_text = "".join(list(filter(lambda char: char in self.encoder, filtered_text)))` —
  deletion, not `<unk>`, with nothing logged. Measured in §9: `mms-tts-eng` has 37 pronounceable
  symbols, `mms-tts-fra` 43, `mms-tts-ara` 38, and **not one of the three has a full stop, a comma,
  an exclamation mark or a question mark**. English carries digits 0 to 6 and stops there, so even in
  English "1987" is only partly sayable and "1989" is not. Arabic has no digits, no punctuation of
  any kind, not even an apostrophe, and no harakat. Unit 6's SpeechT5 at least emits an `<unk>` you
  can count.
- **The chance baseline for the diarization scorer is not 50%.** `max((pred == truth).mean(),
  (pred != truth).mean())` takes the better of the two cluster-to-speaker mappings, because cluster
  ids are arbitrary — so it *cannot* report below 50%, and on a short track it sits near 58% on luck
  alone. Measured by Monte Carlo at the track's own window count: 59% at n=19, 57% at n=28, 54% at
  n=100. Quote 50% and a null result reads as a weak positive one. Both `walkthrough.py` and
  `meeting.py` compute the null at runtime and print it next to every accuracy.
- **A latency table that includes model loading is wrong by an order of magnitude.** §4 timed the
  first checkpoint's load inside its own timer and reported whisper-tiny's 3-stage row at **14.57s**
  against a 2-stage **3.68s**, pricing the translation hop at about eleven seconds. Warm, both rows
  land near 1.7s and the hop costs **+0.07s**. §1 now prints the cold and warm rows side by side
  rather than picking one.
- **`return_language=True` does not report Whisper's detected language here.** On transformers 4.57
  it leaves `chunks[0]["language"]` as `None` on *both* the chunked and the sequential path, so an
  assertion written against it passes **vacuously** — `None != "english"` is true and means nothing.
  `model.detect_language()` works and returns a token id you invert through
  `generation_config.lang_to_id`.
- **The Space contract is one positional input and one output** — see the hands-on section above.
  Break it and you get a `gradio_client` traceback, not a grading message. Related: **gradio 6 names
  an `Interface`'s endpoint after its FUNCTION** (`gradio/blocks.py:773`, *"If api_name is None or
  empty string, use the function name"*), where gradio 5 always used `predict`. `space/app.py` names
  its function `predict` so the endpoint is `/predict` under both, and renaming it is a silent way to
  fail. A `TabbedInterface` of two Interfaces registers `/predict` **and** `/predict_1`, so the
  endpoint the grader reaches depends on tab order.
- **The `speechbox` merge has three unguarded failure modes**: `argmin` of an array it has already
  emptied, a `None` end timestamp (Whisper emits one for a final unterminated chunk), and a single
  segment, where its loop body never runs at all. It also drops trailing words no segment absorbed.
  §7 and `meeting.py` handle all four.
- **The AST checkpoint's `id2label[27]` is `"marvin"`, but `google/speech_commands`' own index for it
  is 26.** Compare label **strings**, never ints. A hard-coded index is off by one and quietly wakes
  on a different word.
- **Reporting only the wake word's score flatters the wake word.** Synthesised `marvin` scores
  **0.997**, which looks like a classifier that works. Run the controls and **0 of 3** are classified
  correctly — `stop` comes back as `go`, `yes` and `house` as `stop`. What saves it is that **0 of 3**
  falsely wake it: AST is unreliable on synthesised speech in general and reliable about this one
  word, which is all a wake word has to do. `assistant.py` prints both numbers, because the second
  one is what a gate is actually judged on.
- **`(x * 32767).astype(np.int16)` wraps the sign above 1.0.** Measured: `1.05 → -31131`,
  `1.5 → -16386`, `-1.2 → +26216` — 3 of 5 sample signs flipped. The course's Space template does
  exactly this cast. `mms-tts` peaks below 1.0 today so it does not fire, but it is one
  `np.clip(x, -1.0, 1.0)` from being safe, and a louder checkpoint would invert every loud sample
  without a word.
- **`ffmpeg_microphone_live`'s `_get_microphone_name()` takes device `[0]`** from ffmpeg's device
  list. On this machine device 0 is a Voicemeeter loopback, so it records a virtual cable and never
  triggers — with no error, because a silent stream is a valid stream. The course's own snippet also
  has a **`return` that reads its loop variable from outside the loop**: an empty stream is an
  unbound-name error rather than an empty transcript, and a loop that never breaks returns the last
  *partial* hypothesis as though it were final.
- **Two source voices 78.2 Hz apart in median f0 produce output waveforms that are bit-identical**
  (`max abs diff 0.000e+00`). Nothing failed — a string simply cannot carry prosody. If you expect a
  cascade to preserve *who was speaking*, it will disappoint you without ever raising.

**Substituted:**

- **Three stages, not the course's two.** `task="translate"` → TTS is simpler and only ever emits
  English, so it cannot pass the hands-on. This unit inserts a machine-translation hop:
  ASR → MT → TTS. §4 prices it at **+0.07s**, which is the whole cost of getting a target language.
- **`Helsinki-NLP/opus-mt-en-fr` (301 MB, 75.1M params) rather than NLLB-200-600M** (2460 MB), which
  is eight times the download and needs explicit source and target language codes threaded through
  the tokenizer. Marian is one direction per checkpoint, which is exactly what a fixed-target Space
  wants.
- **`facebook/mms-tts-*` rather than `kakao-enterprise/vits-ljs`**, which needs espeak-ng.
- **`MBZUAI/LaMini-Flan-T5-248M` run locally rather than `tiiuae/falcon-7b-instruct` over the
  Inference API.** The API host no longer resolves, and 7B parameters is not a CPU proposition.
- **Gradio push-to-talk rather than `ffmpeg_microphone_live`** — see the device-index bug above.
  Push-to-talk also makes the demo reproducible on a machine with no usable microphone at all.
- **The merge reimplemented inline rather than `speechbox`**: about thirty lines, no `torchaudio`,
  and the guards `speechbox` lacks.
- **pyannote → nothing, reported as a result rather than papered over.** The only CPU-only substitute
  available here is librosa MFCCs plus `AgglomerativeClustering`, and it is **not a diarizer**.
  `meeting.py --sweep` scores all **10** speaker pairs of `english_dialects` `northern_female`
  against each track's own measured null: mean **60.3%** against a mean chance baseline of **57.8%**,
  range **51.7% to 68.2%**, **7 of 10** pairs beat their own null, **3 of 10 fall below it**, and
  **0 of 10 reach a usable 80%**. The average edge over chance is **+2.5 points**. The walkthrough's
  single track scores **57.1%** against a **57.4%** null — 0.3 points *below* chance. MFCCs encode
  *what* was said far more strongly than *who* said it, so without a speaker-embedding model you are
  clustering phonetics. It ships as a measured negative result, not as a feature.
- **Arabic rather than French in `space/`.** The grader's whole test is "confidently not English",
  and a different script is the least ambiguous way to pass it. Verified with the cached
  `sanchit-gandhi/whisper-medium-fleurs-lang-id` standing in for `mms-lid-126`: top-1 `Arabic` at
  **1.000** against the grader's 0.5 threshold, with Whisper's own `detect_language()` independently
  returning `<|ar|>`. A substitute classifier agreeing is evidence, not the grade.
- **No Italian in `gradio_demo.py`'s dropdown, and not by preference:** `facebook/mms-tts-ita` does
  not exist. Every Arabic and European MMS code was checked against the Hub; MMS-TTS publishes no
  Italian voice at all, so the pair cannot be built however good `opus-mt-en-it` is. The dropdown
  offers French, Arabic, German, Dutch and Russian, lazily, with the download cost in each label.

**Other:**

- The full cascade is `whisper-base → opus-mt-en-fr → mms-tts-fra`: **184M parameters** (72.6M +
  75.1M + 36.3M). §1 reports first-call and warm timings side by side, because on a cold process most
  of the first row is weights arriving from disk rather than compute. Measured on 5.86s of audio:

  | | ASR | MT | TTS | total | real-time factor |
  |---|---|---|---|---|---|
  | first call | 8.44 | 3.53 | 3.45 | **15.41** | 2.63 |
  | warm | 1.23 | 0.50 | 1.04 | **2.77** | **0.47** |

  **12.64 seconds of that first row was loading weights, not running them.** Warm, the cascade is
  comfortably faster than real time; quote the cold row and you would conclude it is unusable.
- Whisper decodes with `num_beams=5` by default: ~5× slower on CPU for a barely better transcript.
  Every call here passes `num_beams=1`.
- The dialect corpus is **48 kHz**, so `cast_column("audio", Audio(sampling_rate=16_000))` is
  mandatory. Streaming caches nothing, so §7, `meeting.py` and the demo's meeting tab need the
  network on **every** run and re-read that ~59 MB row group each time. `meeting.py` streams once per
  process and reuses it across all ten pairs; doing it per pair would be ~590 MB.
- `scikit-learn` became a **core** dependency for this unit. It was already installed, but only
  transitively through `librosa`, and depending on another package's pin is how a working import
  disappears in a minor release.
- **`sklearn` has intermittent Windows Application Control DLL blocks on a cold import.** It lets the
  identical import through a moment later, so `meeting.py` and `gradio_demo.py` retry once rather
  than reporting a missing dependency.
- Download budget: **~1.95 GB** new on a cold machine. `facebook/mms-lid-126` (3.86 GB) is
  deliberately not downloaded.
- `figures/` holds 18 generated artifacts from the walkthrough plus a handful from `assistant.py` and
  `meeting.py`, and is git-ignored.
- Benign warnings on this stack: a "symlinks not supported" cache notice on Windows, a
  `return_token_timestamps is deprecated` notice from the Whisper feature extractor, a
  `Recommended: pip install sacremoses` notice from `MarianTokenizer`, and an "at least one mel
  filter has all zero values" notice from the AST feature extractor.

## Supplemental reading

- [Textless speech-to-speech translation on real data](https://ai.meta.com/research/publications/textless-speech-to-speech-translation-on-real-data/) — STST with discrete units, no text in the middle
- [Translatotron 2](https://arxiv.org/abs/2107.08661) — direct speech-to-speech, the model §5 is arguing for
- [Accurate detection of wake word start and end using a CNN](https://www.amazon.science/publications/accurate-detection-of-wake-word-start-and-end-using-a-cnn) — how a real wake word is done
- [pyannote.audio 2.1](https://huggingface.co/pyannote/speaker-diarization) — the diarizer §7 could not run
- [WhisperX](https://arxiv.org/abs/2303.00747) — forced alignment for word timestamps that survive the merge
