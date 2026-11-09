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
| 4 — Build a music genre classifier | [`units/unit4_music_genre_classifier/`](units/unit4_music_genre_classifier/) | Pre-trained classification survey (keyword spotting, zero-shot CLAP, language ID); fine-tuning DistilHuBERT on GTZAN (CPU smoke test + full GPU/Colab run with Hub push); a genre-classifier Gradio demo |
| 5 — Automatic speech recognition | [`units/unit5_automatic_speech_recognition/`](units/unit5_automatic_speech_recognition/) | CTC vs seq2seq on one clip (Wav2Vec2 vs Whisper); the Whisper checkpoint family and RTFx; transcribe vs translate; the 30-second wall, chunking and timestamps; the English ASR dataset landscape; WER by hand with the S/I/D alignment, CER, orthographic vs normalised; what the data collator builds; fine-tuning Whisper (CPU smoke test + Colab hands-on with Hub push); a tabbed transcription demo |
| 6 — From text to speech | [`units/unit6_text_to_speech/`](units/unit6_text_to_speech/) | TTS as two models (a SpeechT5 acoustic model plus a frozen HiFi-GAN vocoder); the 81-symbol character vocabulary and why you audit it by tokenizing rather than by vocab keys; 512-dim x-vector speaker conditioning; SpeechT5 vs Bark vs MMS/VITS, and the one-to-many problem proved via inference-time dropout; the TTS dataset landscape and which of the course's corpora still load; spectrogram labels, `-100` masking and reduction-factor truncation; evaluation the course says is impossible (round-trip WER, a median-f0 check, a blind MOS sheet); fine-tuning SpeechT5 (CPU smoke test + Colab hands-on with Hub push and the `pipeline_tag` the grader queries); a synthesis and engine-comparison demo |
| 7 — Putting it all together | [`units/unit7_putting_it_all_together/`](units/unit7_putting_it_all_together/) | Speech-to-speech translation as a **three**-stage cascade (`whisper-base` → `opus-mt-en-fr` → `mms-tts-fra`) rather than the course's two, because `task="translate"` only ever emits English and the hands-on forbids it; the language-forcing "trick" measured and rejected; error attributed to the seam that caused it by running the translation hop twice, once on the transcript and once on the human reference; latency separated from model loading, where a cold first call costs 15.41s and a warm one 2.77s for the same 5.86s of audio; what the text bottleneck destroys, proved by two voices 78 Hz apart coming out bit-identical; the four-stage voice assistant, including the two stages that no longer exist (`HfAgent`, the Inference API) and the wrong `pipeline_tag` that turns the language model into a parrot; meeting transcription with word timestamps, the `speechbox` merge reimplemented with the guards it lacks, and MFCC clustering scored against a measured chance baseline that is **not** 50%; the hands-on Space contract, one input and one output, with the int16 cast that wraps the sign; and a 38-symbol TTS vocabulary that silently deletes every comma and full stop you give it |

Each unit folder has roughly the same shape:

```
units/unitN_*/
├── walkthrough.py       # runnable script; saves plots/clips to figures/
├── notebook.ipynb       # the same material with inline plots + audio
├── finetune.py          # fine-tuning: CPU smoke test + full GPU/Colab run
├── colab_handson.ipynb  # ready-to-run Colab notebook for the unit's hands-on
├── gradio_demo.py       # optional local demo
├── assistant.py         # Unit 7: the four-stage voice assistant
├── meeting.py           # Unit 7: meeting transcription and the diarization sweep
├── space/               # Unit 7: a Hugging Face Space, ready to upload
├── figures/             # generated outputs (git-ignored)
└── README.md            # unit-specific setup, run steps, and notes
```

Not every unit has every file. `finetune.py` and `colab_handson.ipynb` appear only in the units with
a graded fine-tuning hands-on (4, 5 and 6). Unit 7 trains nothing, so it has neither, and carries
`assistant.py`, `meeting.py` and a `space/` folder instead: its hands-on is a deployed Space rather
than a fine-tuned checkpoint.

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
