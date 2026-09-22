"""
Hugging Face Audio Course - Unit 6: From text to speech
========================================================

Unit 5 turned speech into text. Unit 6 runs the tape backwards, and the first thing you
learn is that it is not symmetric: ASR has one right answer and a metric to score it with,
TTS has many right answers and, the course says, no metric at all. This script makes every
one of those claims runnable on CPU:

 1. Two models, not one    - SpeechT5 writes a spectrogram, HiFi-GAN makes the sound
 2. The 81-token front end - a character vocabulary, and the <unk> that never warns you
 3. Speaker control        - 512 floats decide whose voice it is, and who consented
 4. Three designs          - SpeechT5 vs MMS/VITS vs Bark, and the one-to-many problem
 5. TTS data               - what the course lists, what still loads, what "good" means
 6. Fine-tuning innards    - what the TTS collator builds, and why it is Unit 5 inverted
 7. How TTS fails          - the stop token, babbling, and text the vocabulary cannot hold
 8. Evaluating for real    - round-trip WER, a pitch check, and a blind MOS sheet

Unit 2 already generated speech with `pipeline("text-to-speech")`; this unit is about what
is inside that call, how to condition it, and how to judge the result. The training run
lives in `finetune.py`, the demo in `gradio_demo.py`, and the graded hands-on in
`colab_handson.ipynb`.

Run with:

    uv run python units/unit6_text_to_speech/walkthrough.py

Models used: microsoft/speecht5_tts (~585 MB), microsoft/speecht5_hifigan (~51 MB),
facebook/mms-tts-eng (~145 MB), openai/whisper-tiny (~151 MB, already cached from Units 2-5)
and the CMU ARCTIC x-vectors (~18 MB). The dialect corpus is STREAMED, so it costs one
~59 MB parquet row group rather than the full 395 MB config. A cold machine pulls about
0.86 GB into ~/.cache/huggingface, not the repo. Everything runs on CPU in about 6-10
minutes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters like "→" or the
# sentencepiece word-boundary marker "▁". Force UTF-8 so the prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import soundfile as sf

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save figures, never block
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL_ID = "microsoft/speecht5_tts"             # acoustic model: text -> log-mel spectrogram
VOCODER_ID = "microsoft/speecht5_hifigan"       # vocoder: log-mel spectrogram -> waveform
MMS_ID = "facebook/mms-tts-eng"                 # VITS: text -> waveform, no vocoder needed
BARK_ID = "suno/bark-small"                     # Unit 2's generation_demo.py already runs this
ASR_ID = "openai/whisper-tiny"                  # cached since Unit 2; the judge in section 8

DATASET_ID = "ylacombe/english_dialects"        # parquet-native, so no trust_remote_code
CONFIG = "northern_female"                      # 750 clips, 5 speakers x 150, 48 kHz source
XVECTOR_ID = "Matthijs/cmu-arctic-xvectors"     # pre-computed 512-d x-vectors, ~18 MB
XVECTOR_SPLIT = "validation"                    # the ONLY split this dataset publishes

SAMPLING_RATE = 16_000        # SpeechT5, HiFi-GAN and MMS all speak 16 kHz
NUM_MEL_BINS = 80             # config.num_mel_bins
HOP_SAMPLES = 256             # hop 16 ms x 16 kHz; HiFi-GAN's upsample_rates multiply to 256
REDUCTION_FACTOR = 2          # config.reduction_factor: 2 mel frames per decoder step
SPEAKER_EMBEDDING_DIM = 512   # config.speaker_embedding_dim
VOICES = ["slt", "clb", "bdl", "rms", "ksp"]    # CMU ARCTIC: slt + clb female, rest male
DEMO_TEXT = "the sun provides energy for life on earth"   # lowercase, no digits: in-vocabulary
N_STREAM = 40                 # rows pulled from the stream; 40 < 100, so one parquet row group
SEED = 6                      # SpeechT5 applies dropout at INFERENCE, so every call is seeded

# Whisper decodes with num_beams=5 by default, which is ~5x slower on CPU for a barely
# better transcript. Greedy is the right call for a round-trip check.
ASR_GEN = {"task": "transcribe", "language": "english", "num_beams": 1}

EVAL_SENTENCES = [
    "the sun provides energy for life on earth",
    "she took the early train from the station",
    "please check my account balance today",
    "a cheaper way to travel is by coach",
    "the mastering was done by a local studio",
    "flights to the northern airport were delayed",
]

RUN_BARK = False   # set True to download suno/bark-small (~1.7 GB) for one non-verbal clip in
                   # section 4; Unit 2's generation_demo.py already covers Bark end to end


def save_fig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   saved {path.relative_to(Path(__file__).parent)}")


def save_wav(name: str, audio, sampling_rate: int = SAMPLING_RATE) -> None:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:  # VITS occasionally overshoots; players clip float WAVs above 1.0
        audio = audio / peak
    path = FIG_DIR / name
    sf.write(path, audio, sampling_rate)
    print(f"   saved {path.relative_to(Path(__file__).parent)}  "
          f"({len(audio) / sampling_rate:.1f}s @ {sampling_rate} Hz, peak {peak:.2f})")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_tts():
    """SpeechT5 + its vocoder + the processor. ~636 MB on a cold machine."""
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    processor = SpeechT5Processor.from_pretrained(MODEL_ID)
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID)
    model.eval()
    torch.set_grad_enabled(False)
    return processor, model, vocoder


def speak(processor, model, speaker, text: str, vocoder=None, seed: int = SEED):
    """One seeded generation. Returns (output, seconds).

    Seeding is not optional here: the speech decoder pre-net applies dropout even in
    eval(), so two identical calls give different-length output. Section 4 proves it.
    """
    import torch

    ids = processor(text=text, return_tensors="pt")["input_ids"]
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    out = model.generate_speech(ids, speaker, vocoder=vocoder)
    return out, time.perf_counter() - t0


def load_xvectors():
    """The pre-computed CMU ARCTIC x-vectors, plus one fixed vector per named voice."""
    import torch
    from datasets import load_dataset

    xv = load_dataset(XVECTOR_ID, split=XVECTOR_SPLIT)
    filenames = xv["filename"]
    chosen = {}
    for voice in VOICES:
        prefix = f"cmu_us_{voice}_"
        idx = next(i for i, f in enumerate(filenames) if f.startswith(prefix))
        chosen[voice] = torch.tensor(xv[idx]["xvector"]).unsqueeze(0)
    return xv, chosen


def stream_clips(n: int = N_STREAM):
    """Stream n rows of the dialect corpus, resampled to 16 kHz.

    The config is one parquet file of 8 row groups, 100 rows each, so streaming n < 100
    range-reads roughly 59 MB instead of the full 395 MB. The source audio is 48 kHz, so
    the cast is mandatory: hand SpeechT5 48 kHz and every duration is three times wrong.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(row)
    return rows


# ===========================================================================
# 1. TWO MODELS, NOT ONE
# ===========================================================================
def section_two_stage(processor, model, vocoder, speaker):
    banner("1. Text to speech is two models: an acoustic model and a vocoder")

    print(
        "\n   SpeechT5 does NOT output audio. It outputs a log-mel spectrogram - a picture of\n"
        "   which frequencies are loud over time. A second, separately trained network (the\n"
        "   vocoder, here HiFi-GAN) turns that picture into a waveform you can play. You\n"
        "   fine-tune the first model; the vocoder stays frozen."
    )

    mel, t_mel = speak(processor, model, speaker, DEMO_TEXT)
    wav, t_wav = speak(processor, model, speaker, DEMO_TEXT, vocoder=vocoder)

    ids = processor(text=DEMO_TEXT, return_tensors="pt")["input_ids"]
    print(f"\n     text            : {DEMO_TEXT!r}")
    print(f"     input_ids       : {tuple(ids.shape)}  ({ids.shape[1]} character tokens)")
    print(f"     spectrogram     : {tuple(mel.shape)}  (frames x {NUM_MEL_BINS} mel bins)  {t_mel:.1f}s")
    print(f"     waveform        : {tuple(wav.shape)}  {t_wav:.1f}s")
    print(f"     duration        : {len(wav) / SAMPLING_RATE:.2f}s")

    print(
        f"\n   The two models meet at one number. A mel frame is {HOP_SAMPLES} samples of hop\n"
        f"   (16 ms x 16 kHz), and HiFi-GAN's upsample_rates multiply to 4*4*4*4 = {HOP_SAMPLES}.\n"
        f"   Those being the same number is the entire contract: any vocoder trained on\n"
        f"   {NUM_MEL_BINS}-bin mels with the same hop can be swapped in."
    )
    expected = mel.shape[0] * HOP_SAMPLES
    print(f"\n     frames x {HOP_SAMPLES}    : {expected}")
    print(f"     actual samples  : {len(wav)}   (difference: {len(wav) - expected})")
    print("     Exactly equal - but only because both calls above were seeded. Compare two")
    print("     UNSEEDED generations and they differ, because they are different lengths, not")
    print("     because the arithmetic changed. That is section 4's subject.")

    mel_np = np.asarray(mel)
    wav_np = np.asarray(wav)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    secs = mel_np.shape[0] * HOP_SAMPLES / SAMPLING_RATE
    im = ax1.imshow(mel_np.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, secs, 0, NUM_MEL_BINS])
    ax1.set(ylabel="mel bin", title="SpeechT5 writes this (log-mel spectrogram)")
    fig.colorbar(im, ax=ax1, pad=0.01)
    ax2.plot(np.arange(len(wav_np)) / SAMPLING_RATE, wav_np, lw=0.4, color="tab:blue")
    ax2.set(xlabel="seconds", ylabel="amplitude",
            title="HiFi-GAN turns it into this (waveform)")
    ax2.set_xlim(0, secs)
    fig.tight_layout()
    save_fig("01_mel_to_waveform.png")
    save_wav("02_speecht5_demo.wav", wav_np)


# ===========================================================================
# 2. THE 81-TOKEN FRONT END
# ===========================================================================
def section_text_frontend(processor, clips):
    banner("2. The 81-token front end, and the <unk> that never warns you")

    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    unk_id = tokenizer.unk_token_id
    print(
        f"\n   SpeechT5's vocabulary is {len(vocab)} CHARACTERS, not word pieces. Anything outside\n"
        "   it becomes <unk> silently - no warning, no error, just wrong speech. So auditing\n"
        "   the text is a real step, and the obvious way to do it is wrong."
    )
    print(f"\n     len(get_vocab())     : {len(vocab)}   (matches config.vocab_size)")
    print(f"     tokenizer.vocab_size : {tokenizer.vocab_size}   (excludes the added <mask>, <ctc_blank>)")
    print(f"     <unk> id             : {unk_id}")

    texts = [c["text"] for c in clips]
    chars = set("".join(texts))

    # The wrong audit, run first so you watch it fail.
    naive_bad = sorted(chars - set(vocab))
    print("\n   The WRONG audit - compare characters against get_vocab() keys:")
    print(f"     flagged as missing   : {''.join(naive_bad)!r}")
    print("     note the leading SPACE. This is a sentencepiece model: a space is stored as")
    print("     the word-boundary marker U+2581, so a literal ' ' is absent from the keys.")
    naive_keep = [t for t in texts if not (set(t) - set(vocab))]
    print(f"     rows this check keeps: {len(naive_keep)} of {len(texts)}  <- it rejects everything")

    # The right audit: ask the tokenizer.
    def unsupported(text):
        return unk_id in tokenizer(text, add_special_tokens=False).input_ids

    real_bad = sorted(c for c in chars if unsupported(c))
    print("\n   The RIGHT audit - ask whether the tokenizer emits <unk>:")
    print(f"     genuinely missing    : {''.join(real_bad)!r}")
    print(f"     space is fine        : {not unsupported(' ')}")
    real_keep = [t for t in texts if not unsupported(t)]
    print(f"     rows this check keeps: {len(real_keep)} of {len(texts)}")

    hostile = "In 1984 he paid 25 pounds"
    ids = tokenizer(hostile, add_special_tokens=False).input_ids
    toks = tokenizer.convert_ids_to_tokens(ids)
    print(f"\n   Round-tripping {hostile!r}:")
    print(f"     tokens : {' '.join(toks[:24])}{' ...' if len(toks) > 24 else ''}")
    print(f"     <unk> count: {toks.count('<unk>')}  <- every digit")

    # The escape hatch the course never mentions.
    normalized = None
    try:
        from transformers import SpeechT5Tokenizer

        ntok = SpeechT5Tokenizer.from_pretrained(MODEL_ID, normalize=True)
        nids = ntok(hostile, add_special_tokens=False).input_ids
        normalized = ntok.decode(nids)
        print("\n   SpeechT5Tokenizer ships an EnglishNumberNormalizer, off by default:")
        print(f"     normalize=True -> {normalized!r}")
        print(f"     <unk> now      : {ntok.unk_token_id in nids}")
        print("     So digits are recoverable. The hands-on still DROPS those rows, because a")
        print("     training run wants a predictable corpus more than it wants 30 extra clips.")
    except Exception as exc:
        print(f"\n     normalize=True unavailable: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:90]}")

    policies = ["raw", "lowercase", "lowercase + normalize"]
    kept = [
        len([t for t in texts if not unsupported(t)]),
        len([t for t in texts if not unsupported(t.lower())]),
        len(texts) if normalized else len([t for t in texts if not unsupported(t.lower())]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(policies, kept, color=["tab:red", "tab:orange", "tab:green"])
    for b, k in zip(bars, kept):
        ax.text(b.get_x() + b.get_width() / 2, k, str(k), ha="center", va="bottom")
    ax.set(ylabel=f"rows kept of {len(texts)}",
           title="Cleaning policy decides how much of your corpus survives")
    ax.set_ylim(0, len(texts) * 1.15)
    fig.tight_layout()
    save_fig("03_vocabulary_and_unk.png")


# ===========================================================================
# 3. SPEAKER CONTROL
# ===========================================================================
def section_speaker_control(processor, model, vocoder, xv, chosen):
    banner("3. Five hundred and twelve floats decide whose voice it is")

    import torch

    print(
        "\n   SpeechT5 is multi-speaker: alongside the text it takes a 512-dimensional x-vector\n"
        "   saying WHOSE voice to use. The course computes these with SpeechBrain; we use the\n"
        "   pre-computed CMU ARCTIC set instead, because speechbrain.pretrained was renamed to\n"
        "   speechbrain.inference (the course's import is a straight ImportError) and SpeechBrain\n"
        "   drags in torchaudio, which this repo deliberately does not install."
    )
    print(f"\n     {XVECTOR_ID} [{XVECTOR_SPLIT}]: {len(xv)} rows x {SPEAKER_EMBEDDING_DIM} dims")
    print("\n     I-Vectors  - GMM statistics, unsupervised, low-dimensional")
    print("     X-Vectors  - a DNN trained to discriminate speakers, with temporal context")
    print("                  (the course: state of the art, and what SpeechT5 expects)")

    durations = {}
    waves = {}
    for voice, vec in chosen.items():
        wav, secs = speak(processor, model, vec, DEMO_TEXT, vocoder=vocoder)
        waves[voice] = np.asarray(wav)
        durations[voice] = len(wav) / SAMPLING_RATE
        print(f"     cmu_us_{voice:<4} -> {durations[voice]:.2f}s of audio  ({secs:.1f}s to generate)")
    print("\n   Same text, same seed, five vectors, five different durations. The x-vector does")
    print("   not just colour the timbre - it reaches the duration mechanism too.")

    # Magnitude is irrelevant: the decoder pre-net L2-normalises the vector before use.
    v = chosen[VOICES[0]]
    a, _ = speak(processor, model, v, DEMO_TEXT)
    b, _ = speak(processor, model, v * 10.0, DEMO_TEXT)
    same = a.shape == b.shape and float((a - b).abs().max()) < 1e-5
    print(f"\n     scaling an x-vector by 10 changes the output: {not same}")
    print("     (the speech decoder pre-net calls F.normalize on it, so only direction matters -")
    print("      which is also why section 8 compares voices with cosine, not Euclidean distance)")

    print(
        "\n   The course's introduction page ends on ethics, and this is the right place for it:\n"
        "   a 512-float vector is all it takes to put words in someone's voice. Voice data needs\n"
        "   explicit, informed consent covering purpose, scope and risk. The five vectors used\n"
        "   here come from CMU ARCTIC, a public research corpus recorded for exactly this, which\n"
        "   is why they are the ones in this file."
    )

    gap = np.zeros(int(0.4 * SAMPLING_RATE), dtype=np.float32)
    save_wav("04_five_voices.wav", np.concatenate([np.concatenate([waves[v], gap]) for v in VOICES]))

    mat = torch.cat([chosen[v] for v in VOICES], dim=0)
    mat = torch.nn.functional.normalize(mat, dim=1)
    cos = (mat @ mat.T).numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    im = ax1.imshow(cos, cmap="viridis", vmin=-0.2, vmax=1.0)
    ax1.set(xticks=range(len(VOICES)), yticks=range(len(VOICES)),
            xticklabels=VOICES, yticklabels=VOICES,
            title="x-vector cosine similarity\n(slt + clb female, bdl + rms + ksp male)")
    for i in range(len(VOICES)):
        for j in range(len(VOICES)):
            ax1.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                     color="white" if cos[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax1, pad=0.01)
    ax2.bar(VOICES, [durations[v] for v in VOICES], color=[f"C{k}" for k in range(len(VOICES))])
    for i, v in enumerate(VOICES):
        ax2.text(i, durations[v], f"{durations[v]:.2f}", ha="center", va="bottom")
    ax2.set(ylabel="seconds", title="Same sentence, same seed, five voices")
    fig.tight_layout()
    save_fig("05_xvector_space.png")
    return waves


# ===========================================================================
# 4. THREE DESIGNS, AND ONE-TO-MANY
# ===========================================================================
def section_designs(processor, model, vocoder, speaker):
    banner("4. Three designs, and the one-to-many problem underneath all of them")

    rows = [
        ("SpeechT5", "log-mel", "yes (HiFi-GAN)", "x-vector", "English", "585 MB"),
        ("Bark", "waveform", "no (EnCodec)", "voice presets", "multilingual", "1.7 GB"),
        ("MMS / VITS", "waveform", "no (built in)", "none", "1100+ langs", "145 MB each"),
    ]
    print(f"\n     {'model':<12}{'output':<10}{'vocoder?':<17}{'speaker ctrl':<16}"
          f"{'languages':<14}{'size'}")
    for r in rows:
        print(f"     {r[0]:<12}{r[1]:<10}{r[2]:<17}{r[3]:<16}{r[4]:<14}{r[5]}")
    print("\n   MMS covers 1100+ languages but ships ONE CHECKPOINT PER LANGUAGE and gives you no")
    print("   voice control at all. Ten languages means ten downloads, each with a fixed voice.")

    try:
        import torch
        from transformers import VitsModel, VitsTokenizer

        vt = VitsTokenizer.from_pretrained(MMS_ID)
        vm = VitsModel.from_pretrained(MMS_ID)
        t0 = time.perf_counter()
        with torch.no_grad():
            vw = vm(**vt(text=DEMO_TEXT, return_tensors="pt")).waveform
        vw = np.asarray(vw).squeeze()
        print(f"\n     {MMS_ID}: {len(vw) / vm.config.sampling_rate:.2f}s "
              f"in {time.perf_counter() - t0:.1f}s, sr={vm.config.sampling_rate}")
        save_wav("06_mms_vits_eng.wav", vw, vm.config.sampling_rate)
    except Exception as exc:
        print(f"\n     {MMS_ID} unavailable: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:100]}")

    if RUN_BARK:
        try:
            from transformers import AutoProcessor, BarkModel

            bp = AutoProcessor.from_pretrained(BARK_ID)
            bm = BarkModel.from_pretrained(BARK_ID)
            inp = bp("This is a test [laughter] and now a pause ...",
                     voice_preset="v2/en_speaker_3")
            bw = np.asarray(bm.generate(**inp)).squeeze()
            save_wav("06b_bark_nonverbal.wav", bw, bm.generation_config.sample_rate)
        except Exception as exc:
            print(f"     Bark unavailable: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    else:
        print(f"\n     (RUN_BARK=False, so {BARK_ID} is not downloaded; Unit 2's")
        print("      generation_demo.py already runs Bark end to end.)")

    print(
        "\n   Now the problem the datasets page opens with: TTS is ONE-TO-MANY. The same text has\n"
        "   many valid spoken forms. You can watch it happen - two identical calls, same weights,\n"
        "   same x-vector, model.eval(), no_grad, and still different output:"
    )
    a, _ = speak(processor, model, speaker, DEMO_TEXT, seed=SEED)
    b, _ = speak(processor, model, speaker, DEMO_TEXT, seed=SEED)
    import torch

    torch.manual_seed(SEED)
    ids = processor(text=DEMO_TEXT, return_tensors="pt")["input_ids"]
    c1 = model.generate_speech(ids, speaker)
    c2 = model.generate_speech(ids, speaker)   # no reseed
    print(f"\n     same seed   : {a.shape[0]} vs {b.shape[0]} frames, identical = "
          f"{a.shape == b.shape and float((a - b).abs().max()) == 0.0}")
    print(f"     no reseed   : {c1.shape[0]} vs {c2.shape[0]} frames  <- different lengths")
    print("\n   The cause is the speech decoder pre-net, which applies dropout p=0.5 even in eval()")
    print("   (the source comment says so outright, citing Tacotron 2). The practical rule:")
    print("   MODEL.EVAL() DOES NOT MAKE THIS MODEL DETERMINISTIC - torch.manual_seed() does.")

    m1, m2 = np.asarray(c1), np.asarray(c2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, m, label in ((ax1, m1, "run 1"), (ax2, m2, "run 2")):
        ax.imshow(m.T, aspect="auto", origin="lower", cmap="magma")
        ax.set(xlabel="frame", title=f"{label}: {m.shape[0]} frames")
    ax1.set(ylabel="mel bin")
    fig.suptitle("Identical inputs, identical weights, different speech "
                 "(dropout is on at inference)")
    fig.tight_layout()
    save_fig("07_one_to_many.png")
    save_wav("08_one_to_many.wav",
             np.concatenate([np.asarray(speak(processor, model, speaker, DEMO_TEXT,
                                              vocoder=vocoder, seed=s)[0])
                             for s in (SEED, SEED + 1)]))


# ===========================================================================
# 5. TTS DATA
# ===========================================================================
def section_datasets(clips):
    banner("5. Choosing TTS data, and why an ASR corpus is the wrong tool")

    print(
        "\n   The course names three challenges. One-to-many (section 4 just proved it).\n"
        "   Long-distance dependencies: prosody has to stay coherent over a whole sentence.\n"
        "   And the data itself: the things that make a corpus GOOD for ASR - background noise,\n"
        "   varied channels, spontaneous speech - are exactly what you do not want in TTS. It is\n"
        "   useful to transcribe a noisy street; it is not useful for your assistant to reply with\n"
        "   traffic behind it."
    )

    corpora = [
        ("LJSpeech", "1 speaker", "13,100 clips", "script", "no"),
        ("Multilingual LibriSpeech", "many", "7 languages", "parquet", "yes"),
        ("VCTK", "110 speakers", "~400 each", "script", "no"),
        ("LibriTTS-R", "many", "~585 h, 24 kHz", "script", "no"),
        (f"{CONFIG} (used here)", "5 speakers", "750 clips", "parquet", "yes"),
    ]
    print(f"\n     {'corpus':<28}{'speakers':<14}{'size':<18}{'builder':<10}{'loads on 3.6.0?'}")
    for c in corpora:
        print(f"     {c[0]:<28}{c[1]:<14}{c[2]:<18}{c[3]:<10}{c[4]}")
    print("\n   THREE OF THE FOUR CORPORA THE COURSE RECOMMENDS NO LONGER LOAD. lj_speech, vctk and")
    print("   libritts-r-aligned are loading-script datasets, and script execution was removed in")
    print("   datasets 3.0. Only multilingual_librispeech has been converted to parquet.")

    durations = [len(c["audio"]["array"]) / SAMPLING_RATE for c in clips]
    lengths = [len(c["text"]) for c in clips]
    speakers = {}
    for c in clips:
        speakers[c["speaker_id"]] = speakers.get(c["speaker_id"], 0) + 1
    print(f"\n   Streamed {len(clips)} rows of {CONFIG} (~59 MB, one parquet row group):")
    print(f"     duration   : {min(durations):.1f}-{max(durations):.1f}s, mean {np.mean(durations):.1f}s")
    print(f"     characters : {min(lengths)}-{max(lengths)}, mean {np.mean(lengths):.0f}")
    print(f"     speakers   : {dict(sorted(speakers.items()))}")
    print("\n   The course's three characteristics of good TTS data: clean and varied recordings,")
    print("   a transcription for every clip, and linguistic variety. This corpus has all three")
    print("   at small scale - which is the point, since the hands-on has no metric to hit.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.scatter(lengths, durations, alpha=0.7, color="tab:blue")
    ax1.set(xlabel="characters of text", ylabel="seconds of audio",
            title="Text length predicts audio length\n(the relationship a TTS model has to learn)")
    ax2.bar([str(s) for s in sorted(speakers)], [speakers[s] for s in sorted(speakers)],
            color="tab:green")
    ax2.set(xlabel="speaker_id", ylabel="clips in the stream",
            title=f"Five speakers, evenly split\n(streaming the first {len(clips)} rows still sees all of them)")
    fig.tight_layout()
    save_fig("09_dataset_landscape.png")


# ===========================================================================
# 6. FINE-TUNING INNARDS
# ===========================================================================
def section_finetune_innards(processor, clips):
    banner("6. What the TTS collator builds, and why it is Unit 5 inverted")

    import torch

    print(
        "\n   In Unit 5 the labels were token ids. Here they are a SPECTROGRAM: one 80-bin vector\n"
        "   per frame. That one difference drives every quirk of the TTS collator."
    )

    batch = []
    for c in clips[:4]:
        out = processor(
            text=c["text"], audio_target=c["audio"]["array"],
            sampling_rate=SAMPLING_RATE, return_attention_mask=False,
        )
        raw = out["labels"]
        out["labels"] = out["labels"][0]
        batch.append(out)
        if len(batch) == 1:
            print(f"\n     processor(...) returns labels with shape {np.asarray(raw).shape}")
            print(f"     after out['labels'][0]          : {np.asarray(out['labels']).shape}")
            print("     THAT INDEX IS REQUIRED HERE. In Unit 5 the same line was the bug: there")
            print("     the labels are a flat list of ids and [0] takes a single integer.")

    lens = [len(b["labels"]) for b in batch]
    print(f"\n     four clips, label frames: {lens}")

    padded = processor.pad(
        input_ids=[{"input_ids": b["input_ids"]} for b in batch],
        labels=[{"input_values": b["labels"]} for b in batch],
        return_tensors="pt",
    )
    labels = padded["labels"].masked_fill(padded.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
    target_lengths = torch.tensor(lens)
    target_lengths = target_lengths.new([l - l % REDUCTION_FACTOR for l in target_lengths])
    truncated = labels[:, : max(target_lengths)]

    print(f"     padded labels           : {tuple(padded['labels'].shape)}")
    print(f"     after -100 masking      : {int((labels == -100).all(-1).sum())} padded frames ignored")
    print(f"     reduction_factor={REDUCTION_FACTOR} rounds : {lens} -> {target_lengths.tolist()}")
    print(f"     final labels            : {tuple(truncated.shape)}")
    print("\n   The decoder emits 2 mel frames per step, so an odd target leaves the loss")
    print("   misaligned against the predictions. And decoder_attention_mask is deleted after")
    print("   masking - the model does not take it during training.")

    mask = (truncated[0] == -100).numpy()
    fig, ax = plt.subplots(figsize=(13, 4))
    show = truncated[0].numpy().copy()
    show[mask] = np.nan
    ax.imshow(show.T, aspect="auto", origin="lower", cmap="magma")
    ax.axvline(target_lengths[0].item(), color="tab:red", ls="--", lw=2,
               label=f"real audio ends at frame {target_lengths[0].item()}")
    ax.legend(loc="upper right")
    ax.set(xlabel="frame", ylabel="mel bin",
           title="Padded label batch — blank columns are -100 and contribute no loss")
    fig.tight_layout()
    save_fig("10_label_padding.png")


# ===========================================================================
# 7. HOW TTS FAILS
# ===========================================================================
def section_failure_modes(processor, model, vocoder, speaker):
    banner("7. How TTS fails: the stop token, babbling, and text it cannot hold")

    print(
        "\n   An autoregressive TTS model decides for itself when to stop: a small head predicts a\n"
        "   stop probability per step, and generation ends when it crosses a threshold. Two things\n"
        "   can go wrong - stopping early and clipping the sentence, or never stopping and babbling\n"
        "   until the length cap (maxlen = text_length * maxlenratio / reduction_factor)."
    )

    short, _ = speak(processor, model, speaker, "hello", vocoder=vocoder)
    long_text = " ".join([DEMO_TEXT] * 4)
    long_out, secs = speak(processor, model, speaker, long_text, vocoder=vocoder)
    short_rate = len(short) / len("hello") / SAMPLING_RATE * 1000
    long_rate = len(long_out) / len(long_text) / SAMPLING_RATE * 1000
    print(f"\n     'hello' ({len('hello')} chars)        -> {len(short) / SAMPLING_RATE:.2f}s  "
          f"({short_rate:.0f} ms/char)")
    print(f"     the demo text x4 ({len(long_text)} chars) -> {len(long_out) / SAMPLING_RATE:.2f}s  "
          f"({long_rate:.0f} ms/char)  {secs:.1f}s to generate")
    print("\n   Both stopped correctly here, so this is the healthy case, not the failure. The short")
    print("   utterance costs more per character simply because a fixed lead-in and tail are spread")
    print("   over five characters instead of 160. Runaway generation is real but not reproducible")
    print("   on demand from a healthy checkpoint - it shows up on under-trained models, which is")
    print("   why the hands-on watches the loss rather than trusting one listen.")

    tokenizer = processor.tokenizer
    unk_id = tokenizer.unk_token_id
    broken = "In 1984 he paid 25"
    ids = tokenizer(broken, add_special_tokens=False).input_ids
    print(f"\n     {broken!r} -> {ids.count(unk_id)} <unk> of {len(ids)} tokens")
    unk_wav, _ = speak(processor, model, speaker, broken, vocoder=vocoder)
    print(f"     it still generates {len(unk_wav) / SAMPLING_RATE:.2f}s of confident-sounding audio.")
    print("     Nothing raised. Nothing warned. That is the failure mode section 2 is about.")
    save_wav("11_unk_speech.wav", np.asarray(unk_wav))

    fig, ax = plt.subplots(figsize=(13, 3.5))
    w = np.asarray(long_out)
    ax.plot(np.arange(len(w)) / SAMPLING_RATE, w, lw=0.3, color="tab:purple")
    ax.set(xlabel="seconds", ylabel="amplitude",
           title="A long prompt: watch the tail for the stop token firing late")
    fig.tight_layout()
    save_fig("12_stop_token.png")


# ===========================================================================
# 8. EVALUATING FOR REAL
# ===========================================================================
def section_evaluate(processor, model, vocoder, chosen, voice_waves):
    banner("8. Evaluating TTS: round-trip WER, a pitch check, and a blind MOS sheet")

    print(
        "\n   The course's evaluation page recommends NO automatic metric. Its position is that TTS\n"
        "   is one-to-many, so quality is subjective and MOS - humans scoring 1 to 5 - is the only\n"
        "   honest measure. That is true, and it is also not very actionable, so this section adds\n"
        "   two proxies that are cheap and say something real."
    )

    from transformers import pipeline
    import jiwer

    asr = pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)
    speaker = chosen[VOICES[0]]
    refs, hyps = [], []
    print("\n   (a) Round-trip intelligibility: synthesise, then transcribe with whisper-tiny.")
    for text in EVAL_SENTENCES:
        wav, _ = speak(processor, model, speaker, text, vocoder=vocoder)
        out = asr({"array": np.asarray(wav), "sampling_rate": SAMPLING_RATE},
                  generate_kwargs=ASR_GEN)
        hyp = out["text"].strip().lower().rstrip(".").strip()
        refs.append(text)
        hyps.append(hyp)
        mark = "ok " if hyp == text else "DIFF"
        print(f"     {mark} {hyp}")
    wer = jiwer.wer(refs, hyps)
    print(f"\n     round-trip WER: {wer:.4f}")
    print("     A LOW WER MEANS INTELLIGIBLE, NOT NATURAL. It cannot see prosody, pacing or")
    print("     whether the voice sounds human - only whether an ASR model recovered the words.")
    if wer == 0.0:
        print("     A perfect 0.0 is itself the lesson: these are short, common, in-vocabulary")
        print("     sentences, so the metric has almost no room to discriminate. Treat it as a")
        print("     regression alarm - it catches a model that broke - not as a quality score.")

    print("\n   (b) Did the x-vector actually change the VOICE, or only the x-vector?")
    print("       Measure the generated audio, not the embedding that produced it: estimate the")
    print("       fundamental frequency of each clip and see whether the voices separate.")
    import librosa

    pitches = {}
    for voice in VOICES:
        y = np.asarray(voice_waves[voice], dtype=np.float32)
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=SAMPLING_RATE)
        pitches[voice] = float(np.median(f0))
    for voice in VOICES:
        sex = "female" if voice in ("slt", "clb") else "male"
        print(f"     cmu_us_{voice:<4} median f0: {pitches[voice]:6.1f} Hz   ({sex} in CMU ARCTIC)")
    female = np.mean([pitches[v] for v in ("slt", "clb")])
    male = np.mean([pitches[v] for v in ("bdl", "rms", "ksp")])
    print(f"\n     female mean {female:.1f} Hz vs male mean {male:.1f} Hz "
          f"-> separation {female - male:+.1f} Hz")
    print("     This is a measurement of the OUTPUT. Comparing the input x-vectors to each other")
    print("     would tell you nothing about whether the model used them.")

    print("\n   (c) A blind MOS sheet. The only measure the course endorses, made runnable:")
    print("     figures/13_mos_a.wav, _b.wav, _c.wav are the same sentence from three voices.")
    print("     Listen without looking, score each 1-5 for naturalness, then check the key below.")
    key = VOICES[:3]
    for letter, voice in zip("abc", key):
        save_wav(f"13_mos_{letter}.wav", voice_waves[voice])
    print(f"     key: a={key[0]}  b={key[1]}  c={key[2]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    per = [jiwer.wer(r, h) for r, h in zip(refs, hyps)]
    ax1.bar(range(len(per)), per, color="tab:blue")
    ax1.axhline(wer, color="tab:red", ls="--", label=f"corpus WER {wer:.3f}")
    ax1.legend()
    ax1.set(xlabel="sentence", ylabel="WER",
            title="Round-trip WER per sentence\n(intelligibility, not naturalness)")
    colours = ["tab:pink" if v in ("slt", "clb") else "tab:blue" for v in VOICES]
    ax2.bar(VOICES, [pitches[v] for v in VOICES], color=colours)
    for i, v in enumerate(VOICES):
        ax2.text(i, pitches[v], f"{pitches[v]:.0f}", ha="center", va="bottom")
    ax2.set(ylabel="median f0 (Hz)",
            title="The x-vector reached the output\n(pink = female voices, blue = male)")
    fig.tight_layout()
    save_fig("14_tts_evaluation.png")


def main() -> None:
    print("Hugging Face Audio Course — Unit 6: From text to speech")
    print(f"Figures will be written to: {FIG_DIR}")

    print("\nLoading SpeechT5 + HiFi-GAN (~636 MB on a cold machine)...")
    processor, model, vocoder = load_tts()
    print("Loading the CMU ARCTIC x-vectors (~18 MB)...")
    xv, chosen = load_xvectors()
    speaker = chosen[VOICES[0]]
    print(f"Streaming {N_STREAM} rows of {DATASET_ID} [{CONFIG}] (~59 MB)...")
    clips = stream_clips()

    section_two_stage(processor, model, vocoder, speaker)
    section_text_frontend(processor, clips)
    voice_waves = section_speaker_control(processor, model, vocoder, xv, chosen)
    section_designs(processor, model, vocoder, speaker)
    section_datasets(clips)
    section_finetune_innards(processor, clips)
    section_failure_modes(processor, model, vocoder, speaker)
    section_evaluate(processor, model, vocoder, chosen, voice_waves)

    banner("Done!  See figures/ for the plots.")
    print("Next steps:")
    print("   notebook.ipynb       - the same material with inline plots and playable audio")
    print("   finetune.py          - fine-tune SpeechT5 (CPU smoke test, or full run on a GPU)")
    print("   gradio_demo.py       - type a sentence, pick a voice, hear it")
    print("   colab_handson.ipynb  - the graded hands-on: fine-tune, push, and fix the tag")
    print("\nRemember: SPEECHT5'S VOCABULARY IS 81 CHARACTERS AND ANYTHING OUTSIDE IT BECOMES")
    print("<unk> SILENTLY. Audit by tokenizing, never by comparing against get_vocab() keys.")
    print("\nSupplemental reading from the course:")
    print("   HiFi-GAN vocoder    https://arxiv.org/pdf/2010.05646.pdf")
    print("   X-Vectors           https://www.danielpovey.com/files/2018_icassp_xvectors.pdf")
    print("   FastSpeech 2        https://arxiv.org/pdf/2006.04558.pdf")
    print("   MQTTS               https://arxiv.org/pdf/2302.04215v1.pdf")


if __name__ == "__main__":
    main()
