"""
Hugging Face Audio Course - Unit 7: Putting it all together
============================================================

Units 1-6 built the parts. Unit 7 bolts them together, and the lesson is that composition
is where the errors live: every seam is a place to lose information, add latency, or be
silently wrong. So this script measures each seam rather than the end of the pipe:

 1. The cascade        - three models, one sentence, and what each seam costs
 2. Language forcing   - Whisper translates into English only, and the "trick" that isn't
 3. Error propagation  - an ASR mistake is unrecoverable, because stage 2 never sees audio
 4. Latency            - additive, per stage, and the fast cascade cannot pass the hands-on
 5. The text bottleneck- two voices in, one voice out
 6. The voice assistant- four stages, two of which no longer exist
 7. Transcribe a meeting - word timestamps, the merge, and diarization you cannot fake
 8. The hands-on       - what the assessor actually calls, and the int16 trap
 9. The last seam      - a 39-symbol vocabulary that deletes what it cannot say

Unit 5 gave us ASR, Unit 6 gave us TTS. Neither needed the other. This unit is the first
where the failure of one component shows up as a confusing failure of a different one.

Run with:

    uv run python units/unit7_putting_it_all_together/walkthrough.py

Models used: openai/whisper-base (~290 MB) and whisper-tiny (~151 MB, both cached since
Units 2-5), Helsinki-NLP/opus-mt-en-fr (~301 MB), facebook/mms-tts-fra and -eng (~145 MB
each), MBZUAI/LaMini-Flan-T5-248M (~990 MB), MIT/ast-finetuned-speech-commands-v2 (~342 MB,
cached since Unit 4), and Unit 6's SpeechT5 stack for section 5. Section 9 adds the Arabic pair,
Helsinki-NLP/opus-mt-en-ar (~307 MB) and facebook/mms-tts-ara (~145 MB). Section 7 streams ~59 MB
of ylacombe/english_dialects.

Itemised, that is 301 + 145 + 990 + 307 + 145 + 59 = about 1.95 GB new on a cold machine, into
~/.cache/huggingface, not the repo. Everything runs on CPU in about 6-11 minutes.

Deliberately NOT downloaded: facebook/mms-lid-126 (3.86 GB), the language classifier the
grading Space uses. Section 8 explains why, and why that Space is currently down.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which can't encode "→" or the French accents this
# unit prints constantly. Force UTF-8 so the prints never crash.
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

ASR_ID = "openai/whisper-base"          # 290 MB, cached since Unit 5; stage 1 of the cascade
ASR_TINY = "openai/whisper-tiny"        # 151 MB, cached since Unit 2; the section 4 sweep
ASR_SMALL = "openai/whisper-small"      # 967 MB, cached since Unit 5; section 4, behind a flag
MT_ID = "Helsinki-NLP/opus-mt-en-fr"    # 301 MB, 75M params; MarianTokenizer is sentencepiece-only
MT_ARA = "Helsinki-NLP/opus-mt-en-ar"   # 307 MB; section 9's second target, and the Space's
TTS_FRA = "facebook/mms-tts-fra"        # 145 MB, phonemize:false so no espeak-ng needed
TTS_ARA = "facebook/mms-tts-ara"        # 145 MB; 39 symbols, no digits, NO punctuation at all
TTS_ENG = "facebook/mms-tts-eng"        # 145 MB, cached since Unit 6; the 2-stage baseline's voice
LLM_ID = "MBZUAI/LaMini-Flan-T5-248M"   # 990 MB; the Hub's pipeline_tag on this model is WRONG
WAKE_ID = "MIT/ast-finetuned-speech-commands-v2"   # 342 MB, cached since Unit 4
T5_ID = "microsoft/speecht5_tts"        # Unit 6's stack, used in section 5 as the SOURCE voice
VOC_ID = "microsoft/speecht5_hifigan"
XVECTOR_ID = "Matthijs/cmu-arctic-xvectors"
DUMMY_ID = "hf-internal-testing/librispeech_asr_dummy"   # cached since Unit 5, ~9 MB
DIALECTS_ID = "ylacombe/english_dialects"   # Unit 6's corpus; section 7 streams ~59 MB of it
DIALECTS_CONFIG = "northern_female"         # 5 real speakers, same accent and sex: the hard case

SAMPLING_RATE = 16_000
WAKE_WORD = "marvin"          # AST id2label[27]; google/speech_commands' own index is 26
SEED = 7                      # VITS has a stochastic duration predictor: seed every call
N_SEAM = 6                    # sentences scored in the section 3 seam analysis

# Whisper decodes with num_beams=5 by default, ~5x slower on CPU for a barely better
# transcript. Greedy is the right call for a walkthrough.
ASR_GEN = {"task": "translate", "num_beams": 1}

USE_WHISPER_SMALL = False     # set True to add openai/whisper-small (~967 MB) to section 4

_CACHE: dict[str, object] = {}


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


def asr_pipe(model_id: str = ASR_ID):
    """Build an ASR pipeline once and reuse it (loading weights is the slow part)."""
    from transformers import pipeline

    key = f"asr:{model_id}"
    if key not in _CACHE:
        _CACHE[key] = pipeline("automatic-speech-recognition", model=model_id, device=-1)
    return _CACHE[key]


def translator(model_id: str = MT_ID):
    """MarianMT en->xx. Sentencepiece-only tokenizer: there is no fast variant."""
    from transformers import MarianMTModel, MarianTokenizer

    key = f"mt:{model_id}"
    if key not in _CACHE:
        tok = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def speaker(model_id: str = TTS_FRA):
    """MMS/VITS: text straight to waveform, no vocoder. phonemize:false, so no espeak-ng."""
    from transformers import VitsModel, VitsTokenizer

    key = f"tts:{model_id}"
    if key not in _CACHE:
        _CACHE[key] = (VitsTokenizer.from_pretrained(model_id),
                       VitsModel.from_pretrained(model_id))
    return _CACHE[key]


def translate_text(text: str, model_id: str = MT_ID) -> str:
    import torch

    tok, model = translator(model_id)
    with torch.no_grad():
        out = model.generate(**tok(text, return_tensors="pt"),
                             num_beams=1, max_new_tokens=256)
    return tok.decode(out[0], skip_special_tokens=True)


def synthesise(text: str, model_id: str = TTS_FRA, seed: int = SEED):
    import torch

    tok, model = speaker(model_id)
    torch.manual_seed(seed)   # VITS's duration predictor is stochastic
    with torch.no_grad():
        wav = model(**tok(text=text, return_tensors="pt")).waveform
    return np.asarray(wav, dtype=np.float32).squeeze()


def load_dummy():
    """73 clean LibriSpeech validation clips with human references and speaker ids."""
    from datasets import load_dataset

    if "dummy" not in _CACHE:
        _CACHE["dummy"] = load_dataset(DUMMY_ID, "clean", split="validation")
    return _CACHE["dummy"]


def audio_dict(array) -> dict:
    """A FRESH dict for every pipeline call.

    AutomaticSpeechRecognitionPipeline.preprocess mutates the dict you hand it - it moves
    "array" into "raw" in place - so reusing one dict across two calls makes the second
    raise `ValueError: ... the dict needs to contain a "raw" key`, which reads like your
    input was malformed rather than already consumed.
    """
    return {"array": np.asarray(array), "sampling_rate": SAMPLING_RATE}


# ===========================================================================
# 1. THE CASCADE
# ===========================================================================
def section_cascade(clip):
    banner("1. The cascade: three models, one sentence, and what each seam costs")

    print(
        "\n   The course builds speech-to-speech as TWO stages: Whisper with task='translate',\n"
        "   then TTS. That produces ENGLISH, always. The hands-on requires a non-English target,\n"
        "   so this unit uses THREE stages: ASR -> machine translation -> TTS. The course's own\n"
        "   introduction warns that more components means more error propagation and more\n"
        "   latency. Sections 3 and 4 measure both instead of taking its word for it."
    )

    arr = clip["audio"]["array"]
    src_secs = len(arr) / SAMPLING_RATE

    t0 = time.perf_counter()
    english = asr_pipe()(
        audio_dict(arr),
        generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
    )["text"].strip()
    t_asr = time.perf_counter() - t0

    t0 = time.perf_counter()
    french = translate_text(english)
    t_mt = time.perf_counter() - t0

    t0 = time.perf_counter()
    wav = synthesise(french)
    t_tts = time.perf_counter() - t0

    # Run the whole thing a SECOND time, now that every checkpoint is resident. The first
    # pass above is what a cold process costs, because each stage loaded its own weights
    # inside its own timer; the second is what the models actually cost to run. Reporting
    # only one of the two is how a latency table ends up an order of magnitude out, which is
    # the mistake section 4 used to make.
    t0 = time.perf_counter()
    asr_pipe()(audio_dict(arr),
               generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1})
    w_asr = time.perf_counter() - t0
    t0 = time.perf_counter()
    translate_text(english)
    w_mt = time.perf_counter() - t0
    t0 = time.perf_counter()
    synthesise(french)
    w_tts = time.perf_counter() - t0

    print(f"\n     source audio    : {src_secs:.2f}s")
    print(f"     1. ASR   : {english}")
    print(f"     2. MT    : {french}")
    print(f"     3. TTS   : {len(wav) / SAMPLING_RATE:.2f}s of French audio")

    total, warm = t_asr + t_mt + t_tts, w_asr + w_mt + w_tts
    print(f"\n     {'':<14}{'ASR':>9}{'MT':>9}{'TTS':>9}{'total':>9}{'RTF':>8}")
    print(f"     {'first call':<14}{t_asr:9.2f}{t_mt:9.2f}{t_tts:9.2f}{total:9.2f}"
          f"{total / src_secs:8.2f}")
    print(f"     {'warm':<14}{w_asr:9.2f}{w_mt:9.2f}{w_tts:9.2f}{warm:9.2f}"
          f"{warm / src_secs:8.2f}")
    print(f"\n     {total - warm:.2f}s of that first row was loading weights, not running them.")
    print("     Quote the wrong row and you will conclude this cascade is unusable. Section 4")
    print("     times the warm one, and every number in it is deliberate.")

    print(
        "\n   Look at what crosses each seam: stage 1 hands stage 2 a STRING. Not audio, not a\n"
        "   lattice, not a confidence - a string. That is the whole reason error propagation is\n"
        "   possible, and the reason section 5's result is what it is."
    )

    save_wav("01_cascade_source.wav", arr)
    save_wav("02_cascade_french.wav", wav)

    stages = ["ASR\nwhisper-base", "MT\nopus-mt-en-fr", "TTS\nmms-tts-fra"]
    params = [72.6, 75.1, 36.3]   # millions, as reported by the checkpoints
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for row, (times, label) in enumerate(((( t_asr, t_mt, t_tts), "first call"),
                                          ((w_asr, w_mt, w_tts), "warm"))):
        left = 0.0
        for i, (name, secs, colour) in enumerate(
                zip(stages, times, ["tab:blue", "tab:orange", "tab:green"])):
            ax1.barh(row, secs, left=left, color=colour, height=0.55,
                     label=name.replace("\n", " ") if row == 0 else None)
            if secs > 0.25:
                ax1.text(left + secs / 2, row, f"{secs:.1f}", ha="center", va="center",
                         color="white", fontsize=8)
            left += secs
    ax1.axvline(src_secs, color="tab:red", ls="--", lw=2, label=f"audio length {src_secs:.1f}s")
    ax1.set(yticks=[0, 1], yticklabels=["first call", "warm"], xlabel="seconds",
            title="Latency is additive, and the first call is mostly loading")
    ax1.legend(loc="lower right", fontsize=8)
    ax2.bar(stages, params, color=["tab:blue", "tab:orange", "tab:green"])
    for i, p in enumerate(params):
        ax2.text(i, p, f"{p:.0f}M", ha="center", va="bottom")
    ax2.set(ylabel="parameters (millions)", title=f"{sum(params):.0f}M parameters in total")
    fig.tight_layout()
    save_fig("03_cascade_stages.png")
    return english, french


# ===========================================================================
# 2. LANGUAGE FORCING
# ===========================================================================
def section_language_forcing(clip, mt_french):
    banner("2. Whisper translates INTO English only, and the \"trick\" that isn't one")

    print(
        "\n   The course's speech-to-speech page offers a shortcut: rather than adding a translation\n"
        "   model, 'trick' Whisper into X->Y by setting task='transcribe' and language=Y. It is a\n"
        "   tempting shortcut, because it removes a whole stage. Here is what it actually produces."
    )

    combos = [
        ("transcribe, language=english", {"task": "transcribe", "language": "english"}),
        ("transcribe, language=es", {"task": "transcribe", "language": "spanish"}),
        ("transcribe, language=french", {"task": "transcribe", "language": "french"}),
        ("translate,  language=es", {"task": "translate", "language": "spanish"}),
        ("translate   (no language)", {"task": "translate"}),
    ]
    results = {}
    for label, gen in combos:
        gen = dict(gen, num_beams=1)
        out = asr_pipe()(audio_dict(clip["audio"]["array"]),
                         generate_kwargs=gen)["text"].strip()
        results[label] = out
        print(f"\n     {label:<30} {out}")

    print(f"\n     MT hop (the correct way)       {mt_french}")

    same = results["translate,  language=es"] == results["translate   (no language)"]
    print(f"\n   translate+language=es identical to plain translate: {same}")
    print("   -> with task='translate', the language kwarg is SILENTLY IGNORED. Whisper's")
    print("      translation objective has exactly one target, and it is English.")

    import jiwer

    trick = results["transcribe, language=french"]
    cers = {label: jiwer.cer(mt_french, text) for label, text in results.items()}
    print(f"\n   Character error rate against the MT French, for each decode:")
    for label, cer in cers.items():
        print(f"     {label:<30} {cer:.3f}")

    print(
        "\n   The forced-French decode is not a translation; it is a transcription that has been\n"
        "   nudged toward French spelling. Listen to 05_trick_vs_mt.wav: the trick first, then the\n"
        "   MT hop. Nothing raised, nothing warned - it just quietly produced worse French."
    )

    gap = np.zeros(int(0.4 * SAMPLING_RATE), dtype=np.float32)
    save_wav("05_trick_vs_mt.wav",
             np.concatenate([synthesise(trick), gap, synthesise(mt_french)]))

    labels = [k.replace(", ", ",\n") for k in cers]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colours = ["tab:red" if "translate" in k else "tab:orange" for k in cers]
    bars = ax.bar(labels, list(cers.values()), color=colours)
    for b, v in zip(bars, cers.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set(ylabel="CER vs the MT French",
           title="No Whisper setting reaches the MT hop\n(red = translate task, which only ever emits English)")
    fig.tight_layout()
    save_fig("04_whisper_language_forcing.png")


# ===========================================================================
# 3. ERROR PROPAGATION
# ===========================================================================
def section_error_propagation(ds):
    banner("3. Error propagation: measure every seam, not just the end of the pipe")

    import jiwer

    print(
        "\n   A cascade hides where the damage happened. The end user hears bad French and blames\n"
        "   the TTS. So measure each seam separately: run the MT hop TWICE, once on what the ASR\n"
        "   heard and once on the human reference, and the difference between the two French\n"
        "   strings is exactly the damage the ASR caused."
    )

    rows = []
    for i in range(N_SEAM):
        clip = ds[i]
        ref = clip["text"].lower()
        hyp = asr_pipe()(
            audio_dict(clip["audio"]["array"]),
            generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
        )["text"].strip().lower().rstrip(".")
        asr_wer = jiwer.wer(ref, hyp)
        fr_hyp = translate_text(hyp)
        fr_ref = translate_text(ref)
        mt_cer = jiwer.cer(fr_ref, fr_hyp)
        rows.append((asr_wer, mt_cer))
        print(f"\n     clip {i}  ASR WER {asr_wer:.3f}  |  French divergence CER {mt_cer:.3f}")
        if asr_wer > 0:
            print(f"       ref : {ref[:74]}")
            print(f"       hyp : {hyp[:74]}")

    asr_wers = [r[0] for r in rows]
    mt_cers = [r[1] for r in rows]
    print(f"\n     mean ASR WER              : {np.mean(asr_wers):.3f}")
    print(f"     mean French divergence CER: {np.mean(mt_cers):.3f}")
    print(
        "\n   Where the ASR was clean the French is identical; where it slipped, the French\n"
        "   inherited the slip and made it worse. Stage 2 cannot recover, because stage 2 never\n"
        "   sees the audio - only the string. That is the price of the extra stage, and it is the\n"
        "   thing the course's introduction warns about in one sentence without measuring."
    )

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, asr_wers, 0.4, label="ASR seam (WER vs human reference)", color="tab:blue")
    ax.bar(x + 0.2, mt_cers, 0.4, label="MT seam (CER: ASR-French vs gold-French)", color="tab:orange")
    ax.axhline(np.mean(asr_wers), color="tab:blue", ls="--", lw=1)
    ax.axhline(np.mean(mt_cers), color="tab:orange", ls="--", lw=1)
    ax.set(xticks=x, xlabel="clip", ylabel="error rate",
           title="Damage attributed to the seam that caused it")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig("06_error_propagation.png")


# ===========================================================================
# 4. LATENCY
# ===========================================================================
def section_latency(ds):
    banner("4. Latency is additive, and the fast cascade cannot pass the hands-on")

    import jiwer

    models = [ASR_TINY, ASR_ID] + ([ASR_SMALL] if USE_WHISPER_SMALL else [])
    clip = ds[0]
    secs = len(clip["audio"]["array"]) / SAMPLING_RATE
    ref = clip["text"].lower()

    # Warm EVERY checkpoint this section will time. Without this the first model's load
    # lands inside t_asr and the first voice's load lands inside t_two, and the table below
    # reports import cost as latency. Measured on this machine: cold, whisper-tiny's 3-stage
    # row came out at 14.57s against a 2-stage 3.68s, which prices the MT hop at eleven
    # seconds. Warm, the same two rows are 1.79s and 1.58s, so the hop costs about two
    # tenths of a second. The conclusion survives either way; the magnitude does not, and
    # the magnitude is what anyone reading a latency table takes away from it.
    print("\n   Warming every checkpoint first, so the table below times inference and not")
    print("   import. A model load in the wrong place is worth ten seconds of imaginary MT.")
    for model_id in models:
        asr_pipe(model_id)
    translator()
    speaker(TTS_FRA)
    speaker(TTS_ENG)

    print(f"\n   One clip ({secs:.1f}s), each ASR checkpoint in slot 1, both cascades:")
    print(f"\n     {'model':<22}{'ASR s':>7}{'WER':>8}{'3-stage s':>11}{'2-stage s':>11}")
    points, deltas = [], []
    for model_id in models:
        t0 = time.perf_counter()
        hyp = asr_pipe(model_id)(
            audio_dict(clip["audio"]["array"]),
            generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
        )["text"].strip().lower().rstrip(".")
        t_asr = time.perf_counter() - t0
        wer = jiwer.wer(ref, hyp)

        t0 = time.perf_counter()
        fr = translate_text(hyp)
        synthesise(fr)
        t_rest3 = time.perf_counter() - t0

        t0 = time.perf_counter()
        en = asr_pipe(model_id)(audio_dict(clip["audio"]["array"]),
                                generate_kwargs=ASR_GEN)["text"].strip()
        synthesise(en, TTS_ENG)
        t_two = time.perf_counter() - t0

        name = model_id.split("/")[-1]
        print(f"     {name:<22}{t_asr:7.2f}{wer:8.3f}{t_asr + t_rest3:11.2f}{t_two:11.2f}")
        points.append((name, (t_asr + t_rest3) / secs, wer))
        deltas.append((t_asr + t_rest3) - t_two)

    gap = float(np.mean(deltas))
    print(f"\n   The third stage costs {gap:+.2f}s on average here, and it is what buys a target")
    print("   language at all: the 2-stage cascade produces ENGLISH, which is exactly what the")
    print("   hands-on forbids. You are not choosing between fast and slow. You are paying a")
    print("   fraction of a second for the only output that can pass.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, rtf, wer in points:
        ax.scatter(rtf, wer, s=90)
        ax.annotate(name, (rtf, wer), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set(xlabel="real-time factor of the 3-stage cascade (lower is faster)",
           ylabel="ASR WER (lower is better)",
           title="Pick your stage-1 checkpoint\n(bottom-left is what you want)")
    fig.tight_layout()
    save_fig("07_latency_frontier.png")


# ===========================================================================
# 5. THE TEXT BOTTLENECK
# ===========================================================================
def section_prosody_bottleneck():
    banner("5. What the text bottleneck destroys: two voices in, one voice out")

    import librosa
    import torch
    from datasets import load_dataset
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    print(
        "\n   Unit 6 showed that a 512-float x-vector decides whose voice SpeechT5 uses. Feed two\n"
        "   clearly different voices into this cascade and see what comes out the far end."
    )

    proc = SpeechT5Processor.from_pretrained(T5_ID)
    t5 = SpeechT5ForTextToSpeech.from_pretrained(T5_ID).eval()
    voc = SpeechT5HifiGan.from_pretrained(VOC_ID)
    xv = load_dataset(XVECTOR_ID, split="validation")
    names = xv["filename"]

    sentence = "the sun provides energy for life on earth"
    outputs = {}
    for voice in ("slt", "bdl"):
        idx = next(i for i, f in enumerate(names) if f.startswith(f"cmu_us_{voice}_"))
        vec = torch.tensor(xv[idx]["xvector"]).unsqueeze(0)
        ids = proc(text=sentence, return_tensors="pt")["input_ids"]
        torch.manual_seed(SEED)
        with torch.no_grad():
            src = t5.generate_speech(ids, vec, vocoder=voc)
        src = np.asarray(src, dtype=np.float32)
        f0 = float(np.median(librosa.yin(src, fmin=60, fmax=400, sr=SAMPLING_RATE)))

        heard = asr_pipe()(
            audio_dict(src),
            generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
        )["text"].strip()
        fr = translate_text(heard)
        out = synthesise(fr)
        outputs[voice] = (src, f0, heard, fr, out)
        print(f"\n     {voice}: source f0 {f0:6.1f} Hz, {len(src) / SAMPLING_RATE:.2f}s")
        print(f"        heard : {heard}")
        print(f"        french: {fr}")
        print(f"        output: {len(out) / SAMPLING_RATE:.2f}s")

    a, b = outputs["slt"][4], outputs["bdl"][4]
    identical = a.shape == b.shape and float(np.max(np.abs(a - b))) == 0.0
    print(f"\n     source f0 differ by {abs(outputs['slt'][1] - outputs['bdl'][1]):.1f} Hz")
    print(f"     output waveforms identical: {identical}"
          f"  (max abs diff {float(np.max(np.abs(a - b))) if a.shape == b.shape else float('nan'):.3e})")
    print(
        "\n   Everything about HOW the sentence was said - pitch, pace, speaker - died at the first\n"
        "   seam, because a string cannot carry it. This is the argument for direct speech-to-speech\n"
        "   models (the course's supplemental reading covers Translatotron), which skip the text\n"
        "   bottleneck entirely. None of them run on a CPU in a few seconds, which is why the\n"
        "   cascade is still what you build."
    )

    save_wav("08_two_voices_in.wav",
             np.concatenate([outputs["slt"][0], np.zeros(int(0.4 * SAMPLING_RATE), np.float32),
                             outputs["bdl"][0]]))
    save_wav("09_one_voice_out.wav",
             np.concatenate([a, np.zeros(int(0.4 * SAMPLING_RATE), np.float32), b]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    for voice, style in (("slt", "-"), ("bdl", "--")):
        src = outputs[voice][0]
        f0 = librosa.yin(src, fmin=60, fmax=400, sr=SAMPLING_RATE)
        ax1.plot(np.linspace(0, len(src) / SAMPLING_RATE, len(f0)), f0, style,
                 label=f"{voice} ({outputs[voice][1]:.0f} Hz)")
    ax1.set(ylabel="f0 (Hz)", title="Two voices in - clearly different")
    ax1.legend(fontsize=8)
    for voice, style, w in (("slt", "-", 2.5), ("bdl", "--", 1.2)):
        out = outputs[voice][4]
        f0 = librosa.yin(out, fmin=60, fmax=400, sr=SAMPLING_RATE)
        ax2.plot(np.linspace(0, len(out) / SAMPLING_RATE, len(f0)), f0, style, lw=w, label=voice)
    ax2.set(xlabel="seconds", ylabel="f0 (Hz)",
            title="One voice out - the two curves are superimposed")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save_fig("10_prosody_erased.png")


# ===========================================================================
# 6. THE VOICE ASSISTANT
# ===========================================================================
def section_assistant(ds):
    banner("6. The voice assistant's four stages, and the two that no longer exist")

    from transformers import pipeline

    print(
        "\n   Wake word -> transcribe -> ask a language model -> speak the answer. Two of the four\n"
        "   stages in the course are no longer runnable, and neither failure is obvious."
    )

    print("\n   Stage 1: wake word. We synthesise the words rather than needing a microphone.")
    clf = pipeline("audio-classification", model=WAKE_ID, device=-1)
    scores = {}
    for word in (WAKE_WORD, "stop", "yes", "house"):
        wav = synthesise(word, TTS_ENG)
        top = clf(audio_dict(wav), top_k=1)[0]
        scores[word] = (top["label"], top["score"])
        hit = "WAKE" if top["label"] == WAKE_WORD and top["score"] > 0.5 else "    "
        print(f"     {hit} {word:<8} -> {top['label']:<12} {top['score']:.3f}  "
              f"({len(wav) / SAMPLING_RATE:.2f}s)")
    print(f"\n     The AST label set has {len(clf.model.config.id2label)} classes and "
          f"'{WAKE_WORD}' is id {clf.model.config.label2id[WAKE_WORD]}.")
    print("     google/speech_commands puts it at 26 of 36. Compare label STRINGS, never ints.")

    print("\n   Stage 2: transcribe. Reusing whisper-base from section 1.")
    clip = ds[2]
    command = asr_pipe()(
        audio_dict(clip["audio"]["array"]),
        generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
    )["text"].strip()
    print(f"     heard: {command}")

    print("\n   Stage 3: the language model. The course calls tiiuae/falcon-7b-instruct over")
    print("   api-inference.huggingface.co - a host whose DNS record no longer resolves, and a")
    print("   model whose inferenceProviderMapping is now empty. So we run a small one locally.")

    question = "Answer in one short sentence: what is the capital of France?"
    # transformers logs every supported architecture here - a single 3,273-character line
    # listing 200+ class names. The warning is real and is exactly the point, but printing
    # it in full buries the lesson. It is logged at ERROR level, not WARNING, so disable() has
    # to be told ERROR or it sails straight through.
    import logging

    logging.disable(logging.ERROR)
    wrong = pipeline("text-generation", model=LLM_ID, device=-1)
    echoed = wrong(question, max_new_tokens=40)[0]["generated_text"].strip()
    logging.disable(logging.NOTSET)
    right = pipeline("text2text-generation", model=LLM_ID, device=-1)
    t0 = time.perf_counter()
    answer = right(question, max_new_tokens=40)[0]["generated_text"].strip()
    t_llm = time.perf_counter() - t0

    print("\n     (transformers does log an error here - \"T5ForConditionalGeneration is not")
    print("      supported for text-generation\" - muted above because it lists 200+ class")
    print("      names. An error you have to scroll past is an error you stop reading.)")
    print(f"\n     pipeline('text-generation', ...)      -> {echoed!r}")
    print(f"     pipeline('text2text-generation', ...) -> {answer!r}   ({t_llm:.2f}s)")
    print(f"     echoed the prompt back verbatim: {echoed.strip() == question.strip()}")
    print("\n     The Hub's pipeline_tag for this model says 'text-generation', which is wrong -")
    print("     it is a T5, a seq2seq. Pass the task explicitly or your assistant is a parrot")
    print("     that never raises. This is the quietest bug in the unit.")

    print("\n   Stage 4: speak the answer.")
    reply = synthesise(answer, TTS_ENG)
    save_wav("12_assistant_reply.wav", reply)

    print("\n   Also dead: `from transformers import HfAgent`, which the course's closing section")
    print("   uses to generalise the assistant. Agents were removed from transformers entirely.")

    fig, ax = plt.subplots(figsize=(10, 4))
    words = list(scores)
    vals = [scores[w][1] for w in words]
    colours = ["tab:green" if scores[w][0] == WAKE_WORD else "tab:grey" for w in words]
    bars = ax.bar(words, vals, color=colours)
    for b, w in zip(bars, words):
        ax.text(b.get_x() + b.get_width() / 2, scores[w][1], scores[w][0],
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="tab:red", ls="--", label="wake threshold")
    ax.set(ylabel="top-1 score", ylim=(0, 1.15),
           title="Wake-word detection on synthesised words\n(green = classified as the wake word)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig("11_wakeword_scores.png")


# ===========================================================================
# 7. TRANSCRIBE A MEETING
# ===========================================================================
def section_meeting(ds):
    banner("7. Transcribing a meeting: word timestamps, the merge, and fake diarization")

    print(
        "\n   Two jobs: work out WHAT was said and WHO said it, then align them. Whisper does the\n"
        "   first with word timestamps. The second needs a speaker-embedding model, and that is\n"
        "   where this stack runs out of road - so we measure exactly how far short it falls."
    )

    # The LibriSpeech dummy is a single speaker (1272 x 73 rows), so it cannot make a
    # two-speaker track. Stream Unit 6's dialect corpus instead: five real speakers, same
    # accent, same sex - the HARD case, which is what makes the number below credible.
    meeting, truth, pair = build_meeting()
    print(f"\n     built a {len(meeting) / SAMPLING_RATE:.1f}s meeting from speakers {pair}, "
          f"{len(truth)} turns, boundaries exact by construction")

    words = asr_pipe()(audio_dict(meeting),
                       generate_kwargs={"task": "transcribe", "language": "english",
                                        "num_beams": 1},
                       return_timestamps="word")
    chunks = words["chunks"]
    print(f"     whisper word timestamps: {len(chunks)} chunks, "
          f"first {chunks[0]['text']!r} at {chunks[0]['timestamp']}")

    merged = merge_speakers(truth, chunks, len(meeting) / SAMPLING_RATE)
    print("\n   The speechbox merge, against the TRUE boundaries:")
    for seg in merged[:4]:
        print(f"     [{seg['speaker']}] ({seg['start']:.1f}-{seg['end']:.1f}s) {seg['text'][:56]}")

    acc, n_win = mfcc_diarization_accuracy(meeting, truth)
    chance = chance_baseline(n_win)
    print(f"\n   Now the part the course does with pyannote, which is gated and needs torchaudio.")
    print(f"   The only CPU-only substitute here is librosa MFCCs plus sklearn clustering:")
    print(f"\n     frame accuracy on this track : {acc:.1%}  ({n_win} one-second windows)")
    print(f"     RANDOM clustering scores     : {chance:.1%}  (measured, 20k trials at n={n_win})")
    print(f"     so this method beats chance by {acc - chance:+.1%}")
    print("\n   That second number is the one everybody omits, including the first draft of this")
    print("   script. The scorer takes max() over the two cluster-to-speaker mappings, because")
    print("   cluster ids are arbitrary, so it CANNOT report below 50% and on a short track it")
    print("   lands near 60% on luck alone. Quoting 50% as the coin flip turns a null result into")
    print("   a weak positive one. MFCCs encode WHAT was said far more strongly than WHO said it,")
    print("   so without a speaker-embedding model you are clustering phonetics. meeting.py sweeps")
    print("   every speaker pair and scores each one against this same measured null.")

    fig, ax = plt.subplots(figsize=(13, 3.5))
    colours = {pair[0]: "tab:blue", pair[1]: "tab:orange"}
    for start, end, sid in truth:
        ax.barh(1, end - start, left=start, height=0.6, color=colours[sid])
    for c in chunks:
        ts = c["timestamp"]
        if ts and ts[0] is not None:
            ax.plot([ts[0], ts[0]], [0.35, 0.65], color="0.3", lw=0.6)
    ax.set(yticks=[1], yticklabels=["true speaker"], xlabel="seconds",
           title=f"Two-speaker track: true turns (bars) and Whisper word onsets (ticks), "
                 f"MFCC clustering scored {acc:.0%} against a {chance:.0%} null")
    fig.tight_layout()
    save_fig("13_meeting_timeline.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["this track"], [acc * 100], color="tab:red" if acc < chance else "tab:green")
    ax.axhline(50, color="0.75", ls=":", label="the 50% everyone quotes")
    ax.axhline(chance * 100, color="tab:red", ls=":",
               label=f"RANDOM clustering, measured ({chance:.0%} at n={n_win})")
    ax.axhline(80, color="tab:green", ls="--", label="usable threshold")
    ax.text(0, acc * 100, f"{acc:.1%}", ha="center", va="bottom")
    ax.set(ylabel="frame accuracy (%)", ylim=(0, 105),
           title="MFCC + AgglomerativeClustering is not diarization\n"
                 "(the null is not 50%: the scorer picks the better label mapping)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig("14_diarization_accuracy.png")


def build_meeting(n_turns: int = 4):
    """A two-speaker track with boundaries that are exact by construction.

    Streams Unit 6's dialect corpus (one ~59 MB parquet row group) and interleaves turns
    from the two most frequent speakers. Source audio is 48 kHz, so the cast is mandatory.
    """
    from datasets import Audio, load_dataset

    stream = load_dataset(DIALECTS_ID, DIALECTS_CONFIG, split="train", streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    per_speaker: dict[int, list] = {}
    for _, row in zip(range(60), stream):
        per_speaker.setdefault(row["speaker_id"], []).append(row["audio"]["array"])
    pair = sorted(per_speaker, key=lambda s: -len(per_speaker[s]))[:2]

    turns, truth, cursor = [], [], 0.0
    for t in range(n_turns):
        sid = pair[t % 2]
        arr = np.asarray(per_speaker[sid][t // 2], dtype=np.float32)
        turns.append(arr)
        truth.append((cursor, cursor + len(arr) / SAMPLING_RATE, sid))
        cursor += len(arr) / SAMPLING_RATE
    return np.concatenate(turns), truth, pair


def merge_speakers(segments, chunks, total_s):
    """The speechbox ASRDiarizationPipeline merge, reimplemented with three guards.

    speechbox aligns ONLY on segment end times and consumes the transcript greedily, which
    is fine on well-behaved input and raises on everything else. The guards below are the
    three cases it does not handle: an emptied timestamp array, a None end timestamp (which
    Whisper emits for a final unterminated chunk), and a single segment, where its merge
    loop body never runs at all.
    """
    transcript = list(chunks)
    ends = np.array(
        [c["timestamp"][-1] if c["timestamp"] and c["timestamp"][-1] is not None else total_s
         for c in transcript],
        dtype=float,
    )
    out = []
    for start, end, sid in segments:
        if ends.size == 0:   # guard: speechbox raises "argmin of an empty sequence" here
            break
        upto = int(np.argmin(np.abs(ends - end)))
        out.append({
            "speaker": sid,
            "start": start,
            "end": end,
            "text": "".join(c["text"] for c in transcript[: upto + 1]).strip(),
        })
        transcript = transcript[upto + 1:]
        ends = ends[upto + 1:]
    return out


def mfcc_diarization_accuracy(audio, truth, win_s: float = 1.0):
    """Cluster 1-second windows by MFCC mean and score against the true speaker labels.

    Returns (accuracy, n_windows). The window count is not decoration: it is the only thing
    the chance baseline below depends on, and the accuracy is misleading without it.
    """
    import librosa
    from sklearn.cluster import AgglomerativeClustering

    step = int(win_s * SAMPLING_RATE)
    feats, labels = [], []
    for start in range(0, len(audio) - step, step):
        centre = (start + step / 2) / SAMPLING_RATE
        who = next((sid for s, e, sid in truth if s <= centre < e), None)
        if who is None:
            continue
        mfcc = librosa.feature.mfcc(y=audio[start:start + step], sr=SAMPLING_RATE, n_mfcc=20)
        feats.append(mfcc.mean(axis=1))
        labels.append(who)
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    pred = AgglomerativeClustering(n_clusters=2, metric="cosine",
                                   linkage="average").fit_predict(np.array(feats))
    truth_arr = np.array([sorted(set(labels)).index(l) for l in labels])
    acc = max((pred == truth_arr).mean(), (pred != truth_arr).mean())   # labels are arbitrary
    return float(acc), len(labels)


def chance_baseline(n_windows: int, trials: int = 20_000, seed: int = SEED) -> float:
    """What the scorer above returns when the clustering is RANDOM. It is not 50%.

    Cluster ids are arbitrary, so that scorer takes max() over the two cluster-to-speaker
    mappings. The max can never fall below 0.5, and on a short track it sits well above it
    on luck alone: with n windows the expected score is about 0.5 + sqrt(2 / (pi * n)) / 2,
    which is 59% at n=19 and still 54% at n=100.

    So "50% is a coin flip" flatters every number quoted beside it, and an accuracy in the
    high 50s on a 20-second track is not a weak positive result - it is chance. This measures
    the null on the track you actually have instead of assuming one.

    The arrangement of the true labels does not enter into it: for iid uniform predictions
    the agreement is Binomial(n, 1/2) / n whatever the truth looks like.
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_windows, dtype=int)
    truth[n_windows // 2:] = 1
    agree = (rng.integers(0, 2, size=(trials, n_windows)) == truth).mean(axis=1)
    return float(np.maximum(agree, 1.0 - agree).mean())


# ===========================================================================
# 8. THE HANDS-ON CONTRACT
# ===========================================================================
def section_handson(french):
    banner("8. The hands-on contract: what the assessor actually calls")

    print(
        "\n   The hands-on asks you to duplicate the course's Space and make it translate into a\n"
        "   NON-English language. The grader does not read your code. It does this:\n"
        "\n     client = Client(repo_id)\n"
        "     audio_file = client.predict(\"test_short.wav\", api_name=\"/predict\")\n"
        "     # then facebook/mms-lid-126 on the returned audio, and it fails you if the\n"
        "     # top label is 'eng' or its score is below 0.5\n"
        "\n   Two consequences most people miss. ONE POSITIONAL ARGUMENT IN, ONE RETURN VALUE OUT:\n"
        "   add a language dropdown or a second output and you fail with a gradio_client traceback\n"
        "   rather than a grading message. And the check is only 'not English, confidently' - it\n"
        "   never verifies you hit the language you claimed."
    )

    wav = synthesise(french)
    peak = float(np.max(np.abs(wav)))
    print(f"\n     our French output: {len(wav) / SAMPLING_RATE:.2f}s, peak {peak:.3f}")

    naive = (np.array([0.9, 1.0, 1.05, 1.5, -1.2], dtype=np.float32) * 32767).astype(np.int16)
    clipped = (np.clip(np.array([0.9, 1.0, 1.05, 1.5, -1.2], dtype=np.float32), -1.0, 1.0)
               * 32767).astype(np.int16)
    print(f"\n     the template's cast, (x * 32767).astype(np.int16):")
    print(f"       float  : [ 0.9, 1.0, 1.05, 1.5, -1.2]")
    print(f"       naive  : {naive.tolist()}   <- 1.05 became NEGATIVE")
    print(f"       clipped: {clipped.tolist()}")
    print(f"\n     mms-tts peaks below 1.0 today ({peak:.3f}), so this does not fire - but it is one")
    print("     np.clip from being safe, and a louder checkpoint would wrap the sign silently.")

    out = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    save_wav("16_handson_output.wav", out.astype(np.float32) / 32767.0)
    print(f"\n     the exact bytes the grader would consume: {out.dtype}, {out.shape}, "
          f"returned as (16000, array)")

    print("\n   The grading Space is currently RUNTIME_ERROR: it loads facebook/mms-lid-126")
    print("   (3.86 GB) at import and dies on 'No space left on device'. We deliberately do not")
    print("   download that model here. Build the Space, make it public, and retry later.")

    x = np.array([0.9, 1.0, 1.05, 1.5, -1.2], dtype=np.float32)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.6))
    ax1.stem(x)
    ax1.axhline(1.0, color="tab:red", ls="--")
    ax1.set(title="float samples", ylabel="amplitude")
    ax2.stem(naive)
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set(title="(x * 32767).astype(int16)\nsign wraps above 1.0")
    ax3.stem(clipped)
    ax3.axhline(0, color="0.5", lw=0.8)
    ax3.set(title="np.clip first\nsaturates correctly")
    fig.tight_layout()
    save_fig("15_int16_overflow.png")


# ===========================================================================
# 9. THE LAST SEAM
# ===========================================================================
def section_vocabulary(english, french):
    banner("9. The last seam: a vocabulary that deletes what it cannot say")

    print(
        "\n   Section 1 said stage 1 hands stage 2 a string. Stage 3 has the same problem in\n"
        "   reverse, and it is worse, because VITS is character-level and its vocabulary is\n"
        "   tiny. VitsTokenizer is built with normalize=True, and that path ends in this line\n"
        "   of transformers/models/vits/tokenization_vits.py:\n"
        "\n     filtered_text = \"\".join(list(filter(lambda char: char in self.encoder, ...)))\n"
        "\n   Not <unk>. DELETED. Unit 6's SpeechT5 at least emits an <unk> you can count. Here\n"
        "   the character is dropped from the string before the model ever sees it, nothing is\n"
        "   logged, and the audio comes out fluent and complete."
    )

    arabic = translate_text(english, MT_ARA)
    print(f"\n     the same English sentence, two targets:")
    print(f"       en -> fr : {french}")
    print(f"       en -> ar : {arabic}")

    # ".,!?;:" plus the Arabic comma, question mark and semicolon. Every claim below is
    # computed from these lists rather than written down, so the prose cannot drift away
    # from the vocabularies if a checkpoint is ever revised.
    SENTENCE_PUNCT = ".,!?;:" + "\u060c\u061f\u061b"

    rows = []
    for label, tts_id, text in (("English", TTS_ENG, english),
                                ("French", TTS_FRA, french),
                                ("Arabic", TTS_ARA, arabic)):
        tok, _ = speaker(tts_id)
        # get_vocab() folds in the added special tokens (<unk>, <pad>), which are not
        # symbols the model can pronounce. Count only what it can actually say.
        symbols = sorted(set(tok.get_vocab()) - set(tok.all_special_tokens))
        letters = [c for c in symbols if c.isalpha()]
        digits = [c for c in symbols if c.isdigit()]
        sentence = [c for c in SENTENCE_PUNCT if c in symbols]
        other = [c for c in symbols if not c.isalnum() and c not in sentence]

        normalised = tok.normalize_text(text)
        filtered, _ = tok.prepare_for_tokenization(text)
        dropped = sorted(set(normalised) - set(filtered))
        n_dropped = len(normalised) - len(filtered)
        rows.append({"label": label, "id": tts_id, "symbols": len(symbols),
                     "letters": len(letters), "digits": digits, "sentence": sentence,
                     "other": other, "n_dropped": n_dropped, "dropped": dropped})

        print(f"\n     {tts_id}")
        print(f"       pronounceable symbols : {len(symbols)}  ({len(letters)} of them letters)")
        print(f"       digits                : {len(digits)}  {''.join(digits) or '(none)'}")
        print(f"       sentence punctuation  : {len(sentence)}  {''.join(sentence) or '(NONE)'}")
        print(f"       everything else       : {''.join(other) or '(none)'}")
        print(f"       deleted from our text : {n_dropped} characters  "
              f"{''.join(dropped) or '(none)'}")

    no_punct = [r["label"] for r in rows if not r["sentence"]]
    no_digits = [r["label"] for r in rows if not r["digits"]]
    ara = next(r for r in rows if r["label"] == "Arabic")
    allof = " - which is all of them" if len(no_punct) == len(rows) else ""
    print(
        f"\n   Voices with NO sentence punctuation at all: {', '.join(no_punct)}{allof}.\n"
        f"   Voices with NO digits at all: {', '.join(no_digits)}.\n"
        "\n   A full stop is not mispronounced by these models, it is REMOVED, and with it every\n"
        "   prosodic cue a reader would take from it. English keeps digits 0 to 6 and stops\n"
        "   there, so even in English '1987' is only partly sayable and '1989' is not.\n"
        f"\n   Arabic is the extreme case: {ara['symbols']} symbols, {ara['letters']} of them "
        "bare letters, no digits, no\n"
        "   sentence punctuation, not even an apostrophe or a hyphen, and no harakat - so the\n"
        "   short vowels a diacritised text carries are deleted along with everything else."
    )
    print(
        "\n   This matters for the hands-on, because space/ targets Arabic. The grader only asks\n"
        "   'is this confidently not English', and deleted punctuation cannot make it fail. Your\n"
        "   Space passes while quietly dropping characters, which is this unit's whole thesis in\n"
        "   one line: the seam was lossy and the score did not notice."
    )

    save_wav("17_arabic_output.wav", synthesise(arabic, TTS_ARA), SAMPLING_RATE)

    # Latin labels only. DejaVu Sans renders Arabic unshaped and left-to-right, so a figure
    # that printed the dropped glyphs would misrepresent the script it is complaining about.
    # The counts are the measurement; the glyphs are printed above where the console can
    # render them properly.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    names = [r["label"] for r in rows]
    ax1.bar(names, [r["symbols"] for r in rows], color=["tab:blue", "tab:orange", "tab:green"])
    for i, r in enumerate(rows):
        ax1.text(i, r["symbols"],
                 f"{r['symbols']} symbols\n{r['letters']} letters\n{len(r['digits'])} digits\n"
                 f"{len(r['sentence'])} sentence punct",
                 ha="center", va="bottom", fontsize=8)
    ax1.set(ylabel="pronounceable symbols", ylim=(0, max(r["symbols"] for r in rows) * 1.5),
            title="Every MMS voice is a character vocabulary of about 40 symbols")
    ax2.bar(names, [r["n_dropped"] for r in rows],
            color=["tab:blue", "tab:orange", "tab:green"])
    for i, r in enumerate(rows):
        ax2.text(i, r["n_dropped"], str(r["n_dropped"]), ha="center", va="bottom")
    ax2.set(ylabel="characters deleted", title="Characters silently deleted from our own\n"
                                               "pipeline output, before the model saw it")
    fig.tight_layout()
    save_fig("18_vocabulary_filter.png")


def main() -> None:
    print("Hugging Face Audio Course — Unit 7: Putting it all together")
    print(f"Figures will be written to: {FIG_DIR}")

    ds = load_dummy()
    clip = ds[0]
    print(f"Demo clip: row 0 of {len(ds)} in {DUMMY_ID}")

    english, french = section_cascade(clip)
    section_language_forcing(clip, french)
    section_error_propagation(ds)
    section_latency(ds)
    section_prosody_bottleneck()
    section_assistant(ds)
    section_meeting(ds)
    section_handson(french)
    section_vocabulary(english, french)

    banner("Done!  See figures/ for the plots.")
    print("Next steps:")
    print("   notebook.ipynb   - the same material with inline plots and playable audio")
    print("   assistant.py     - the four-stage voice assistant (canned, or --live in Gradio)")
    print("   meeting.py       - word timestamps + the merge, and the diarization sweep")
    print("   gradio_demo.py   - four tabs: translate, compare cascades, assistant, meeting")
    print("   space/           - the hands-on Space, ready to upload (not published)")
    print("\nAlso remember: EVERY mms-tts VOICE SILENTLY DELETES CHARACTERS IT HAS NO SYMBOL")
    print("FOR. None of them has a full stop, a comma or a question mark.")
    print("\nRemember: WHISPER'S task='translate' ONLY EVER PRODUCES ENGLISH. THE HANDS-ON")
    print("REQUIRES NON-ENGLISH. ADD A TRANSLATION HOP - DO NOT FORCE A LANGUAGE TOKEN.")
    print("\nSupplemental reading from the course:")
    print("   STST with discrete units   https://ai.meta.com/research/publications/textless-speech-to-speech-translation-on-real-data/")
    print("   Translatotron 2            https://arxiv.org/abs/2107.08661")
    print("   Wake word detection (CNN)  https://www.amazon.science/publications/accurate-detection-of-wake-word-start-and-end-using-a-cnn")
    print("   pyannote.audio 2.1         https://huggingface.co/pyannote/speaker-diarization")
    print("   WhisperX                   https://arxiv.org/abs/2303.00747")


if __name__ == "__main__":
    main()
