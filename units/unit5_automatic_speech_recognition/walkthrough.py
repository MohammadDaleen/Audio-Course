"""
Hugging Face Audio Course - Unit 5: Automatic speech recognition
=================================================================

Unit 5 is about turning speech into text: which pre-trained models to use, which
dataset to train on, how to *measure* a transcription, and how to fine-tune. This
script makes every one of those concepts runnable on CPU:

 1. CTC vs seq2seq      - Wav2Vec2 and Whisper transcribe the same clip, differently
 2. The Whisper family  - tiny/base/small/medium/large, .en twins, speed vs accuracy
 3. Multilingual        - one model, two tasks: "transcribe" and "translate"
 4. Long-form audio     - the 30-second wall, chunking, and timestamps
 5. Choosing a dataset  - the eight English ASR corpora and the four axes that matter
 6. WER by hand         - substitutions, insertions, deletions; word accuracy; CER
 7. Evaluating for real - orthographic vs normalised WER, and why the ranking flips
 8. Fine-tuning innards - what `prepare_dataset` and the data collator actually build

Unit 3 already covered the *mechanics* (CTC blank collapse, Whisper's task tokens);
this unit is about choosing, measuring and adapting. The training run itself lives in
`finetune.py`, the demo in `gradio_demo.py`, and the graded hands-on in
`colab_handson.ipynb`.

Run with:

    uv run python units/unit5_automatic_speech_recognition/walkthrough.py

Models used: facebook/wav2vec2-base-960h (~360 MB), openai/whisper-tiny (~150 MB),
openai/whisper-base (~290 MB) and openai/whisper-small (~970 MB, behind a flag).
All four are already cached from Units 2-3, so a re-run downloads nothing; a cold
machine pulls ~1.77 GB into ~/.cache/huggingface, not the repo. Everything runs on
CPU in about 4-6 minutes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters like "→" or
# Whisper's "<|...|>" task tokens. Force UTF-8 so the prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save figures, never block
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

CTC_MODEL = "facebook/wav2vec2-base-960h"  # cached; the course uses base-100h (same 32-char vocab)
WHISPER_TINY = "openai/whisper-tiny"
WHISPER_BASE = "openai/whisper-base"
WHISPER_SMALL = "openai/whisper-small"
DUMMY_ID = "hf-internal-testing/librispeech_asr_dummy"
SAMPLING_RATE = 16_000
N_EVAL = 8  # clips used in the section 7 evaluation

# Whisper decodes with num_beams=5 by default, which is ~5x slower on CPU for a
# barely-better transcript. Greedy is the right call for a walkthrough.
GEN = {"task": "transcribe", "language": "english", "num_beams": 1}

USE_WHISPER_SMALL = True  # set False to skip the ~970 MB model in sections 2 and 7
TRY_ENGLISH_ONLY = False  # set True to download whisper-tiny.en (~150 MB) in section 2


def save_fig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   saved {path.relative_to(Path(__file__).parent)}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _device() -> int:
    """The course writes `device = "cuda:0" if torch.cuda.is_available() else "cpu"`.
    The pipeline also accepts an int: -1 means CPU, 0 means the first GPU. Returning
    the int keeps this script honest if you ever paste it into Colab."""
    import torch

    return 0 if torch.cuda.is_available() else -1


def load_dummy():
    """73 clean LibriSpeech validation clips, 16 kHz, with human references.
    Public, no auth, ~9 MB - the same dataset the course uses on its first page."""
    from datasets import load_dataset

    return load_dataset(DUMMY_ID, "clean", split="validation")


def find_clip(ds, keyword: str = "CHRISTMAS") -> int:
    """The course quotes one specific sentence. Look it up by keyword rather than
    hard-coding a row index, which would silently drift if the dataset is revised."""
    for i, text in enumerate(ds["text"]):
        if keyword in text:
            return i
    return 0


def load_minds14(name: str = "en-AU", split: str = "train"):
    """Load MINDS-14 robustly whether it resolves as parquet or a script."""
    from datasets import load_dataset

    kwargs = dict(path="PolyAI/minds14", name=name, split=split)
    try:
        return load_dataset(**kwargs)
    except Exception:
        return load_dataset(**kwargs, trust_remote_code=True)


_PIPES: dict[str, object] = {}


def asr_pipe(model_id: str):
    """Build an ASR pipeline once and reuse it (loading weights is the slow part)."""
    from transformers import pipeline

    if model_id not in _PIPES:
        _PIPES[model_id] = pipeline(
            "automatic-speech-recognition", model=model_id, device=_device()
        )
    return _PIPES[model_id]


def transcribe(model_id: str, array, sr: int = SAMPLING_RATE, **kwargs) -> tuple[str, float]:
    """Run one clip through one model. Returns (text, seconds_elapsed).

    We hand the pipeline a dict rather than a file path so nothing shells out to
    ffmpeg. The sampling rate MUST already be 16 kHz: transformers 4.57 raises
    ImportError("torchaudio is required to resample") otherwise, and torchaudio is
    deliberately not installed here.
    """
    pipe = asr_pipe(model_id)
    gen = {} if "wav2vec2" in model_id else dict(GEN)
    if gen:
        kwargs.setdefault("generate_kwargs", gen)
    t0 = time.perf_counter()
    out = pipe({"array": np.asarray(array), "sampling_rate": sr}, **kwargs)
    return out, time.perf_counter() - t0


# ===========================================================================
# METRIC HELPERS - the machinery behind sections 6 and 7
# ===========================================================================
def edit_ops(ref: list[str], hyp: list[str]):
    """Levenshtein alignment with a backtrace.

    Returns (ops, counts) where `ops` is the edit tape - a list of
    (op, ref_token, hyp_token) with op in {"ok", "sub", "ins", "del"} - and
    `counts` holds S, I, D, and N (= len(ref)). This one function feeds the WER
    numbers, the CER numbers and the alignment figure, so there is exactly one
    place where "how many errors" is decided.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion:    a reference word went missing
                d[i][j - 1] + 1,      # insertion:   the model invented a word
                d[i - 1][j - 1] + cost,  # match or substitution
            )

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        cost = 0 if (i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]) else 1
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + cost:
            ops.append(("ok" if cost == 0 else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()

    counts = {
        "S": sum(1 for o, _, _ in ops if o == "sub"),
        "I": sum(1 for o, _, _ in ops if o == "ins"),
        "D": sum(1 for o, _, _ in ops if o == "del"),
        "N": n,
    }
    return ops, counts


def wer_of(reference: str, hypothesis: str) -> tuple[float, dict]:
    """WER for a single sentence pair: (S + I + D) / N."""
    _, c = edit_ops(reference.split(), hypothesis.split())
    return (c["S"] + c["I"] + c["D"]) / max(c["N"], 1), c


def corpus_wer(references: list[str], hypotheses: list[str]) -> tuple[float, dict]:
    """Corpus WER: sum the errors, sum the reference words, divide ONCE.

    This is not the same as averaging per-utterance WERs, and the difference is not
    academic - a 2-word clip and a 40-word clip would otherwise count equally.
    `evaluate.load("wer")` and `jiwer` both aggregate this way, so this is the
    definition to match.
    """
    tot = {"S": 0, "I": 0, "D": 0, "N": 0}
    for ref, hyp in zip(references, hypotheses):
        _, c = edit_ops(ref.split(), hyp.split())
        for k in tot:
            tot[k] += c[k]
    return (tot["S"] + tot["I"] + tot["D"]) / max(tot["N"], 1), tot


def corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    """Same arithmetic, but the tokens are characters instead of words."""
    errs = chars = 0
    for ref, hyp in zip(references, hypotheses):
        _, c = edit_ops(list(ref), list(hyp))
        errs += c["S"] + c["I"] + c["D"]
        chars += c["N"]
    return errs / max(chars, 1)


def drop_empty_refs(references: list[str], hypotheses: list[str]):
    """WER divides by the number of reference words, so a reference that normalises
    to "" would divide by zero. The course's compute_metrics does this same filter."""
    keep = [i for i, r in enumerate(references) if len(r.strip()) > 0]
    return [references[i] for i in keep], [hypotheses[i] for i in keep]


def normalizers():
    """Whisper ships two text normalisers. The English one is much more aggressive:
    it expands contractions, spells out numbers and applies a 1740-entry British ->
    American spelling map that rides along with the tokenizer."""
    from transformers import WhisperTokenizer
    from transformers.models.whisper.english_normalizer import (
        BasicTextNormalizer,
        EnglishTextNormalizer,
    )

    # NOTE: the course writes `from transformers import BasicTextNormalizer`, which
    # raises ImportError on transformers 4.57 - it lives in the whisper subpackage.
    tok = WhisperTokenizer.from_pretrained(WHISPER_BASE)
    spelling = getattr(tok, "english_spelling_normalizer", None) or {}
    return BasicTextNormalizer(), EnglishTextNormalizer(spelling), len(spelling)


def metric_backend():
    """Two-tier WER/CER backend, mirroring unit4's build_compute_metrics() idiom.

    The course calls `evaluate.load("wer")`. That needs the optional `evaluate`
    package AND downloads a small metric module from the Hub on first use. `jiwer`
    is a core dependency here and does the same arithmetic offline, so we prefer
    `evaluate` when it is present and fall back to `jiwer` otherwise.
    """
    try:
        import evaluate

        wer_m, cer_m = evaluate.load("wer"), evaluate.load("cer")

        def _wer(refs, hyps):
            return wer_m.compute(references=refs, predictions=hyps)

        def _cer(refs, hyps):
            return cer_m.compute(references=refs, predictions=hyps)

        return _wer, _cer, "evaluate.load('wer')"
    except Exception:
        import jiwer

        return (
            lambda refs, hyps: jiwer.wer(refs, hyps),
            lambda refs, hyps: jiwer.cer(refs, hyps),
            "jiwer (evaluate not installed; `uv sync --extra training` adds it)",
        )


def diff_line(ops) -> str:
    """Render an edit tape as readable text: [-deleted-] [+inserted+] [ref>hyp]."""
    out = []
    for op, r, h in ops:
        if op == "ok":
            out.append(h)
        elif op == "sub":
            out.append(f"[{r}>{h}]")
        elif op == "del":
            out.append(f"[-{r}-]")
        else:
            out.append(f"[+{h}+]")
    return " ".join(out)


# ===========================================================================
# 1. CTC VS SEQ2SEQ
# ===========================================================================
def section_models(ds, idx):
    banner("1. Two families of ASR model on one clip: CTC vs sequence-to-sequence")

    ex = ds[idx]
    arr, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
    reference = ex["text"]
    print(f"\n   clip {idx}: {len(arr) / sr:.1f}s @ {sr} Hz")
    print(f"   reference (human): {reference}")

    print(
        "\n   CTC models (Wav2Vec2, HuBERT, XLSR) are encoder-only. They classify every\n"
        "   ~20 ms frame independently into a character, then collapse repeats and blanks.\n"
        "   Nothing in that pipeline knows what a word IS, so the errors are phonetic.\n"
        "   Seq2seq models (Whisper) add a decoder that generates text token by token,\n"
        "   conditioned on the audio AND on everything it has written so far. That decoder\n"
        "   is a language model, trained on 680,000 hours of weakly-labelled audio, so it\n"
        "   fixes spelling, adds punctuation and restores casing for free."
    )

    rows = []
    for model_id in (CTC_MODEL, WHISPER_TINY, WHISPER_BASE):
        out, secs = transcribe(model_id, arr, sr)
        hyp = out["text"].strip()
        # CTC output is uppercase and unpunctuated, so compare like with like.
        ref_cmp, hyp_cmp = reference.lower(), hyp.lower().replace(",", "").replace(".", "")
        w, c = wer_of(ref_cmp, hyp_cmp)
        ops, _ = edit_ops(ref_cmp.split(), hyp_cmp.split())
        rows.append((model_id, hyp, w, c, secs))

        kind = "CTC     " if "wav2vec2" in model_id else "seq2seq "
        print(f"\n   [{kind}] {model_id}   ({secs:.1f}s)")
        print(f"     text : {hyp}")
        print(f"     WER  : {w:.3f}   S={c['S']} I={c['I']} D={c['D']} N={c['N']}")
        wrong = [f"{op}:{r}->{h}" for op, r, h in ops if op != "ok"]
        if wrong:
            print(f"     errors: {', '.join(wrong[:6])}")

    print(
        "\n   Read the errors, not just the numbers. Wav2Vec2 hears the sounds correctly and\n"
        "   spells them out as best it can - that is what a phonetic error looks like. Whisper\n"
        "   returns cased, punctuated, correctly-spelled English. The catch is the mirror image:\n"
        "   a CTC error stays local, while a decoder can hallucinate a fluent wrong word.\n"
        "   (Unit 3 section 2 showed the frame-by-frame collapse that produces the CTC output.)"
    )

    labels = [m.split("/")[-1] for m, _, _, _, _ in rows]
    wers = [w for _, _, w, _, _ in rows]
    colors = ["tab:orange" if "wav2vec2" in m else "tab:blue" for m, _, _, _, _ in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.bar(labels, wers, color=colors)
    ax1.set(ylabel="WER (lower is better)", title="Case/punctuation-insensitive WER on one clip")
    for i, w in enumerate(wers):
        ax1.text(i, w, f"{w:.3f}", ha="center", va="bottom")
    bottom = np.zeros(len(rows))
    for key, color in (("S", "tab:red"), ("I", "tab:purple"), ("D", "tab:brown")):
        vals = np.array([c[key] for _, _, _, c, _ in rows], dtype=float)
        ax2.bar(labels, vals, bottom=bottom, label=key, color=color)
        bottom += vals
    ax2.set(ylabel="error count", title="Substitutions / Insertions / Deletions")
    ax2.legend()
    fig.tight_layout()
    save_fig("01_ctc_vs_whisper.png")
    return reference, arr, sr


# ===========================================================================
# 2. THE WHISPER FAMILY
# ===========================================================================
def section_family(arr, sr):
    banner("2. The Whisper checkpoint family: five sizes, two flavours")

    # Parameter counts and Hub ids as published by OpenAI. The ".en" column marks
    # which sizes ship an English-only twin (large has none).
    family = [
        ("tiny", 39, True), ("base", 74, True), ("small", 244, True),
        ("medium", 769, True), ("large", 1550, False),
    ]
    print("\n     size     params   multilingual         English-only")
    for name, params, has_en in family:
        en = f"openai/whisper-{name}.en" if has_en else "(none)"
        print(f"     {name:<8} {params:>5} M   openai/whisper-{name:<8}  {en}")

    print(
        "\n   Bigger is more accurate and slower, roughly linearly in parameters. The .en\n"
        "   checkpoints are trained on English only: better on English, and they REFUSE a\n"
        "   language or task argument (transformers raises ValueError, because there is no\n"
        "   language to choose). Everything in this repo uses the multilingual ones."
    )

    to_measure = [WHISPER_TINY, WHISPER_BASE] + ([WHISPER_SMALL] if USE_WHISPER_SMALL else [])
    print("\n   Measured on this machine (CPU, greedy decoding):")
    print("     model                  params      time    RTFx")
    measured = []
    for model_id in to_measure:
        out, secs = transcribe(model_id, arr, sr)
        pipe = asr_pipe(model_id)
        params = pipe.model.num_parameters() / 1e6
        rtfx = (len(arr) / sr) / secs  # audio seconds processed per wall-clock second
        measured.append((model_id.split("/")[-1], params, secs, rtfx))
        print(f"     {model_id:<22} {params:>6.0f} M  {secs:>6.1f}s  {rtfx:>6.1f}x")
    print(
        "\n   RTFx (inverse real-time factor) = audio duration / processing time. Above 1.0\n"
        "   means faster than real time. It is the metric you quote next to WER when someone\n"
        "   asks whether a model is deployable, not just accurate."
    )

    if TRY_ENGLISH_ONLY:
        print("\n   Demonstrating the English-only restriction (downloads whisper-tiny.en, ~150 MB):")
        try:
            pipe = asr_pipe("openai/whisper-tiny.en")
            plain = pipe({"array": np.asarray(arr), "sampling_rate": sr})
            print(f"     transcribes fine: {plain['text'].strip()[:70]}…")
            pipe({"array": np.asarray(arr), "sampling_rate": sr},
                 generate_kwargs={"task": "transcribe", "language": "english"})
            print("     unexpectedly accepted a language argument")
        except Exception as exc:
            print(f"     passing language= raises {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:90]}")
    else:
        print("\n   (TRY_ENGLISH_ONLY=False, so whisper-tiny.en is not downloaded.)")

    names = [n for n, _, _, _ in measured]
    params = [p for _, p, _, _ in measured]
    rtfx = [r for _, _, _, r in measured]
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(names, params, color="tab:blue", alpha=0.75)
    ax1.set_ylabel("parameters (millions)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(names, rtfx, "o-", color="tab:red")
    ax2.set_ylabel("RTFx (higher = faster)", color="tab:red")
    ax1.set_title("Whisper: model size vs speed on CPU")
    fig.tight_layout()
    save_fig("06_whisper_family.png")


# ===========================================================================
# 3. MULTILINGUAL: TRANSCRIBE VS TRANSLATE
# ===========================================================================
def section_multilingual():
    banner("3. One multilingual model, two tasks: transcribe and translate")

    print(
        "\n   Whisper's decoder is prompted with special tokens that say which language it is\n"
        "   hearing and which job to do. Flip 'transcribe' to 'translate' and the SAME weights\n"
        "   produce English text instead. Translation always targets English - there is no\n"
        "   token for any other output language."
    )

    example = None
    try:
        from datasets import load_dataset

        print("\n   Trying facebook/multilingual_librispeech (spanish, streamed)…")
        # The course says split="validation"; this dataset actually publishes
        # dev/test/train, so "validation" raises ValueError on datasets 3.6.0.
        mls = load_dataset("facebook/multilingual_librispeech", "spanish",
                           split="test", streaming=True)
        row = next(iter(mls))
        example = (row["audio"]["array"], row["audio"]["sampling_rate"],
                   row.get("transcript") or row.get("text") or "(none)", "spanish", None)
    except Exception as exc:
        print(f"     unavailable: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        print("     falling back to MINDS-14 de-DE (cached, and it ships a gold English line).")
        try:
            from datasets import Audio

            de = load_minds14("de-DE").cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
            row = de[0]
            example = (row["audio"]["array"], SAMPLING_RATE, row["transcription"],
                       "german", row.get("english_transcription"))
        except Exception as exc2:
            print(f"     MINDS-14 de-DE also unavailable: {type(exc2).__name__}: "
                  f"{str(exc2).splitlines()[0][:100]}. Skipping section 3.")
            return

    arr, sr, reference, language, gold_en = example
    model_id = WHISPER_SMALL if USE_WHISPER_SMALL else WHISPER_BASE
    for task in ("transcribe", "translate"):
        out, secs = transcribe(model_id, arr, sr,
                               generate_kwargs={"task": task, "language": language, "num_beams": 1})
        print(f"\n     task={task:<10} ({secs:.1f}s): {out['text'].strip()}")
    print(f"\n     reference ({language}) : {reference}")
    if gold_en:
        print(f"     gold English        : {gold_en}")
    print(
        "\n   Do not judge translation quality from tiny/base - they are weak translators and\n"
        "   it looks like a bug when it is just capacity. The point is that one task token,\n"
        "   not a second model, is what changed the output language."
    )


# ===========================================================================
# 4. LONG-FORM AUDIO
# ===========================================================================
def section_longform(ds):
    banner("4. Audio longer than 30 seconds: the wall, chunking, and timestamps")

    # Whisper's encoder takes a FIXED 30-second log-mel window (80 x 3000). Anything
    # longer has to be split. Stitch a few clips together to get past the limit.
    parts, total = [], 0.0
    for i in range(len(ds)):
        a = ds[i]["audio"]["array"]
        parts.append(a)
        total += len(a) / SAMPLING_RATE
        if total > 36:
            break
    long_arr = np.concatenate(parts)
    print(f"\n   built a {total:.1f}s clip by concatenating {len(parts)} LibriSpeech utterances")
    print("   Whisper's encoder input is fixed at 80 mel bins x 3000 frames = exactly 30 s.")

    print("\n   (a) Naive call, no chunking - this is expected to fail:")
    try:
        transcribe(WHISPER_BASE, long_arr)
        print("     it did not fail (transformers behaviour changed?)")
    except Exception as exc:
        print(f"     {type(exc).__name__}: {str(exc).splitlines()[0][:210]}")
        print("     Whisper switches to long-form mode past 30 s and long-form needs timestamps.")

    print("\n   (b) Chunked: slice into 30 s windows with overlap, transcribe in batches, stitch.")
    out_chunked, secs_chunked = transcribe(
        WHISPER_BASE, long_arr,
        chunk_length_s=30, batch_size=8, max_new_tokens=256, ignore_warning=True,
    )
    print(f"     {secs_chunked:.1f}s  RTFx={total / secs_chunked:.1f}x")
    print(f"     {out_chunked['text'].strip()[:200]}…")

    print("\n   (c) Sequential long-form: no chunk_length_s, but return_timestamps=True.")
    print("       Whisper predicts timestamps itself and slides its own window forward.")
    out_ts, secs_ts = transcribe(WHISPER_BASE, long_arr, return_timestamps=True)
    print(f"     {secs_ts:.1f}s  RTFx={total / secs_ts:.1f}x")

    chunks = out_ts.get("chunks") or []
    print(f"\n   (d) {len(chunks)} timestamped segments:")
    for ch in chunks[:8]:
        start, end = ch["timestamp"]
        end_s = f"{end:6.2f}" if end is not None else "  ....."
        print(f"     [{start:6.2f} → {end_s}]  {ch['text'].strip()[:64]}")
    if len(chunks) > 8:
        print(f"     … and {len(chunks) - 8} more")
    print(
        "\n   Chunking is faster (it batches) but can cut a word in half at a boundary;\n"
        "   sequential decoding is slower but keeps Whisper's own context across the whole\n"
        "   file. Timestamps are what make subtitles, diarisation and search possible."
    )

    t = np.arange(len(long_arr)) / SAMPLING_RATE
    plt.figure(figsize=(13, 3.5))
    plt.plot(t, long_arr, lw=0.3, color="0.6")
    for k, ch in enumerate(chunks):
        start, end = ch["timestamp"]
        if end is None:
            end = total
        plt.axvspan(start, end, alpha=0.25, color=f"C{k % 10}")
        plt.text((start + end) / 2, 0.8 * np.max(np.abs(long_arr)), str(k + 1),
                 ha="center", fontsize=8)
    plt.xlabel("time (s)")
    plt.title(f"Sequential long-form decoding: {len(chunks)} timestamped segments over {total:.0f}s")
    save_fig("05_longform_timestamps.png")


# ===========================================================================
# 5. CHOOSING A DATASET
# ===========================================================================
def section_datasets(ds):
    banner("5. Choosing a dataset: the eight English ASR corpora")

    # Reproduced from the Unit 5 "Choosing a dataset" page.
    corpora = [
        ("LibriSpeech",     960, "Audiobook",              "Narrated",   False, False, "CC-BY-4.0"),
        ("Common Voice 11", 3000, "Wikipedia",             "Narrated",   True,  True,  "CC0-1.0"),
        ("VoxPopuli",       540, "EU Parliament",          "Oratory",    False, True,  "CC0"),
        ("TED-LIUM",        450, "TED talks",              "Oratory",    False, False, "CC-BY-NC-ND"),
        ("GigaSpeech",    10000, "Audiobook/podcast/YT",   "Narr.+spon.", False, True,  "apache-2.0"),
        ("SPGISpeech",     5000, "Financial meetings",     "Orat.+spon.", True,  True,  "User agmt."),
        ("Earnings-22",     119, "Financial meetings",     "Orat.+spon.", True,  True,  "CC-BY-SA-4.0"),
        ("AMI",             100, "Meetings",               "Spontaneous", True,  True,  "CC-BY-4.0"),
    ]
    print(f"\n     {'dataset':<16}{'hours':>6}  {'domain':<22}{'style':<13}{'case':<6}{'punct':<7}license")
    for name, hours, domain, style, case, punct, lic in corpora:
        print(f"     {name:<16}{hours:>6}  {domain:<22}{style:<13}"
              f"{'yes' if case else 'no':<6}{'yes' if punct else 'no':<7}{lic}")

    print(
        "\n   Four axes decide which one you want:\n"
        "     hours       - more data generalises better. Under ~100 h, fine-tune; don't train.\n"
        "     domain      - a studio audiobook model falls apart on a noisy phone call.\n"
        "     style       - narrated speech is read aloud; spontaneous speech has disfluencies.\n"
        "     formatting  - if the labels are uppercase and unpunctuated, your model will be too.\n"
        "\n     narrated    : \"Consider the task of training a model on a speech recognition dataset\"\n"
        "     spontaneous : \"Let's uhh let's take a look at how you'd go about training a model\n"
        "                    on uhm a sp- speech recognition dataset\"\n"
        "\n   The ESB benchmark (arxiv 2210.13352) exists because a model that wins on LibriSpeech\n"
        "   often loses everywhere else; it scores one model across eight of these at once.\n"
        "   The course's fine-tuning chapter uses Common Voice 13 (mozilla-foundation/\n"
        "   common_voice_13_0), Dhivehi subset - a GATED dataset: you must accept its terms on\n"
        "   the Hub and log in before load_dataset() will work. This script deliberately avoids\n"
        "   gated data, and `finetune.py` uses MINDS-14 instead (which is what the hands-on asks\n"
        "   for anyway)."
    )

    print(f"\n   Let's actually measure a domain change. Same whisper-base weights, same")
    print(f"   normaliser, {N_EVAL} clips from each of two very different corpora:")
    basic, _, _ = normalizers()
    probes = []

    n = min(N_EVAL, len(ds))
    refs = [ds[i]["text"] for i in range(n)]
    hyps = [transcribe(WHISPER_BASE, ds[i]["audio"]["array"])[0]["text"] for i in range(n)]
    r, h = drop_empty_refs([basic(x) for x in refs], [basic(x) for x in hyps])
    probes.append(("LibriSpeech clean (16 kHz, narrated, studio)", corpus_wer(r, h)[0]))

    try:
        from datasets import Audio

        minds = load_minds14("en-AU").cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        refs = [minds[i]["english_transcription"] for i in range(n)]
        hyps = [transcribe(WHISPER_BASE, minds[i]["audio"]["array"])[0]["text"] for i in range(n)]
        r, h = drop_empty_refs([basic(x) for x in refs], [basic(x) for x in hyps])
        probes.append(("MINDS-14 en-AU (8 kHz phone, spontaneous)", corpus_wer(r, h)[0]))
    except Exception as exc:
        print(f"     MINDS-14 probe skipped: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")

    for name, w in probes:
        print(f"     {name:<46} normalised WER {w:.3f}")

    if len(probes) == 2:
        lib, phone = probes[0][1], probes[1][1]
        if phone > lib * 1.25:
            print(f"\n     The phone-call set is {phone / max(lib, 1e-9):.1f}x worse on identical "
                  f"weights. That gap is\n     the domain: 8 kHz telephony, spontaneous speech, "
                  f"background noise.")
        else:
            print(
                f"\n     Interesting - the gap is small here ({lib:.3f} vs {phone:.3f}), and it is\n"
                f"     worth understanding why rather than pretending otherwise. These MINDS-14\n"
                f"     utterances are short, scripted-sounding banking requests, and Whisper was\n"
                f"     pre-trained on a huge amount of noisy web audio, so 8 kHz phone speech is\n"
                f"     well within its range. Domain shift bites hardest on things neither corpus\n"
                f"     here contains: overlapping meeting speech (AMI), heavy accents, and jargon.\n"
                f"     The honest lesson is that you MEASURE the shift on your own data - you do\n"
                f"     not assume it from the dataset description."
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    names = [c[0] for c in corpora]
    hours = [c[1] for c in corpora]
    cols = ["tab:green" if c[4] and c[5] else "tab:blue" if c[5] else "tab:red" for c in corpora]
    ax1.barh(names[::-1], hours[::-1], color=cols[::-1])
    ax1.set_xscale("log")
    ax1.set_xlabel("hours (log scale)")
    ax1.set_title("English ASR corpora\ngreen = cased + punctuated, blue = punct only, red = neither")
    if probes:
        ax2.bar([p[0] for p in probes], [p[1] for p in probes], color=["tab:green", "tab:red"])
        for i, p in enumerate(probes):
            ax2.text(i, p[1], f"{p[1]:.3f}", ha="center", va="bottom")
    ax2.set_ylabel("normalised WER")
    ax2.set_title("Same model, different domain (whisper-base)")
    fig.tight_layout()
    save_fig("02_dataset_landscape.png")


# ===========================================================================
# 6. WER BY HAND
# ===========================================================================
def section_wer_by_hand():
    banner("6. Word Error Rate, computed by hand")

    reference = "the cat sat on the mat"
    prediction = "the cat sit on the"
    ops, c = edit_ops(reference.split(), prediction.split())

    print(f"\n   reference  : {reference}")
    print(f"   prediction : {prediction}")
    print("\n   Line the two up and every word gets exactly one label:")
    for op, r, h in ops:
        sym = {"ok": "  ok ", "sub": " SUB ", "del": " DEL ", "ins": " INS "}[op]
        print(f"     {sym}  {str(r or '-'):<10} → {str(h or '-'):<10}")

    wer = (c["S"] + c["I"] + c["D"]) / c["N"]
    print(f"\n     S (substitutions) = {c['S']}   'sat' came back as 'sit'")
    print(f"     I (insertions)    = {c['I']}   nothing was invented")
    print(f"     D (deletions)     = {c['D']}   'mat' went missing")
    print(f"     N (reference words) = {c['N']}")
    print(f"\n     WER = (S + I + D) / N = ({c['S']} + {c['I']} + {c['D']}) / {c['N']} = {wer:.3f}")

    import jiwer

    print(f"     jiwer agrees      : {jiwer.wer(reference, prediction):.3f}")
    _wer, _cer, backend = metric_backend()
    print(f"     metric backend    : {backend}")
    print(f"     backend agrees    : {_wer([reference], [prediction]):.3f}")

    print(
        "\n   Two things beginners get bitten by:\n"
        "     1. WER has no upper bound. The denominator is the REFERENCE length, so a model\n"
        "        that rambles can score above 1.0."
    )
    chatty = "the cat sat on the mat and then it got up and left the room entirely"
    w2, c2 = wer_of(reference, chatty)
    print(f"        reference  : {reference}")
    print(f"        prediction : {chatty}")
    print(f"        S={c2['S']} I={c2['I']} D={c2['D']} N={c2['N']} → WER = {w2:.3f}")
    print(f"        'Word accuracy' = 1 - WER = {1 - w2:.3f}, i.e. negative. That is why the")
    print("        course warns against quoting accuracy for speech.")

    print(
        "\n     2. Corpus WER is not the mean of per-utterance WERs. Sum the errors and sum the\n"
        "        reference words, THEN divide - otherwise a 2-word clip outvotes a 40-word one."
    )
    refs = [reference, "hello world"]
    hyps = [prediction, "goodbye world"]
    per_utt = np.mean([wer_of(r, h)[0] for r, h in zip(refs, hyps)])
    corpus, tot = corpus_wer(refs, hyps)
    print(f"        mean of per-utterance WERs : {per_utt:.3f}")
    print(f"        corpus WER (the right one) : {corpus:.3f}   "
          f"(S={tot['S']} I={tot['I']} D={tot['D']} N={tot['N']})")

    print("\n   CER does the same arithmetic over characters, which is kinder to near misses:")
    for r, h in (("similes", "similarly"), ("christmas", "christmanus")):
        print(f"     {r:<10} → {h:<12}  WER {wer_of(r, h)[0]:.3f}   CER {corpus_cer([r], [h]):.3f}")
    print("     Useful for languages without spaces, and for spotting 'almost right' spellings.")

    # Figure: the alignment grid.
    ref_w, hyp_w = reference.split(), prediction.split()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colours = {"ok": "tab:green", "sub": "tab:red", "del": "tab:brown", "ins": "tab:purple"}
    for k, (op, r, h) in enumerate(ops):
        ax.add_patch(plt.Rectangle((k, 0), 0.92, 0.9, color=colours[op], alpha=0.35))
        ax.text(k + 0.46, 1.15, r or "—", ha="center", fontsize=10)
        ax.text(k + 0.46, 0.42, op.upper(), ha="center", fontsize=9, weight="bold")
        ax.text(k + 0.46, -0.3, h or "—", ha="center", fontsize=10)
    ax.set(xlim=(-0.2, len(ops)), ylim=(-0.6, 1.5))
    ax.text(-0.15, 1.15, "ref", ha="right", fontsize=9, style="italic")
    ax.text(-0.15, -0.3, "hyp", ha="right", fontsize=9, style="italic")
    ax.axis("off")
    ax.set_title(f"WER alignment:  (S={c['S']} + I={c['I']} + D={c['D']}) / N={c['N']} = {wer:.3f}")
    fig.tight_layout()
    save_fig("03_wer_alignment.png")


# ===========================================================================
# 7. EVALUATING REAL MODELS
# ===========================================================================
def section_evaluate(ds):
    banner("7. Evaluating real models: orthographic vs normalised WER")

    basic, english, n_spell = normalizers()
    print(f"\n   Loaded the English spelling map: {n_spell} entries.")
    probe = "Mr. Quilter's 20 dollars isn't cheap."
    print(f"     raw     : {probe!r}")
    print(f"     basic   : {basic(probe)!r}")
    print(f"     english : {english(probe)!r}")
    print(
        "\n   BasicTextNormalizer lowercases and strips punctuation - safe for any language.\n"
        "   EnglishTextNormalizer additionally expands contractions and numbers and applies\n"
        "   a British→American spelling map. Whisper writes '$20'; LibriSpeech's reference\n"
        "   says 'TWENTY DOLLARS'. Without normalisation you are scoring formatting, not\n"
        "   speech recognition."
    )

    n = min(N_EVAL, len(ds))
    refs = [ds[i]["text"] for i in range(n)]
    audio = [ds[i]["audio"]["array"] for i in range(n)]
    secs_audio = sum(len(a) / SAMPLING_RATE for a in audio)
    _wer, _cer, backend = metric_backend()

    models = [CTC_MODEL, WHISPER_TINY, WHISPER_BASE] + ([WHISPER_SMALL] if USE_WHISPER_SMALL else [])
    print(f"\n   Transcribing {n} clips ({secs_audio:.0f}s of audio) with {len(models)} models…")
    print(f"   metric backend: {backend}")

    results = []
    for model_id in models:
        hyps, elapsed = [], 0.0
        for a in audio:
            out, secs = transcribe(model_id, a)
            hyps.append(out["text"])
            elapsed += secs

        # Orthographic WER: compare the strings as they come out, no cleanup.
        wer_ortho, tot = corpus_wer([r for r in refs], [h.strip() for h in hyps])
        # Normalised WER: put both sides through the same normaliser first.
        nr, nh = drop_empty_refs([english(r) for r in refs], [english(h) for h in hyps])
        wer_norm, tot_n = corpus_wer(nr, nh)
        cer = corpus_cer(nr, nh)
        rtfx = secs_audio / elapsed
        results.append((model_id.split("/")[-1], wer_ortho, wer_norm, cer, rtfx))

        print(f"\n     {model_id}")
        print(f"       orthographic WER {wer_ortho:.3f}   "
              f"S={tot['S']} I={tot['I']} D={tot['D']} N={tot['N']}")
        print(f"       normalised   WER {wer_norm:.3f}   "
              f"S={tot_n['S']} I={tot_n['I']} D={tot_n['D']} N={tot_n['N']}")
        print(f"       CER {cer:.3f}      RTFx {rtfx:.1f}x      {elapsed:.1f}s total")
        print(f"       backend cross-check: {_wer(nr, nh):.3f}")

    ctc = next((r for r in results if "wav2vec2" in r[0]), None)
    whisper = [r for r in results if "whisper" in r[0]]
    best_ortho = min(results, key=lambda r: r[1])
    best_norm = min(results, key=lambda r: r[2])
    print("\n   Read those two WER columns carefully, because they disagree wildly:")
    print(f"     best by orthographic WER : {best_ortho[0]} ({best_ortho[1]:.3f})")
    print(f"     best by normalised WER   : {best_norm[0]} ({best_norm[2]:.3f})")

    if whisper and min(w[1] for w in whisper) > 0.9:
        print(
            "\n     Whisper's orthographic WER is around 1.0 - close to 'every single word wrong'.\n"
            "     It is not. Look back at section 1: the transcripts are excellent. LibriSpeech's\n"
            "     references are UPPERCASE WITH NO PUNCTUATION, so 'He' vs 'HE' counts as a\n"
            "     substitution, and nearly every word is therefore 'wrong'."
        )
    if ctc is not None:
        print(
            f"\n     Meanwhile wav2vec2 scores {ctc[1]:.3f} orthographically - not because it is the\n"
            "     better model, but because it happens to output the same SHOUTY format the\n"
            f"     reference is written in. Normalise both sides and it lands at {ctc[2]:.3f}, "
            f"which\n     {'now loses to' if best_norm[0] != ctc[0] else 'still beats'} "
            f"{best_norm[0]} at {best_norm[2]:.3f}."
        )
    print(
        "\n   The lesson is not 'always normalise'. It is that orthographic WER only means\n"
        "   something when the reference is formatted the way you actually want output. On\n"
        "   LibriSpeech it is not, so it measures format agreement instead of recognition.\n"
        "   That is exactly why the ESB paper argues for orthographic WER *and* for datasets\n"
        "   whose labels carry real casing and punctuation - the green rows in section 5.\n"
        "   Normalisation is also not a magic fix: on the course's Dhivehi example it only\n"
        "   took WER from 168% down to 126%."
    )

    labels = [r[0] for r in results]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(x - 0.2, [r[1] for r in results], 0.4, label="orthographic", color="tab:red")
    ax1.bar(x + 0.2, [r[2] for r in results], 0.4, label="normalised", color="tab:green")
    ax1.set_xticks(x, labels, rotation=15, ha="right")
    ax1.set_ylabel("WER")
    ax1.set_title("Normalisation forgives casing and punctuation")
    ax1.legend()
    ax2.scatter([r[4] for r in results], [r[2] for r in results], s=90, color="tab:blue")
    for r in results:
        ax2.annotate(r[0], (r[4], r[2]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax2.set(xlabel="RTFx (higher = faster)", ylabel="normalised WER (lower = better)",
            title="The deployment trade-off\n(bottom-right is what you want)")
    fig.tight_layout()
    save_fig("04_wer_normalisation.png")


# ===========================================================================
# 8. FINE-TUNING INNARDS
# ===========================================================================
def section_finetune_innards(ds):
    from dataclasses import dataclass

    from transformers import WhisperProcessor

    banner("8. What fine-tuning actually feeds the model")

    processor = WhisperProcessor.from_pretrained(
        WHISPER_TINY, language="english", task="transcribe"
    )
    tok = processor.tokenizer
    print("\n   The processor is a feature extractor + a tokenizer in one object.")
    print(f"     advertised prefix : {tok.convert_ids_to_tokens(tok.prefix_tokens)}")
    print("     Those prefix tokens are prepended to every label sequence: they are how the")
    print("     model is told 'this is English, and your job is transcription'.")

    # --- A REAL BUG YOU WILL HIT, and it is silent ------------------------------
    # On transformers 4.57 the *fast* tokenizer (the default) ignores the language
    # and task passed to from_pretrained() when it actually encodes text, even
    # though .prefix_tokens reports them correctly. You get labels beginning
    # <|startoftranscript|><|notimestamps|> with <|en|><|transcribe|> missing.
    # At inference generation_config DOES force those tokens, so you would train on
    # one prompt format and decode with another. set_prefix_tokens() repairs it.
    before = tok("hello world")["input_ids"]
    tok.set_prefix_tokens(language="english", task="transcribe")
    after = tok("hello world")["input_ids"]
    print(f"\n     labels actually encoded : {tok.convert_ids_to_tokens(before)[:4]}")
    print(f"     after set_prefix_tokens : {tok.convert_ids_to_tokens(after)[:4]}")
    if before[:4] != after[:4]:
        print("     ^ the language and task tokens were MISSING until we called")
        print("       processor.tokenizer.set_prefix_tokens(language=…, task=…).")
        print("       finetune.py and the Colab hands-on both do this; the course does not")
        print("       need to, because older transformers wired it up in from_pretrained().")

    def prepare_dataset(example):
        """The course's preprocessing function, unchanged in spirit.

        WARNING: some versions of the course write `out["labels"][0]`. For a single
        string the tokenizer returns a FLAT list of ids, so `[0]` yields one integer
        and the collator's pad() later explodes. Return the list itself.
        """
        audio = example["audio"]
        out = processor(
            audio=audio["array"], sampling_rate=audio["sampling_rate"], text=example["text"]
        )
        out["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return out

    rows = [prepare_dataset(ds[i]) for i in (0, 1)]
    print("\n   Two examples after preprocessing:")
    for k, row in enumerate(rows):
        feats = np.asarray(row["input_features"])
        print(f"     [{k}] input_features {feats.shape}   "
              f"labels {len(row['labels'])} tokens   {row['input_length']:.1f}s of audio")
    print(
        "\n   Notice the asymmetry. Every clip becomes the SAME (80, 3000) log-mel block -\n"
        "   Whisper pads or truncates to exactly 30 seconds, so the audio side needs no\n"
        "   padding logic at all. The labels are different lengths, so they do. That is the\n"
        "   entire reason the data collator exists."
    )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: object
        decoder_start_token_id: int

        def __call__(self, features):
            # 1. audio: already uniform, just stack into a tensor
            input_features = [{"input_features": f["input_features"][0]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            # 2. labels: pad to the longest in the batch
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            # 3. -100 tells cross_entropy to ignore those positions (ignore_index)
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # 4. the model prepends the start token itself, so drop a duplicated one
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_TINY)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id)

    padded = processor.tokenizer.pad([{"input_ids": r["labels"]} for r in rows], return_tensors="pt")
    batch = collator(rows)
    # Pick the SHORTER row - the longer one defines the batch width and so has no
    # padding at all, which would make the before/after comparison look identical.
    short = int(np.argmin([len(r["labels"]) for r in rows]))
    print("\n   Running the collator on those two examples:")
    print(f"     input_features → {tuple(batch['input_features'].shape)}  (batch, mels, frames)")
    print(f"     labels padded  → {tuple(padded['input_ids'].shape)}, pad id "
          f"{processor.tokenizer.pad_token_id}")
    print(f"     labels masked  → {tuple(batch['labels'].shape)}")
    print(f"\n     row {short} is the shorter one, so it is the one that gets padded:")
    print(f"       tail before masking: {padded['input_ids'][short][-8:].tolist()}")
    print(f"       tail after masking : {batch['labels'][short][-8:].tolist()}")
    print(f"       every {processor.tokenizer.pad_token_id} became -100; the real "
          f"<|endoftext|> is kept.")
    start_id = model.config.decoder_start_token_id
    print(f"\n     decoder_start_token_id = {start_id} "
          f"({tok.convert_ids_to_tokens([start_id])[0]})")
    print(f"     labels[:, 0] before the strip was {padded['input_ids'][:, 0].tolist()} "
          f"= the start token,")
    print(f"     and after it is {batch['labels'][:, 0].tolist()} "
          f"({tok.convert_ids_to_tokens(batch['labels'][:, 0].tolist())}).")
    print("     We strip it because the model shifts labels right internally and re-adds it;")
    print("     leaving it in would train the model to emit two start tokens.")
    print(
        "\n   Why -100 and not the pad id? torch.nn.functional.cross_entropy takes an\n"
        "   ignore_index argument that defaults to -100. Marking padding with -100 means\n"
        "   those positions contribute no loss - otherwise the model would be rewarded for\n"
        "   predicting padding, which is the single most common silent bug in this pipeline."
    )

    lab = batch["labels"].numpy().astype(float)
    plt.figure(figsize=(11, 2.6))
    plt.imshow(np.where(lab == -100, np.nan, lab), aspect="auto", cmap="viridis",
               interpolation="nearest")
    plt.colorbar(label="token id")
    plt.xlabel("position in the label sequence")
    plt.ylabel("example")
    plt.yticks([0, 1])
    plt.title("Padded label batch — blank cells are -100 and contribute no loss")
    save_fig("07_label_padding.png")
    del model


def main() -> None:
    print("Hugging Face Audio Course — Unit 5: Automatic speech recognition")
    print(f"Figures will be written to: {FIG_DIR}")
    print(f"Device: {'GPU' if _device() >= 0 else 'CPU'}")

    ds = load_dummy()
    idx = find_clip(ds, "CHRISTMAS")
    print(f"Demo clip: row {idx} of {len(ds)} in {DUMMY_ID}")

    _, arr, sr = section_models(ds, idx)
    section_family(arr, sr)
    section_multilingual()
    section_longform(ds)
    section_datasets(ds)
    section_wer_by_hand()
    section_evaluate(ds)
    section_finetune_innards(ds)

    banner("Done!  See figures/ for the plots.")
    print("Next steps:")
    print("   finetune.py          - fine-tune Whisper (CPU smoke test, or full run on a GPU)")
    print("   gradio_demo.py       - a tabbed microphone + file transcription demo")
    print("   colab_handson.ipynb  - the graded hands-on: whisper-tiny on MINDS-14, WER < 0.37")
    print("\nRemember: the hands-on wants WER as a FRACTION (0.37), not a percentage (37).")
    print("\nSupplemental reading from the course:")
    print("   Whisper talk (Jong Wook Kim)  https://www.youtube.com/live/fZMiD8sDzzg")
    print("   ESB benchmark paper           https://arxiv.org/abs/2210.13352")
    print("   Fine-tuning Whisper           https://huggingface.co/blog/fine-tune-whisper")
    print("   MMS adapter fine-tuning       https://huggingface.co/blog/mms_adapters")
    print("   Wav2Vec2 with n-grams         https://huggingface.co/blog/wav2vec2-with-ngram")


if __name__ == "__main__":
    main()
