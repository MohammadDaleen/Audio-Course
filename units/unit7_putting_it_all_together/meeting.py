"""
Hugging Face Audio Course - Unit 7: transcribing a meeting.

Two jobs, WHAT was said and WHO said it, and one of them does not work on this stack. ONE
module, THREE ways to run it (flags, or the env var UNIT7_AUDIO; default = synthetic):

  --audio synthetic   (default, SCORED) Builds a two-speaker track by interleaving turns
                      from two ylacombe/english_dialects [northern_female] speakers, streamed.
                      The boundaries are exact BY CONSTRUCTION, so the diarization accuracy
                      at the end is a real measured number with a real reference. Five
                      speakers of the same accent and the same sex is the HARD case, which is
                      exactly what makes a negative result here credible.

  --audio course      (UNSCORED) sanchit-gandhi/concatenated_librispeech, the dataset the
                      course itself uses. One audio column, one row, 661 KB: no transcript,
                      no speaker labels, nothing to score against. This mode prints the
                      course's own tidy table and then says plainly that there is no ground
                      truth under it and the course never computes one.

  --sweep             All 10 speaker pairs (5 choose 2), diarization only, no ASR. Prints
                      each pair against its OWN measured chance baseline. Implies synthetic.

    uv run python units/unit7_putting_it_all_together/meeting.py
    uv run python units/unit7_putting_it_all_together/meeting.py --audio course
    uv run python units/unit7_putting_it_all_together/meeting.py --sweep
    uv run python units/unit7_putting_it_all_together/meeting.py --help

Units 4 to 6 toggle one binary mode with `"--full" in sys.argv`. That idiom does not cover a
flag which takes a VALUE, and a hand-rolled parser that silently ignores `--audio corse` is
the exact class of bug this unit is about, so Unit 7 uses argparse and gets the whitelist
for free. The env var is kept so both idioms agree.

Models: openai/whisper-base (~290 MB, cached). Network: synthetic and sweep stream ONE ~59 MB
parquet row group, read once and reused for every pair; course mode downloads 661 KB. Audio
and figures go to figures/, which is git-ignored.


THE COURSE'S RECIPE, PRESERVED BECAUSE IT CANNOT BE INSTALLED HERE
-----------------------------------------------------------------
    from speechbox import ASRDiarizationPipeline
    from transformers import pipeline
    from pyannote.audio import Pipeline

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=True)
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    pipeline = ASRDiarizationPipeline(
        asr_pipeline=asr_pipeline, diarization_pipeline=diarization_pipeline)

Three blockers, none of which is a version bump you can wait out. `pyannote.audio` requires
`torchaudio`, and `speechbox` pulls `pyannote.audio` in transitively; this repo deliberately
installs neither. `pyannote/speaker-diarization@2.1` is a GATED repo, so it needs an accepted
licence and a token. And `speechbox` is unmaintained. So the MERGE is reimplemented below,
with the three guards it lacks, and the DIARIZER is replaced by librosa MFCCs plus sklearn
clustering, which is measured rather than trusted, and scored against a chance baseline this
module computes instead of assuming. Run --sweep for the distribution across every speaker
pair: what it reports is the finding, not a bug in this file.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import soundfile as sf

import matplotlib

matplotlib.use("Agg")   # non-interactive backend: save figures, never block
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

ASR_ID = "openai/whisper-base"
DIALECTS_ID = "ylacombe/english_dialects"
DIALECTS_CONFIG = "northern_female"        # 5 real speakers, same accent, same sex
COURSE_ID = "sanchit-gandhi/concatenated_librispeech"   # 1 column, 1 row, NO labels

SAMPLING_RATE = 16_000
N_SPEAKERS = 5              # what northern_female publishes
N_TURNS = 4                 # turns per built track: 2 clips from each of 2 speakers
MIN_CLIPS = 3               # clips kept per speaker: N_TURNS // 2 plus one of headroom
MAX_ROWS = 100              # HARD CAP. 100 rows is exactly one parquet row group (~59 MB);
                            # row 100 opens the next one and doubles the transfer.
WIN_S = 1.0                 # diarization window
N_MFCC = 20
TRIALS = 20_000             # Monte Carlo trials for the chance baseline
USABLE = 0.80               # the "would you ship this" line
SEED = 7
AUDIO_CHOICES = ("synthetic", "course")

_CACHE: dict[str, object] = {}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_wav(name: str, audio, sampling_rate: int = SAMPLING_RATE) -> None:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    path = OUT_DIR / name
    sf.write(path, audio, sampling_rate)
    print(f"     saved {path.name}  ({len(audio) / sampling_rate:.1f}s @ {sampling_rate} Hz)")


def save_fig(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     saved {path.name}")


def audio_dict(array) -> dict:
    """A FRESH dict for every pipeline call.

    The ASR pipeline moves "array" into "raw" in place, so a reused dict makes the SECOND
    call raise about a missing "raw" key as though the input were malformed.
    """
    return {"array": np.asarray(array, dtype=np.float32), "sampling_rate": SAMPLING_RATE}


def asr_pipe():
    from transformers import pipeline

    if "asr" not in _CACHE:
        _CACHE["asr"] = pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)
    return _CACHE["asr"]


def _agglomerative():
    """Import sklearn's clusterer, retrying once.

    Windows Application Control intermittently blocks a scikit-learn extension DLL on a cold
    import and then lets the identical import through a moment later. One retry keeps a
    transient OS block from being read as a missing dependency.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except (ImportError, OSError):
        time.sleep(1.0)
        from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering


# ---------------------------------------------------------------------------
# The corpus, streamed ONCE
# ---------------------------------------------------------------------------
def stream_speaker_clips(min_clips: int = MIN_CLIPS, max_rows: int = MAX_ROWS) -> dict:
    """Stream the dialect corpus until every speaker has `min_clips` turns, then stop.

    Read once and cached for the whole process. The sweep needs 10 pairs, and re-streaming
    per pair would range-read the same row group ten times: about 590 MB for bytes already
    in memory.

    The walkthrough's build_meeting() takes a fixed first 60 rows and then the two most
    frequent speakers, which is right for one pair and wrong for a sweep, because nothing in
    it guarantees the other three speakers appeared at all. This collects by COVERAGE
    instead, with max_rows as the backstop so a pathological ordering cannot turn into an
    unbounded read. The source is 48 kHz, so the cast is mandatory: skip it and every
    duration is three times wrong with nothing raised.
    """
    if "clips" in _CACHE:
        return _CACHE["clips"]

    from datasets import Audio, load_dataset

    stream = load_dataset(DIALECTS_ID, DIALECTS_CONFIG, split="train", streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    per_speaker: dict[int, list] = {}
    rows = 0
    for row in stream:
        rows += 1
        held = per_speaker.setdefault(row["speaker_id"], [])
        if len(held) < min_clips:   # cap memory at 5 speakers x 3 clips, not 100 rows
            held.append(np.asarray(row["audio"]["array"], dtype=np.float32))
        covered = (len(per_speaker) >= N_SPEAKERS
                   and all(len(c) >= min_clips for c in per_speaker.values()))
        if covered or rows >= max_rows:
            break

    counts = {sid: len(c) for sid, c in sorted(per_speaker.items())}
    print(f"     read {rows} streamed rows (cap {max_rows}); clips per speaker: {counts}")

    if len(per_speaker) < N_SPEAKERS:
        # Not fatal, and not silently ignored either: say what we got and what it costs.
        pairs = len(per_speaker) * (len(per_speaker) - 1) // 2
        print(f"     NOTE: expected {N_SPEAKERS} speakers, found {len(per_speaker)} in "
              f"{rows} rows. A sweep will cover {pairs} pairs instead of "
              f"{N_SPEAKERS * (N_SPEAKERS - 1) // 2}. Raise MAX_ROWS to read further: each "
              f"extra 100 rows is another ~59 MB.")
    if len(per_speaker) < 2:
        sys.exit(f"{DIALECTS_CONFIG} yielded {len(per_speaker)} speaker(s) in {rows} rows; "
                 "a two-speaker track needs two.")

    _CACHE["clips"] = per_speaker
    return per_speaker


def build_meeting(clips: dict, pair, n_turns: int = N_TURNS):
    """A two-speaker track whose boundaries are exact BY CONSTRUCTION.

    That construction is the only reason this mode can be scored at all: the turns are
    concatenated by us, so the reference is arithmetic rather than annotation.
    """
    turns, truth, cursor = [], [], 0.0
    for t in range(n_turns):
        sid = pair[t % 2]
        arr = clips[sid][t // 2]
        turns.append(arr)
        truth.append((cursor, cursor + len(arr) / SAMPLING_RATE, sid))
        cursor += len(arr) / SAMPLING_RATE
    return np.concatenate(turns), truth


def enough_clips(clips: dict, pair, n_turns: int = N_TURNS) -> bool:
    return all(len(clips[sid]) >= (n_turns + 1) // 2 for sid in pair)


# ---------------------------------------------------------------------------
# WHAT was said
# ---------------------------------------------------------------------------
def transcribe_words(audio):
    """Whisper word timestamps.

    Whisper's encoder window is a fixed 30 s. Under that it is one forward pass; over it the
    pipeline has to chunk. Chunking is requested EXPLICITLY when the track is long rather
    than left to whichever path the pipeline picks, so both --audio modes behave the same way
    and the switch is visible in the output instead of inferred.
    """
    secs = len(audio) / SAMPLING_RATE
    extra = {}
    if secs > 30.0:
        print(f"     track is {secs:.1f}s, past Whisper's 30 s window: chunking at 30 s")
        extra = {"chunk_length_s": 30, "batch_size": 1, "ignore_warning": True}
    out = asr_pipe()(
        audio_dict(audio),
        generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
        return_timestamps="word",
        **extra,
    )
    return out["chunks"]


# ---------------------------------------------------------------------------
# WHO said it: the merge, reimplemented with the guards speechbox lacks
# ---------------------------------------------------------------------------
def merge_speakers(segments, chunks, total_s):
    """Attach a speaker to every word by aligning on segment END times.

    speechbox's ASRDiarizationPipeline aligns only on ends and consumes the transcript
    greedily, which is fine on well-behaved input and raises on everything else. The guards
    below are the three cases it does not handle:

      1. an EMPTIED timestamp array. Once the transcript is consumed, np.argmin raises
         "attempt to get argmin of an empty sequence" on the next segment;
      2. a None end timestamp. Whisper emits one for a final unterminated chunk, and
         np.abs(None - end) is a TypeError;
      3. a SINGLE segment, where its merge loop body never runs, so the transcript comes
         back empty rather than wholly attributed to the one speaker it must belong to.
    """
    transcript = list(chunks)
    ends = np.array(
        [c["timestamp"][-1] if c["timestamp"] and c["timestamp"][-1] is not None else total_s
         for c in transcript],
        dtype=float,
    )
    out = []
    for start, end, sid in segments:
        if ends.size == 0:          # guard 1
            break
        upto = int(np.argmin(np.abs(ends - end)))
        out.append({
            "speaker": sid if isinstance(sid, str) else f"SPEAKER_{sid}",
            "start": start,
            "end": end,
            "text": "".join(c["text"] for c in transcript[: upto + 1]).strip(),
        })
        transcript = transcript[upto + 1:]
        ends = ends[upto + 1:]
    if transcript and out:
        # guard 3's other half: whatever the last segment did not absorb still belongs to
        # somebody. speechbox drops it on the floor without saying so.
        out[-1]["text"] = (out[-1]["text"] + " "
                           + "".join(c["text"] for c in transcript)).strip()
    return out


def format_as_transcription(segments) -> str:
    """The course's own tidy table."""
    return "\n".join(
        f"     {seg['speaker']:<12} ({seg['start']:5.1f}, {seg['end']:5.1f})  {seg['text']}"
        for seg in segments
    )


# ---------------------------------------------------------------------------
# The stand-in diarizer, and the null it has to beat
# ---------------------------------------------------------------------------
def mfcc_windows(audio, win_s: float = WIN_S):
    """One feature vector per win_s of audio: the mean of 20 MFCCs.

    std and delta features were tried and make the accuracy WORSE, which is the point. MFCCs
    describe the SHAPE OF THE SPECTRUM, so they encode what was said far more strongly than
    who said it, and there is no cheap feature to bolt on that turns phonetics into speaker
    identity. That takes a speaker-embedding model, and every CPU-only one on this stack
    wants torchaudio.
    """
    import librosa

    step = int(win_s * SAMPLING_RATE)
    centres, feats = [], []
    for start in range(0, len(audio) - step, step):
        centres.append((start + step / 2) / SAMPLING_RATE)
        feats.append(librosa.feature.mfcc(y=audio[start:start + step], sr=SAMPLING_RATE,
                                          n_mfcc=N_MFCC).mean(axis=1))
    return np.array(centres), np.array(feats)


def cluster(feats, n_clusters: int = 2):
    cls = _agglomerative()
    return cls(n_clusters=n_clusters, metric="cosine",
               linkage="average").fit_predict(np.asarray(feats))


def diarization_accuracy(audio, truth, win_s: float = WIN_S):
    """Frame accuracy against the TRUE speaker of each window. Returns (accuracy, n)."""
    centres, feats = mfcc_windows(audio, win_s)
    who = [next((sid for s, e, sid in truth if s <= c < e), None) for c in centres]
    keep = [i for i, w in enumerate(who) if w is not None]
    labels = [who[i] for i in keep]
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    pred = cluster(feats[keep])
    order = sorted(set(labels))
    truth_arr = np.array([order.index(x) for x in labels])
    # Cluster ids are arbitrary, so take the better of the two assignments.
    return float(max((pred == truth_arr).mean(), (pred != truth_arr).mean())), len(labels)


def chance_baseline(n_windows: int, trials: int = TRIALS, seed: int = SEED) -> float:
    """What the scorer above returns when the clustering is RANDOM. It is not 50%.

    Because cluster ids are arbitrary, the scorer takes max() over the two cluster-to-speaker
    mappings. That max can never fall below 0.5 and on a short track it sits well above it on
    luck alone: with n windows the expected score is about 0.5 + sqrt(2 / (pi * n)) / 2,
    which is 59% at n=19 and still 54% at n=100.

    So "50% is a coin flip" flatters every number quoted beside it. Every accuracy this
    module prints is printed next to this number instead.
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_windows, dtype=int)
    truth[n_windows // 2:] = 1
    agree = (rng.integers(0, 2, size=(trials, n_windows)) == truth).mean(axis=1)
    return float(np.maximum(agree, 1.0 - agree).mean())


def predicted_segments(audio, win_s: float = WIN_S):
    """Contiguous runs of one cluster, as segments.

    This is the stand-in for pyannote's output, used where there is nothing to score against.
    """
    centres, feats = mfcc_windows(audio, win_s)
    total = len(audio) / SAMPLING_RATE
    if len(feats) < 2:
        return [(0.0, total, "SPEAKER_0")]
    pred = cluster(feats)
    segs, start = [], 0.0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[i - 1]:
            end = float(centres[i - 1] + win_s / 2)
            segs.append((start, end, f"SPEAKER_{int(pred[i - 1])}"))
            start = end
    segs[-1] = (segs[-1][0], total, segs[-1][2])   # mfcc_windows drops the tail; give it back
    return segs


# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------
def run_synthetic() -> None:
    banner("A two-speaker meeting with boundaries that are exact by construction")

    print(f"\n   Streaming {DIALECTS_ID} [{DIALECTS_CONFIG}]: five real speakers, one accent,")
    print("   one sex. That is the HARD case, and it is why the number at the bottom of this")
    print("   run is worth printing at all.\n")
    clips = stream_speaker_clips()

    # Every speaker is capped at MIN_CLIPS, so "the two most frequent" would be a coin toss
    # between equals. Take the two lowest ids: reproducible, and --sweep covers all of them.
    pair = sorted(clips)[:2]
    meeting, truth = build_meeting(clips, pair)
    secs = len(meeting) / SAMPLING_RATE
    print(f"\n     speakers {pair}, {len(truth)} turns, {secs:.1f}s total")
    for start, end, sid in truth:
        print(f"       SPEAKER_{sid}  {start:5.1f} - {end:5.1f}s")
    save_wav("meeting_synthetic.wav", meeting)

    banner("WHAT was said: Whisper word timestamps")
    t0 = time.perf_counter()
    chunks = transcribe_words(meeting)
    print(f"\n     {len(chunks)} word chunks in {time.perf_counter() - t0:.1f}s; first "
          f"{chunks[0]['text']!r} at {chunks[0]['timestamp']}")
    nones = sum(1 for c in chunks if not c["timestamp"] or c["timestamp"][-1] is None)
    print(f"     chunks with a missing end timestamp: {nones}  "
          f"(each one is a TypeError inside speechbox's merge)")

    banner("WHO said it: the merge, against the TRUE boundaries")
    print()
    print(format_as_transcription(merge_speakers(truth, chunks, secs)))
    print("\n   That table is only this clean because the segments handed to the merge are the")
    print("   TRUE ones. Swap in a real diarizer's segments and the merge is unchanged: the")
    print("   quality of this table is the quality of the diarization and nothing else.")

    banner("The diarization itself, scored against its own null")
    acc, n_win = diarization_accuracy(meeting, truth)
    chance = chance_baseline(n_win)
    print(f"\n     frame accuracy on this track : {acc:.1%}  ({n_win} one-second windows)")
    print(f"     RANDOM clustering scores     : {chance:.1%}  (measured, {TRIALS:,} trials)")
    print(f"     this method beats chance by  : {acc - chance:+.1%}")
    print("\n   The second line is the one everybody omits. The scorer takes max() over the two")
    print("   cluster-to-speaker mappings, because cluster ids are arbitrary, so it cannot")
    print("   report below 50%, and on a short track it lands near 60% on luck alone. So the")
    print(f"   honest comparison for {acc:.1%} is {chance:.1%}, not 50%.")
    if acc < chance:
        print("\n   On this track it does not even reach its own null: the clustering is worse")
        print("   than assigning speakers at random would have been.")
    elif acc - chance < 0.10:
        print(f"\n   On this track it clears that null by {acc - chance:.1%}, which against a")
        print("   quoted 50% would have read as a comfortable pass and is in fact noise.")
    else:
        print(f"\n   On this track it clears that null by {acc - chance:.1%}, which is a real")
        print("   margin. One track is not a method, though.")
    print("\n   Run --sweep before forming an opinion from any single track: the spread across")
    print("   the ten speaker pairs is far wider than the gap you are looking at here.")


def run_course() -> None:
    from datasets import Audio, get_dataset_split_names, load_dataset

    banner("The course's own meeting audio, and what there is to check it against")

    splits = get_dataset_split_names(COURSE_ID)
    split = "train" if "train" in splits else splits[0]
    print(f"\n   Downloading {COURSE_ID} (661 KB), splits={splits}, using {split!r}.")
    stream = load_dataset(COURSE_ID, split=split, streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    rows = list(stream)
    row = rows[0]
    audio = np.asarray(row["audio"]["array"], dtype=np.float32)
    secs = len(audio) / SAMPLING_RATE
    print(f"\n     columns : {list(row)}")
    print(f"     rows    : {len(rows)}")
    print(f"     audio   : {secs:.1f}s @ {SAMPLING_RATE} Hz")
    save_wav("meeting_course.wav", audio)

    chunks = transcribe_words(audio)
    print(f"     words   : {len(chunks)} chunks with timestamps")

    merged = merge_speakers(predicted_segments(audio), chunks, secs)

    banner("The course's tidy table")
    print()
    print(format_as_transcription(merged))

    banner("And now the part the course does not print")
    print(f"\n   That table has {len(merged)} labelled segments and it looks authoritative.")
    print("   Here is the complete list of things available to check it against:\n")
    print(f"     reference transcript : none. {COURSE_ID} has the columns {list(row)}")
    print("     speaker labels       : none")
    print("     turn boundaries      : none")
    print(f"     rows to average over : {len(rows)}")
    print("\n   So there is no accuracy to report here. Not a low one. None. The course runs")
    print("   this dataset through pyannote plus speechbox, prints exactly this shape of")
    print("   table, and moves on, and nothing in the chapter ever computes a number for it.")
    print("   A demo you cannot score is a demo, not a result.")
    print("\n   The speaker labels above came from the SAME MFCC clustering that --audio")
    print("   synthetic scores against real boundaries and --sweep scores across all ten")
    print("   speaker pairs. Run either one to see what it is actually worth; this mode")
    print("   cannot tell you, and neither can the course.")
    print("\n   The table it produces here is just as tidy and just as confident either way.")
    print("   That is the whole contrast: the formatting is identical whether the diarization")
    print("   is right or a coin flip, and only a scored run can tell the two apart.")


def run_sweep() -> None:
    banner("Every speaker pair: is one bad track bad luck, or is this just what it does?")

    print(f"\n   Streaming {DIALECTS_ID} [{DIALECTS_CONFIG}] ONCE and reusing it for every")
    print("   pair. Re-streaming per pair would re-read the same ~59 MB row group ten times.")
    print("   No ASR in this mode: the transcript is not what is being measured.\n")
    clips = stream_speaker_clips()

    speakers = sorted(clips)
    pairs = list(itertools.combinations(speakers, 2))
    print(f"\n     {len(speakers)} speakers -> {len(pairs)} pairs\n")
    print(f"     {'pair':<12}{'seconds':>9}{'windows':>9}{'accuracy':>10}"
          f"{'chance':>9}{'delta':>9}")

    names, accs, chances, skipped = [], [], [], []
    for pair in pairs:
        if not enough_clips(clips, pair):
            skipped.append(pair)
            continue
        meeting, truth = build_meeting(clips, list(pair))
        acc, n_win = diarization_accuracy(meeting, truth)
        if np.isnan(acc):
            skipped.append(pair)
            continue
        ch = chance_baseline(n_win)
        label = f"{pair[0]}+{pair[1]}"
        names.append(label)
        accs.append(acc)
        chances.append(ch)
        flag = "" if acc >= ch else "   <- below chance"
        print(f"     {label:<12}{len(meeting) / SAMPLING_RATE:9.1f}{n_win:9d}"
              f"{acc:10.1%}{ch:9.1%}{acc - ch:+9.1%}{flag}")

    if not accs:
        sys.exit("no pair produced a scorable track; check the streamed clip counts above")

    arr, ch_arr = np.array(accs), np.array(chances)
    beats = int((arr >= ch_arr).sum())
    usable = int((arr >= USABLE).sum())
    print(f"\n     scored {len(accs)} of {len(pairs)} pairs"
          + (f" ({len(skipped)} skipped: {skipped})" if skipped else ""))
    print(f"     mean accuracy   : {arr.mean():.1%}   (mean chance {ch_arr.mean():.1%})")
    print(f"     min / max       : {arr.min():.1%} / {arr.max():.1%}")
    print(f"     beats its null  : {beats} of {len(accs)}")
    print(f"     reaches {USABLE:.0%}     : {usable} of {len(accs)}")
    print(f"\n   {usable} of {len(accs)} pairs reach a usable {USABLE:.0%}, and the average edge")
    print(f"   over chance is {(arr - ch_arr).mean():+.1%}. That is the headline.")
    print(f"\n   The mean is not the story, the SPREAD is: {arr.min():.1%} to {arr.max():.1%}")
    print("   on the same corpus, the same accent and the same sex, with the same code. A")
    print("   method whose accuracy ranges that far has not learned who is speaking; it has")
    print("   learned what they happened to say. Quote one track and you can report almost")
    print("   any number you like, which is exactly why the walkthrough's single track is")
    print("   not the whole answer and why this mode exists.")
    print("\n   This is a measured negative result, not a diarizer. For the real thing the")
    print("   course is right: pyannote, which needs torchaudio, a Hub token and an accepted")
    print("   gate, and is therefore not installable here by choice.")

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(x, [a * 100 for a in accs],
                  color=["tab:green" if a >= c else "tab:red" for a, c in zip(accs, chances)])
    ax.plot(x, [c * 100 for c in chances], "k_", markersize=26, markeredgewidth=2,
            label="that pair's measured chance baseline")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a * 100, f"{a:.0%}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color="0.75", ls=":", label="the 50% everyone quotes")
    ax.axhline(USABLE * 100, color="tab:green", ls="--", label=f"usable ({USABLE:.0%})")
    ax.set(xticks=x, xticklabels=names, xlabel="speaker pair", ylabel="frame accuracy (%)",
           ylim=(0, 105),
           title=f"MFCC + AgglomerativeClustering across {len(accs)} speaker pairs\n"
                 f"{beats} of {len(accs)} beat their own null; {usable} reach {USABLE:.0%}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    print()
    save_fig("meeting_sweep.png")


def parse_args(argv=None) -> argparse.Namespace:
    # argparse validates what it is GIVEN, never its own default, so an env var that is not
    # a real choice would sail past choices= and fail deep inside load_dataset instead.
    env = os.environ.get("UNIT7_AUDIO", AUDIO_CHOICES[0]).strip().lower()
    if env not in AUDIO_CHOICES:
        sys.exit(f"UNIT7_AUDIO={env!r} is not one of: {', '.join(AUDIO_CHOICES)}")

    parser = argparse.ArgumentParser(
        prog="meeting.py",
        description="Unit 7's meeting transcriber: word timestamps, the merge, and the "
                    "diarization that does not work.",
        epilog="The env var UNIT7_AUDIO sets the default track; an explicit flag beats it.",
    )
    parser.add_argument(
        "--audio", choices=AUDIO_CHOICES, default=env,
        help=f"synthetic builds a two-speaker track with exact boundaries and SCORES it; "
             f"course uses the dataset the course uses, which has no labels to score "
             f"against (default: {env})",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="score all 10 speaker pairs against their own chance baselines; implies "
             "--audio synthetic",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.sweep and args.audio != AUDIO_CHOICES[0]:
        # Say it rather than silently doing something else: the course track has one row and
        # no labels, so there is nothing for a sweep to sweep over.
        print(f"NOTE: --sweep is synthetic by definition (the {args.audio} track has one row "
              f"and no labels), so --audio {args.audio} is ignored.")
    mode = "sweep" if args.sweep else f"audio={args.audio}"
    print(f"Unit 7 meeting transcriber | {mode} | out={OUT_DIR}")

    if args.sweep:
        run_sweep()
    elif args.audio == "course":
        run_course()
    else:
        run_synthetic()


if __name__ == "__main__":
    main()
