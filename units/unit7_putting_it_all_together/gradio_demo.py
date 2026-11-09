"""
Unit 7 demo: four tabs, one cascade, and every seam visible - on CPU.

Tab 1 runs the three-stage cascade (ASR -> machine translation -> TTS) into a language you
pick, and reports which characters the chosen voice silently deleted. Tab 2 races that
against the course's two-stage shortcut and shows why the faster one cannot pass the
hands-on. Tab 3 is the four-stage voice assistant. Tab 4 does word timestamps, the speechbox
merge, and the MFCC clustering that is not a diarizer.

    uv run python units/unit7_putting_it_all_together/gradio_demo.py
    # then open http://127.0.0.1:7860

Models: openai/whisper-base (~290 MB, cached since Unit 5) loads at import. Everything else
loads the first time a tab asks for it, and the dropdown labels say what that costs. Nothing
is uploaded (share=False).

THIS FILE HAS MANY INPUTS AND MANY OUTPUTS BECAUSE IT IS LOCAL. space/app.py is the same
cascade with exactly ONE input and ONE output, because the hands-on grader calls
client.predict(file, api_name="/predict") and a second input or a second output fails there
with a gradio_client traceback instead of a grading message. Tab 1's language dropdown is
the single clearest reason those two files cannot be the same file.
"""

from __future__ import annotations

import time

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import pipeline

ASR_ID = "openai/whisper-base"          # stage 1 everywhere in this file
TTS_ENG = "facebook/mms-tts-eng"        # the 2-stage baseline's voice, and the assistant's
LLM_ID = "MBZUAI/LaMini-Flan-T5-248M"   # the Hub's pipeline_tag on this model is WRONG
WAKE_ID = "MIT/ast-finetuned-speech-commands-v2"
DIALECTS_ID = "ylacombe/english_dialects"
DIALECTS_CONFIG = "northern_female"     # 5 real speakers, one accent, one sex: the hard case

SAMPLING_RATE = 16_000
SEED = 7                 # VITS has a stochastic duration predictor: seed every call
WAKE_WORD = "marvin"     # AST id2label[27]; google/speech_commands' own index is 26
WAKE_THRESHOLD = 0.5
WAKE_WINDOW_S = 1.0      # AST was trained on ~1s isolated words, not on sentences
MIN_CLIP_S = 0.25
N_MFCC = 20
WIN_S = 1.0

# (Marian en->xx, MMS/VITS voice) per target. French and Arabic are cached by the
# walkthrough; the other three are one ~301 MB Marian checkpoint plus one ~145 MB VITS
# checkpoint each, downloaded ONLY when somebody picks them. The cost lives in the DISPLAYED
# label, so nobody clicks it blind - gradio's (label, value) choices keep that prose out of
# the value the function receives.
#
# There is no Italian here on purpose: facebook/mms-tts-ita does not exist. MMS-TTS publishes
# no Italian voice at all, so the pair cannot be built however good opus-mt-en-it is.
TARGETS = {
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "facebook/mms-tts-fra"),
    "ar": ("Helsinki-NLP/opus-mt-en-ar", "facebook/mms-tts-ara"),
    "de": ("Helsinki-NLP/opus-mt-en-de", "facebook/mms-tts-deu"),
    "nl": ("Helsinki-NLP/opus-mt-en-nl", "facebook/mms-tts-nld"),
    "ru": ("Helsinki-NLP/opus-mt-en-ru", "facebook/mms-tts-rus"),
}
TARGET_CHOICES = [
    ("French  - cached, downloads nothing", "fr"),
    ("Arabic  - cached, downloads nothing (non-Latin script)", "ar"),
    ("German  - downloads ~446 MB the first time", "de"),
    ("Dutch   - downloads ~446 MB the first time", "nl"),
    ("Russian - downloads ~446 MB the first time (Cyrillic)", "ru"),
]

# ".,!?;:" plus the Arabic comma, question mark and semicolon.
SENTENCE_PUNCT = ".,!?;:" + "،؟؛"

# Plain ASCII: this module has no sys.stdout.reconfigure block, so a "..." would
# mojibake on a cp1252 Windows console.
print(f"Loading {ASR_ID} (~290 MB on first run)...")
asr = pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)
torch.set_grad_enabled(False)

# One registry for everything that is not whisper, keyed by CHECKPOINT rather than by tab.
# Unit 6 could use `_mms = {"tok": None, "model": None}` because one tab had one lazy model;
# here four tabs overlap, and keying by checkpoint is what stops tab 1 at target "fr" and
# tab 2's three-stage leg from building two separate copies of facebook/mms-tts-fra.
_LAZY: dict[str, object] = {}


# ---------------------------------------------------------------- lazy model accessors
def _mt(model_id: str):
    """MarianMT en->xx. Sentencepiece-only tokenizer: there is no fast variant."""
    from transformers import MarianMTModel, MarianTokenizer

    key = f"mt:{model_id}"
    if key not in _LAZY:
        print(f"Loading {model_id} (~301 MB if this is the first time)...")
        _LAZY[key] = (MarianTokenizer.from_pretrained(model_id),
                      MarianMTModel.from_pretrained(model_id))
    return _LAZY[key]


def _tts(model_id: str):
    """MMS/VITS: text straight to waveform, no vocoder."""
    from transformers import VitsModel, VitsTokenizer

    key = f"tts:{model_id}"
    if key not in _LAZY:
        print(f"Loading {model_id} (~145 MB if this is the first time)...")
        tok = VitsTokenizer.from_pretrained(model_id)
        # Every facebook/mms-tts-* checkpoint ships phonemize:false, so this never fires for
        # the five targets above. VitsTokenizer's own DEFAULT is phonemize=True, and
        # kakao-enterprise/vits-ljs ships that: it raises from inside the tokenizer asking
        # for espeak-ng. Checking the flag turns that into a sentence you can act on.
        if tok.phonemize:
            raise RuntimeError(
                f"{model_id} has phonemize=true and needs espeak-ng installed system-wide. "
                f"Pick an MMS checkpoint instead; they all ship phonemize=false."
            )
        _LAZY[key] = (tok, VitsModel.from_pretrained(model_id))
    return _LAZY[key]


def _llm():
    """LaMini is a T5, a seq2seq, but the Hub's pipeline_tag says "text-generation".

    With no task, pipeline() believes the tag, builds a text-generation pipeline, logs one
    3,273-character error naming 200+ classes at ERROR level, and then ECHOES THE PROMPT
    BACK VERBATIM without ever raising. Passing the task explicitly is the entire fix, and
    it is the quietest bug in this unit.
    """
    if "llm" not in _LAZY:
        print(f"Loading {LLM_ID} (~990 MB if this is the first time)...")
        _LAZY["llm"] = pipeline("text2text-generation", model=LLM_ID, device=-1)
    return _LAZY["llm"]


def _wake_clf():
    if "wake" not in _LAZY:
        print(f"Loading {WAKE_ID} (~342 MB if this is the first time)...")
        _LAZY["wake"] = pipeline("audio-classification", model=WAKE_ID, device=-1)
    return _LAZY["wake"]


def _builtin_meeting(n_turns: int = 4):
    """A two-speaker track whose boundaries are exact BY CONSTRUCTION.

    That construction is the only reason tab 4 can report an accuracy at all. It streams one
    ~59 MB parquet row group of Unit 6's dialect corpus, and streaming=True does NOT populate
    ~/.cache/huggingface, so without this memo it would re-stream on every press. Memoised
    here it is once per process.

    northern_female is five real speakers with the same accent and the same sex, the hard
    case. A cheaper track built from two SpeechT5 x-vectors would download nothing and score
    much higher, and reporting THAT number under this tab's headline would be a quiet lie.
    """
    if "meeting" not in _LAZY:
        from datasets import Audio, load_dataset

        print(f"Streaming {DIALECTS_ID}/{DIALECTS_CONFIG} (~59 MB, once per process)...")
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
        _LAZY["meeting"] = (np.concatenate(turns), truth, pair)
    return _LAZY["meeting"]


# ---------------------------------------------------------------------------- the seams
def audio_dict(array) -> dict:
    """A FRESH dict for every pipeline call.

    AutomaticSpeechRecognitionPipeline.preprocess mutates the dict you hand it - it moves
    "array" into "raw" in place - so reusing one dict across two calls makes the second
    raise `ValueError: ... the dict needs to contain a "raw" key`, which reads like your
    input was malformed rather than already consumed.
    """
    return {"array": np.asarray(array), "sampling_rate": SAMPLING_RATE}


def _decode(filepath):
    """(array, error). librosa straight to 16 kHz mono avoids two traps: the pipeline never
    shells out to ffmpeg, and it never hits transformers'
    ImportError("torchaudio is required to resample")."""
    if filepath is None:
        return None, "Record or upload some audio first."
    array, _ = librosa.load(filepath, sr=SAMPLING_RATE, mono=True)
    if len(array) < int(MIN_CLIP_S * SAMPLING_RATE):
        return None, f"That clip is under {MIN_CLIP_S}s, too short for Whisper to say anything."
    return array, None


def transcribe(array, task="transcribe", timestamps=False):
    gen = {"task": task, "num_beams": 1}
    if task == "transcribe":
        gen["language"] = "english"
    # Deliberately NOT passing language with task="translate": Whisper's translation
    # objective has exactly one target and it is English, so the kwarg is SILENTLY IGNORED.
    # Passing it would imply, in the code, that it does something.
    return asr(
        audio_dict(array),
        chunk_length_s=30,      # so a file past Whisper's 30 s encoder window still works
        batch_size=8,
        ignore_warning=True,
        return_timestamps=timestamps,
        generate_kwargs=gen,
    )


def translate_text(text: str, model_id: str) -> str:
    tok, model = _mt(model_id)
    out = model.generate(**tok(text, return_tensors="pt"), num_beams=1, max_new_tokens=256)
    return tok.decode(out[0], skip_special_tokens=True)


def synthesise(text: str, model_id: str) -> np.ndarray:
    tok, model = _tts(model_id)
    torch.manual_seed(SEED)   # VITS's duration predictor is stochastic
    wav = model(**tok(text=text, return_tensors="pt")).waveform
    return np.asarray(wav, dtype=np.float32).squeeze()


def deleted_characters(text: str, model_id: str):
    """(n_deleted, the distinct characters) this voice will drop before it sees the text.

    VitsTokenizer is built with normalize=True, and that path ends in
        filtered_text = "".join(list(filter(lambda char: char in self.encoder, ...)))
    so anything outside the vocabulary is DELETED rather than turned into <unk>, with
    nothing logged. None of the five voices here has a full stop, a comma, an exclamation
    mark or a question mark, so every sentence loses its punctuation on the way in.
    """
    tok, _ = _tts(model_id)
    normalised = tok.normalize_text(text)
    filtered, _ = tok.prepare_for_tokenization(text)
    return len(normalised) - len(filtered), "".join(sorted(set(normalised) - set(filtered)))


# ======================================================================== 1. TRANSLATE
def translate_speech(filepath, target):
    array, err = _decode(filepath)
    if err:
        return None, err
    mt_id, tts_id = TARGETS[target]
    src_s = len(array) / SAMPLING_RATE

    t0 = time.perf_counter()
    english = transcribe(array)["text"].strip()
    t_asr = time.perf_counter() - t0

    t0 = time.perf_counter()
    foreign = translate_text(english, mt_id)
    t_mt = time.perf_counter() - t0

    t0 = time.perf_counter()
    wav = synthesise(foreign, tts_id)
    t_tts = time.perf_counter() - t0

    n_dropped, dropped = deleted_characters(foreign, tts_id)
    tok, _ = _tts(tts_id)
    symbols = sorted(set(tok.get_vocab()) - set(tok.all_special_tokens))
    punct = [c for c in SENTENCE_PUNCT if c in symbols]

    total = t_asr + t_mt + t_tts
    note = (
        f"1. ASR  {t_asr:6.2f}s  {ASR_ID}\n"
        f"        {english}\n"
        f"2. MT   {t_mt:6.2f}s  {mt_id}\n"
        f"        {foreign}\n"
        f"3. TTS  {t_tts:6.2f}s  {tts_id}\n"
        f"        {len(wav) / SAMPLING_RATE:.2f}s of audio\n\n"
        f"{total:.2f}s total for {src_s:.2f}s in (real-time factor {total / src_s:.2f}). "
        f"Latency is additive:\nit is three models' compute, not the cost of the slowest one.\n\n"
        f"WHAT THE VOICE SILENTLY DELETED\n"
        f"  {tts_id} has {len(symbols)} pronounceable symbols and "
        f"{len(punct)} of ,.!?;: in them.\n"
        f"  It dropped {n_dropped} characters from the translation above: {dropped or '(none)'}\n"
        f"  Not <unk> - deleted, before the model saw the text, with nothing logged.\n"
        f"  A full stop is not mispronounced by these models; it is removed, and every\n"
        f"  prosodic cue a reader would take from it goes with it.\n\n"
        f"And look at what crosses each seam: stage 1 hands stage 2 a STRING. Not audio, not\n"
        f"a lattice, not a confidence. That is why an ASR mistake here is unrecoverable -\n"
        f"stage 2 never sees the audio - and why pitch, pace and speaker die at the first seam."
    )
    # gradio wants (sampling_rate, array) in THAT order - the most common mistake here is
    # returning (array, sampling_rate), which plays as a fraction of a second of noise.
    return (SAMPLING_RATE, wav), note


# ==================================================================== 2. TWO CASCADES
def compare_cascades(filepath):
    array, err = _decode(filepath)
    if err:
        return None, None, err
    # Pinned to French: this tab compares two ARCHITECTURES, not two languages, and pinning
    # it means pressing Submit here can never trigger a 446 MB download.
    mt_id, tts_id = TARGETS["fr"]

    # Load both voices and the translator BEFORE either clock starts. Otherwise the first
    # cascade to run pays for its own imports and the second looks artificially quick: with
    # mms-tts-eng loading inside the 2-stage timer, the 2-stage leg measured 4.17s against
    # the 3-stage's 2.15s, which reverses the result this tab exists to show.
    _mt(mt_id), _tts(tts_id), _tts(TTS_ENG)

    t0 = time.perf_counter()
    english = transcribe(array)["text"].strip()
    french = translate_text(english, mt_id)
    three = synthesise(french, tts_id)
    t_three = time.perf_counter() - t0

    t0 = time.perf_counter()
    direct = transcribe(array, task="translate")["text"].strip()
    two = synthesise(direct, TTS_ENG)
    t_two = time.perf_counter() - t0

    # Say which one was faster from the clock, not from the expectation. On a warm process
    # the gap is a fraction of a second either way, and it is not the point of this tab.
    gap = t_three - t_two
    verdict = (f"The 3-stage cascade cost {gap:+.2f}s against the 2-stage one"
               if gap > 0 else
               f"The 3-stage cascade was actually {-gap:.2f}s FASTER on this run")

    note = (
        f"3-stage  ASR -> opus-mt-en-fr -> mms-tts-fra      {t_three:6.2f}s\n"
        f"   heard : {english}\n"
        f"   spoken: {french}\n\n"
        f"2-stage  ASR(task='translate') -> mms-tts-eng     {t_two:6.2f}s\n"
        f"   spoken: {direct}\n\n"
        f"{verdict}. Both models were loaded before either clock started, so these are\n"
        f"inference numbers; time an unwarmed cascade and you measure imports instead.\n\n"
        f"The timing is not the point, though - the LANGUAGE is. The 2-stage cascade is\n"
        f"cheaper precisely because it deletes the MT hop, and the MT hop is the only reason\n"
        f"the output is not English. Whisper's task='translate' has exactly one target\n"
        f"language, and passing language= alongside it is silently ignored. The hands-on\n"
        f"grader runs a language classifier on whatever your Space returns and fails you if\n"
        f"the top label is 'eng' or its score is under 0.5, so the cheaper cascade cannot\n"
        f"pass at all. You are not choosing between fast and slow. You are paying a fraction\n"
        f"of a second for the only output that can pass."
    )
    return (SAMPLING_RATE, three), (SAMPLING_RATE, two), note


# ====================================================================== 3. ASSISTANT
def assistant(filepath, gate):
    array, err = _decode(filepath)
    if err:
        return None, err

    # Stage 1: wake word. Score the FIRST SECOND, not the whole clip. AST's classes are
    # isolated ~1s command words, and handing it six seconds of a sentence asks a question
    # it was never trained to answer.
    clf = _wake_clf()
    head = array[: int(WAKE_WINDOW_S * SAMPLING_RATE)]
    top = clf(audio_dict(head), top_k=1)[0]
    woke = top["label"] == WAKE_WORD and top["score"] > WAKE_THRESHOLD
    wake_line = (
        f"1. wake word   first {WAKE_WINDOW_S:.1f}s -> {top['label']!r} {top['score']:.3f}   "
        f"{'WOULD WAKE' if woke else 'would NOT wake'} "
        f"(needs {WAKE_WORD!r} above {WAKE_THRESHOLD})\n"
        f"   AST has {len(clf.model.config.id2label)} classes and {WAKE_WORD!r} is id "
        f"{clf.model.config.label2id[WAKE_WORD]}; google/speech_commands puts it at 26 of 36.\n"
        f"   Compare label STRINGS, never ints."
    )

    if gate and not woke:
        return None, (
            wake_line + "\n\n"
            "The gate is ON, so the assistant stopped here.\n\n"
            "It is OFF by default on purpose. A whole spoken sentence rarely scores as one\n"
            "isolated command word, so a gate that blocks makes this tab look broken for the\n"
            "right reason in a way you would blame on the wrong stage. Reporting the score\n"
            "and carrying on keeps stage 1's failure attributable to stage 1, which is the\n"
            "entire argument of this unit. Untick the box to see the other three stages run."
        )

    t0 = time.perf_counter()
    heard = transcribe(array)["text"].strip()
    t_asr = time.perf_counter() - t0

    prompt = f"Answer in one short sentence: {heard}"
    t0 = time.perf_counter()
    answer = _llm()(prompt, max_new_tokens=60)[0]["generated_text"].strip()
    t_llm = time.perf_counter() - t0

    t0 = time.perf_counter()
    reply = synthesise(answer, TTS_ENG)
    t_tts = time.perf_counter() - t0

    note = (
        f"{wake_line}\n\n"
        f"2. transcribe  {t_asr:6.2f}s  {ASR_ID}\n"
        f"   {heard}\n\n"
        f"3. language model {t_llm:6.2f}s  {LLM_ID} as text2text-generation\n"
        f"   prompt: {prompt}\n"
        f"   answer: {answer}\n\n"
        f"4. speak       {t_tts:6.2f}s  {TTS_ENG}, {len(reply) / SAMPLING_RATE:.2f}s out\n\n"
        f"Two of the course's four stages no longer exist. It calls tiiuae/falcon-7b-instruct\n"
        f"over api-inference.huggingface.co, a host whose DNS record no longer resolves and a\n"
        f"model whose inferenceProviderMapping is now empty, hence the small local model. And\n"
        f"`from transformers import HfAgent`, which the course's closing section uses to\n"
        f"generalise this assistant, is gone: agents were removed from transformers entirely.\n\n"
        f"Stage 3 has one more trap. The Hub's pipeline_tag for {LLM_ID.split('/')[-1]} says\n"
        f"'text-generation'; it is a T5, a seq2seq. Call pipeline() with no task and you get a\n"
        f"parrot that echoes your prompt back verbatim and never raises.\n\n"
        f"assistant.py runs these same four stages from the command line, and `--live` opens a\n"
        f"push-to-talk page of its own. That file restates this code rather than importing it:\n"
        f"every module in this repo is self-contained on purpose."
    )
    return (SAMPLING_RATE, reply), note


# ======================================================================== 4. MEETING
def _windows(audio, win_s: float = WIN_S):
    """One feature vector per win_s of audio: the mean of 20 MFCCs."""
    step = int(win_s * SAMPLING_RATE)
    centres, feats = [], []
    for start in range(0, len(audio) - step, step):
        centres.append((start + step / 2) / SAMPLING_RATE)
        feats.append(librosa.feature.mfcc(y=audio[start:start + step], sr=SAMPLING_RATE,
                                          n_mfcc=N_MFCC).mean(axis=1))
    return np.array(centres), np.array(feats)


def _cluster(feats, n_clusters: int = 2):
    # Windows Application Control intermittently blocks a scikit-learn extension DLL on a
    # cold import and lets the identical import through a moment later. One retry keeps a
    # transient OS block from being read as a missing dependency.
    try:
        from sklearn.cluster import AgglomerativeClustering
    except (ImportError, OSError):
        time.sleep(1.0)
        from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=n_clusters, metric="cosine",
                                   linkage="average").fit_predict(np.asarray(feats))


def _chance_baseline(n_windows: int, trials: int = 20_000, seed: int = SEED) -> float:
    """What the scorer below returns when the clustering is RANDOM. It is not 50%.

    Cluster ids are arbitrary, so the scorer takes max() over the two cluster-to-speaker
    mappings. That max can never fall below 0.5, and on a short track it sits well above it
    on luck alone. Quoting 50% as the coin flip turns a null result into a weak positive one.
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_windows, dtype=int)
    truth[n_windows // 2:] = 1
    agree = (rng.integers(0, 2, size=(trials, n_windows)) == truth).mean(axis=1)
    return float(np.maximum(agree, 1.0 - agree).mean())


def _score(pred, centres, truth):
    """(accuracy, n_windows) against true speaker labels. Only callable when truth exists."""
    paired = [(int(p), next((sid for s, e, sid in truth if s <= c < e), None))
              for p, c in zip(pred, centres)]
    paired = [(p, who) for p, who in paired if who is not None]
    if len({who for _, who in paired}) < 2:
        return float("nan"), len(paired)
    ids = sorted({who for _, who in paired})
    p = np.array([x[0] for x in paired])
    t = np.array([ids.index(x[1]) for x in paired])
    return float(max((p == t).mean(), (p != t).mean())), len(paired)


def _runs_to_segments(pred, centres, total_s, win_s: float = WIN_S):
    """Collapse per-second cluster labels into contiguous (start, end, cluster) runs."""
    if pred.size == 0:
        return []
    segs, start, current = [], 0.0, int(pred[0])
    for i in range(1, len(pred)):
        if int(pred[i]) != current:
            end = float(centres[i] - win_s / 2)
            segs.append((start, end, f"SPEAKER_{current}"))
            start, current = end, int(pred[i])
    segs.append((start, total_s, f"SPEAKER_{current}"))
    return segs


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
        out.append({"speaker": sid if isinstance(sid, str) else f"SPEAKER_{sid}",
                    "start": start, "end": end,
                    "text": "".join(c["text"] for c in transcript[: upto + 1]).strip()})
        transcript = transcript[upto + 1:]
        ends = ends[upto + 1:]
    if transcript and out:
        out[-1]["text"] = (out[-1]["text"] + " "
                           + "".join(c["text"] for c in transcript)).strip()
    return out


def meeting(source, filepath):
    if source == "builtin":
        audio, truth, pair = _builtin_meeting()
        header = (f"built-in track: {len(audio) / SAMPLING_RATE:.1f}s, speakers {pair}, "
                  f"{len(truth)} turns, boundaries exact by construction")
    else:
        audio, err = _decode(filepath)
        if err:
            return None, err + "  (or switch to the built-in track)"
        truth = None
        header = f"your audio: {len(audio) / SAMPLING_RATE:.1f}s, no ground truth"

    # No chunk_length_s here on purpose: unset, transformers uses Whisper's SEQUENTIAL
    # long-form algorithm, whose word timestamps are the ones speechbox's merge was designed
    # around. The chunked path re-stitches timestamps across window overlaps.
    out = asr(audio_dict(audio), return_timestamps="word",
              generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1})
    chunks = out["chunks"]

    centres, feats = _windows(audio)
    total_s = len(audio) / SAMPLING_RATE
    pred = _cluster(feats) if len(feats) >= 2 else np.array([])
    segments = truth if truth is not None else _runs_to_segments(pred, centres, total_s)
    merged = merge_speakers(segments, chunks, total_s)

    lines = [header, ""]
    if chunks:
        lines.append(f"whisper word timestamps: {len(chunks)} words, first "
                     f"{chunks[0]['text'].strip()!r} at {chunks[0]['timestamp']}")
    lines.append("")
    boundary = "TRUE boundaries" if truth is not None else "CLUSTER boundaries"
    lines.append(f"the speechbox merge, against {boundary}:")
    for seg in (merged[:6] or [None]):
        lines.append("  (no segments to merge: the clip is shorter than two windows)"
                     if seg is None else
                     f"  [{seg['speaker']}] ({seg['start']:.1f}-{seg['end']:.1f}s) "
                     f"{seg['text'][:70]}")
    lines.append("")

    if truth is not None:
        acc, n_win = _score(pred, centres, truth)
        chance = _chance_baseline(n_win)
        lines += [
            f"MFCC + AgglomerativeClustering : {acc:.1%}  ({n_win} one-second windows)",
            f"RANDOM clustering scores       : {chance:.1%}  (measured, 20,000 trials)",
            f"this method beats chance by    : {acc - chance:+.1%}",
            "",
            "That second line is the one everybody omits. The scorer takes max() over the",
            "two cluster-to-speaker mappings, because cluster ids are arbitrary, so it",
            "cannot report below 50% and on a short track it lands near 60% on luck alone.",
            "Against a quoted 50% this looks like a weak positive result. Against its own",
            "null it is not a result at all.",
            "",
            "MFCCs encode WHAT was said far more strongly than WHO said it, so without a",
            "speaker-embedding model you are clustering phonetics. The course uses pyannote",
            "here; it is gated and needs torchaudio, and this repo installs neither.",
            "meeting.py --sweep scores all ten speaker pairs the same way.",
        ]
    else:
        lines += [
            "accuracy: not reported.",
            "",
            "Your clip has no speaker labels, so there is nothing to score the clustering",
            "against, and a number invented here would be a number nobody measured. The ids",
            "above are CLUSTER ids, not people: nothing checked that this file contains two",
            "speakers, and the clustering was told to find exactly two whether or not they",
            "are there. Switch to the built-in track for the one input where an accuracy is",
            "computable. That is the whole reason this radio button exists, and the reason",
            "the table above looks just as confident either way.",
        ]
    return (SAMPLING_RATE, audio), "\n".join(lines)


# Each Interface is built at module level, OUTSIDE any `with` block. Wrapping these in
# `with gr.Blocks() as demo:` splices the children into the parent on gradio 6 and renders
# every tab twice. TabbedInterface is already a Blocks.
translate_tab = gr.Interface(
    fn=translate_speech,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speech (English)"),
        gr.Dropdown(choices=TARGET_CHOICES, value="fr", label="Target language"),
    ],
    outputs=[gr.Audio(label="Translated speech", type="numpy"),
             gr.Textbox(label="Every seam, and what it cost", lines=20)],
    flagging_mode="never",  # gradio 6: renamed from allow_flagging="never"
)

compare_tab = gr.Interface(
    fn=compare_cascades,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speech"),
    outputs=[
        gr.Audio(label="3-stage: ASR -> MT -> TTS (French)", type="numpy"),
        gr.Audio(label="2-stage: ASR task='translate' -> TTS (English)", type="numpy"),
        gr.Textbox(label="Both cascades, both clocks", lines=16),
    ],
    flagging_mode="never",
)

assistant_tab = gr.Interface(
    fn=assistant,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath",
                 label=f"Say '{WAKE_WORD}', then ask something"),
        gr.Checkbox(value=False, label="Require the wake word (gate the rest of the pipeline)"),
    ],
    outputs=[gr.Audio(label="Spoken answer", type="numpy"),
             gr.Textbox(label="Four stages, two of which no longer exist", lines=24)],
    flagging_mode="never",
)

meeting_tab = gr.Interface(
    fn=meeting,
    inputs=[
        gr.Radio(
            choices=[("Built-in two-speaker track - SCORED (streams ~59 MB once per run)",
                      "builtin"),
                     ("Your own audio - NOT scored (no ground truth to score against)",
                      "upload")],
            value="builtin",
            label="Track",
        ),
        gr.Audio(sources=["upload", "microphone"], type="filepath",
                 label="Your audio (only used by the second option)"),
    ],
    outputs=[gr.Audio(label="The track", type="numpy"),
             gr.Textbox(label="Timestamps, the merge, and the number (or why there isn't one)",
                        lines=26)],
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [translate_tab, compare_tab, assistant_tab, meeting_tab],
    ["🌍 Translate speech", "⚖️ Two cascades", "🤖 Voice assistant", "📝 Meeting"],
    title="🔗 Putting it all together: speech in, speech out, every seam measured",
)

if __name__ == "__main__":
    demo.launch(share=False)


# gradio 3 (what the course was written against) → gradio 6 changes used above:
#   gr.Audio(source=…)                     → gr.Audio(sources=…)
#   gr.Interface(allow_flagging=…)         → gr.Interface(flagging_mode=…)
#   with gr.Blocks(): TabbedInterface(...) → TabbedInterface used directly
#   gr.Audio outputs return (sampling_rate, array), not (array, sampling_rate)
#   demo.launch(debug=True) is a Colab/notebook idiom; locally share=False is enough.
#   gr.Interface with api_name unset now names its endpoint after the FUNCTION rather than
#     always "/predict" (gradio/blocks.py: "If api_name is None or empty string, use the
#     function name"). Nothing calls THIS file over the API, so it does not matter here -
#     but it is exactly why space/app.py names its function `predict`, and why bumping that
#     Space's sdk_version is not a free action.
