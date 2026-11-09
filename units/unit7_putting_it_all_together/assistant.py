"""
Hugging Face Audio Course - Unit 7: the four-stage voice assistant.

Wake word -> transcribe -> ask a language model -> speak the answer. ONE module, TWO modes
(--mode, the --live shorthand, or the env var UNIT7_MODE), default = canned:

  canned (default)  No microphone, no browser, no network. The wake word is SYNTHESISED with
                    mms-tts-eng and the spoken command comes from the cached LibriSpeech
                    dummy, so the whole chain runs off cached weights with a timing for every
                    stage. It is a regression test as much as a demo: the wake word is a hard
                    gate and the reply is ASSERTED to differ from the prompt, so both of the
                    ways this assistant fails silently become a non-zero exit code.

  live              A Gradio push-to-talk with FOUR outputs, one per stage, so a bad answer
                    is attributable to the stage that produced it rather than blamed on the
                    voice at the end of the pipe. Here the wake word is a REPORTED SCORE and
                    not a gate - see the note in run_live().

    uv run python units/unit7_putting_it_all_together/assistant.py           # canned
    uv run python units/unit7_putting_it_all_together/assistant.py --live    # Gradio
    uv run python units/unit7_putting_it_all_together/assistant.py --help

Unit 7 is the first unit in this repo to use argparse. Units 4 to 6 toggle one binary mode
with `"--full" in sys.argv` plus an env var, which is enough for one switch and stops being
enough here: meeting.py needs a flag that takes a VALUE, and a hand-rolled parser that
silently ignores a typo is exactly the class of bug this unit is about. The env vars are
kept so the two idioms agree.

Models: MIT/ast-finetuned-speech-commands-v2 (~342 MB, cached since Unit 4),
openai/whisper-base (~290 MB, cached since Unit 5), MBZUAI/LaMini-Flan-T5-248M (~990 MB) and
facebook/mms-tts-eng (~145 MB, cached since Unit 6). The walkthrough caches all four, so a
warm run downloads nothing. Audio is written to figures/, which is git-ignored, and nothing
is uploaded: the Gradio mode launches with share=False.


THE COURSE'S RECIPE, PRESERVED BECAUSE NOT ONE LINE OF IT STILL RUNS
-------------------------------------------------------------------
The course builds stages 1 and 2 on a live microphone stream:

    from transformers.pipelines.audio_utils import ffmpeg_microphone_live

    def launch_fn(wake_word="marvin", prob_threshold=0.5,
                  chunk_length_s=2.0, stream_chunk_s=0.25, debug=False):
        if wake_word not in classifier.model.config.label2id.keys():
            raise ValueError(f"Wake word {wake_word} not in set of valid class labels...")
        mic = ffmpeg_microphone_live(
            sampling_rate=classifier.feature_extractor.sampling_rate,
            chunk_length_s=chunk_length_s, stream_chunk_s=stream_chunk_s)
        print("Listening for wake word...")
        for prediction in classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threshold:
                    return True

    def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
        mic = ffmpeg_microphone_live(
            sampling_rate=transcriber.feature_extractor.sampling_rate,
            chunk_length_s=chunk_length_s, stream_chunk_s=stream_chunk_s)
        print("Start speaking...")
        for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
            sys.stdout.write("\033[K")
            print(item["text"], end="\r")
            if not item["partial"][0]:
                break
        return item["text"]                  # <- LOOK AT THIS RETURN

Three separate reasons this module does something else instead.

1. `ffmpeg_microphone_live` shells out to an ffmpeg binary this project does not depend on,
   and its `_get_microphone_name()` helper parses ffmpeg's device list and takes entry [0].
   On this machine [0] is a Voicemeeter loopback, so the "microphone" records the desktop
   output and the wake word never fires, with no error at all, because a silent stream is a
   perfectly valid stream. Gradio asks the BROWSER for a device, so a human picks.

2. That `return item["text"]` reads the loop variable from outside the loop. If the stream
   yields nothing it is an unbound-name error rather than an empty transcript, and if the
   `break` never fires it returns the last PARTIAL hypothesis as though it were final.

3. Stage 3 in the course is HfAgent over the inference API:

       from transformers import HfAgent
       agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/bigcode/starcoder")

   `HfAgent` was removed from transformers entirely - that import is a plain ImportError with
   no deprecation window - and `api-inference.huggingface.co` no longer resolves in DNS, so
   even a raw request to it dies in socket.gaierror before there is an HTTP status to check.
   Two dead things stacked on one line. We run a small seq2seq locally instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252. Both modes print quoted model output, so force UTF-8
# before anything prints. The repo's pure-Gradio demos omit this block and stay on ASCII;
# this module keeps it because its DEFAULT mode is the console one.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import soundfile as sf

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

WAKE_ID = "MIT/ast-finetuned-speech-commands-v2"   # 342 MB, cached since Unit 4
ASR_ID = "openai/whisper-base"                     # 290 MB, cached since Unit 5
LLM_ID = "MBZUAI/LaMini-Flan-T5-248M"              # 990 MB; the Hub's pipeline_tag is WRONG
TTS_ENG = "facebook/mms-tts-eng"                   # 145 MB, cached since Unit 6
DUMMY_ID = "hf-internal-testing/librispeech_asr_dummy"   # ~9 MB, cached since Unit 5

SAMPLING_RATE = 16_000
WAKE_WORD = "marvin"          # AST id2label[27]; google/speech_commands' own index is 26
WAKE_THRESHOLD = 0.5          # the course's prob_threshold
WAKE_WINDOW_S = 1.0           # AST speech-commands was trained on ONE-SECOND single words
NEGATIVES = ("stop", "yes", "house")   # other real speech-commands classes, as controls
SEED = 7                      # VITS has a stochastic duration predictor: seed every call
MAX_NEW_TOKENS = 40
COMMAND_SEARCH = 12           # rows of the dummy considered when picking a "command"

# The language model sees a STRING and nothing else: no audio, no confidence, no speaker.
# Wrapping the transcript in an instruction is the only steering stage 3 ever gets.
PROMPT = "Answer in one short sentence: {command}"

_CACHE: dict[str, object] = {}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_wav(name: str, audio, sampling_rate: int = SAMPLING_RATE) -> None:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:   # VITS occasionally overshoots; players clip float WAVs above 1.0
        audio = audio / peak
    path = OUT_DIR / name
    sf.write(path, audio, sampling_rate)
    print(f"     saved {path.name}  ({len(audio) / sampling_rate:.2f}s @ {sampling_rate} Hz)")


def audio_dict(array) -> dict:
    """A FRESH dict for every pipeline call.

    Both pipelines used here mutate the dict you hand them: they move "array" into "raw" in
    place. Reuse one dict across two calls and the second raises
    `ValueError: ... the dict needs to contain a "raw" key`, which reads as though your
    input were malformed rather than already consumed.
    """
    return {"array": np.asarray(array, dtype=np.float32), "sampling_rate": SAMPLING_RATE}


def wake_clf():
    from transformers import pipeline

    if "wake" not in _CACHE:
        _CACHE["wake"] = pipeline("audio-classification", model=WAKE_ID, device=-1)
    return _CACHE["wake"]


def asr_pipe():
    from transformers import pipeline

    if "asr" not in _CACHE:
        _CACHE["asr"] = pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)
    return _CACHE["asr"]


def llm():
    """LaMini-Flan-T5 with the task passed EXPLICITLY.

    The Hub's pipeline_tag on this checkpoint says "text-generation". It is a T5, a seq2seq,
    so under that task the pipeline ECHOES THE PROMPT BACK VERBATIM and raises nothing.
    transformers does log "T5ForConditionalGeneration is not supported for text-generation",
    at ERROR level, as a single 3,273-character line naming 200+ classes - which is an error
    you scroll past rather than read. Passing "text2text-generation" is the whole fix, and
    stage 3's caller asserts the reply differs from the prompt so that a regression here
    fails the run instead of shipping a parrot.
    """
    from transformers import pipeline

    if "llm" not in _CACHE:
        _CACHE["llm"] = pipeline("text2text-generation", model=LLM_ID, device=-1)
    return _CACHE["llm"]


def speaker(model_id: str = TTS_ENG):
    """MMS/VITS: text straight to waveform, no vocoder. phonemize:false, so no espeak-ng."""
    from transformers import VitsModel, VitsTokenizer

    key = f"tts:{model_id}"
    if key not in _CACHE:
        _CACHE[key] = (VitsTokenizer.from_pretrained(model_id),
                       VitsModel.from_pretrained(model_id))
    return _CACHE[key]


# ---------------------------------------------------------------------------
# The four stages. BOTH modes call exactly these, so the canned run is a real
# rehearsal of the live one rather than a parallel implementation of it.
# ---------------------------------------------------------------------------
def stage_wake(audio, top_k: int = 3):
    """Stage 1. AST over the FIRST SECOND of the audio; returns the pipeline's top-k list.

    The window matters. This checkpoint is fine-tuned on Speech Commands, which is one
    single spoken word per clip at exactly one second. Hand it ten seconds of a sentence and
    the feature extractor pads happily to its 10.24 s window and returns confident nonsense.
    """
    clip = np.asarray(audio, dtype=np.float32)[: int(WAKE_WINDOW_S * SAMPLING_RATE)]
    return wake_clf()(audio_dict(clip), top_k=top_k)


def stage_transcribe(audio) -> str:
    """Stage 2. Greedy whisper-base, language pinned.

    num_beams=1 because the default 5 is about 5x slower on CPU for a barely better
    transcript, and language="english" because letting Whisper auto-detect costs a decode
    pass and adds one more way to be silently wrong.
    """
    return asr_pipe()(
        audio_dict(audio),
        generate_kwargs={"task": "transcribe", "language": "english", "num_beams": 1},
    )["text"].strip()


def stage_answer(command: str) -> tuple[str, str]:
    """Stage 3. Returns (prompt, reply) so the caller can compare the two."""
    prompt = PROMPT.format(command=command.strip())
    reply = llm()(prompt, max_new_tokens=MAX_NEW_TOKENS)[0]["generated_text"].strip()
    return prompt, reply


def stage_speak(text: str, model_id: str = TTS_ENG, seed: int = SEED):
    """Stage 4."""
    import torch

    tok, model = speaker(model_id)
    torch.manual_seed(seed)   # VITS's duration predictor is stochastic
    with torch.no_grad():
        wav = model(**tok(text=text, return_tensors="pt")).waveform
    return np.asarray(wav, dtype=np.float32).squeeze()


# ---------------------------------------------------------------------------
# CANNED
# ---------------------------------------------------------------------------
def pick_command_row(ds) -> tuple[int, str]:
    """Choose the clip that stands in for a spoken command, by a RULE rather than an index.

    Prefer a row whose reference ends in a question mark; failing that take the shortest
    reference in the first COMMAND_SEARCH rows, because a short utterance is the plausible
    one. Reading only the "text" column avoids decoding twelve clips of audio to pick one. A
    hard-coded index would work today and repoint silently if the dataset were ever revised.
    """
    texts = ds["text"][:COMMAND_SEARCH]
    for i, text in enumerate(texts):
        if text.strip().endswith("?"):
            return i, "first reference ending in a question mark"
    best = min(range(len(texts)), key=lambda i: len(texts[i]))
    return best, f"shortest reference in the first {len(texts)} rows"


def run_canned() -> None:
    from datasets import load_dataset

    timings = {}

    # Load all four models BEFORE the clock starts. Without this, stage 1's number is
    # mostly the 342 MB AST checkpoint arriving from disk and stage 3's is mostly LaMini's
    # 990 MB, so the "share" column below would rank import cost and call it latency. A
    # deployed assistant loads once and answers many times, so inference is the honest
    # thing to measure. The load itself is reported separately.
    print("\n   Loading four models (~1.8 GB, all cached by the walkthrough) before timing,")
    print("   so the per-stage numbers below are inference and not import...")
    t0 = time.perf_counter()
    wake_clf(), asr_pipe(), llm(), speaker()
    print(f"   loaded in {time.perf_counter() - t0:.1f}s")

    banner("Stage 1/4  wake word  (synthesised: there is no microphone in this mode)")
    print("\n   The course listens to a live mic. We synthesise the word with mms-tts-eng and")
    print("   classify that, which makes stage 1 reproducible: the same input every run, so a")
    print("   score that moves is a regression rather than a noisy room.\n")

    t0 = time.perf_counter()
    wake_audio = stage_speak(WAKE_WORD)
    top = stage_wake(wake_audio)
    timings["1 wake"] = time.perf_counter() - t0
    label, score = top[0]["label"], top[0]["score"]
    print(f"     {'WAKE' if label == WAKE_WORD else '    '} {WAKE_WORD:<8} -> "
          f"{label:<12} {score:.3f}   ({len(wake_audio) / SAMPLING_RATE:.2f}s)")

    correct = false_wakes = 0
    for word in NEGATIVES:      # controls: a gate has to reject as well as accept
        other = stage_wake(stage_speak(word))[0]
        correct += other["label"] == word
        false_wakes += other["label"] == WAKE_WORD and other["score"] > WAKE_THRESHOLD
        print(f"     {'WAKE' if other['label'] == WAKE_WORD else '    '} {word:<8} -> "
              f"{other['label']:<12} {other['score']:.3f}")

    # Read that block honestly rather than reporting only the number that flatters it. The
    # three controls are real speech-commands classes and the classifier gets most of them
    # WRONG on synthesised audio - it just gets them wrong in the safe direction, as some
    # other command rather than as the wake word. So the score above is not evidence that
    # AST handles synthetic speech well. It is evidence that this one word lands, which is
    # all a wake word has to do, and it is why this mode gates while --live does not.
    print(f"\n     controls classified correctly : {correct} of {len(NEGATIVES)}")
    print(f"     controls that falsely woke it : {false_wakes} of {len(NEGATIVES)}")
    print("\n     The first number is poor and the second is what a gate is judged on. AST is")
    print("     unreliable on synthesised speech in general; it is reliable about this one")
    print("     word, and a wake word only has to be right about its own word.")

    cfg = wake_clf().model.config
    print(f"\n     The AST label set has {len(cfg.id2label)} classes and '{WAKE_WORD}' is id "
          f"{cfg.label2id[WAKE_WORD]}.")
    print("     google/speech_commands puts it at 26 of 36. The two orders disagree, so")
    print("     compare label STRINGS and never ints.")

    # A HARD GATE, and only because this mode controls its own input: the clip is
    # synthesised from a fixed string with a fixed seed. A top-1 that is not the wake word
    # means the TTS, the AST label set or the sampling rate moved under us, which is exactly
    # what a regression test exists to refuse to paper over. Live mode does NOT gate.
    assert label == WAKE_WORD and score > WAKE_THRESHOLD, (
        f"the wake word did not fire on a synthesised {WAKE_WORD!r}: got {label!r} at "
        f"{score:.3f}. Check that mms-tts-eng still emits 16 kHz and that the AST label set "
        f"still contains {WAKE_WORD!r}."
    )
    save_wav("assistant_wake.wav", wake_audio)

    banner("Stage 2/4  transcribe  (the cached LibriSpeech dummy stands in for the mic)")
    ds = load_dataset(DUMMY_ID, "clean", split="validation")
    row, why = pick_command_row(ds)
    clip = ds[row]
    reference = clip["text"].lower()
    secs = len(clip["audio"]["array"]) / SAMPLING_RATE
    print(f"\n   row {row} of {len(ds)} ({why}), {secs:.2f}s")
    print("\n   This dataset is read-aloud audiobook prose, so strictly speaking no row in it")
    print("   is a command. That is worth leaving visible rather than hiding behind a tidier")
    print("   demo: stage 3 receives a STRING and has no way to know it was never addressed")
    print("   to anyone. An odd answer below is the pipeline working exactly as specified.\n")

    t0 = time.perf_counter()
    command = stage_transcribe(clip["audio"]["array"])
    timings["2 asr"] = time.perf_counter() - t0
    print(f"     reference : {reference}")
    print(f"     heard     : {command.lower().rstrip('.')}")
    assert command, "stage 2 returned an empty transcript"
    save_wav("assistant_command.wav", clip["audio"]["array"])

    banner("Stage 3/4  the language model  (locally, because the course's endpoint is gone)")
    t0 = time.perf_counter()
    prompt, reply = stage_answer(command)
    timings["3 llm"] = time.perf_counter() - t0
    print(f"\n     prompt : {prompt}")
    print(f"     reply  : {reply}")

    # THE ASSERTION THIS MODULE EXISTS FOR. pipeline(model=LLM_ID) with no task, or with the
    # Hub's own "text-generation" tag, returns the prompt unchanged and raises nothing. An
    # assistant that repeats you is indistinguishable from one that is thinking until you
    # read the output, so read it here, on every run.
    assert reply and reply != prompt, (
        "stage 3 echoed the prompt back verbatim. That is what LaMini-Flan-T5 does under "
        "task='text-generation': it is a T5 seq2seq and the Hub's pipeline_tag is wrong. "
        "Check that llm() still passes 'text2text-generation' explicitly."
    )
    print("\n     reply differs from the prompt: True   (asserted, not hoped for)")

    banner("Stage 4/4  speak the answer")
    t0 = time.perf_counter()
    audio = stage_speak(reply)
    timings["4 tts"] = time.perf_counter() - t0
    print()
    save_wav("assistant_reply.wav", audio)

    banner("Four stages, four timings")
    total = sum(timings.values())
    slowest = max(timings, key=timings.get)
    print(f"\n     {'stage':<10}{'seconds':>9}{'share':>8}")
    for name, secs in timings.items():
        print(f"     {name:<10}{secs:9.2f}{secs / total:8.0%}")
    print(f"     {'total':<10}{total:9.2f}")
    share = timings[slowest] / total
    print(f"\n   slowest stage: {slowest} at {timings[slowest]:.2f}s ({share:.0%})")
    print("\n   Nothing here overlaps. Stage 2 cannot start until stage 1 decides, and stage 3")
    print("   cannot start until stage 2 has emitted a COMPLETE string, because a partial")
    print("   transcript is not a prompt. So the total is the SUM, not the maximum.")
    if share < 0.5:
        print(f"\n   And no stage dominates: the biggest is {share:.0%} of the wall clock, so")
        print("   there is no single thing to optimise here. Halving even the slowest stage")
        print(f"   would save {timings[slowest] / 2 / total:.0%} of the total and no more.")
        print("   That is the cascade's real cost: four models, four bills, no hot spot.")
    else:
        print(f"\n   One stage owns the experience here, at {share:.0%} of the wall clock,")
        print(f"   so {slowest} is the only thing worth optimising.")
    print("\n   Run --live for the same four numbers on your own voice, one box per stage.")


# ---------------------------------------------------------------------------
# LIVE
# ---------------------------------------------------------------------------
def run_live() -> None:
    try:
        import gradio as gr
    except ImportError:
        sys.exit("--live needs gradio, a core dependency of this project. Run: uv sync")

    import librosa

    # Plain ASCII from here down, matching the repo's other Gradio modules.
    print("Loading four models (~1.8 GB, all cached by the walkthrough)...")
    wake_clf(), asr_pipe(), llm(), speaker()
    print("Ready.")

    def assist(filepath):
        if filepath is None:
            return None, "Record something first.", "", None

        # librosa decodes straight to 16 kHz mono, which avoids two traps at once: the
        # pipeline never shells out to ffmpeg, and it never reaches transformers'
        # ImportError("torchaudio is required to resample") - torchaudio is deliberately
        # not installed in this repo.
        audio, _ = librosa.load(filepath, sr=SAMPLING_RATE, mono=True)
        if audio.size < SAMPLING_RATE // 4:
            return None, "That recording is too short to transcribe.", "", None

        t0 = time.perf_counter()
        top = stage_wake(audio)
        t_wake = time.perf_counter() - t0
        scores = {p["label"]: float(p["score"]) for p in top}

        t0 = time.perf_counter()
        command = stage_transcribe(audio)
        t_asr = time.perf_counter() - t0
        heard = f"[{t_asr:.2f}s]  {command or '(empty - stage 2 returned nothing)'}"
        if not command:
            return scores, heard, "", None

        t0 = time.perf_counter()
        prompt, reply = stage_answer(command)
        t_llm = time.perf_counter() - t0
        answer = f"[{t_llm:.2f}s]  {reply}"
        if reply == prompt:
            # Not an assert. A live server should not die on one bad request; it should say
            # WHICH stage broke, which is the entire reason there are four output boxes.
            answer += ("\n\nSTAGE 3 ECHOED THE PROMPT. That is the text-generation bug: this "
                       "checkpoint is a T5 and the Hub's pipeline_tag is wrong.")

        out = stage_speak(reply)
        # gradio wants (sampling_rate, array) in THAT order - returning it the other way
        # round plays as a fraction of a second of noise.
        return (scores | {f"(wake scored in {t_wake:.2f}s)": 0.0},
                heard, answer, (SAMPLING_RATE, out))

    demo = gr.Interface(
        fn=assist,
        inputs=gr.Audio(sources="microphone", type="filepath",
                        label=f'Push to talk: say "{WAKE_WORD}", then ask something'),
        # FOUR outputs, one per stage. A single audio box would make every failure look like
        # a TTS failure, which is the exact mistake section 3 of the walkthrough is about:
        # attribute the damage to the seam that caused it.
        outputs=[
            gr.Label(num_top_classes=3, label=f"1. Wake word (first {WAKE_WINDOW_S:.0f}s)"),
            gr.Textbox(label="2. Whisper heard", lines=3),
            gr.Textbox(label="3. The model replied", lines=5),
            gr.Audio(label="4. Spoken reply", type="numpy"),
        ],
        flagging_mode="never",   # gradio 6: renamed from allow_flagging="never"
        title="Unit 7 voice assistant: four stages, four outputs",
        description=(
            "Push to talk, then release. The wake word is REPORTED here, not enforced. This "
            "classifier was fine-tuned on one-second single-word clips, so scoring a whole "
            "spoken sentence against it rejects almost everything; the course can gate on it "
            "only because its microphone is a continuous 2-second stream it chunks, and a "
            "push-to-talk button has no such stream. Watch which box goes wrong."
        ),
    )
    demo.launch(share=False)


def parse_args(argv=None) -> argparse.Namespace:
    # argparse validates what it is GIVEN, never its own default, so an env var that is not
    # a real mode would sail through choices= and be discovered four model loads later.
    env = os.environ.get("UNIT7_MODE", "canned").strip().lower()
    if env not in ("canned", "live"):
        sys.exit(f"UNIT7_MODE={env!r} is not one of: canned, live")

    parser = argparse.ArgumentParser(
        prog="assistant.py",
        description="Unit 7's four-stage voice assistant: wake word, transcribe, answer, speak.",
        epilog="The env var UNIT7_MODE sets the default mode; an explicit flag beats it.",
    )
    parser.add_argument(
        "--mode", choices=("canned", "live"), default=env,
        help=f"canned runs off cached audio with no microphone; live opens a Gradio "
             f"push-to-talk page (default: {env})",
    )
    parser.add_argument(
        "--live", action="store_const", const="live", dest="mode",
        help="shorthand for --mode live",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(f"Unit 7 voice assistant | mode={args.mode} | out={OUT_DIR}")
    if args.mode == "live":
        run_live()
    else:
        run_canned()


if __name__ == "__main__":
    main()
