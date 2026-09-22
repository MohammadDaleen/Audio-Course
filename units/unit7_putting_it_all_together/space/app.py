"""
Unit 7 hands-on: the speech-to-speech translation Space. Audio in, ARABIC audio out.

This is the file you upload to a Hugging Face Space. It is deliberately the smallest thing
that satisfies the grader, which does exactly this and nothing more:

    client = Client(repo_id)
    audio_file = client.predict("test_short.wav", api_name="/predict")
    # then facebook/mms-lid-126 on the returned audio; you fail if the top label is
    # 'eng' or if its score is below 0.5

EXACTLY ONE INPUT, EXACTLY ONE OUTPUT. A language dropdown or a second output changes the
endpoint's signature, and the grader then dies with a gradio_client traceback rather than a
grading message. The dropdown lives in ../gradio_demo.py, which is local and has no contract
to honour. Note also that the check is only "not English, confidently": it never verifies
that you reached the language you claimed.

What changed from the course's template (course-demos/speech-to-speech-translation):
  - deleted `import spaces` and `@spaces.GPU`, which are ZeroGPU-only and stop a free CPU
    Space from starting at all
  - deleted the zipfile x-vector block: SpeechT5 is gone, so there is nothing to condition
  - SpeechT5 + HiFi-GAN  ->  Helsinki-NLP/opus-mt-en-ar + facebook/mms-tts-ara
  - translate() gained the MT hop. Whisper's task="translate" only ever emits ENGLISH, and
    English is the one output that cannot pass this hands-on
  - collapsed the template's TabbedInterface to ONE Interface (see the note on predict())
  - np.clip BEFORE the int16 cast (see to_int16)
  - cache_examples=False, and no example .wav is shipped

Arabic rather than French, because "confidently not English" is the whole test and a
different script is the least ambiguous way to pass it. It is also the sharper illustration:
facebook/mms-tts-ara has 38 pronounceable symbols, 35 of them bare letters, no digits and no
punctuation at all, so every comma and full stop in the translation is silently DELETED
before the model sees it. The Space passes anyway. See section 9 of ../walkthrough.py.

Models, all loaded at import (~740 MB on a Space's first boot, then cached in the image
layer): openai/whisper-base (~290 MB), Helsinki-NLP/opus-mt-en-ar (~307 MB),
facebook/mms-tts-ara (~145 MB). Loading them lazily would push that download into the
grader's first call and time it out.
"""

from __future__ import annotations

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer, VitsModel, VitsTokenizer, pipeline

ASR_ID = "openai/whisper-base"
MT_ID = "Helsinki-NLP/opus-mt-en-ar"
TTS_ID = "facebook/mms-tts-ara"
SAMPLING_RATE = 16_000   # whisper's input rate AND mms-tts-ara's config.sampling_rate
SEED = 7                 # VITS has a stochastic duration predictor: seed every call

# Plain ASCII: a Space's build log and a cp1252 Windows console both mangle the rest.
print("Loading whisper-base + opus-mt-en-ar + mms-tts-ara (~740 MB on first run)...")
asr = pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)
mt_tokenizer = MarianTokenizer.from_pretrained(MT_ID)
mt_model = MarianMTModel.from_pretrained(MT_ID)
tts_tokenizer = VitsTokenizer.from_pretrained(TTS_ID)
tts_model = VitsModel.from_pretrained(TTS_ID)
torch.set_grad_enabled(False)


def translate(filepath: str) -> str:
    """Audio in any language -> Arabic text, in two hops.

    Hop 1 is Whisper with task="translate", the template's own call, kept here for the
    OPPOSITE of the template's reason. The template ended there and shipped English, which
    fails. Here it is a normaliser: it turns any input language into English, which is
    exactly what an en->ar model was trained to read. Hop 2 is the Marian model, and it is
    the only reason this Space's output is not English.

    librosa decodes to 16 kHz mono so the pipeline never shells out to ffmpeg and never
    raises transformers' ImportError("torchaudio is required to resample").
    """
    array, _ = librosa.load(filepath, sr=SAMPLING_RATE, mono=True)
    english = asr(
        # A FRESH dict every call: the ASR pipeline moves "array" into "raw" IN PLACE, so a
        # reused dict makes the next call raise 'the dict needs to contain a "raw" key',
        # which reads like a malformed input rather than an already-consumed one.
        {"array": array, "sampling_rate": SAMPLING_RATE},
        chunk_length_s=30,      # so a clip past Whisper's 30 s encoder window still works
        batch_size=8,
        ignore_warning=True,
        generate_kwargs={"task": "translate", "num_beams": 1},
    )["text"].strip()
    tokens = mt_model.generate(**mt_tokenizer(english, return_tensors="pt"),
                               num_beams=1, max_new_tokens=256)
    return mt_tokenizer.decode(tokens[0], skip_special_tokens=True)


def synthesise(text: str) -> np.ndarray:
    """Arabic text -> waveform.

    MMS/VITS goes straight to audio with no vocoder, and its tokenizer config ships
    phonemize:false, so there is no espeak-ng to apt-install on the Space. (VitsTokenizer's
    own default is phonemize=True, and kakao-enterprise/vits-ljs ships that and raises.)

    It also ships normalize:true, which DELETES every character outside its 38-symbol
    vocabulary before tokenizing - not <unk>, deleted, with nothing logged. Arabic has no
    punctuation in that vocabulary at all, so the commas and full stops above never reach
    the model. Nothing here can fix that; it is documented so it is not mistaken for fluency.
    """
    torch.manual_seed(SEED)
    waveform = tts_model(**tts_tokenizer(text=text, return_tensors="pt")).waveform
    return np.asarray(waveform, dtype=np.float32).squeeze()


def to_int16(waveform: np.ndarray) -> np.ndarray:
    """CLIP FIRST, then cast.

    The template's (x * 32767).astype(np.int16) wraps the SIGN on anything above 1.0:
        1.05 -> -31131      1.5 -> -16386      -1.2 -> +26216
    mms-tts peaks below 1.0 today, so the bug is silent, which is the only reason it has
    survived this long. A louder checkpoint would hand the grader a burst of sign-flipped
    noise and no exception. This is a named function rather than an inline expression so
    that a test can call it on exactly that vector.
    """
    return (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)


def predict(filepath):
    """THE graded endpoint. Named `predict` on purpose, and that name is load-bearing.

    gradio <= 5 always exposed an Interface's function as "/predict". gradio 6 changed the
    default: gradio/blocks.py now says "If api_name is None or empty string, use the
    function name". Calling this function `predict` yields "/predict" under BOTH, so the
    grader's api_name="/predict" keeps working whichever gradio the Space resolves, and
    renaming this function is a silent way to fail the hands-on.
    """
    if filepath is None:
        return None
    return SAMPLING_RATE, to_int16(synthesise(translate(filepath)))


# ONE Interface, ONE endpoint. The template puts two Interfaces inside a TabbedInterface,
# which registers "/predict" and "/predict_1" (gradio appends a unique suffix to a duplicate
# api_name), so which one the grader reaches depends on tab order. A single Audio component
# with both sources gives the same two ways in and only one endpoint to get wrong.
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath",
                    label="Speech (any language)"),
    outputs=gr.Audio(label="Arabic speech", type="numpy"),
    title="Speech-to-speech translation: anything to Arabic",
    description=(
        "Whisper transcribes and translates to English, Helsinki-NLP/opus-mt-en-ar turns "
        "that into Arabic, and facebook/mms-tts-ara speaks it. Three models, ~740 MB, CPU."
    ),
    flagging_mode="never",   # gradio 5 renamed allow_flagging to flagging_mode
    # No example .wav is committed, so there is nothing to cache and this flag is inert
    # TODAY. It is here as a guard for the next edit: gradio's docs say this parameter
    # defaults to True in HuggingFace Spaces, so the moment anyone adds examples=[[...]] the
    # Space runs the whole cascade at startup. If that raises - a missing file, a Hub hiccup
    # - the Space never reaches RUNNING, and a Space that is not running fails the grader
    # with a connection error rather than with a score.
    cache_examples=False,
)

if __name__ == "__main__":
    # Spaces runs `python app.py`, so this guard fires there and launch() is reached. It is
    # also what lets a test import this file and call predict() directly without starting a
    # server. Do not add share=True: a Space is already public.
    demo.launch()
