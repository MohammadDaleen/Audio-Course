"""
Unit 6 demo: type a sentence, pick a voice, hear it - on CPU.

Three tabs, matching the course's Unit 6 pages: synthesise with SpeechT5 and a chosen
x-vector, compare SpeechT5 against MMS/VITS on the same sentence, and inspect what the
tokenizer will actually do to your text before you generate it.

    uv run python units/unit6_text_to_speech/gradio_demo.py
    # then open http://127.0.0.1:7860

Models: microsoft/speecht5_tts (~585 MB) + microsoft/speecht5_hifigan (~51 MB) +
facebook/mms-tts-eng (~145 MB), all cached by the walkthrough. Nothing is uploaded
(share=False). To use your own fine-tuned model from the hands-on, change TTS_ID to
"<your-username>/speecht5_finetuned_english_dialects" - the vocoder and x-vectors stay
exactly the same, because fine-tuning only changes the acoustic model.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    VitsModel,
    VitsTokenizer,
)

TTS_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
MMS_ID = "facebook/mms-tts-eng"
XVECTOR_ID = "Matthijs/cmu-arctic-xvectors"
XVECTOR_SPLIT = "validation"
SAMPLING_RATE = 16_000
VOICES = ["slt", "clb", "bdl", "rms", "ksp"]

# Plain ASCII: this module has no sys.stdout.reconfigure block, so a "..." would
# mojibake on a cp1252 Windows console.
print(f"Loading {TTS_ID} + vocoder (~636 MB on first run)...")
processor = SpeechT5Processor.from_pretrained(TTS_ID)
model = SpeechT5ForTextToSpeech.from_pretrained(TTS_ID)
vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID)
model.eval()
torch.set_grad_enabled(False)

print(f"Loading {XVECTOR_ID} (~18 MB)...")
_xv = load_dataset(XVECTOR_ID, split=XVECTOR_SPLIT)
_filenames = _xv["filename"]

# Match by filename prefix rather than a hard-coded row index: an index would silently
# repoint to a different speaker if the dataset were ever revised.
SPEAKERS = {}
for _voice in VOICES:
    _prefix = f"cmu_us_{_voice}_"
    _idx = next(i for i, f in enumerate(_filenames) if f.startswith(_prefix))
    SPEAKERS[_voice] = torch.tensor(_xv[_idx]["xvector"]).unsqueeze(0)

_mms = {"tok": None, "model": None}   # loaded lazily; tab 2 is the only user


def _synthesise(text: str, voice: str, seed: int = 6):
    """SpeechT5 -> log-mel -> HiFi-GAN -> waveform, seeded."""
    ids = processor(text=text, return_tensors="pt")["input_ids"]
    torch.manual_seed(seed)   # the decoder pre-net applies dropout even in eval()
    speech = model.generate_speech(ids, SPEAKERS[voice], vocoder=vocoder)
    return np.asarray(speech, dtype=np.float32)


def speak(text, voice):
    if not text or not text.strip():
        return None, "Type something first."
    audio = _synthesise(text.strip(), voice)
    note = f"{len(audio) / SAMPLING_RATE:.2f}s as cmu_us_{voice}"
    # gradio wants (sampling_rate, array) in THAT order - the most common mistake here
    # is returning (array, sampling_rate), which plays as a fraction of a second of noise.
    return (SAMPLING_RATE, audio), note


def compare(text):
    if not text or not text.strip():
        return None, None, "Type something first."
    text = text.strip()
    t5 = _synthesise(text, VOICES[0])

    if _mms["model"] is None:
        _mms["tok"] = VitsTokenizer.from_pretrained(MMS_ID)
        _mms["model"] = VitsModel.from_pretrained(MMS_ID)
    mms_out = _mms["model"](**_mms["tok"](text=text, return_tensors="pt")).waveform
    mms_out = np.asarray(mms_out, dtype=np.float32).squeeze()

    note = (
        f"SpeechT5: {len(t5) / SAMPLING_RATE:.2f}s, spectrogram + a separate vocoder, "
        f"voice chosen by x-vector.\n"
        f"MMS/VITS: {len(mms_out) / _mms['model'].config.sampling_rate:.2f}s, waveform "
        f"straight out, no vocoder and no way to pick a voice."
    )
    return (SAMPLING_RATE, t5), (_mms["model"].config.sampling_rate, mms_out), note


def inspect(text):
    """What the 81-token character vocabulary will do to this text, before you generate."""
    if not text or not text.strip():
        return "Type something first."
    tok = processor.tokenizer
    ids = tok(text, add_special_tokens=False).input_ids
    pieces = tok.convert_ids_to_tokens(ids)
    n_unk = pieces.count("<unk>")
    bad = sorted({c for c in set(text)
                  if tok.unk_token_id in tok(c, add_special_tokens=False).input_ids})
    lines = [
        f"tokens ({len(ids)}): {' '.join(pieces)}",
        "",
        f"<unk> count: {n_unk}",
        f"characters the tokenizer cannot represent: {''.join(bad)!r}" if bad
        else "every character is in the vocabulary",
    ]
    if n_unk:
        lines += [
            "",
            "Those characters become <unk> SILENTLY - the model will still generate",
            "confident-sounding audio for them. Digits are the usual culprit; either spell",
            "them out or load the tokenizer with normalize=True.",
        ]
    return "\n".join(lines)


# Each Interface is built at module level, OUTSIDE any `with` block. Wrapping these in
# `with gr.Blocks() as demo:` splices the children into the parent on gradio 6 and renders
# every tab twice. TabbedInterface is already a Blocks.
speak_tab = gr.Interface(
    fn=speak,
    inputs=[
        gr.Textbox(label="Text", value="the sun provides energy for life on earth", lines=2),
        gr.Dropdown(choices=VOICES, value=VOICES[0], label="CMU ARCTIC voice"),
    ],
    outputs=[gr.Audio(label="Speech", type="numpy"), gr.Textbox(label="Notes", lines=2)],
    flagging_mode="never",  # gradio 6: renamed from allow_flagging="never"
)

compare_tab = gr.Interface(
    fn=compare,
    inputs=gr.Textbox(label="Text", value="the sun provides energy for life on earth", lines=2),
    outputs=[
        gr.Audio(label="SpeechT5 + HiFi-GAN", type="numpy"),
        gr.Audio(label="MMS / VITS", type="numpy"),
        gr.Textbox(label="Notes", lines=4),
    ],
    flagging_mode="never",
)

inspect_tab = gr.Interface(
    fn=inspect,
    inputs=gr.Textbox(label="Text", value="In 1984 he paid 25 pounds", lines=2),
    outputs=gr.Textbox(label="What the tokenizer sees", lines=10),
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [speak_tab, compare_tab, inspect_tab],
    ["🗣️ Speak", "⚖️ Compare engines", "🔤 Inspect the text"],
    title="🔊 Text to speech with SpeechT5",
)

if __name__ == "__main__":
    demo.launch(share=False)


# gradio 3 (what the course was written against) → gradio 6 changes used above:
#   gr.Interface(allow_flagging=…)         → gr.Interface(flagging_mode=…)
#   with gr.Blocks(): TabbedInterface(...) → TabbedInterface used directly
#   gr.Audio outputs return (sampling_rate, array), not (array, sampling_rate)
#   demo.launch(debug=True) is a Colab/notebook idiom; locally share=False is enough.
