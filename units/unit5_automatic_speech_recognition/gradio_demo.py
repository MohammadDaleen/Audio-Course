"""
Unit 5 demo: transcribe speech from your microphone or a file, on CPU.

Three tabs, matching the course's "Building a demo" page plus the translation task:
microphone transcription, file transcription (with optional timestamps), and
speech-to-English translation.

    uv run python units/unit5_automatic_speech_recognition/gradio_demo.py
    # then open http://127.0.0.1:7860

Model: openai/whisper-base (~290 MB, already cached from the walkthrough). Nothing is
uploaded (share=False). To use your own fine-tuned model from the hands-on, change
MODEL_ID to "<your-username>/whisper-tiny-finetuned-minds14-en".
"""

from __future__ import annotations

import gradio as gr
import librosa
import torch
from transformers import pipeline

MODEL_ID = "openai/whisper-base"
DEVICE = 0 if torch.cuda.is_available() else -1

# Plain ASCII: this module has no sys.stdout.reconfigure block, so a "…" would
# mojibake on a cp1252 Windows console.
print(f"Loading {MODEL_ID} (~290 MB on first run)...")
pipe = pipeline("automatic-speech-recognition", model=MODEL_ID, device=DEVICE)


def _run(filepath, task="transcribe", timestamps=False):
    if filepath is None:
        return "Record or upload some audio first."

    # Decode with librosa straight to 16 kHz mono. Handing the pipeline a dict of
    # {array, sampling_rate} avoids two traps: it never shells out to ffmpeg, and it
    # never hits transformers' ImportError("torchaudio is required to resample").
    array, _ = librosa.load(filepath, sr=16_000, mono=True)

    out = pipe(
        {"array": array, "sampling_rate": 16_000},
        # chunk_length_s slices long audio into 30 s windows with overlap, so files
        # longer than Whisper's fixed 30 s encoder window still work.
        chunk_length_s=30,
        batch_size=8,
        ignore_warning=True,
        return_timestamps=timestamps,
        generate_kwargs={"task": task, "language": "english", "num_beams": 1},
    )

    if timestamps and out.get("chunks"):
        lines = []
        for chunk in out["chunks"]:
            start, end = chunk["timestamp"]
            end = f"{end:6.2f}" if end is not None else "  ... "
            lines.append(f"[{start:6.2f} → {end}]  {chunk['text'].strip()}")
        return "\n".join(lines)
    return out["text"].strip()


def transcribe_mic(filepath):
    return _run(filepath)


def transcribe_file(filepath, timestamps):
    return _run(filepath, timestamps=timestamps)


def translate_file(filepath):
    return _run(filepath, task="translate")


# Each Interface is built at module level, OUTSIDE any `with` block. The course wraps
# these in `with gr.Blocks() as demo:`, which on gradio 6 splices the child components
# into the parent and renders every tab twice. TabbedInterface is already a Blocks.
mic_transcribe = gr.Interface(
    fn=transcribe_mic,
    inputs=gr.Audio(sources="microphone", type="filepath", label="Speak"),
    outputs=gr.Textbox(label="Transcription", lines=6),
    flagging_mode="never",  # gradio 6: renamed from allow_flagging="never"
)

file_transcribe = gr.Interface(
    fn=transcribe_file,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file"),
        gr.Checkbox(value=False, label="Show timestamps"),
    ],
    outputs=gr.Textbox(label="Transcription", lines=12),
    flagging_mode="never",
)

file_translate = gr.Interface(
    fn=translate_file,
    inputs=gr.Audio(sources="upload", type="filepath", label="Audio file (any language)"),
    outputs=gr.Textbox(label="English translation", lines=8),
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [mic_transcribe, file_transcribe, file_translate],
    ["🎙️ Microphone", "📁 File + timestamps", "🌍 Translate to English"],
    title="🗣️ Speech recognition with Whisper",
)

if __name__ == "__main__":
    demo.launch(share=False)

# gradio 3 (what the course was written against) → gradio 6 changes used above:
#   gr.Audio(source=…)             → gr.Audio(sources=…)
#   gr.Interface(allow_flagging=…) → gr.Interface(flagging_mode=…)
#   with gr.Blocks(): TabbedInterface(...)  → TabbedInterface used directly
#   demo.launch(debug=True) is a Colab/notebook idiom; locally share=False is enough.
