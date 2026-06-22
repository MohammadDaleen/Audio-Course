"""
Optional Unit 2 demo: pick a random MINDS-14 clip and see what the pipelines predict.

Loads the audio-classification and ASR pipelines once, then for a random clip shows
the predicted intent (with the true label) and the ASR transcription, next to the audio.

    uv run python units/unit2_audio_applications/gradio_demo.py

Then open http://127.0.0.1:7860. First run downloads the classifier (~1.2 GB) and the
ASR model (~360 MB). CPU only; nothing is uploaded (share=False).
"""

from __future__ import annotations

import random

import gradio as gr
from datasets import Audio, load_dataset
from transformers import pipeline


def load_minds14(name: str = "en-AU", split: str = "train"):
    kwargs = dict(path="PolyAI/minds14", name=name, split=split)
    try:
        return load_dataset(**kwargs)
    except Exception:
        return load_dataset(**kwargs, trust_remote_code=True)


print("Loading MINDS-14 + pipelines (first run downloads ~1.5 GB of models)…")
minds = load_minds14().cast_column("audio", Audio(sampling_rate=16_000))
id2label = minds.features["intent_class"].int2str
classifier = pipeline("audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")
asr = pipeline("automatic-speech-recognition")


def analyze_random():
    idx = random.randrange(minds.num_rows)
    example = minds[idx]
    array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    top = classifier(array, top_k=3)
    transcription = asr(array)["text"]
    true_label = id2label(example["intent_class"])

    intent_md = "\n".join(f"- **{p['label']}** - {p['score']:.2f}" for p in top)
    caption = (
        f"### Predicted intent\n{intent_md}\n\n"
        f"(true intent: `{true_label}`)\n\n"
        f"### Transcription\n{transcription}"
    )
    return (sr, array), caption


with gr.Blocks(title="MINDS-14 classification + ASR") as demo:
    gr.Markdown(
        "# 🎯 Unit 2 demo - classification + ASR\n"
        "A random `PolyAI/minds14` (en-AU) clip, run through the audio-classification and "
        "ASR pipelines. Click the button to try another."
    )
    audio_out = gr.Audio(label="Audio clip")
    info_out = gr.Markdown()
    gr.Button("🎲 Analyze a random clip").click(fn=analyze_random, outputs=[audio_out, info_out])
    demo.load(fn=analyze_random, outputs=[audio_out, info_out])


if __name__ == "__main__":
    demo.launch(share=False)
