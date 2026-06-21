"""
Optional listening demo for Unit 1 (mirrors the course's Gradio example).

Launches a small *local* web app that plays a random MINDS-14 clip together with
its intent label and transcription. Nothing is uploaded anywhere (share=False).

Run with:

    uv run python gradio_demo.py

Then open http://127.0.0.1:7860 in your browser. (Windows may show a one-time
firewall prompt the first time a local server opens a port.)
"""

from __future__ import annotations

import random

import gradio as gr
from datasets import load_dataset


def load_minds14():
    """Load MINDS-14 robustly whether it resolves as parquet or a script."""
    kwargs = dict(path="PolyAI/minds14", name="en-AU", split="train")
    try:
        return load_dataset(**kwargs)
    except Exception:
        return load_dataset(**kwargs, trust_remote_code=True)


print("Loading MINDS-14 (en-AU)… first run downloads the data.")
minds = load_minds14()
id2label = minds.features["intent_class"].int2str


def random_example():
    """Return ((sampling_rate, samples), caption) for a random clip."""
    idx = random.randrange(minds.num_rows)
    example = minds[idx]
    audio = example["audio"]
    label = id2label(example["intent_class"])
    transcription = (
        example.get("english_transcription")
        or example.get("transcription")
        or ""
    )
    caption = f"Intent: {label}\n\n“{transcription}”"
    return (audio["sampling_rate"], audio["array"]), caption


with gr.Blocks(title="MINDS-14 listening demo") as demo:
    gr.Markdown(
        "# 🎧 MINDS-14 listening demo\n"
        "A clip from the `PolyAI/minds14` (en-AU) dataset used in Unit 1. "
        "Click the button to hear a random example and its intent label."
    )
    audio_out = gr.Audio(label="Audio clip")
    caption_out = gr.Textbox(label="Intent & transcription", lines=3)
    play_btn = gr.Button("🎲 Play a random clip")

    play_btn.click(fn=random_example, outputs=[audio_out, caption_out])
    demo.load(fn=random_example, outputs=[audio_out, caption_out])


if __name__ == "__main__":
    demo.launch(share=False)
