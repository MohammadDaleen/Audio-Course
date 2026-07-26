"""
Unit 4 demo: classify the genre of a music clip on CPU.

Uses the course's published fine-tuned model, so it works without any training.

    uv run python units/unit4_music_genre_classifier/gradio_demo.py
    # then open http://127.0.0.1:7860

Model: sanchit-gandhi/distilhubert-finetuned-gtzan (~90 MB, fast on CPU). Nothing is
uploaded (share=False).
"""

from __future__ import annotations

import gradio as gr
from transformers import pipeline

MODEL_ID = "sanchit-gandhi/distilhubert-finetuned-gtzan"
print(f"Loading {MODEL_ID} (~90 MB on first run)…")
pipe = pipeline("audio-classification", model=MODEL_ID)


def classify(filepath):
    if filepath is None:
        return {}
    # The pipeline reads the file and resamples to 16 kHz internally.
    preds = pipe(filepath, top_k=10)
    return {p["label"]: float(p["score"]) for p in preds}  # gr.Label wants {label: prob}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(type="filepath", sources=["upload", "microphone"], label="Music clip"),
    outputs=gr.Label(num_top_classes=5, label="Predicted genre"),
    title="🎵 Music genre classifier (DistilHuBERT fine-tuned on GTZAN)",
    description=(
        "Upload or record a short music clip; the model predicts one of 10 GTZAN genres "
        "(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)."
    ),
)

if __name__ == "__main__":
    demo.launch(share=False)
