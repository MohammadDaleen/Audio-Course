"""
Optional Unit 3 demo: one audio clip through all three architecture families.

Runs a chosen clip through Wav2Vec2 (CTC), Whisper-tiny (seq2seq), and the Audio
Spectrogram Transformer (classification), and shows how differently each one works.

    uv run python units/unit3_transformer_architectures/gradio_demo.py

Then open http://127.0.0.1:7860. First launch downloads ~1.3 GB of models (whisper-small
+ AST; wav2vec2 is already cached). CPU only; nothing is uploaded (share=False).
Each analysis runs a Whisper generation on CPU, so expect a short wait per clip.
"""

from __future__ import annotations

import random

import gradio as gr
import librosa
import torch
from datasets import Audio, load_dataset
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

print("Loading three models (first launch downloads ~1.3 GB)…")

# MINDS-14 (en-AU) at 16 kHz gives us labelled example clips to pick from.
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train").cast_column(
    "audio", Audio(sampling_rate=16_000)
)

ctc_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
ctc_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()

whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval()

ast_fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
).eval()


def analyze(array, sr):
    if sr != 16_000:
        array = librosa.resample(array, orig_sr=sr, target_sr=16_000)

    # CTC (character-level, per-frame collapse)
    inputs = ctc_proc(array, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        ctc_logits = ctc_model(inputs.input_values).logits
    ctc_text = ctc_proc.batch_decode(ctc_logits.argmax(-1))[0]
    n_frames = ctc_logits.shape[1]

    # Whisper (seq2seq, autoregressive BPE)
    feats = whisper_proc(array, sampling_rate=16_000, return_tensors="pt").input_features
    with torch.no_grad():
        gen = whisper_model.generate(feats, language="english", task="transcribe", max_new_tokens=128)
    whisper_text = whisper_proc.decode(gen[0], skip_special_tokens=True).strip()

    # AST (classification, spectrogram patches)
    ast_inputs = ast_fe(array, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        ast_probs = ast_model(**ast_inputs).logits.softmax(-1)[0]
    top = ast_probs.topk(3)
    ast_labels = ", ".join(ast_model.config.id2label[i] for i in top.indices.tolist())

    md = (
        f"### CTC - Wav2Vec2 (char-level, 32-token vocab)\n"
        f"`{ctc_text}`\n\n"
        f"*{n_frames} frames, each collapsed from per-frame character predictions.*\n\n"
        f"### Seq2seq - Whisper (autoregressive, ~50k BPE vocab)\n"
        f"{whisper_text}\n\n"
        f"*Generated one token at a time, cased and punctuated.*\n\n"
        f"### Classification - AST (527 AudioSet classes, spectrogram patches)\n"
        f"{ast_labels}"
    )
    return md


def analyze_random():
    idx = random.randrange(minds.num_rows)
    a = minds[idx]["audio"]
    return (a["sampling_rate"], a["array"]), analyze(a["array"], a["sampling_rate"])


def analyze_upload(audio):
    if audio is None:
        return "Upload or record a clip, or use the random button."
    sr, array = audio
    array = array.astype("float32")
    if array.ndim > 1:  # stereo -> mono
        array = array.mean(axis=1)
    array = array / (abs(array).max() or 1.0)
    return analyze(array, sr)


with gr.Blocks(title="Three audio transformer architectures") as demo:
    gr.Markdown(
        "# 🧠 Unit 3 - three architectures, one clip\n"
        "The same audio through **CTC** (Wav2Vec2), **seq2seq** (Whisper), and "
        "**classification** (AST). Notice how the input, vocabulary, and output style differ."
    )
    audio_out = gr.Audio(label="Audio clip")
    info_out = gr.Markdown()
    with gr.Row():
        gr.Button("🎲 Random MINDS-14 clip").click(fn=analyze_random, outputs=[audio_out, info_out])
    gr.Markdown("Or analyze your own:")
    upload = gr.Audio(label="Upload / record", sources=["upload", "microphone"])
    upload.change(fn=analyze_upload, inputs=upload, outputs=info_out)

    demo.load(fn=analyze_random, outputs=[audio_out, info_out])


if __name__ == "__main__":
    demo.launch(share=False)
