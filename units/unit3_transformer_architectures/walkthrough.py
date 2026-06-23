"""
Hugging Face Audio Course - Unit 3: Transformer architectures for audio
=======================================================================

Unit 3 is conceptual in the course: it explains how transformers are applied to
audio and the three main architecture families. This script makes each concept
*runnable* so you can see the mechanism, not just read about it:

 1. Input representations - raw waveform (Wav2Vec2) vs log-mel spectrogram (Whisper)
 2. CTC mechanics         - per-frame predictions -> blank/repeat collapse -> text
 3. Seq2seq mechanics     - Whisper encoder/decoder, task tokens, ASR + translation
 4. Classification        - the Audio Spectrogram Transformer (ViT over a spectrogram)

Run with:

    uv run python units/unit3_transformer_architectures/walkthrough.py

First run downloads ~1.3 GB of models to the Hugging Face cache (not the repo):
openai/whisper-small (~970 MB) and MIT/ast-finetuned-audioset (~350 MB).
facebook/wav2vec2-base-960h and MINDS-14 are already cached from earlier units.
Everything runs on CPU; Whisper generation is a little slow (tens of seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters like "|" runs,
# "→", or Whisper's "<|...|>" task tokens. Force UTF-8 so the prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save figures, never block
import matplotlib.pyplot as plt

import librosa
import librosa.display

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   saved {path.relative_to(Path(__file__).parent)}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_minds14(name: str = "en-AU", split: str = "train"):
    """Load MINDS-14 robustly whether it resolves as parquet or a script."""
    from datasets import load_dataset

    kwargs = dict(path="PolyAI/minds14", name=name, split=split)
    try:
        return load_dataset(**kwargs)
    except Exception:
        return load_dataset(**kwargs, trust_remote_code=True)


def load_demo_clip():
    """One English 16 kHz clip shared by the CTC and seq2seq sections."""
    from datasets import load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ex = ds[0]
    return ex["audio"]["array"], ex["audio"]["sampling_rate"]


# ===========================================================================
# 1. INPUT REPRESENTATIONS
# ===========================================================================
def section_inputs(array, sr):
    banner("1. Two ways audio enters a transformer: waveform vs log-mel")

    dur = len(array) / sr
    print(f"   clip: {len(array)} samples @ {sr} Hz = {dur:.2f}s")
    print(f"   [A] waveform: {len(array)} float samples  (Wav2Vec2 ingests this directly)")

    # Whisper-style front-end: 80 mel bins, hop 160 @ 16 kHz -> 100 frames/sec.
    mel = librosa.feature.melspectrogram(
        y=array, sr=sr, n_mels=80, n_fft=400, hop_length=160, fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    print(f"   [B] log-mel : {log_mel.shape}  (80 mel bins x {log_mel.shape[1]} frames; Whisper input)")
    print(f"       {len(array)} samples -> {log_mel.shape[1]} frames "
          f"(~{len(array) // log_mel.shape[1]}x shorter sequence)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(len(array)) / sr
    ax1.plot(t, array, lw=0.4)
    ax1.set(title=f"(A) Raw waveform - {len(array)} samples (Wav2Vec2 input)",
            xlabel="time (s)", ylabel="amplitude")
    img = librosa.display.specshow(log_mel, x_axis="time", y_axis="mel", sr=sr,
                                   hop_length=160, fmax=8000, ax=ax2)
    ax2.set(title=f"(B) Log-mel spectrogram - 80 x {log_mel.shape[1]} (Whisper input)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    fig.tight_layout()
    save_fig("01_input_representations.png")


# ===========================================================================
# 2. CTC MECHANICS
# ===========================================================================
def section_ctc(array, sr):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    banner("2. CTC mechanics: per-frame predictions -> collapse -> text")

    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    inputs = proc(array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # [1, frames, 32]
    frames = logits.shape[1]
    dur = len(array) / sr
    print(f"   input_values: {tuple(inputs.input_values.shape)}")
    print(f"   logits      : {tuple(logits.shape)}  -> {frames} frames, vocab {logits.shape[-1]}")
    print(f"   the CNN feature encoder downsamples: {dur:.2f}s -> {frames} frames "
          f"= {frames / dur:.1f} frames/sec (~50/s)")

    pred_ids = logits.argmax(-1)[0]
    blank_id = proc.tokenizer.pad_token_id  # the pad token doubles as the CTC blank
    delim = proc.tokenizer.word_delimiter_token  # "|"
    tokens = proc.tokenizer.convert_ids_to_tokens(pred_ids.tolist())
    n_blank = tokens.count(proc.tokenizer.convert_ids_to_tokens(blank_id))
    print(f"   blank/pad id = {blank_id} ({proc.tokenizer.convert_ids_to_tokens(blank_id)!r}); "
          f"word delimiter = {delim!r}")
    print(f"   RAW per-frame tokens ({frames} frames, {n_blank} are blank):")
    print("     " + "".join(tokens))

    def collapse(ids, blank):
        out, prev = [], None
        for i in ids:
            if i != prev and i != blank:  # 1) merge repeats  2) drop blanks
                out.append(i)
            prev = i
        return out

    kept = proc.tokenizer.convert_ids_to_tokens(collapse(pred_ids.tolist(), blank_id))
    manual = "".join(" " if t == delim else t for t in kept)  # 3) "|" -> space
    final = proc.batch_decode(pred_ids.unsqueeze(0))[0]
    print(f"   after collapse : {manual.strip()!r}")
    print(f"   batch_decode   : {final.strip()!r}")
    print(f"   manual == batch_decode: {manual.strip() == final.strip()}")

    # plot the argmax token id per frame: blanks sit on the baseline, characters spike
    plt.figure(figsize=(12, 3))
    plt.step(range(frames), pred_ids.tolist(), where="mid", lw=0.8)
    plt.axhline(blank_id, color="tab:red", ls="--", lw=0.8, label=f"blank id={blank_id}")
    plt.xlabel("frame (~20 ms each)")
    plt.ylabel("argmax token id")
    plt.title("CTC: predicted token per frame (spikes between long blank runs)")
    plt.legend(loc="upper right")
    save_fig("02_ctc_logits.png")

    print("   (HuBERT, e.g. facebook/hubert-large-ls960-ft, is the same CTC architecture, ~1.2 GB - not run.)")


# ===========================================================================
# 3. SEQ2SEQ MECHANICS
# ===========================================================================
def section_seq2seq(array, sr):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    banner("3. Seq2seq mechanics: encoder hidden states + autoregressive decoder")

    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()

    feats = proc(array, sampling_rate=sr, return_tensors="pt").input_features
    print(f"   input_features: {tuple(feats.shape)}  (1, 80, 3000 = fixed 30s log-mel)")

    with torch.no_grad():
        enc = model.model.encoder(feats).last_hidden_state
    print(f"   encoder hidden states: {tuple(enc.shape)}  (1, frames, d_model)")

    # reveal the decoder's task-token prompt (generate() strips these from its output)
    prompt = proc.get_decoder_prompt_ids(language="english", task="transcribe")
    ids = [proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")] + [i for _, i in prompt]
    print(f"   decoder task prompt: {proc.tokenizer.convert_ids_to_tokens(ids)}")

    with torch.no_grad():
        out = model.generate(feats, language="english", task="transcribe", max_new_tokens=128)
    print(f"   transcription: {proc.decode(out[0], skip_special_tokens=True)!r}")
    print(f"   Whisper vocab: {model.config.vocab_size} BPE tokens (vs CTC's 32 characters)")

    # save the Whisper input (the log-mel the encoder consumes)
    plt.figure(figsize=(12, 3))
    plt.imshow(feats[0].numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("frame")
    plt.ylabel("mel bin")
    plt.title("Whisper encoder input: log-mel features (80 x 3000)")
    plt.colorbar()
    save_fig("03_whisper_input_features.png")
    return proc, model


def section_translate(proc, model):
    import torch
    from datasets import Audio

    banner("3d. Whisper translation: German speech -> English text")
    try:
        de = load_minds14(name="de-DE").cast_column("audio", Audio(sampling_rate=16_000))
        ex = de[0]
    except Exception as exc:
        print(f"   could not load MINDS-14 de-DE: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        return

    feats = proc(ex["audio"]["array"], sampling_rate=16_000, return_tensors="pt").input_features
    with torch.no_grad():
        transc = model.generate(feats, language="german", task="transcribe", max_new_tokens=128)
        transl = model.generate(feats, language="german", task="translate", max_new_tokens=128)
    print(f"   reference (de)  : {ex['transcription']!r}")
    print(f"   transcribe (de) : {proc.decode(transc[0], skip_special_tokens=True)!r}")
    print(f"   translate  (en) : {proc.decode(transl[0], skip_special_tokens=True)!r}")
    print(f"   gold english    : {ex['english_transcription']!r}")
    print("   (translation captures the meaning; small models still aren't perfect, but note")
    print("    the SAME model produced both outputs, switched only by the task token.)")


# ===========================================================================
# 4. CLASSIFICATION ARCHITECTURES
# ===========================================================================
def section_classification():
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, AutoConfig

    banner("4. Classification: Audio Spectrogram Transformer (ViT over a spectrogram)")

    array, sr0 = librosa.load(librosa.ex("trumpet"))  # 22050 Hz
    array = librosa.resample(array, orig_sr=sr0, target_sr=16_000)  # AST requires 16 kHz

    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(model_id)
    model = ASTForAudioClassification.from_pretrained(model_id)
    model.eval()

    inputs = fe(array, sampling_rate=16_000, return_tensors="pt")
    print(f"   input_values: {tuple(inputs.input_values.shape)}  (1, 1024 frames, 128 mel)")
    print("   AST splits this 1024x128 spectrogram into 16x16 patches, like a Vision Transformer.")
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, 527]
    probs = logits.softmax(-1)[0]
    top = probs.topk(5)
    print("   top-5 AudioSet labels:")
    labels, scores = [], []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        lab = model.config.id2label[i]
        print(f"     {p:.3f}  {lab}")
        labels.append(lab)
        scores.append(p)

    plt.figure(figsize=(10, 4))
    plt.barh(labels[::-1], scores[::-1], color="tab:purple")
    plt.xlim(0, 1)
    plt.xlabel("probability")
    plt.title("Audio Spectrogram Transformer - top-5 (trumpet clip)")
    save_fig("04_ast_topk.png")

    # connect back to Unit 2 without loading the 1.2 GB weights (config only)
    cfg = AutoConfig.from_pretrained("anton-l/xtreme_s_xlsr_300m_minds14")
    print(f"\n   Connect back: Unit 2's intent classifier is a {cfg.architectures}")
    print("   = the same Wav2Vec2 encoder, mean-pooled over frames + a linear head,")
    print("   whereas AST is a pure transformer over spectrogram patches.")


def main() -> None:
    print("Hugging Face Audio Course — Unit 3: Transformer architectures for audio")
    print(f"Figures will be written to: {FIG_DIR}")

    array, sr = load_demo_clip()
    section_inputs(array, sr)
    section_ctc(array, sr)
    proc, model = section_seq2seq(array, sr)
    section_translate(proc, model)
    section_classification()

    banner("Done!  See figures/ for the plots.")


if __name__ == "__main__":
    main()
