"""
Hugging Face Audio Course - Unit 1: Working with Audio Data
===========================================================

A complete, runnable walkthrough of every concept in Unit 1.

This is the *script* version of the companion notebook (`unit1_audio_data.ipynb`).
Because a script can't show plots or play audio inline, every figure is saved to
the ``figures/`` folder and a few short ``.wav`` demo clips are written there too,
so you can open them and listen.

Run it with:

    uv run python unit1_audio_data.py

Sections
--------
 2. Audio fundamentals (sampling, Nyquist, decibels, bit depth)  -- synthetic demos
 3. Loading audio & the waveform
 4. The frequency spectrum (DFT)
 5. The spectrogram (STFT)
 6. The mel spectrogram
 7. Loading a 🤗 audio dataset (MINDS-14)
 8. Listening to audio
 9. Resampling
10. Filtering by duration
11. Pre-processing with a feature extractor (Whisper)
12. Streaming audio data (GigaSpeech, with a public fallback)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't encode characters like "≈".
# Force UTF-8 so the explanatory prints below never crash the script.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

# Use a non-interactive backend so the script never blocks on a window and can
# save figures on a headless machine.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure into figures/ and close it."""
    path = FIG_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   saved {path.relative_to(Path(__file__).parent)}")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# 2. AUDIO FUNDAMENTALS  (synthetic demos)
# ===========================================================================
def section_fundamentals() -> None:
    banner("2. Audio fundamentals: sampling, Nyquist, decibels, bit depth")

    # --- 2a. Sampling & sampling rate --------------------------------------
    # Sampling = measuring a continuous signal at fixed time steps. The
    # sampling rate (Hz) is how many samples we take per second. The same 1s
    # 220 Hz tone contains very different numbers of samples at each rate.
    print("\n[2a] Sampling rate -> number of samples in 1 second of a 220 Hz tone:")
    freq = 220.0
    for sr in (8_000, 16_000, 44_100):
        t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        print(f"     {sr:>6d} Hz  ->  {len(tone):>6d} samples")

    # Visualise sample density: plot a smooth reference curve plus the actual
    # samples taken at 8 kHz vs 16 kHz over the first 5 ms.
    window_s = 0.005
    t_ref = np.linspace(0, window_s, 2000)
    ref = 0.5 * np.sin(2 * np.pi * freq * t_ref)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, sr in zip(axes, (8_000, 16_000)):
        n = int(sr * window_s)
        ts = np.arange(n) / sr
        ax.plot(t_ref * 1000, ref, color="lightgray", label="continuous signal")
        ax.stem(ts * 1000, 0.5 * np.sin(2 * np.pi * freq * ts), basefmt=" ")
        ax.set_title(f"{sr} Hz  ({n} samples in 5 ms)")
        ax.set_xlabel("time (ms)")
    axes[0].set_ylabel("amplitude")
    fig.suptitle("Higher sampling rate = more samples per second")
    save_fig("01_sampling_rate.png")

    # --- 2b. Nyquist limit & aliasing --------------------------------------
    # Nyquist limit = sampling_rate / 2: the highest frequency we can faithfully
    # capture. A 6 kHz tone sampled at 8 kHz (Nyquist 4 kHz) is ALIASED: it
    # masquerades as |6000 - 8000| = 2000 Hz. Sampled at 16 kHz (Nyquist 8 kHz)
    # it is captured correctly.
    print("\n[2b] Nyquist & aliasing: a 6000 Hz tone")
    f_sig = 6_000

    def peak_freq(signal: np.ndarray, sr: int) -> float:
        spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        freqs = np.fft.rfftfreq(len(signal), 1 / sr)
        return float(freqs[np.argmax(spec)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sr in zip(axes, (8_000, 16_000)):
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * f_sig * t)
        detected = peak_freq(sig, sr)
        nyq = sr // 2
        verdict = "ALIASED!" if detected < f_sig - 1 else "captured correctly"
        print(f"     sampled at {sr} Hz (Nyquist {nyq} Hz): peak detected at "
              f"{detected:.0f} Hz  -> {verdict}")
        # save a clip so you can *hear* the aliasing
        sf.write(FIG_DIR / f"02_alias_6000Hz_at_{sr}.wav", sig.astype(np.float32), sr)
        # plot the first 2 ms of samples
        n = int(sr * 0.002)
        ax.plot(t[:n] * 1000, sig[:n], marker="o")
        ax.set_title(f"6 kHz @ {sr} Hz (Nyquist {nyq} Hz)\ndetected peak {detected:.0f} Hz")
        ax.set_xlabel("time (ms)")
    axes[0].set_ylabel("amplitude")
    save_fig("02_nyquist_aliasing.png")
    print("     -> listen to 02_alias_6000Hz_at_8000.wav vs ..._at_16000.wav")

    # --- 2c. Amplitude -> decibels -----------------------------------------
    # Amplitude is perceived as loudness. In digital audio we usually express it
    # on a logarithmic decibel (dB) scale where 0 dB is the loudest value and
    # quieter sounds are negative. Every -6 dB halves the amplitude.
    print("\n[2c] Amplitude on a linear vs decibel scale")
    amp = np.logspace(0, -4, 500)  # 1.0 down to 0.0001
    amp_db = librosa.amplitude_to_db(amp, ref=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(amp)
    ax1.set_title("linear amplitude")
    ax1.set_xlabel("sample index")
    ax2.plot(amp_db, color="tab:red")
    ax2.set_title("same data in decibels (dB)")
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("dB")
    for half in (-6, -12, -18):
        ax2.axhline(half, ls="--", lw=0.7, color="gray")
    save_fig("03_amplitude_db.png")
    print(f"     amplitude 1.0 -> {librosa.amplitude_to_db(np.array([1.0]))[0]:.1f} dB; "
          f"0.5 -> {librosa.amplitude_to_db(np.array([0.5]))[0]:.1f} dB (≈ -6 dB, half)")

    # --- 2d. Bit depth & dynamic range -------------------------------------
    # Bit depth = how precisely each amplitude sample is stored. More bits =>
    # more discrete levels => lower quantization noise => wider dynamic range
    # (roughly 6.02 dB per bit).
    print("\n[2d] Bit depth -> number of levels and dynamic range")
    for bits in (16, 24, 32):
        levels = 2 ** bits
        print(f"     {bits:>2d}-bit -> {levels:>14,d} levels, "
              f"dynamic range ≈ {6.02 * bits:5.1f} dB")

    def quantize(x: np.ndarray, bits: int) -> np.ndarray:
        levels = 2 ** bits
        return np.round(x * (levels / 2 - 1)) / (levels / 2 - 1)

    t = np.linspace(0, 1, 200, endpoint=False)
    clean = np.sin(2 * np.pi * 3 * t)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, clean, color="lightgray", lw=3, label="original (high precision)")
    ax.step(t, quantize(clean, 3), where="mid", label="3-bit (8 levels)")
    ax.step(t, quantize(clean, 4), where="mid", label="4-bit (16 levels)")
    ax.set_title("Low bit depth = visible 'staircase' quantization steps")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")
    save_fig("04_bit_depth.png")


# ===========================================================================
# 3-6. ONE SOUND, FOUR REPRESENTATIONS (the trumpet example)
# ===========================================================================
def section_representations() -> tuple[np.ndarray, int]:
    banner("3-6. Waveform, spectrum, spectrogram, mel spectrogram")

    # librosa.load downloads (first run only, via pooch) and decodes the clip.
    # By default it resamples to 22050 Hz and mixes to mono.
    print("\n[3] Loading librosa's 'trumpet' example with librosa.load(...)")
    array, sampling_rate = librosa.load(librosa.ex("trumpet"))
    print(f"     array shape={array.shape}, dtype={array.dtype}, sr={sampling_rate} Hz")
    print(f"     duration={len(array) / sampling_rate:.2f}s, "
          f"min={array.min():.3f}, max={array.max():.3f}")
    sf.write(FIG_DIR / "05_trumpet.wav", array, sampling_rate)

    # --- 3. Waveform (time domain) -----------------------------------------
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(array, sr=sampling_rate)
    plt.title("Waveform (time domain)")
    save_fig("05_waveform.png")

    # --- 4. Frequency spectrum (DFT of a short slice) ----------------------
    print("[4] Frequency spectrum via the Discrete Fourier Transform")
    dft_input = array[:4096]
    window = np.hanning(len(dft_input))          # taper edges to reduce leakage
    windowed_input = dft_input * window
    dft = np.fft.rfft(windowed_input)            # real FFT -> complex values
    amplitude = np.abs(dft)                      # magnitude of each frequency
    amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
    frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))
    plt.figure().set_figwidth(12)
    plt.plot(frequency, amplitude_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.xscale("log")
    plt.title("Frequency spectrum (one slice) - peaks are harmonics")
    save_fig("06_spectrum.png")

    # --- 5. Spectrogram (STFT: how the spectrum changes over time) ---------
    print("[5] Spectrogram via the Short-Time Fourier Transform")
    D = librosa.stft(array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure().set_figwidth(12)
    librosa.display.specshow(S_db, x_axis="time", y_axis="hz", sr=sampling_rate)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (time x frequency x amplitude)")
    save_fig("07_spectrogram.png")

    # --- 6. Mel spectrogram ------------------------------------------------
    print("[6] Mel spectrogram (perceptually-spaced frequency bins)")
    S = librosa.feature.melspectrogram(
        y=array, sr=sampling_rate, n_mels=128, fmax=8000
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure().set_figwidth(12)
    librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram (log-mel, in dB)")
    save_fig("08_mel_spectrogram.png")

    return array, sampling_rate


# ===========================================================================
# 7-11. WORKING WITH A 🤗 AUDIO DATASET (MINDS-14)
# ===========================================================================
def load_minds14():
    """Load MINDS-14 robustly whether it resolves as parquet or a script."""
    from datasets import load_dataset

    kwargs = dict(path="PolyAI/minds14", name="en-AU", split="train")
    try:
        return load_dataset(**kwargs)
    except Exception:
        # Older script-based datasets need trust_remote_code=True on datasets 3.x
        return load_dataset(**kwargs, trust_remote_code=True)


def section_dataset() -> None:
    from datasets import Audio
    from transformers import WhisperFeatureExtractor

    banner("7-11. Loading & preprocessing a real audio dataset (MINDS-14)")

    # --- 7. Load the dataset & inspect the Audio feature -------------------
    print("\n[7] load_dataset('PolyAI/minds14', name='en-AU', split='train')")
    minds = load_minds14()
    print(minds)
    example = minds[0]
    audio = example["audio"]
    print("\n     The 'audio' column is a dict with keys:", list(audio.keys()))
    print(f"       array:         {type(audio['array']).__name__} shape={audio['array'].shape}")
    print(f"       sampling_rate: {audio['sampling_rate']} Hz")
    print(f"       path:          {audio['path']}")
    id2label = minds.features["intent_class"].int2str
    print(f"     intent of example 0: {id2label(example['intent_class'])!r}")

    # --- 8. Listen (script mode: save a .wav instead of inline playback) ---
    print("\n[8] Listening — in the notebook this is IPython.display.Audio(...).")
    sf.write(FIG_DIR / "09_minds14_example.wav", audio["array"], audio["sampling_rate"])
    print("     saved figures/09_minds14_example.wav")
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(audio["array"], sr=audio["sampling_rate"])
    plt.title(f"MINDS-14 example (intent: {id2label(example['intent_class'])})")
    save_fig("09_minds14_waveform.png")

    # --- 9. Resampling to 16 kHz -------------------------------------------
    # Most speech models expect 16 kHz. cast_column resamples on the fly when
    # each example is accessed.
    print("\n[9] Resampling to 16 kHz with cast_column(...)")
    print(f"     before: {minds[0]['audio']['sampling_rate']} Hz")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    print(f"     after:  {minds[0]['audio']['sampling_rate']} Hz")

    # --- 10. Filtering by duration -----------------------------------------
    print("\n[10] Filtering out clips longer than 20 seconds")
    MAX_DURATION_IN_SECONDS = 20.0

    def is_audio_length_in_range(input_length):
        return input_length < MAX_DURATION_IN_SECONDS

    # The course does: [librosa.get_duration(path=x) for x in minds["path"]].
    # We compute from the decoded array instead, which is equivalent and works
    # whether the audio lives in local files or inline in parquet.
    new_column = [
        librosa.get_duration(y=ex["audio"]["array"], sr=ex["audio"]["sampling_rate"])
        for ex in minds
    ]
    minds = minds.add_column("duration", new_column)
    before = minds.num_rows
    minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])
    after = minds.num_rows
    minds = minds.remove_columns(["duration"])  # tidy up the helper column
    print(f"     {before} examples -> {after} after filtering "
          f"({before - after} removed)")

    # --- 11. Feature extractor (Whisper) -----------------------------------
    print("\n[11] Pre-processing with WhisperFeatureExtractor + dataset.map(...)")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    def prepare_dataset(example):
        audio = example["audio"]
        features = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], padding=True
        )
        return features

    minds = minds.map(prepare_dataset)
    print("     columns after map:", minds.column_names)
    feats = np.array(minds[0]["input_features"])
    n_frames = feats.shape[-1]
    print(f"     input_features shape for one example: {feats.shape}")
    print(f"     -> 80 mel bins x {n_frames} frames. With padding=True each clip is")
    print("        padded to the longest in its call; use padding='max_length' for")
    print("        Whisper's fixed 30-second input of 80 x 3000 frames.")

    # visualise the model-ready log-mel features
    log_mel = feats[0] if feats.ndim == 3 else feats
    plt.figure().set_figwidth(12)
    librosa.display.specshow(log_mel, x_axis="time", y_axis="mel", sr=16_000)
    plt.colorbar()
    plt.title("Whisper log-mel input features (what the model sees)")
    save_fig("10_whisper_features.png")


# ===========================================================================
# 12. STREAMING (GigaSpeech, with a public fallback)
# ===========================================================================
def section_streaming() -> None:
    from datasets import load_dataset

    banner("12. Streaming audio data")

    def stream_dataset():
        try:
            ds = load_dataset(
                "speechcolab/gigaspeech", "xs", streaming=True, trust_remote_code=True
            )
            return "speechcolab/gigaspeech", ds
        except Exception as exc:  # gated / not logged in / no access yet
            print("\n   GigaSpeech is gated and could not be loaded:")
            print(f"     {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            print("   -> To use it: accept the terms on the dataset page and run")
            print("      `uv run huggingface-cli login` (see README, section 4).")
            print("   -> Falling back to a small public dataset for the demo.\n")
            ds = load_dataset(
                "hf-internal-testing/librispeech_asr_dummy", "clean", streaming=True
            )
            return "hf-internal-testing/librispeech_asr_dummy", ds

    name, streamed = stream_dataset()
    print(f"   streaming from: {name}")

    # A streamed dataset is an IterableDataset — nothing is downloaded up front.
    split = "train" if "train" in streamed else list(streamed.keys())[0]
    stream = streamed[split]

    # Grab examples one at a time, on the fly.
    first = next(iter(stream))
    print(f"   first example keys: {list(first.keys())}")
    print(f"   first example sampling_rate: {first['audio']['sampling_rate']} Hz")

    # .take(n) previews the first n examples without loading the whole set.
    head = list(stream.take(2))
    print(f"   stream.take(2) -> {len(head)} examples pulled on demand")

    print(
        "\n   Why stream?\n"
        "     * Disk space: examples load one-by-one; nothing is stored locally,\n"
        "       so you can use datasets of arbitrary size.\n"
        "     * Speed: processing happens on the fly — start as soon as example 1\n"
        "       is ready, no waiting for a full download.\n"
        "     * Easy experimentation: try your code on a handful of examples\n"
        "       before committing to the full dataset."
    )


def main() -> None:
    print("Hugging Face Audio Course — Unit 1: Working with Audio Data")
    print(f"Figures and audio clips will be written to: {FIG_DIR}")
    section_fundamentals()
    section_representations()
    section_dataset()
    section_streaming()
    banner("Done!  Open the figures/ folder to view the plots and hear the clips.")


if __name__ == "__main__":
    main()
