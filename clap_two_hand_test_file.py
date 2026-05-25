import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write
from msclap import CLAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from two_hand_audio import load_two_hand_wav

TARGET_SR = 48000


def load_labels(label_file: Path | None):
    if label_file is None:
        label_file = Path(__file__).resolve().parent / "default_everyday_labels.txt"
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {label_file}")
    return labels


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def print_ranked(title, labels, scores):
    probs = softmax(scores)
    top_idx = np.argsort(probs)[::-1]
    print(f"\n{title}")
    for idx in top_idx[:5]:
        print(f"{labels[idx]:45s} {probs[idx]:.4f}")


def main(audio_path, label_file):
    labels = load_labels(label_file)
    left, right, _ = load_two_hand_wav(audio_path, TARGET_SR)

    with tempfile.NamedTemporaryFile(suffix="_left.wav", delete=False) as f1,          tempfile.NamedTemporaryFile(suffix="_right.wav", delete=False) as f2:
        left_path = f1.name
        right_path = f2.name

    try:
        write(left_path, TARGET_SR, left.astype(np.float32))
        write(right_path, TARGET_SR, right.astype(np.float32))

        print("Loading CLAP...")
        clap_model = CLAP(version="2023", use_cuda=False)
        text_embeddings = clap_model.get_text_embeddings(labels)
        audio_embeddings = clap_model.get_audio_embeddings([left_path, right_path])

        similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        sims = similarities.detach().cpu().numpy()

        print_ranked("Left hand predictions:", labels, sims[0])
        print_ranked("Right hand predictions:", labels, sims[1])

    finally:
        if os.path.exists(left_path):
            os.remove(left_path)
        if os.path.exists(right_path):
            os.remove(right_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLAP on a stereo two-hand recording.")
    parser.add_argument("audio_file", help="Stereo WAV file with left/right hand microphones.")
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional text file with one CLAP label/prompt per line.",
    )
    args = parser.parse_args()

    label_file = Path(args.labels_file).resolve() if args.labels_file else None
    main(args.audio_file, label_file)
