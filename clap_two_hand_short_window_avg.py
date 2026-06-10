#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from two_hand_audio import load_two_hand_wav
from short_window_utils import iter_windows, print_topk, read_simple_label_file

from msclap import CLAP

TARGET_SR = 48000
TOP_K = 10


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def write_chunks_to_temp(chunks: np.ndarray, prefix: str):
    tmpdir = tempfile.TemporaryDirectory()
    paths = []
    for i, chunk in enumerate(chunks):
        path = Path(tmpdir.name) / f"{prefix}_{i:04d}.wav"
        wavfile.write(str(path), TARGET_SR, np.clip(chunk, -1.0, 1.0))
        paths.append(str(path))
    return tmpdir, paths


def aggregate_hand(clap_model, hand_audio: np.ndarray, text_embeddings, label_texts: list[str], window_seconds: float, hop_seconds: float) -> np.ndarray:
    chunks = iter_windows(hand_audio, TARGET_SR, window_seconds, hop_seconds)
    tmpdir, paths = write_chunks_to_temp(chunks, "chunk")
    try:
        audio_embeddings = clap_model.get_audio_embeddings(paths)
        with torch.no_grad():
            sims = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        sims = sims.detach().cpu().numpy()
        if sims.ndim == 1:
            sims = sims[None, :]
        probs = np.stack([softmax(row) for row in sims], axis=0)
        return probs.mean(axis=0)
    finally:
        tmpdir.cleanup()


def main():
    if len(sys.argv) < 2:
        print("Usage: python clap_two_hand_short_window_avg.py stereo_audio.wav [window_seconds] [hop_seconds] [labels_file]")
        sys.exit(1)

    audio_path = sys.argv[1]
    window_seconds = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
    hop_seconds = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.5
    labels_file = Path(sys.argv[4]) if len(sys.argv) >= 5 else (PROJECT_ROOT / "CLAP" / "default_everyday_labels.txt")

    label_texts = read_simple_label_file(labels_file)

    print("Loading CLAP...")
    clap_model = CLAP(use_cuda=False)
    text_embeddings = clap_model.get_text_embeddings(label_texts)

    left, right, _ = load_two_hand_wav(audio_path, TARGET_SR)
    left_scores = aggregate_hand(clap_model, left, text_embeddings, label_texts, window_seconds, hop_seconds)
    right_scores = aggregate_hand(clap_model, right, text_embeddings, label_texts, window_seconds, hop_seconds)

    print_topk("Left hand predictions:", left_scores, label_texts, min(TOP_K, len(label_texts)))
    print_topk("Right hand predictions:", right_scores, label_texts, min(TOP_K, len(label_texts)))


if __name__ == "__main__":
    main()
