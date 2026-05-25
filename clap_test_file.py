import sys
import numpy as np
from msclap import CLAP

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def main(audio_path):
    # You can change these prompts to match your experiment.
    class_labels = [
        "the sound of typing on a keyboard",
        "the sound of washing hands",
        "the sound of drying hands",
        "the sound of running water",
        "the sound of zipping a backpack"
    ]

    print("Loading CLAP...")
    clap_model = CLAP(version="2023", use_cuda=False)

    print("Embedding text labels...")
    text_embeddings = clap_model.get_text_embeddings(class_labels)

    print(f"Embedding audio file: {audio_path}")
    audio_embeddings = clap_model.get_audio_embeddings([audio_path])

    similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    scores = similarities.detach().cpu().numpy()[0]

    probs = softmax(scores)
    top_idx = np.argsort(probs)[::-1]

    print("\nTop predictions:")
    for idx in top_idx:
        print(f"{class_labels[idx]:35s} {probs[idx]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clap_test_file.py your_audio.wav")
        sys.exit(1)
    main(sys.argv[1])