import os
import queue
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from msclap import CLAP

SR = 48000
CHANNELS = 1
BLOCK_SEC = 2.5
BUFFER_SEC = 5.0

class_labels = [
    "brushing teeth"
    "washing hands"
    "drying hands with a towel"
    "typing on a keyboard"
    "writing on paper"
    "zipping or unzipping a backpack or jacket"
    "folding paper"
    "cutting paper"
    "stirring a spoon in a mug"
    "pressing microwave buttons"
    "folding clothes"
    "opening or closing blinds"
    "tying shoelaces"
    "sorting coins"
    "pouring water into a cup"
    "opening or closing a drawer"
    "clicking a mouse"
    "turning notebook pages"
    "crumpling paper"
    "tearing paper"
    "handling keys"
    "opening or closing a door"
    "chopping on a cutting board"
    "rinsing dishes in a sink"
    "opening or closing a bottle cap"
    "using a soap or lotion pump"
    "zipping or unzipping a pencil case"
    "stirring in a bowl"
]

audio_q = queue.Queue()

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

def main():
    print("Loading CLAP...")
    clap_model = CLAP(version="2023", use_cuda=False)
    text_embeddings = clap_model.get_text_embeddings(class_labels)

    buffer_samples = int(BUFFER_SEC * SR)
    ring = np.zeros(buffer_samples, dtype=np.float32)

    print("Listening live. Press Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(BLOCK_SEC * SR),
        callback=audio_callback,
    ):
        while True:
            block = audio_q.get().reshape(-1)

            n = len(block)
            ring = np.roll(ring, -n)
            ring[-n:] = block

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            write(tmp_path, SR, ring)

            audio_embeddings = clap_model.get_audio_embeddings([tmp_path])
            similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
            scores = similarities.detach().cpu().numpy()[0]
            probs = softmax(scores)
            top_idx = np.argsort(probs)[::-1]

            print("\nTop predictions:")
            for idx in top_idx[:5]:
                print(f"  {class_labels[idx]:35s} {probs[idx]:.4f}")

            os.remove(tmp_path)

if __name__ == "__main__":
    main()