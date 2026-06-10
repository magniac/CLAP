import sounddevice as sd
from scipy.io.wavfile import write

sr = 48000
duration_sec = 5

print("Recording...")
audio = sd.rec(int(duration_sec * sr), samplerate=sr, channels=1, dtype="float32")
sd.wait()
print("Done.")

write("clap_test.wav", sr, audio)
print("Saved to clap_test.wav")