# CLAP-Controlled Audio Amplifier ("Audio Augmentation for Mindfulness")

A low-latency, live audio pass-through system that **boosts the volume of sounds
you want to pay attention to**. You tell it which activities/sounds to listen for
(for example "typing" or "rubbing hands"); the [CLAP](https://github.com/microsoft/CLAP)
audio model continuously listens and, whenever one of those sounds is present, the
system raises the gain on your monitored audio. When the sound stops, it returns to
the normal level.

It can be driven two ways:

- **A web UI** (recommended) — a single local page where you type the sounds to
  amplify, watch live detection scores, move a gain slider, and pick your devices.
- **The command line** — the same engine exposed as flags, useful for scripting
  and debugging.

The live audio path is handled by [Pedalboard](https://spotify.github.io/pedalboard/)'s
`AudioStream`, while CLAP runs separately and only decides whether the stream
should use *base* gain or *boosted* gain. Audio playback never waits for CLAP, so
the model can be slow, stall, or crash without interrupting what you hear.

## Files

```text
app.py                    # local web server (the web UI backend)
index.html                # single-page web frontend
paddle_stream_new.py      # core engine + command-line program; owns live audio
live_clap_detector.py     # CLAP-only detector worker
```

`paddle_stream_new.py` owns the microphone-to-headphone audio path and switches the
Pedalboard gain between base and boost. `live_clap_detector.py` runs as a separate,
input-only detector process. `app.py` drives the exact same engine as the command
line, so both front-ends behave identically.

## Why this design exists

Earlier versions put audio streaming, CLAP inference, gain control, and status
printing in one script. That worked, but it added noticeable monitoring latency
because the real-time audio path was tightly coupled to the CLAP system. The
current design separates responsibilities:

```text
paddle_stream_new.py
  -> opens the low-latency Pedalboard AudioStream
  -> streams microphone input to headphones/output
  -> applies Pedalboard Gain at either base or boost level
  -> polls a shared CLAP state flag

live_clap_detector.py
  -> opens an independent input-only audio stream
  -> runs CLAP on recent audio windows
  -> scores each input channel independently
  -> publishes only "boost on / off" plus status messages
```

If CLAP is slow, stalls, or crashes, the Pedalboard audio stream keeps running with
the most recent gain state.

---

## Requirements

- **Python 3.10 or newer** (developed on 3.11).
- A working audio input (microphone) and output (headphones recommended).
- macOS, Linux, or Windows. Most testing has been on macOS.
- Roughly 2 GB of disk/RAM headroom — the CLAP model is downloaded automatically
  on first run and loaded into memory (this is why startup takes ~10–15 seconds).

> **Headphones strongly recommended.** If you monitor a microphone through
> speakers, the amplified output can feed back into the mic. See
> [Using built-in speakers](#using-built-in-speakers-the-may-cause-feedback-error).

---

## Installation

Clone the repository and enter it:

```bash
git clone <REPO_URL>
cd <REPO_DIRECTORY>
```

Create and activate a virtual environment (recommended), then install the
dependencies:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
# python -m venv .venv
# .venv\Scripts\Activate.ps1

python3 -m pip install --upgrade pip
python3 -m pip install numpy sounddevice soundfile torch msclap pedalboard
```

The web server itself uses only the Python standard library; the packages above are
for the audio path and the CLAP model. The first run downloads the CLAP model
weights, so allow extra time and a network connection the first time.

---

## Quick start (web UI — recommended)

From the repository directory, with your virtual environment activated:

```bash
python3 app.py
```

Then open <http://127.0.0.1:8765> in a browser. To use a different address or
port:

```bash
python3 app.py --host 127.0.0.1 --port 8765
```

Press `Ctrl+C` in the terminal to stop the server.

### Using the web UI

1. **Pick your devices** in the left sidebar — an **Input (microphone)** and an
   **Output (headphones)** dropdown, populated from the devices on your machine.
   Choose your devices *before* pressing Start; the selection locks while running.
2. **Enter the sounds to amplify** under **"I want to be mindful of …"**. Type an
   activity and press **Add** (or Enter). Each becomes its own card. You can also
   paste a comma-separated list. These become the CLAP text labels.
3. **Set the boost gain** with the slider (`0`–`24` dB). While running, moving the
   slider changes the boost **live**, with no restart.
4. **Press Start.** The model takes ~10–15 seconds to load — the status shows
   **"Loading…"** during this time, then **"Listening"** once it is running.
5. When a target sound is detected, the status switches to a green **"Boost"** and
   the corresponding activity bar(s) turn green. When the sound stops, it returns
   to **"Listening"** at the base level.

#### Score bars

While running, each activity card shows a live bar of the CLAP cosine-similarity
score for that sound (the maximum across input channels). A tick marks the
on-threshold. A bar turns **green only when it is actually causing the boost** —
i.e. the boost is on and that activity crossed the threshold. It stays green for as
long as the boost is active (even if the score momentarily dips), and returns to
gray when the boost turns off. Bars that never triggered the boost stay gray.

#### Changing the list

Activities are fixed when you press Start, because they become CLAP text labels.
To change which sounds are amplified, edit the list and press Stop, then Start. The
gain slider, by contrast, always takes effect immediately.

#### Hidden developer panel

A faint gear icon at the bottom of the sidebar opens a **Developer settings** panel
for tuning detector parameters. You can also open it with `Ctrl+Shift+D`
(`Cmd+Shift+D` on macOS), or by loading <http://127.0.0.1:8765/#debug>. It exposes:

```text
window_seconds, hop_seconds,
on_threshold, off_threshold (blank = same as on),
on_windows_required, off_windows_required,
base_gain_db, input_channels,
torch_num_threads, analysis_queue_size,
allow_feedback (checkbox)
```

Each field has a plain-English label describing what it does. Values are pre-filled
from the server's current settings. Changes apply the next time you press Start
(boost gain remains live). See [Detector concepts](#detector-concepts) for what
these mean.

#### Using built-in speakers (the "may cause feedback" error)

Pedalboard refuses to open a stream when the input looks like a microphone and the
output looks like a speaker, because amplifying the mic back out through speakers
can create a feedback loop. If you run with, for example, a laptop's built-in mic
and speakers, you will see:

```text
The audio input device passed to AudioStream looks like a microphone, and the
output device looks like a speaker. This setup may cause feedback. To create an
AudioStream anyways, pass `allow_feedback=True` to the AudioStream constructor.
```

To run anyway, open the developer panel, tick **Allow feedback**, and press Start.
Keep the boost gain low to avoid loud feedback. This is off by default and intended
for testing; the normal microphone + headphones setup does not need it.

#### Web server HTTP endpoints

The UI is a single page that polls the server. The endpoints are:

```text
GET  /          # the UI page
GET  /status    # running state, gain state, per-activity scores, settings
GET  /devices   # available input/output device names
POST /start     # start with activities, gain, devices, and detector params
POST /stop      # stop the stream and detector
POST /gain      # change the boost gain live
```

#### Changing the web UI's default devices

The device dropdowns start on a default name. You can set those defaults when
launching the server (useful if you always use the same hardware):

```bash
python3 app.py \
  --input-device-name "YOUR INPUT DEVICE NAME" \
  --output-device-name "YOUR OUTPUT DEVICE NAME"
```

If a configured default name is not currently present, it is still shown as a
selectable option so your normal setup stays selected.

---

## Quick start (command line)

The command-line program is `paddle_stream_new.py`. A minimal run:

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --input-device-name "YOUR INPUT DEVICE NAME" \
  --output-device-name "YOUR OUTPUT DEVICE NAME" \
  --base-gain-db 0 \
  --boost-gain-db 12
```

Stop with `Ctrl+C`. Replace the device names with your own (see
[Listing devices](#listing-devices)). If you omit `--input-device-name` /
`--output-device-name`, the program falls back to its built-in defaults, which may
not match your machine — so it is usually best to pass them explicitly.

### Listing devices

```bash
python3 paddle_stream_new.py --list-devices
```

This prints two device lists:

1. Pedalboard input/output devices for the main audio stream.
2. `sounddevice` devices for the CLAP detector.

Use a Pedalboard `AudioStream` name for `--input-device-name` /
`--output-device-name`. If the CLAP detector needs a different device than
Pedalboard uses, pass it separately:

```bash
--detector-input-device "DEVICE_NAME_OR_INDEX"
```

### Main command-line options

#### Audio stream

```bash
--input-device-name  "YOUR INPUT DEVICE NAME"
--output-device-name "YOUR OUTPUT DEVICE NAME"
--input-channels 2
--output-channels 2
--buffer-size 128            # optional; leave unset first
--allow-feedback             # only if monitoring a mic through speakers
```

`--buffer-size` is optional. Leave it unset first; only try `--buffer-size 128` if
you need to experiment with latency.

#### Gain

```bash
--base-gain-db 0       # level when no target sound is detected (0 dB = unchanged)
--boost-gain-db 12     # level while a target sound is detected (+12 dB = strong boost)
```

The stream switches Pedalboard plugins when the detector changes state:

```text
base state  -> Pedalboard([Gain(gain_db=base_gain_db)])
boost state -> Pedalboard([Gain(gain_db=boost_gain_db)])
```

#### CLAP detector

```bash
--command "typing, rubbing hands"   # the ONLY CLAP text labels; comma-separated
--detector-input-device "..."       # if CLAP needs a different device than Pedalboard
--detector-channels 2                # logical channels CLAP analyzes (defaults to input channels)
--sample-rate 48000
--window-seconds 1.0
--hop-seconds 1.0
--on-threshold 0.20
--off-threshold 0.20                 # defaults to --on-threshold if omitted
--on-windows-required 3
--off-windows-required 7
--torch-num-threads 1
```

`--command` supplies the only CLAP text labels — there is no predefined label set.
For example, `--command "typing, rubbing hands"` makes CLAP score only `typing` and
`rubbing hands`.

### Running without CLAP (test the audio path only)

```bash
python3 paddle_stream_new.py \
  --no-clap \
  --input-device-name "YOUR INPUT DEVICE NAME" \
  --output-device-name "YOUR OUTPUT DEVICE NAME" \
  --base-gain-db 0
```

This keeps the stream at base gain and never starts the detector process.

### Running the CLAP detector by itself (debugging)

```bash
python3 live_clap_detector.py \
  --command "typing, rubbing hands" \
  --input-device "YOUR INPUT DEVICE NAME" \
  --channels 2

# list detector devices only:
python3 live_clap_detector.py --list-devices
```

---

## Detector concepts

These apply to both the web UI and the command line.

### Scores are not probabilities

CLAP scores are cosine similarities between the audio embedding and the text-label
embeddings. They are good for ranking and thresholding, but they are **not
calibrated probabilities** and do not sum to 1. Use them empirically: observe the
scores for target and non-target sounds, then tune the threshold.

### "Neutral" is not a CLAP label

`neutral` is never embedded, scored, or passed to CLAP — it is display/control logic
only. A channel is shown as `neutral` when none of its target scores reaches the
threshold. For example, with `--on-threshold 0.20`:

```text
hand1: neutral (scores: typing=0.151, rubbing hands=0.086)
```

means neither target score reached `0.20` for that channel.

### Channels / "hands"

Each input channel is analyzed independently. The code refers to them as `hand1`,
`hand2`, etc. (the project originally used two wrist microphones). `--detector-channels`
controls how many logical channels CLAP analyzes; it defaults to the input channel
count.

### Threshold behavior (hysteresis)

```text
Boost is OFF:
  turn ON when any channel's max target score >= on_threshold

Boost is ON:
  stay ON while any channel's max target score >= off_threshold
  turn OFF when all channels drop below off_threshold
```

If `--off-threshold` is omitted it equals `--on-threshold`. For simple behavior,
set only `--on-threshold`. For hysteresis (harder to trigger, easier to keep on),
set a lower off-threshold:

```bash
--on-threshold 0.25 \
--off-threshold 0.18
```

### Window and debounce settings

```bash
--window-seconds 1.0          # how much audio CLAP sees per prediction
--hop-seconds 1.0             # how often a new prediction is attempted
--on-windows-required 3       # consecutive positive windows before boost turns ON
--off-windows-required 7      # consecutive neutral windows before boost turns OFF
```

With a 1.0 s window and 1.0 s hop, the window counts map roughly to seconds: ~3
consecutive seconds of detection to turn the boost on, ~7 to turn it off. Larger
windows and higher "required" counts make detection more stable but less responsive;
smaller values do the opposite. These affect detection timing only — they do not add
latency to the audio path, which is owned by Pedalboard in the main process.

More responsive:

```bash
--window-seconds 1.0 --hop-seconds 0.5 \
--on-windows-required 1 --off-windows-required 1
```

More stable:

```bash
--window-seconds 2.0 --hop-seconds 1.0 \
--on-windows-required 3 --off-windows-required 5
```

### What `torch-num-threads` and the analysis queue do

- **`torch_num_threads`** caps how many CPU threads PyTorch uses for each CLAP
  inference. The default is `1` to keep the model from starving the real-time audio
  thread (which would cause glitches). Raising it makes each score compute faster at
  the cost of more CPU contention.
- **`analysis_queue_size`** is the buffer (default `4`) between the audio capture
  thread and the slower analysis thread. If analysis falls behind and the queue
  fills, the **oldest** pending window is dropped and the newest kept, so detection
  stays in sync with the present rather than drifting behind real time.

---

## Status output (command line)

Example startup:

```text
Starting Pedalboard audio stream as the main low-latency process.
Input device: 'YOUR INPUT DEVICE'; output device: 'YOUR OUTPUT DEVICE'
AudioStream channels: input=2, output=2
Gain: base=0.0 dB, boost=12.0 dB
Starting CLAP detector in a separate process. Audio streaming will not wait for CLAP.
```

When CLAP changes the boost state:

```text
AUDIO GAIN -> BOOST (12.0 dB)
AUDIO GAIN -> base (0.0 dB)
```

Detector status lines:

```text
CLAP=base | hand1: neutral (scores: typing=0.151, rubbing hands=0.086) | hand2: typing=0.237 (scores: typing=0.237, rubbing hands=0.094)
```

- `CLAP=base`: the detector is not requesting boost.
- `hand1: neutral`: channel 1 did not meet the threshold.
- `hand2: typing=0.237`: channel 2's top target is `typing` and it exceeded the threshold.
- `scores: ...`: raw CLAP cosine similarities for each target activity.

---

## Common command-line recipes

Standard run:

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --input-device-name "YOUR INPUT DEVICE NAME" \
  --output-device-name "YOUR OUTPUT DEVICE NAME" \
  --base-gain-db 0 \
  --boost-gain-db 12
```

Stricter threshold:

```bash
python3 paddle_stream_new.py --command "typing, rubbing hands" --on-threshold 0.30
```

Hysteresis:

```bash
python3 paddle_stream_new.py --command "typing, rubbing hands" \
  --on-threshold 0.25 --off-threshold 0.18
```

More stable detection:

```bash
python3 paddle_stream_new.py --command "typing, rubbing hands" \
  --on-windows-required 3 --off-windows-required 7
```

Faster detection updates:

```bash
python3 paddle_stream_new.py --command "typing, rubbing hands" \
  --window-seconds 1.0 --hop-seconds 0.5
```

---

## Troubleshooting

### `No module named 'pedalboard'` (or `torch`, `msclap`, …)

The virtual environment is not active or the dependencies are not installed. Run:

```bash
python3 -m pip install numpy sounddevice soundfile torch msclap pedalboard
```

### `unrecognized arguments: --command ...`

You are running an older streaming-only script. The main program must accept
`--command`, `--base-gain-db`, and `--boost-gain-db`. Confirm with:

```bash
grep -nE -- "--command|--base-gain-db|--boost-gain-db" paddle_stream_new.py
```

### Startup seems to hang for 10–15 seconds

That is expected — the CLAP model is loading (and downloading on first run). The web
UI shows **"Loading…"** during this time.

### No boost is happening

Check the status. If every channel is `neutral`, lower the threshold or reword the
prompt:

```bash
--on-threshold 0.15
```

Confirm the detector opened input:

```text
CLAP detector opened input: physical_channels=2, logical_hands=2
```

### Headphone audio has latency

The audio path is Pedalboard `AudioStream`. If latency persists, check hardware:

- avoid Bluetooth headphones; use wired
- leave `--buffer-size` unset first; if experimenting, try `--buffer-size 128`

### CLAP detector fails but audio still plays

That is by design — the audio stream is independent and continues with the last gain
state. Try running the detector alone (see
[Running the CLAP detector by itself](#running-the-clap-detector-by-itself-debugging)).
If it fails, use `--list-devices` and pass a sounddevice-compatible input via
`--detector-input-device`.

### Only one channel ("hand") appears

Ensure two logical channels and check the startup message:

```bash
--detector-channels 2
```

```text
CLAP detector opened input: physical_channels=2, logical_hands=2
```

If it fell back to mono, the detector could not open a two-channel input. Use a true
two-channel device or a macOS Aggregate Device.

### The mic is used by two readers at once

The audio path and the CLAP detector are intentionally independent and may both read
the same physical input device. On many setups this works, but if a driver does not
allow two readers, one of them may fail. Workarounds:

- use a macOS Aggregate Device (Audio MIDI Setup)
- pass a separate detector device via `--detector-input-device`
- test `paddle_stream_new.py --no-clap` first, then `live_clap_detector.py` alone

### `input overflow` in CLAP status

This refers to the detector input, not the main audio stream — detection may drop
windows but headphone audio should continue. Try:

```bash
--torch-num-threads 1 --window-seconds 1.0 --hop-seconds 0.5
```

or reduce other CPU load.

---

## Development notes

- `paddle_stream_new.py` is the core engine and command-line program; `app.py`
  drives the same `StreamController`.
- `paddle_stream_new.py` owns all output; `live_clap_detector.py` never applies gain
  and never opens an output stream.
- The only communication from CLAP to the stream is simple shared state: boost on or
  off.
- The audio stream never waits for CLAP inference. If CLAP falls behind, analysis
  windows are dropped; audio playback continues normally.
</content>
</invoke>
