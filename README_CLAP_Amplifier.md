# Two-Hand CLAP-Controlled Pedalboard Amplifier

This project runs a low-latency live audio pass-through system for two wrist microphones. The main audio stream is handled by Pedalboard's `AudioStream`, while CLAP runs separately and only decides whether the stream should use base gain or boosted gain.

The current version uses two files in the same directory:

```text
paddle_stream_new.py      # main program; owns live audio streaming
live_clap_detector.py     # CLAP-only detector worker
```

The important design change is that `paddle_stream_new.py` is the main file. It owns the microphone-to-headphone audio path and changes the Pedalboard plugin chain between base gain and boost gain. `live_clap_detector.py` runs separately as an input-only detector process. Audio playback never waits for CLAP predictions.

## Why this design exists

Earlier versions put audio streaming, CLAP inference, gain control, and status printing in the same live CLAP amplifier script. That worked functionally, but it introduced noticeable monitoring latency because the real-time audio path was too tightly coupled to the CLAP system.

The current design separates responsibilities:

```text
paddle_stream_new.py
  -> opens the low-latency Pedalboard AudioStream
  -> streams wrist-mic input to headphones
  -> applies Pedalboard Gain at either base or boost level
  -> polls a shared CLAP state flag

live_clap_detector.py
  -> opens an independent input-only sounddevice stream
  -> runs CLAP on recent audio windows
  -> scores each hand independently
  -> publishes only boost on/off plus status messages
```

If CLAP is slow, stalls, or crashes, the Pedalboard audio stream keeps running with the most recent gain state.

## What it does

At runtime:

1. `paddle_stream_new.py` opens a Pedalboard `AudioStream` for the live audio path.
2. It starts `live_clap_detector.py` in a separate Python process.
3. The detector listens to the wrist-mic input independently.
4. Each input channel is treated as one hand:
   - `hand1` = input channel 1
   - `hand2` = input channel 2
5. The detector uses only the target activities entered by the user as CLAP text labels.
6. The detector publishes a boolean state: target active or not active.
7. The main Pedalboard stream switches between:
   - base gain, usually `0 dB`
   - boost gain, usually `+12 dB`

## Neutral is not a CLAP label

`neutral` is not embedded, scored, or passed to CLAP. It is display and control logic only.

A hand is shown as `neutral` when none of its target scores reaches the activity threshold.

For example, with:

```bash
--on-threshold 0.20
```

this detector status:

```text
hand1: neutral (scores: typing=0.151, rubbing hands=0.086)
```

means that neither target score reached `0.20` for `hand1`.

## Scores are not probabilities

The CLAP scores are cosine similarity scores between the audio embedding and the text-label embeddings. They are useful for ranking and thresholding, but they are not calibrated probabilities and they do not sum to 1.

Use the scores empirically. Record or observe scores for target and non-target sounds, then tune the threshold.

## Installation

Activate your project environment and install the dependencies:

```bash
python3 -m pip install numpy sounddevice soundfile torch msclap pedalboard
```

You may already have these installed in your `clap311` environment. From your project root:

```bash
cd /Users/magniac/Mindful_Audio_Research
source clap311/bin/activate
```

## Quick start

From `/Users/magniac/Mindful_Audio_Research`, run:

```bash
cd /Users/magniac/Mindful_Audio_Research && \
source clap311/bin/activate && \
cd CLAP && \
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --input-device-name "Sonic Presence SP-15 V2.0" \
  --output-device-name "External Headphones" \
  --base-gain-db 0 \
  --boost-gain-db 12
```

Stop with `Ctrl+C`.

## Listing devices

Use:

```bash
python3 paddle_stream_new.py --list-devices
```

This prints two device lists:

1. Pedalboard input/output devices for the main audio stream.
2. `sounddevice` devices for the CLAP detector.

Usually you should set:

```bash
--input-device-name "Sonic Presence SP-15 V2.0"
--output-device-name "External Headphones"
```

These are Pedalboard `AudioStream` device names.

If the CLAP detector needs a different device name or index than Pedalboard uses, pass:

```bash
--detector-input-device "DEVICE_NAME_OR_INDEX"
```

## Main command-line options

### Audio stream options

```bash
--input-device-name "Sonic Presence SP-15 V2.0"
--output-device-name "External Headphones"
--input-channels 2
--output-channels 2
--buffer-size 128
--allow-feedback
```

`--input-device-name` and `--output-device-name` control the Pedalboard stream. This is the real-time monitoring path.

`--buffer-size` is optional. Leave it unset first, because the original low-latency Pedalboard stream worked without forcing a buffer size. Try `--buffer-size 128` only if you need to experiment.

### Gain options

```bash
--base-gain-db 0
--boost-gain-db 12
```

`0 dB` means unchanged audio level. `+12 dB` is a strong boost.

The main stream switches Pedalboard plugins when the CLAP detector changes state:

```text
base state  -> Pedalboard([Gain(gain_db=base_gain_db)])
boost state -> Pedalboard([Gain(gain_db=boost_gain_db)])
```

### CLAP detector options

```bash
--command "typing, rubbing hands"
--detector-input-device "Sonic Presence SP-15 V2.0"
--detector-channels 2
--sample-rate 48000
--window-seconds 2.0
--hop-seconds 1.0
--on-threshold 0.20
--off-threshold 0.20
--on-windows-required 1
--off-windows-required 1
--torch-num-threads 1
```

`--command` supplies the only CLAP text labels. For example:

```bash
--command "typing, rubbing hands"
```

means CLAP only scores:

```text
typing
rubbing hands
```

There is no predefined label dataset.

`--detector-channels` controls how many logical hand channels CLAP analyzes. It defaults to the main input channel count.

## Threshold behavior

Default threshold:

```bash
--on-threshold 0.20
```

If `--off-threshold` is omitted, it defaults to the same value as `--on-threshold`. That means boost stays on as long as at least one hand remains above the activity threshold.

Default behavior:

```text
Boost is OFF:
  turn ON when any hand's max target score >= on_threshold

Boost is ON:
  stay ON while any hand's max target score >= off_threshold
  turn OFF when all hands are below off_threshold
```

For simple behavior, use only:

```bash
--on-threshold 0.20
```

For hysteresis, set a lower off-threshold:

```bash
--on-threshold 0.25 \
--off-threshold 0.18
```

That makes boost harder to trigger, but easier to keep active once triggered.

## Window and debounce settings

The detector analyzes recent audio windows with CLAP.

Common defaults:

```bash
--window-seconds 2.0
--hop-seconds 1.0
--on-windows-required 1
--off-windows-required 1
```

Meaning:

- `window-seconds`: how much audio CLAP sees for each prediction.
- `hop-seconds`: how often the detector attempts a new prediction.
- `on-windows-required`: how many consecutive positive windows are required before boost turns on.
- `off-windows-required`: how many consecutive neutral windows are required before boost turns off.

More responsive detection:

```bash
--window-seconds 1.0 \
--hop-seconds 0.5 \
--on-windows-required 1 \
--off-windows-required 1
```

More stable detection:

```bash
--window-seconds 2.0 \
--hop-seconds 1.0 \
--on-windows-required 2 \
--off-windows-required 2
```

These settings affect detection timing only. They should not add latency to the headphone audio path, because audio streaming is owned by Pedalboard in the main process.

## Status output

Example startup output:

```text
Starting Pedalboard audio stream as the main low-latency process.
Input device: 'Sonic Presence SP-15 V2.0'; output device: 'External Headphones'
AudioStream channels: input=2, output=2
Gain: base=0.0 dB, boost=12.0 dB
Starting CLAP detector in a separate process. Audio streaming will not wait for CLAP.
```

When CLAP changes the boost state, the main process prints:

```text
AUDIO GAIN -> BOOST (12.0 dB)
```

or:

```text
AUDIO GAIN -> base (0.0 dB)
```

Detector status messages look like:

```text
CLAP=base | hand1: neutral (scores: typing=0.151, rubbing hands=0.086) | hand2: typing=0.237 (scores: typing=0.237, rubbing hands=0.094)
```

Interpretation:

- `CLAP=base`: the detector is not requesting boost.
- `hand1: neutral`: hand1 did not meet the activity threshold.
- `hand2: typing=0.237`: hand2's top target score is typing and it exceeded the threshold.
- `scores: ...`: raw CLAP cosine similarity scores for each target activity.

## Running without CLAP

To test only the Pedalboard stream path:

```bash
python3 paddle_stream_new.py \
  --no-clap \
  --input-device-name "Sonic Presence SP-15 V2.0" \
  --output-device-name "External Headphones" \
  --base-gain-db 0
```

This keeps the stream at base gain and does not start the detector process.

## Running the CLAP detector by itself

For debugging CLAP without starting Pedalboard output:

```bash
python3 live_clap_detector.py \
  --command "typing, rubbing hands" \
  --input-device "Sonic Presence SP-15 V2.0" \
  --channels 2
```

To list detector devices only:

```bash
python3 live_clap_detector.py --list-devices
```

## Important device note

The main audio path and the CLAP detector are intentionally independent. That means they may both try to read from the same physical input device:

```text
Pedalboard AudioStream reads the mic for live monitoring.
sounddevice InputStream reads the mic for CLAP detection.
```

On many macOS setups this can work, but if the device or driver does not allow two readers at once, the Pedalboard stream may work while the CLAP detector fails, or vice versa.

If this happens, try one of these:

- Use an Aggregate Device in macOS Audio MIDI Setup.
- Use a separate detector input device.
- Pass a specific detector device with `--detector-input-device`.
- Test `paddle_stream_new.py --no-clap` first, then test `live_clap_detector.py` by itself.

## Common commands

### Standard two-hand run

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --input-device-name "Sonic Presence SP-15 V2.0" \
  --output-device-name "External Headphones"
```

### Standard run from project root

```bash
cd /Users/magniac/Mindful_Audio_Research && \
source clap311/bin/activate && \
cd CLAP && \
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --input-device-name "Sonic Presence SP-15 V2.0" \
  --output-device-name "External Headphones" \
  --base-gain-db 0 \
  --boost-gain-db 12
```

### Use a stricter CLAP threshold

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --on-threshold 0.30
```

### Use hysteresis

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --on-threshold 0.25 \
  --off-threshold 0.18
```

### More stable detection

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --on-windows-required 2 \
  --off-windows-required 2
```

### Faster detection updates

```bash
python3 paddle_stream_new.py \
  --command "typing, rubbing hands" \
  --window-seconds 1.0 \
  --hop-seconds 0.5
```

### Test stream only

```bash
python3 paddle_stream_new.py \
  --no-clap \
  --input-device-name "Sonic Presence SP-15 V2.0" \
  --output-device-name "External Headphones"
```

## Troubleshooting

### `unrecognized arguments: --command ...`

You are running an older streaming-only file. The new main script must accept:

```text
--command
--base-gain-db
--boost-gain-db
```

Check with:

```bash
grep -nE -- "--command|--base-gain-db|--boost-gain-db" paddle_stream_new.py
```

### No boost is happening

Check the CLAP status messages. If every hand is `neutral`, lower the threshold or change the prompt wording:

```bash
--on-threshold 0.15
```

Also confirm that the detector actually started and opened input:

```text
CLAP detector opened input: physical_channels=2, logical_hands=2
```

### Headphone audio has latency

The main audio path is now Pedalboard `AudioStream`, matching the original low-latency design. If latency remains, check hardware first:

- avoid Bluetooth headphones
- use wired headphones
- use the same devices that worked in the original `paddle_stream.py`
- leave `--buffer-size` unset at first
- if experimenting, try `--buffer-size 128`

### CLAP detector fails but audio still plays

That is expected for this design. The main stream is independent and continues with the last gain state.

Try running the detector alone:

```bash
python3 live_clap_detector.py \
  --command "typing, rubbing hands" \
  --input-device "Sonic Presence SP-15 V2.0" \
  --channels 2
```

If that fails, use `--list-devices` and pass a sounddevice-compatible input name or index via `--detector-input-device` in the main script.

### Only one hand appears in CLAP status

Make sure the detector is using two logical channels:

```bash
--detector-channels 2
```

Also check the detector startup message:

```text
CLAP detector opened input: physical_channels=2, logical_hands=2
```

If it says it fell back to mono, the detector could not open a two-channel input stream. Use a true two-channel input device or macOS Aggregate Device.

### `No module named 'pedalboard'`

Install Pedalboard in the active environment:

```bash
python3 -m pip install pedalboard
```

Unlike the older robust CLAP-streaming script, the new main program uses Pedalboard directly for the audio stream, so `pedalboard` is required.

### `input overflow` in CLAP status

This refers to the detector input stream, not the main Pedalboard audio stream. It means CLAP detection may drop or skip input, but live headphone audio should continue.

Try:

```bash
--torch-num-threads 1
--window-seconds 1.0
--hop-seconds 0.5
```

or reduce other CPU load.

## Development notes

- `paddle_stream_new.py` should remain the main program.
- `paddle_stream_new.py` should own all headphone output.
- `live_clap_detector.py` should never apply gain and should never open an output stream.
- The only communication from CLAP to the stream should be simple shared state: boost on or boost off.
- The audio stream should never wait for CLAP inference.
- If CLAP falls behind, detector analysis blocks can be dropped; audio playback should continue normally.
