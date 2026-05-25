# Two-Hand CLAP Activity Amplifier

This project runs a live audio pass-through system that listens to two wrist microphone channels, scores each hand independently with CLAP, and applies a gain boost when either hand appears to be producing one of the target activities you specify.

The current version consists of two files that should live in the same directory:

```text
live_clap_amplifier_pedalboard_gain_robust.py
paddle_stream.py
```

`live_clap_amplifier_pedalboard_gain_robust.py` handles live audio streaming, CLAP scoring, two-hand detection logic, thresholds, and status printing. `paddle_stream.py` provides the gain processor used by the amplifier. It uses `pedalboard.Gain(gain_db=...)` when the `pedalboard` package is installed, and falls back to equivalent NumPy dB gain when `pedalboard` is unavailable.

## What it does

At runtime, the script:

1. Opens a live microphone-to-headphone audio stream.
2. Treats the input channels as hand microphones, usually:
   - `hand1` = input channel 1
   - `hand2` = input channel 2
3. Uses only the target activities entered by the user as CLAP text labels.
4. Scores each hand independently against those target labels.
5. Prints the top prediction and all target scores for each hand.
6. Applies base gain when no target is active.
7. Applies boost gain when any hand's target score meets the activity threshold.

Example target labels:

```text
typing, rubbing hands
```

These become the only CLAP labels. There is no predefined output-label dataset.

## Neutral is not a CLAP label

`neutral` is not embedded or scored by CLAP. It is display and control logic only.

A hand is shown as `neutral` when its best target score is below the activity threshold.

For example, with:

```bash
--on-threshold 0.20
```

this output:

```text
hand1: neutral (scores: typing=0.151, rubbing hands=0.086)
```

means that none of hand1's target scores reached `0.20`.

## Scores are not probabilities

The displayed CLAP scores are cosine similarity scores between the audio embedding and the text-label embeddings. They are useful for ranking and thresholding, but they are not calibrated probabilities and they do not sum to 1.

Use them empirically: observe score ranges for target and non-target audio, then tune thresholds.

## Installation

Create or activate your Python environment, then install the required packages.

```bash
python -m pip install numpy sounddevice soundfile scipy torch msclap
```

Optional, but recommended if you want the actual Pedalboard gain backend:

```bash
python -m pip install pedalboard
```

If `pedalboard` is missing, the script still runs using a NumPy dB-gain fallback. At startup, the script prints which backend is active:

```text
Gain processing: base_gain_db=0.0, boost_gain_db=12.0, backend=pedalboard
```

or:

```text
Gain processing: base_gain_db=0.0, boost_gain_db=12.0, backend=numpy-db-gain
```

## Quick start

Basic run:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands"
```

Recommended explicit device run:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --input-device "Sonic Presence SP-15 V2.0" \
  --output-device "External Headphones" \
  --channels 2 \
  --output-channels 2 \
  --on-threshold 0.20 \
  --base-gain-db 0 \
  --boost-gain-db 12
```

Stop with `Ctrl+C`.

## Listing devices

To see available input and output devices:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py --list-devices
```

Then pass either a device name or a device index:

```bash
--input-device "Sonic Presence SP-15 V2.0"
--output-device "External Headphones"
```

or:

```bash
--input-device 2
--output-device 5
```

Use `--input-device` and `--output-device` when the wrist microphones and headphones are separate devices. Use `--device` only when one device handles both input and output.

## Important channel behavior

By default, the script expects two logical hand channels:

```bash
--channels 2
```

This means CLAP scores two hand streams independently.

The output channel count is separate:

```bash
--output-channels 2
```

If your headphones or output device are mono, use:

```bash
--output-channels 1
```

while keeping:

```bash
--channels 2
```

for two-hand analysis.

If the selected input device cannot open as 2-channel input, the script may fall back to opening a 1-channel physical input and duplicating it to both `hand1` and `hand2`. This keeps the script running, but it does not create two independent wrist-mic signals. For real two-hand sensing, use a true stereo/two-channel input device, or on macOS create an Aggregate Device that exposes both microphones as separate channels.

To disable the mono fallback and force failure when real two-channel input is unavailable:

```bash
--no-mono-input-fallback
```

## Gain control

Gain is controlled in dB, using the helper in `paddle_stream.py`.

Default values:

```bash
--base-gain-db 0
--boost-gain-db 12
```

Meaning:

- `0 dB` = unchanged audio level
- `+12 dB` = boosted audio level

The old linear gain flags are still accepted internally as hidden legacy aliases, but new runs should use dB:

```bash
--base-gain-db 0
--boost-gain-db 12
```

## Detection thresholds

Default activity threshold:

```bash
--on-threshold 0.20
```

If `--off-threshold` is omitted, it is automatically set equal to `--on-threshold`. That means boost stays on as long as at least one hand remains above the activity threshold.

Default behavior:

```text
Gain is OFF:
  turn ON when any hand's max target score >= on_threshold

Gain is ON:
  stay ON while any hand's max target score >= off_threshold
  turn OFF when all hands are below off_threshold
```

For simple behavior, use the same threshold for on/off:

```bash
--on-threshold 0.20
```

For hysteresis, use a lower off threshold:

```bash
--on-threshold 0.25 \
--off-threshold 0.18
```

That makes boost harder to turn on but easier to keep on once active.

## Window and debounce settings

The script analyzes recent audio windows with CLAP.

Common parameters:

```bash
--window-seconds 2.0
--hop-seconds 1.0
--on-windows-required 1
--off-windows-required 1
```

Meaning:

- `window-seconds`: how much audio CLAP sees for each prediction.
- `hop-seconds`: how often a new CLAP prediction is made.
- `on-windows-required`: how many consecutive positive windows are required before turning boost on.
- `off-windows-required`: how many consecutive neutral windows are required before turning boost off.

More responsive but potentially noisier:

```bash
--window-seconds 1.0 \
--hop-seconds 0.5 \
--on-windows-required 1 \
--off-windows-required 1
```

More stable but slower:

```bash
--window-seconds 2.0 \
--hop-seconds 1.0 \
--on-windows-required 2 \
--off-windows-required 2
```

## Audio latency and buffering

The script uses `sounddevice` for live audio I/O. Default latency is conservative:

```bash
--latency 0.10
```

This is often more stable while CLAP inference is also running. For lower latency, try:

```bash
--latency 0.05
```

or:

```bash
--latency low
```

If you get `input overflow`, increase latency rather than lowering it:

```bash
--latency 0.15
```

or:

```bash
--latency 0.20
```

By default, `blocksize` is left to the audio host API:

```bash
--blocksize 0
```

You can force a callback size with either:

```bash
--blocksize 512
```

or:

```bash
--block-duration 0.02
```

For real-time monitoring, wired headphones and a wired audio interface are strongly recommended. Bluetooth output can add noticeable delay independent of CLAP.

## Status output

Example output:

```text
gain=0.0 dB (base) | hand1: neutral (scores: typing=0.151, rubbing hands=0.086) | hand2: typing=0.237 (scores: typing=0.237, rubbing hands=0.094)
```

Interpretation:

- `gain=0.0 dB (base)`: no activity-triggered boost is active.
- `hand1: neutral`: hand1 did not meet the activity threshold.
- `hand2: typing=0.237`: hand2's top score is typing, and it exceeded the threshold.
- `scores: ...`: raw CLAP cosine similarity scores for every target activity.

You may also see:

```text
audio status: input overflow x2
```

This means the audio input buffer overflowed. Increase `--latency`, reduce CPU load, close other audio apps, or use a more stable audio interface.

You may see:

```text
dropped analysis blocks: 3
```

This means CLAP inference could not keep up with every queued analysis block, so stale blocks were discarded to keep predictions real-time. Audio pass-through still continues.

## Common commands

Two-hand default run:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands"
```

Two-hand run with explicit devices:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --input-device "Sonic Presence SP-15 V2.0" \
  --output-device "External Headphones"
```

Boost by +12 dB when activity is detected:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --base-gain-db 0 \
  --boost-gain-db 12
```

Use a stricter threshold:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --on-threshold 0.30
```

Use hysteresis:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --on-threshold 0.25 \
  --off-threshold 0.18
```

More stable detection:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --on-windows-required 2 \
  --off-windows-required 2
```

Lower-latency attempt:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --latency 0.05 \
  --window-seconds 1.0 \
  --hop-seconds 0.5
```

More stable audio if you see overflows:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py \
  --command "typing, rubbing hands" \
  --latency 0.20 \
  --torch-num-threads 1
```

## Troubleshooting

### Only one hand is printed

Make sure you are running the latest script and that `--channels` is 2:

```bash
--channels 2
```

Also check the startup line:

```text
Logical hand channels: 2; physical input channels opened: 2
```

If it says physical input channels opened: 1, the selected input device is mono or the script fell back to mono input.

### `Invalid number of channels`

The selected/default input or output device does not support the requested channel count.

Try:

```bash
python live_clap_amplifier_pedalboard_gain_robust.py --list-devices
```

Then select devices explicitly:

```bash
--input-device "YOUR TWO-CHANNEL INPUT"
--output-device "YOUR HEADPHONES"
```

If your output device is mono:

```bash
--output-channels 1
```

### `No module named 'pedalboard'`

Install Pedalboard:

```bash
python -m pip install pedalboard
```

The robust script can still run without Pedalboard by using the NumPy fallback, but if your helper is not import-safe or is an older version, replace it with the updated `paddle_stream.py`.

### `input overflow`

The audio callback is not receiving input quickly enough. Try:

```bash
--latency 0.15
```

or:

```bash
--latency 0.20
```

Also try closing CPU-heavy apps and keeping:

```bash
--torch-num-threads 1
```

### CLAP predictions are delayed

Lower `--window-seconds` and `--hop-seconds`:

```bash
--window-seconds 1.0 --hop-seconds 0.5
```

This affects detection timing, not raw audio monitoring latency.

### Headphone audio itself is delayed

That is audio I/O latency, not CLAP. Avoid Bluetooth headphones, use wired headphones, select the correct input/output devices, and tune `--latency`.

## Notes for development

- Keep `paddle_stream.py` import-safe. It should not open an audio stream at import time.
- Do not print from the audio callback. Printing in the real-time callback can worsen overflows and corrupt terminal output.
- CLAP analysis runs in a background thread. If the analysis queue fills, stale blocks are dropped so predictions remain roughly current.
- The audio stream still passes through immediately even if CLAP analysis falls behind.
- The gain backend should expose a `PaddleGainProcessor` class with a `process(audio, gain_db)` method.
