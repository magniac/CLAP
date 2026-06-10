from pedalboard import (
    Pedalboard,
    Chorus,
    Compressor,
    Delay,
    Gain,
    Reverb,
    Phaser,
    PitchShift,
    HighpassFilter,
    LowpassFilter,
    Distortion
)
from pedalboard.io import AudioStream

INPUT_DEVICE_NAMES = [n for n in AudioStream.input_device_names]

OUTPUT_DEVICE_NAMES = [n for n in AudioStream.output_device_names]


print(INPUT_DEVICE_NAMES)
print(OUTPUT_DEVICE_NAMES)

# Open up an audio stream:
with AudioStream(
    input_device_name="Sonic Presence SP-15 V2.0",  # Guitar interface
    output_device_name="External Headphones",
    num_input_channels=2,                           # Force mono input
    num_output_channels=2,   
    # allow_feedback=True,
    # buffer_size=128,
) as stream:
    # Audio is now streaming through this pedalboard and out of your speakers!
    stream.plugins = Pedalboard(
        [
            # Delay(delay_seconds=0.5, mix=1.0),
            # Compressor(threshold_db=-50, ratio=25),
            Gain(gain_db=12),
            # HighpassFilter(cutoff_frequency_hz=500),
            # PitchShift(semitones=7),
            # LowpassFilter(cutoff_frequency_hz=500),
            # LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=900),
            # Reverb(room_size=0, damping=0, wet_level=0, dry_level=1, width=1),
            # Reverb(room_size=0.75, damping=0.5, wet_level=0.4, dry_level=0.6, width=1),
            # Reverb(room_size=1, damping=0.5, wet_level=0.6, dry_level=0.4, width=1),
        ]
    )
    input("Press enter to stop streaming...")
