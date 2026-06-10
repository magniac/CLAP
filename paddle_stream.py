from __future__ import annotations

"""Import-safe Pedalboard gain helper for the CLAP amplifier.

This keeps the useful part of the original paddle_stream.py helper -- using
pedalboard.Gain(gain_db=...) -- but avoids opening an AudioStream at import time.
The CLAP script can import PaddleGainProcessor and call process() from its
existing sounddevice callback.
"""

import argparse
from functools import lru_cache
from typing import Any

import numpy as np
from pedalboard import Pedalboard, Gain

try:
    from pedalboard.io import AudioStream
except Exception:  # pragma: no cover
    AudioStream = None  # type: ignore[assignment]


def ensure_2d_float32(audio: np.ndarray) -> np.ndarray:
    """Return audio as shape (frames, channels), dtype float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape (frames,) or (frames, channels), got {audio.shape}.")
    return audio.astype(np.float32, copy=False)


class PaddleGainProcessor:
    """Apply Pedalboard Gain to a real-time audio block.

    Args:
        sample_rate: Audio sample rate in Hz.

    process() expects and returns audio in sounddevice callback format:
    shape (frames, channels). Internally, Pedalboard uses channels-first audio,
    so this class handles the transpose.
    """

    def __init__(self, sample_rate: int | float):
        self.sample_rate = int(sample_rate)
        self._boards: dict[float, Pedalboard] = {}

    def _board_for_gain_db(self, gain_db: float) -> Pedalboard:
        key = round(float(gain_db), 6)
        board = self._boards.get(key)
        if board is None:
            board = Pedalboard([Gain(gain_db=key)])
            self._boards[key] = board
        return board

    def process(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply a Pedalboard Gain(gain_db=...) block and return clipped audio."""
        audio = ensure_2d_float32(audio)
        if audio.size == 0:
            return audio

        # Pedalboard convention is (channels, samples). sounddevice uses
        # (frames, channels). Keep copies contiguous to avoid plugin edge cases.
        channels_first = np.ascontiguousarray(audio.T, dtype=np.float32)
        processed = self._board_for_gain_db(gain_db)(channels_first, self.sample_rate)
        processed = np.asarray(processed, dtype=np.float32)

        if processed.ndim == 1:
            processed = processed[None, :]
        output = np.ascontiguousarray(processed.T, dtype=np.float32)
        if output.shape != audio.shape:
            raise RuntimeError(
                f"Pedalboard gain returned shape {output.shape}, expected {audio.shape}."
            )
        return np.clip(output, -1.0, 1.0)


def input_device_names() -> list[str]:
    if AudioStream is None:
        return []
    return [str(name) for name in AudioStream.input_device_names]


def output_device_names() -> list[str]:
    if AudioStream is None:
        return []
    return [str(name) for name in AudioStream.output_device_names]


def demo_stream(
    input_device_name: str,
    output_device_name: str,
    num_input_channels: int = 2,
    num_output_channels: int = 2,
    gain_db: float = 12.0,
    buffer_size: int | None = None,
) -> None:
    """Run the original standalone Pedalboard demo, but only when called."""
    if AudioStream is None:
        raise RuntimeError("pedalboard.io.AudioStream is unavailable in this environment.")

    kwargs: dict[str, Any] = {
        "input_device_name": input_device_name,
        "output_device_name": output_device_name,
        "num_input_channels": int(num_input_channels),
        "num_output_channels": int(num_output_channels),
    }
    if buffer_size is not None:
        kwargs["buffer_size"] = int(buffer_size)

    with AudioStream(**kwargs) as stream:
        stream.plugins = Pedalboard([Gain(gain_db=float(gain_db))])
        input("Press enter to stop streaming...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Pedalboard gain helper/demo.")
    parser.add_argument("--input-device-name", default=None)
    parser.add_argument("--output-device-name", default=None)
    parser.add_argument("--num-input-channels", type=int, default=2)
    parser.add_argument("--num-output-channels", type=int, default=2)
    parser.add_argument("--gain-db", type=float, default=12.0)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices or args.input_device_name is None or args.output_device_name is None:
        print("Input devices:")
        for name in input_device_names():
            print(f"  {name}")
        print("Output devices:")
        for name in output_device_names():
            print(f"  {name}")
        if args.input_device_name is None or args.output_device_name is None:
            return 0

    demo_stream(
        input_device_name=args.input_device_name,
        output_device_name=args.output_device_name,
        num_input_channels=args.num_input_channels,
        num_output_channels=args.num_output_channels,
        gain_db=args.gain_db,
        buffer_size=args.buffer_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
