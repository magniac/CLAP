#!/usr/bin/env python3
"""Low-latency Pedalboard stream controlled by an independent CLAP process.

This is now the main program. It owns the real-time audio path using
pedalboard.io.AudioStream, matching the original low-latency paddle_stream.py
style. CLAP runs separately and only updates a shared boolean saying whether
boost should be active.

The stream never waits for CLAP predictions. If CLAP is slow, crashes, or has
not produced a first prediction yet, the stream keeps playing with the most
recent gain state.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any

from pedalboard import Pedalboard, Gain
from pedalboard.io import AudioStream


def parse_device_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_detector_device_arg(value: str | None):
    """Parse device selector for live_clap_detector.py / sounddevice."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def make_gain_board(gain_db: float) -> Pedalboard:
    return Pedalboard([Gain(gain_db=float(gain_db))])


def _run_detector_process(detector_kwargs: dict[str, Any], active_value, stop_event, status_queue) -> None:
    """Child-process target.

    Import CLAP and its dependencies only in the detector process so the main
    Pedalboard stream process stays lightweight.
    """
    try:
        from live_clap_detector import run_clap_detector

        run_clap_detector(
            active_value=active_value,
            stop_event=stop_event,
            status_queue=status_queue,
            **detector_kwargs,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            status_queue.put_nowait(f"CLAP detector process exited with error: {exc}")
        except Exception:
            pass
        try:
            active_value.value = 0
        except Exception:
            pass


def drain_status_queue(status_queue, max_messages: int = 20) -> list[str]:
    messages: list[str] = []
    for _ in range(max_messages):
        try:
            messages.append(status_queue.get_nowait())
        except queue.Empty:
            break
        except Exception:
            break
    return messages


def print_devices(include_sounddevice: bool = True) -> None:
    print("Pedalboard input devices:")
    for i, name in enumerate(AudioStream.input_device_names):
        print(f"  [{i}] {name}")
    print()
    print("Pedalboard output devices:")
    for i, name in enumerate(AudioStream.output_device_names):
        print(f"  [{i}] {name}")

    if include_sounddevice:
        print()
        print("sounddevice devices for the CLAP detector:")
        try:
            import sounddevice as sd

            print(sd.query_devices())
        except Exception as exc:
            print(f"  Could not query sounddevice devices: {exc}")


def build_audio_stream_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "input_device_name": args.input_device_name,
        "output_device_name": args.output_device_name,
        "num_input_channels": args.input_channels,
        "num_output_channels": args.output_channels,
    }
    if args.buffer_size is not None:
        kwargs["buffer_size"] = args.buffer_size
    if args.allow_feedback:
        kwargs["allow_feedback"] = True
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-latency Pedalboard audio stream controlled by an independent "
            "CLAP activity detector process."
        )
    )
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "Target activities, separated by commas or 'and'. Example: "
            "\"typing, rubbing hands\". These are the only CLAP labels."
        ),
    )
    parser.add_argument(
        "--input-device-name",
        default="Sonic Presence SP-15 V2.0",
        help="Pedalboard AudioStream input device name for the wrist microphones.",
    )
    parser.add_argument(
        "--output-device-name",
        default="External Headphones",
        help="Pedalboard AudioStream output device name for headphones.",
    )
    parser.add_argument("--input-channels", type=int, default=2)
    parser.add_argument("--output-channels", type=int, default=2)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help=(
            "Optional Pedalboard/AudioStream buffer size. Leave unset to match "
            "the original low-latency AudioStream behavior. Try 128 only if needed."
        ),
    )
    parser.add_argument("--allow-feedback", action="store_true")

    parser.add_argument("--base-gain-db", type=float, default=0.0)
    parser.add_argument("--boost-gain-db", type=float, default=12.0)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.02,
        help="How often paddle_stream.py checks the CLAP shared boost state.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="How often to print stream/detector status messages.",
    )

    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument(
        "--detector-input-device",
        default=None,
        help=(
            "Optional sounddevice input device for CLAP. Defaults to "
            "--input-device-name. Use this if sounddevice names/indices differ."
        ),
    )
    parser.add_argument(
        "--detector-channels",
        type=int,
        default=None,
        help="Logical hand channels for CLAP. Defaults to --input-channels.",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--analysis-buffer-seconds", type=float, default=None)
    parser.add_argument("--analysis-queue-size", type=int, default=4)
    parser.add_argument("--on-threshold", type=float, default=0.20)
    parser.add_argument(
        "--off-threshold",
        type=float,
        default=None,
        help="Defaults to --on-threshold, so boost stays on while any hand remains above threshold.",
    )
    parser.add_argument("--on-windows-required", type=int, default=1)
    parser.add_argument("--off-windows-required", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--no-mono-input-fallback", action="store_true")
    parser.add_argument("--no-clap", action="store_true", help="Run the Pedalboard stream without starting CLAP.")
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_devices(include_sounddevice=True)
        return 0

    command = args.command
    if not command and not args.no_clap:
        command = input("What activities should trigger boost? Separate multiple activities with commas: ").strip()
    if not command and not args.no_clap:
        print("ERROR: no target activities provided.", file=sys.stderr)
        return 2

    input_device_name = parse_device_name(args.input_device_name)
    output_device_name = parse_device_name(args.output_device_name)
    if input_device_name is None or output_device_name is None:
        print("ERROR: input and output device names are required for AudioStream.", file=sys.stderr)
        return 2

    base_board = make_gain_board(args.base_gain_db)
    boost_board = make_gain_board(args.boost_gain_db)

    ctx = mp.get_context("spawn")
    active_value = ctx.Value("b", 0)
    stop_event = ctx.Event()
    status_queue = ctx.Queue(maxsize=100)
    detector_process = None

    detector_input_device = parse_detector_device_arg(
        args.detector_input_device if args.detector_input_device is not None else args.input_device_name
    )
    detector_channels = args.detector_channels if args.detector_channels is not None else args.input_channels

    if not args.no_clap:
        detector_kwargs = {
            "command": command,
            "sample_rate": args.sample_rate,
            "channels": detector_channels,
            "input_device": detector_input_device,
            "window_seconds": args.window_seconds,
            "hop_seconds": args.hop_seconds,
            "analysis_buffer_seconds": args.analysis_buffer_seconds,
            "analysis_queue_size": args.analysis_queue_size,
            "on_threshold": args.on_threshold,
            "off_threshold": args.off_threshold,
            "on_windows_required": args.on_windows_required,
            "off_windows_required": args.off_windows_required,
            "torch_num_threads": args.torch_num_threads,
            "allow_mono_input_fallback": not args.no_mono_input_fallback,
            "status_interval": args.status_interval,
        }
        detector_process = ctx.Process(
            target=_run_detector_process,
            args=(detector_kwargs, active_value, stop_event, status_queue),
            daemon=True,
        )

    stream_kwargs = build_audio_stream_kwargs(args)
    print("Starting Pedalboard audio stream as the main low-latency process.")
    print(f"Input device: {input_device_name!r}; output device: {output_device_name!r}")
    print(f"AudioStream channels: input={args.input_channels}, output={args.output_channels}")
    print(f"Gain: base={args.base_gain_db:.1f} dB, boost={args.boost_gain_db:.1f} dB")
    if args.no_clap:
        print("CLAP detector disabled; stream will stay at base gain.")
    else:
        print("Starting CLAP detector in a separate process. Audio streaming will not wait for CLAP.")
        print(f"CLAP command labels: {command}")
        print(
            f"CLAP detector input: device={detector_input_device!r}, logical_channels={detector_channels}, "
            f"threshold={args.on_threshold:.3f}"
        )
    print("Press Ctrl+C to stop.\n")

    try:
        with AudioStream(**stream_kwargs) as stream:
            # Start clean at base gain before CLAP has produced any decision.
            stream.plugins = base_board
            current_active = False

            if detector_process is not None:
                detector_process.start()

            last_status_print = 0.0
            while True:
                requested_active = bool(active_value.value)
                if requested_active != current_active:
                    current_active = requested_active
                    stream.plugins = boost_board if current_active else base_board
                    print(
                        f"AUDIO GAIN -> {'BOOST' if current_active else 'base'} "
                        f"({args.boost_gain_db if current_active else args.base_gain_db:.1f} dB)",
                        flush=True,
                    )

                now = time.time()
                if now - last_status_print >= args.status_interval:
                    last_status_print = now
                    for message in drain_status_queue(status_queue):
                        print(message, flush=True)
                    if detector_process is not None and not detector_process.is_alive():
                        print("WARNING: CLAP detector process is not running; audio stream continues with last gain state.", flush=True)
                        detector_process = None

                time.sleep(max(0.001, float(args.poll_interval)))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if detector_process is not None and detector_process.is_alive():
            detector_process.join(timeout=2.0)
            if detector_process.is_alive():
                detector_process.terminate()
                detector_process.join(timeout=1.0)
        print("Stopped.")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
