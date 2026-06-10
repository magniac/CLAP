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
import threading
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


class StreamController:
    """Owns the live Pedalboard audio path and an optional CLAP detector.

    Designed to be driven programmatically (e.g. from a web server). The audio
    loop runs in a background thread so the caller stays responsive. Boost gain
    is held in a shared value and can be changed live via set_gain().
    """

    def __init__(
        self,
        command: str | None,
        boost_gain_db: float = 12.0,
        base_gain_db: float = 0.0,
        input_device_name: str = "Sonic Presence SP-15 V2.0",
        output_device_name: str = "External Headphones",
        input_channels: int = 2,
        output_channels: int = 2,
        buffer_size: int | None = None,
        allow_feedback: bool = False,
        poll_interval: float = 0.02,
        status_interval: float = 1.0,
        sample_rate: int = 48000,
        detector_input_device=None,
        detector_channels: int | None = None,
        window_seconds: float = 1.0,
        hop_seconds: float = 1.0,
        analysis_buffer_seconds: float | None = None,
        analysis_queue_size: int = 4,
        on_threshold: float = 0.20,
        off_threshold: float | None = None,
        on_windows_required: int = 3,
        off_windows_required: int = 7,
        torch_num_threads: int = 1,
        allow_mono_input_fallback: bool = True,
        enable_clap: bool = True,
    ) -> None:
        self.command = command
        self.base_gain_db = float(base_gain_db)
        self.input_device_name = parse_device_name(input_device_name)
        self.output_device_name = parse_device_name(output_device_name)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.buffer_size = buffer_size
        self.allow_feedback = bool(allow_feedback)
        self.poll_interval = max(0.001, float(poll_interval))
        self.status_interval = max(0.1, float(status_interval))
        self.enable_clap = bool(enable_clap) and bool(command and command.strip())

        if self.input_device_name is None or self.output_device_name is None:
            raise ValueError("input and output device names are required for AudioStream.")

        self.ctx = mp.get_context("spawn")
        self.active_value = self.ctx.Value("b", 0)
        self.gain_value = self.ctx.Value("d", float(boost_gain_db))
        self.stop_event = self.ctx.Event()
        self.status_queue = self.ctx.Queue(maxsize=100)
        self.detector_process = None

        detector_channels = detector_channels if detector_channels is not None else self.input_channels
        self.detector_kwargs = {
            "command": command,
            "sample_rate": int(sample_rate),
            "channels": int(detector_channels),
            "input_device": (
                detector_input_device
                if detector_input_device is not None
                else self.input_device_name
            ),
            "window_seconds": float(window_seconds),
            "hop_seconds": float(hop_seconds),
            "analysis_buffer_seconds": analysis_buffer_seconds,
            "analysis_queue_size": int(analysis_queue_size),
            "on_threshold": float(on_threshold),
            "off_threshold": off_threshold,
            "on_windows_required": int(on_windows_required),
            "off_windows_required": int(off_windows_required),
            "torch_num_threads": int(torch_num_threads),
            "allow_mono_input_fallback": bool(allow_mono_input_fallback),
            "status_interval": self.status_interval,
        }

        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status_lines: list[str] = []
        self._gain_state = "base"
        self._running = False
        self._error: str | None = None
        self._scores_payload: dict[str, Any] | None = None

    def _stream_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "input_device_name": self.input_device_name,
            "output_device_name": self.output_device_name,
            "num_input_channels": self.input_channels,
            "num_output_channels": self.output_channels,
        }
        if self.buffer_size is not None:
            kwargs["buffer_size"] = self.buffer_size
        if self.allow_feedback:
            kwargs["allow_feedback"] = True
        return kwargs

    def _append_status(self, *messages: str) -> None:
        with self._lock:
            self._status_lines.extend(messages)
            self._status_lines = self._status_lines[-50:]

    def set_gain(self, boost_gain_db: float) -> None:
        self.gain_value.value = float(boost_gain_db)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "gain_state": self._gain_state,
                "base_gain_db": self.base_gain_db,
                "boost_gain_db": float(self.gain_value.value),
                "active": bool(self.active_value.value),
                "clap_enabled": self.enable_clap,
                "command": self.command,
                "on_threshold": self.detector_kwargs["on_threshold"],
                "input_device_name": self.input_device_name,
                "output_device_name": self.output_device_name,
                "scores": self._scores_payload,
                "status_lines": list(self._status_lines),
                "error": self._error,
            }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.stop_event.clear()
        with self._lock:
            self._error = None
            self._scores_payload = None
        if self.enable_clap:
            self.detector_process = self.ctx.Process(
                target=_run_detector_process,
                args=(self.detector_kwargs, self.active_value, self.stop_event, self.status_queue),
                daemon=True,
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.detector_process is not None and self.detector_process.is_alive():
            self.detector_process.join(timeout=2.0)
            if self.detector_process.is_alive():
                self.detector_process.terminate()
                self.detector_process.join(timeout=1.0)
        self.detector_process = None
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            self._running = False

    def _run(self) -> None:
        try:
            with AudioStream(**self._stream_kwargs()) as stream:
                base_board = make_gain_board(self.base_gain_db)
                current_boost_gain = float(self.gain_value.value)
                boost_board = make_gain_board(current_boost_gain)
                stream.plugins = base_board
                current_active = False

                with self._lock:
                    self._running = True
                    self._gain_state = "base"

                if self.detector_process is not None:
                    self.detector_process.start()
                self._append_status(
                    f"Audio stream started. base={self.base_gain_db:.1f} dB, "
                    f"boost={current_boost_gain:.1f} dB."
                )

                last_status = 0.0
                while not self.stop_event.is_set():
                    requested_active = bool(self.active_value.value)
                    desired_gain = float(self.gain_value.value)

                    if desired_gain != current_boost_gain:
                        current_boost_gain = desired_gain
                        boost_board = make_gain_board(current_boost_gain)
                        if current_active:
                            stream.plugins = boost_board

                    if requested_active != current_active:
                        current_active = requested_active
                        stream.plugins = boost_board if current_active else base_board
                        gain_state = "boost" if current_active else "base"
                        with self._lock:
                            self._gain_state = gain_state
                        self._append_status(
                            f"AUDIO GAIN -> {'BOOST' if current_active else 'base'} "
                            f"({current_boost_gain if current_active else self.base_gain_db:.1f} dB)"
                        )

                    now = time.time()
                    if now - last_status >= self.status_interval:
                        last_status = now
                        text_messages: list[str] = []
                        for message in drain_status_queue(self.status_queue):
                            if isinstance(message, dict) and message.get("type") == "scores":
                                with self._lock:
                                    self._scores_payload = message
                            else:
                                text_messages.append(str(message))
                        if text_messages:
                            self._append_status(*text_messages)
                        if self.detector_process is not None and not self.detector_process.is_alive():
                            self._append_status(
                                "WARNING: CLAP detector stopped; audio continues with last gain state."
                            )
                            self.detector_process = None

                    time.sleep(self.poll_interval)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            self._append_status(f"Audio stream ERROR: {exc}")
        finally:
            self.stop_event.set()
            with self._lock:
                self._running = False


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
    parser.add_argument("--window-seconds", type=float, default=1.0)
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
    parser.add_argument("--on-windows-required", type=int, default=3)
    parser.add_argument("--off-windows-required", type=int, default=7)
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

    detector_input_device = parse_detector_device_arg(
        args.detector_input_device if args.detector_input_device is not None else args.input_device_name
    )
    detector_channels = args.detector_channels if args.detector_channels is not None else args.input_channels

    controller = StreamController(
        command=command,
        boost_gain_db=args.boost_gain_db,
        base_gain_db=args.base_gain_db,
        input_device_name=input_device_name,
        output_device_name=output_device_name,
        input_channels=args.input_channels,
        output_channels=args.output_channels,
        buffer_size=args.buffer_size,
        allow_feedback=args.allow_feedback,
        poll_interval=args.poll_interval,
        status_interval=args.status_interval,
        sample_rate=args.sample_rate,
        detector_input_device=detector_input_device,
        detector_channels=detector_channels,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        analysis_buffer_seconds=args.analysis_buffer_seconds,
        analysis_queue_size=args.analysis_queue_size,
        on_threshold=args.on_threshold,
        off_threshold=args.off_threshold,
        on_windows_required=args.on_windows_required,
        off_windows_required=args.off_windows_required,
        torch_num_threads=args.torch_num_threads,
        allow_mono_input_fallback=not args.no_mono_input_fallback,
        enable_clap=not args.no_clap,
    )

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

    printed = 0
    try:
        controller.start()
        while True:
            status = controller.get_status()
            lines = status["status_lines"]
            for message in lines[printed:]:
                print(message, flush=True)
            printed = len(lines)
            if not status["running"] and status["error"]:
                print(f"ERROR: {status['error']}", file=sys.stderr, flush=True)
                break
            time.sleep(max(0.05, float(args.status_interval)))
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        print("Stopped.")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
