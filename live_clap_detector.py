#!/usr/bin/env python3
"""CLAP-only activity detector for the Pedalboard streaming controller.

This file deliberately does not open an output stream and does not apply gain.
It only listens to the microphone input, runs CLAP in a background analysis
thread, and publishes a boolean "target active" state to the caller.

The intended main entry point is paddle_stream.py, which owns the low-latency
Pedalboard AudioStream. paddle_stream.py starts this detector in a separate
process so audio monitoring never waits for CLAP inference.
"""
from __future__ import annotations

import argparse
import queue
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from msclap import CLAP


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def cosine_similarity_scores(audio_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    audio_emb = row_normalize(audio_emb)
    text_embs = row_normalize(text_embs)
    return (audio_emb @ text_embs.T).squeeze(0)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_target_commands(command: str) -> list[str]:
    """Split a free-form target request into user-defined target labels."""
    parts = re.split(r"\s*(?:,|;|\n|\band\b)\s*", command.strip(), flags=re.IGNORECASE)
    targets: list[str] = []
    seen_normalized: set[str] = set()
    for part in parts:
        target = part.strip()
        if not target:
            continue
        key = normalize_text(target)
        if not key or key in seen_normalized:
            continue
        targets.append(target)
        seen_normalized.add(key)
    return targets


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def ensure_2d_float32(audio: np.ndarray) -> np.ndarray:
    """Return audio as shape (frames, channels) without mixing channels."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError(f"Expected audio with shape (frames,) or (frames, channels), got {audio.shape}.")
    return audio.astype(np.float32, copy=False)


def to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_device_arg(value: str | None):
    """Parse a sounddevice device selector."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _safe_status_put(status_queue: Any | None, message: str) -> None:
    if status_queue is None:
        return
    try:
        status_queue.put_nowait(message)
    except Exception:
        pass


@dataclass
class DetectorState:
    max_target_scores: list[float]
    target_active: bool = False
    top_predictions_by_channel: list[list[tuple[str, float]]] = field(default_factory=list)
    target_scores_by_channel: list[list[tuple[str, float]]] = field(default_factory=list)
    consecutive_target_windows_by_channel: list[int] = field(default_factory=list)
    consecutive_neutral_windows_by_channel: list[int] = field(default_factory=list)
    audio_status_counts: dict[str, int] = field(default_factory=dict)
    dropped_analysis_blocks: int = 0
    running: bool = True


class CLAPActivityDetector:
    """Input-only CLAP detector.

    This class owns an input stream and a CLAP analysis worker. It never touches
    the headphone/audio-output path. The only output is a shared boolean state
    saying whether any requested target is active.
    """

    def __init__(
        self,
        command: str,
        active_value: Any | None = None,
        status_queue: Any | None = None,
        stop_event: Any | None = None,
        sample_rate: int = 48000,
        channels: int = 2,
        input_device=None,
        window_seconds: float = 1.0,
        hop_seconds: float = 1.0,
        analysis_buffer_seconds: float | None = None,
        analysis_queue_size: int = 4,
        on_threshold: float = 0.20,
        off_threshold: float | None = None,
        on_windows_required: int = 3,
        off_windows_required: int = 7,
        torch_num_threads: int | None = 1,
        allow_mono_input_fallback: bool = True,
        status_interval: float = 1.0,
    ) -> None:
        self.labels = split_target_commands(command)
        if not self.labels:
            raise ValueError("No target activities were provided.")

        self.active_value = active_value
        self.status_queue = status_queue
        self.stop_event = stop_event
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.input_device = input_device
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.analysis_buffer_seconds = (
            float(analysis_buffer_seconds)
            if analysis_buffer_seconds is not None
            else self.window_seconds * 3.0
        )
        self.analysis_queue_size = max(1, int(analysis_queue_size))
        self.on_threshold = float(on_threshold)
        self.off_threshold = self.on_threshold if off_threshold is None else float(off_threshold)
        if self.off_threshold > self.on_threshold:
            raise ValueError("off_threshold must be <= on_threshold.")
        self.on_windows_required = max(1, int(on_windows_required))
        self.off_windows_required = max(1, int(off_windows_required))
        self.allow_mono_input_fallback = bool(allow_mono_input_fallback)
        self.status_interval = max(0.1, float(status_interval))

        if self.channels < 1:
            raise ValueError("channels must be >= 1.")
        if self.analysis_buffer_seconds < self.window_seconds:
            raise ValueError("analysis_buffer_seconds must be >= window_seconds.")

        if torch is not None and torch_num_threads is not None:
            try:
                torch.set_num_threads(max(1, int(torch_num_threads)))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(max(1, int(torch_num_threads)))
            except Exception:
                pass

        self.model = CLAP(version="2023", use_cuda=False)
        self.text_embeddings = to_numpy(self.model.get_text_embeddings(self.labels)).astype(np.float32)

        self.window_samples = max(1, int(round(self.sample_rate * self.window_seconds)))
        self.hop_samples = max(1, int(round(self.sample_rate * self.hop_seconds)))
        self.analysis_buffer_samples = max(
            self.window_samples,
            int(round(self.sample_rate * self.analysis_buffer_seconds)),
        )

        self.analysis_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.analysis_queue_size)
        self.state = DetectorState(
            max_target_scores=[0.0 for _ in range(self.channels)],
            top_predictions_by_channel=[[] for _ in range(self.channels)],
            target_scores_by_channel=[[] for _ in range(self.channels)],
            consecutive_target_windows_by_channel=[0 for _ in range(self.channels)],
            consecutive_neutral_windows_by_channel=[0 for _ in range(self.channels)],
        )
        self.analysis_buffer = np.zeros((0, self.channels), dtype=np.float32)
        self.samples_since_last_eval = 0
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.analysis_loop, daemon=True)
        self.status_worker = threading.Thread(target=self.status_loop, daemon=True)

        self.publish_active(False)

    def publish_active(self, active: bool) -> None:
        if self.active_value is not None:
            try:
                self.active_value.value = 1 if active else 0
            except Exception:
                pass

    def analysis_input_from_stream(self, input_audio: np.ndarray) -> np.ndarray:
        """Return exactly self.channels logical hand channels for CLAP."""
        input_audio = ensure_2d_float32(input_audio)
        frames, physical_channels = input_audio.shape

        if physical_channels == self.channels:
            return input_audio
        if physical_channels == 1 and self.channels == 2:
            return np.repeat(input_audio, 2, axis=1)
        if physical_channels > self.channels:
            return input_audio[:, : self.channels]

        out = np.zeros((frames, self.channels), dtype=np.float32)
        out[:, :physical_channels] = input_audio
        out[:, physical_channels:] = input_audio[:, [physical_channels - 1]]
        return out

    def audio_embedding(self, mono_audio: np.ndarray) -> np.ndarray:
        mono_audio = ensure_mono_float32(mono_audio)

        try:
            emb = self.model.get_audio_embeddings([mono_audio])
            return to_numpy(emb)
        except Exception:
            pass

        try:
            emb = self.model.get_audio_embeddings(mono_audio)
            return to_numpy(emb)
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, mono_audio, self.sample_rate)
            emb = self.model.get_audio_embeddings([tmp.name])
            return to_numpy(emb)

    def score_channel_window(
        self, mono_audio: np.ndarray
    ) -> tuple[float, list[tuple[str, float]], list[tuple[str, float]]]:
        audio_emb = self.audio_embedding(mono_audio).astype(np.float32)
        sims_np = cosine_similarity_scores(audio_emb, self.text_embeddings)

        max_target_score = float(np.max(sims_np))
        target_scores = [
            (self.labels[int(idx)], float(sims_np[int(idx)]))
            for idx in range(len(self.labels))
        ]
        top_indices = np.argsort(sims_np)[::-1]
        top_predictions = [(self.labels[int(idx)], float(sims_np[int(idx)])) for idx in top_indices]
        return max_target_score, top_predictions, target_scores

    def score_window(
        self, window_audio: np.ndarray
    ) -> tuple[list[float], list[list[tuple[str, float]]], list[list[tuple[str, float]]]]:
        window_audio = ensure_2d_float32(window_audio)
        max_target_scores: list[float] = []
        top_predictions_by_channel: list[list[tuple[str, float]]] = []
        target_scores_by_channel: list[list[tuple[str, float]]] = []

        for channel_index in range(window_audio.shape[1]):
            channel_audio = window_audio[:, channel_index]
            max_target_score, top_predictions, target_scores = self.score_channel_window(channel_audio)
            max_target_scores.append(max_target_score)
            top_predictions_by_channel.append(top_predictions)
            target_scores_by_channel.append(target_scores)

        return max_target_scores, top_predictions_by_channel, target_scores_by_channel

    def analysis_loop(self) -> None:
        while self.state.running:
            try:
                block = self.analysis_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            block = ensure_2d_float32(block)
            if block.shape[1] != self.channels:
                _safe_status_put(
                    self.status_queue,
                    f"CLAP warning: expected {self.channels} channel(s), got block shape {block.shape}",
                )
                continue

            self.analysis_buffer = np.concatenate([self.analysis_buffer, block], axis=0)
            if self.analysis_buffer.shape[0] > self.analysis_buffer_samples:
                self.analysis_buffer = self.analysis_buffer[-self.analysis_buffer_samples :, :]

            self.samples_since_last_eval += block.shape[0]
            if self.analysis_buffer.shape[0] < self.window_samples:
                continue
            if self.samples_since_last_eval < self.hop_samples:
                continue

            self.samples_since_last_eval = 0
            window = self.analysis_buffer[-self.window_samples :, :]

            try:
                max_target_scores, top_predictions_by_channel, target_scores_by_channel = self.score_window(window)
            except Exception as exc:
                _safe_status_put(self.status_queue, f"CLAP scoring failed: {exc}")
                continue

            with self.lock:
                self.state.max_target_scores = max_target_scores
                self.state.top_predictions_by_channel = top_predictions_by_channel
                self.state.target_scores_by_channel = target_scores_by_channel

                target_window_detected_by_channel = [
                    score >= self.on_threshold for score in max_target_scores
                ]
                neutral_window_detected_by_channel = [
                    score < self.off_threshold for score in max_target_scores
                ]

                if self.state.target_active:
                    for channel_index, neutral_detected in enumerate(neutral_window_detected_by_channel):
                        if neutral_detected:
                            self.state.consecutive_neutral_windows_by_channel[channel_index] += 1
                        else:
                            self.state.consecutive_neutral_windows_by_channel[channel_index] = 0
                        self.state.consecutive_target_windows_by_channel[channel_index] = 0

                    if all(
                        count >= self.off_windows_required
                        for count in self.state.consecutive_neutral_windows_by_channel
                    ):
                        self.state.target_active = False
                        self.state.consecutive_neutral_windows_by_channel = [0 for _ in range(self.channels)]
                else:
                    for channel_index, target_detected in enumerate(target_window_detected_by_channel):
                        if target_detected:
                            self.state.consecutive_target_windows_by_channel[channel_index] += 1
                        else:
                            self.state.consecutive_target_windows_by_channel[channel_index] = 0
                        self.state.consecutive_neutral_windows_by_channel[channel_index] = 0

                    if any(
                        count >= self.on_windows_required
                        for count in self.state.consecutive_target_windows_by_channel
                    ):
                        self.state.target_active = True
                        self.state.consecutive_target_windows_by_channel = [0 for _ in range(self.channels)]

                self.publish_active(self.state.target_active)

    def input_callback(self, indata, frames, time_info, status) -> None:
        if status:
            with self.lock:
                key = str(status)
                self.state.audio_status_counts[key] = self.state.audio_status_counts.get(key, 0) + 1

        analysis_block = self.analysis_input_from_stream(np.array(indata, dtype=np.float32, copy=True))

        try:
            self.analysis_queue.put_nowait(analysis_block)
        except queue.Full:
            try:
                self.analysis_queue.get_nowait()
            except queue.Empty:
                pass
            with self.lock:
                self.state.dropped_analysis_blocks += 1
            try:
                self.analysis_queue.put_nowait(analysis_block)
            except queue.Full:
                pass

    def format_status(self) -> str:
        with self.lock:
            active = self.state.target_active
            top_predictions_by_channel = [list(preds) for preds in self.state.top_predictions_by_channel]
            target_scores_by_channel = [list(scores) for scores in self.state.target_scores_by_channel]
            audio_status_counts = dict(self.state.audio_status_counts)
            self.state.audio_status_counts.clear()
            dropped_analysis_blocks = self.state.dropped_analysis_blocks
            self.state.dropped_analysis_blocks = 0

        parts = [f"CLAP={'BOOST' if active else 'base'}"]
        for channel_index in range(self.channels):
            hand_name = f"hand{channel_index + 1}"
            top_predictions = top_predictions_by_channel[channel_index] if channel_index < len(top_predictions_by_channel) else []
            target_scores = target_scores_by_channel[channel_index] if channel_index < len(target_scores_by_channel) else []
            if not top_predictions:
                prediction_text = "waiting"
            else:
                top_label, top_score = top_predictions[0]
                prediction_text = f"{top_label}={top_score:.3f}" if top_score >= self.on_threshold else "neutral"
                if target_scores:
                    scores_text = ", ".join(f"{label}={score:.3f}" for label, score in target_scores)
                    prediction_text = f"{prediction_text} (scores: {scores_text})"
            parts.append(f"{hand_name}: {prediction_text}")

        if audio_status_counts:
            status_text = ", ".join(f"{name} x{count}" for name, count in sorted(audio_status_counts.items()))
            parts.append(f"input status: {status_text}")
        if dropped_analysis_blocks:
            parts.append(f"dropped analysis blocks: {dropped_analysis_blocks}")
        return " | ".join(parts)

    def scores_payload(self) -> dict[str, Any]:
        """Structured per-activity scores for the UI.

        Each activity's score is the max across logical hand channels, so a
        single bar represents 'how strongly is this sound present right now'.
        """
        with self.lock:
            labels = list(self.labels)
            target_scores_by_channel = [list(scores) for scores in self.state.target_scores_by_channel]
            active = self.state.target_active

        aggregated: dict[str, float | None] = {label: None for label in labels}
        for channel_scores in target_scores_by_channel:
            for label, score in channel_scores:
                if label in aggregated and (aggregated[label] is None or score > aggregated[label]):
                    aggregated[label] = score

        scores = [
            [label, float(aggregated[label]) if aggregated[label] is not None else 0.0]
            for label in labels
        ]
        return {
            "type": "scores",
            "scores": scores,
            "on_threshold": self.on_threshold,
            "active": active,
        }

    def status_loop(self) -> None:
        while self.state.running:
            _safe_status_put(self.status_queue, self.format_status())
            _safe_status_put(self.status_queue, self.scores_payload())
            time.sleep(self.status_interval)

    def open_input_stream(self, input_channels: int) -> sd.InputStream:
        return sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=input_channels,
            dtype="float32",
            callback=self.input_callback,
        )

    def run(self) -> None:
        _safe_status_put(self.status_queue, "CLAP detector starting")
        _safe_status_put(self.status_queue, "CLAP target labels: " + ", ".join(self.labels))
        _safe_status_put(
            self.status_queue,
            f"CLAP detector input request: channels={self.channels}, sample_rate={self.sample_rate}, device={self.input_device!r}",
        )

        self.worker.start()
        self.status_worker.start()

        physical_input_channels = self.channels
        try:
            try:
                stream = self.open_input_stream(physical_input_channels)
            except sd.PortAudioError as first_exc:
                if self.allow_mono_input_fallback and self.channels == 2:
                    _safe_status_put(
                        self.status_queue,
                        "CLAP warning: could not open 2-channel detector input; falling back to mono and duplicating to both hands.",
                    )
                    physical_input_channels = 1
                    try:
                        stream = self.open_input_stream(physical_input_channels)
                    except sd.PortAudioError:
                        raise first_exc
                else:
                    raise

            with stream:
                _safe_status_put(
                    self.status_queue,
                    f"CLAP detector opened input: physical_channels={physical_input_channels}, logical_hands={self.channels}",
                )
                while self.state.running:
                    if self.stop_event is not None and self.stop_event.is_set():
                        break
                    time.sleep(0.1)
        except Exception as exc:
            self.publish_active(False)
            _safe_status_put(self.status_queue, f"CLAP detector ERROR: {exc}")
            raise
        finally:
            self.state.running = False
            self.publish_active(False)
            _safe_status_put(self.status_queue, "CLAP detector stopped")


def run_clap_detector(
    command: str,
    active_value: Any | None = None,
    status_queue: Any | None = None,
    stop_event: Any | None = None,
    sample_rate: int = 48000,
    channels: int = 2,
    input_device=None,
    window_seconds: float = 1.0,
    hop_seconds: float = 1.0,
    analysis_buffer_seconds: float | None = None,
    analysis_queue_size: int = 4,
    on_threshold: float = 0.20,
    off_threshold: float | None = None,
    on_windows_required: int = 3,
    off_windows_required: int = 7,
    torch_num_threads: int | None = 1,
    allow_mono_input_fallback: bool = True,
    status_interval: float = 1.0,
) -> None:
    detector = CLAPActivityDetector(
        command=command,
        active_value=active_value,
        status_queue=status_queue,
        stop_event=stop_event,
        sample_rate=sample_rate,
        channels=channels,
        input_device=input_device,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        analysis_buffer_seconds=analysis_buffer_seconds,
        analysis_queue_size=analysis_queue_size,
        on_threshold=on_threshold,
        off_threshold=off_threshold,
        on_windows_required=on_windows_required,
        off_windows_required=off_windows_required,
        torch_num_threads=torch_num_threads,
        allow_mono_input_fallback=allow_mono_input_fallback,
        status_interval=status_interval,
    )
    detector.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Input-only CLAP detector worker.")
    parser.add_argument("--command", required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--analysis-buffer-seconds", type=float, default=None)
    parser.add_argument("--analysis-queue-size", type=int, default=4)
    parser.add_argument("--on-threshold", type=float, default=0.20)
    parser.add_argument("--off-threshold", type=float, default=None)
    parser.add_argument("--on-windows-required", type=int, default=3)
    parser.add_argument("--off-windows-required", type=int, default=7)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--no-mono-input-fallback", action="store_true")
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return 0

    # Standalone mode: no shared state, just print statuses locally.
    class PrintQueue:
        def put_nowait(self, message: str) -> None:
            print(message, flush=True)

    try:
        run_clap_detector(
            command=args.command,
            active_value=None,
            status_queue=PrintQueue(),
            stop_event=None,
            sample_rate=args.sample_rate,
            channels=args.channels,
            input_device=parse_device_arg(args.input_device),
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
            status_interval=args.status_interval,
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
