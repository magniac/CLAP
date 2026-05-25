#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from msclap import CLAP

try:
    # This module must be the import-safe helper version of paddle_stream.py.
    # It exposes PaddleGainProcessor without opening its own AudioStream.
    from paddle_stream import PaddleGainProcessor
except Exception as exc:  # pragma: no cover
    PADDLE_GAIN_IMPORT_ERROR = exc

    class PaddleGainProcessor:  # type: ignore[no-redef]
        """Emergency fallback if paddle_stream.py cannot be imported.

        This applies mathematically equivalent dB gain with NumPy. Replace
        paddle_stream.py with the import-safe helper to use the shared helper
        module, and install `pedalboard` if you specifically need the
        pedalboard.Gain backend.
        """

        def __init__(self, sample_rate: int | float, prefer_pedalboard: bool = True):
            self.sample_rate = int(sample_rate)
            self.backend = "internal-numpy-db-gain"
            self.import_error = PADDLE_GAIN_IMPORT_ERROR

        def process(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[:, None]
            linear_gain = float(10.0 ** (float(gain_db) / 20.0))
            return np.clip(audio * linear_gain, -1.0, 1.0).astype(np.float32, copy=False)
else:
    PADDLE_GAIN_IMPORT_ERROR = None


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
    """Split a free-form target request into user-defined target labels.

    Examples:
        "washing hands, typing" -> ["washing hands", "typing"]
        "washing hands and typing" -> ["washing hands", "typing"]

    These strings are used directly as the CLAP text labels/prompts.
    No external/predefined label file is used.
    """
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


def resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    return resample_poly(audio, dst_sr, src_sr).astype(np.float32, copy=False)


def to_numpy(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_latency_arg(value: str | float | None):
    """Parse a sounddevice latency value.

    sounddevice accepts "low", "high", a numeric latency in seconds,
    or None to use the host API default.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"none", "default", "auto"}:
        return None
    if text in {"low", "high"}:
        return text
    try:
        latency = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "latency must be 'low', 'high', 'default', or a number of seconds, e.g. 0.02"
        ) from exc
    if latency < 0:
        raise argparse.ArgumentTypeError("latency must be non-negative.")
    return latency


def parse_device_arg(value: str | None):
    """Parse a sounddevice device selector.

    sounddevice accepts either a device index or a device-name substring.
    This lets command-line values like --input-device 2 become integer index 2
    instead of the string "2".
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def linear_gain_to_db(value: float) -> float:
    """Convert a legacy linear gain multiplier into dB."""
    value = float(value)
    if value <= 0:
        raise ValueError("linear gain must be > 0 to convert to dB.")
    return float(20.0 * np.log10(value))


@dataclass
class SharedState:
    current_gain: float
    max_target_scores: list[float] = field(default_factory=list)
    target_active: bool = False
    top_predictions_by_channel: list[list[tuple[str, float]]] = field(default_factory=list)
    target_scores_by_channel: list[list[tuple[str, float]]] = field(default_factory=list)
    consecutive_target_windows_by_channel: list[int] = field(default_factory=list)
    consecutive_neutral_windows_by_channel: list[int] = field(default_factory=list)
    audio_status_counts: dict[str, int] = field(default_factory=dict)
    dropped_analysis_blocks: int = 0
    running: bool = True


class LiveCLAPAmplifier:
    def __init__(
        self,
        command: str,
        sample_rate: int = 48000,
        channels: int = 2,
        output_channels: int = 2,
        block_duration: float | None = None,
        blocksize: int | None = None,
        latency: str | float | None = 0.10,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        analysis_buffer_seconds: float | None = None,
        analysis_queue_size: int = 32,
        base_gain_db: float = 0.0,
        boost_gain_db: float = 12.0,
        on_threshold: float = 0.20,
        off_threshold: float | None = None,
        on_windows_required: int = 1,
        off_windows_required: int = 1,
        device=None,
        torch_num_threads: int | None = 1,
        allow_mono_input_fallback: bool = True,
    ) -> None:
        # The only CLAP text labels are the targets specified by the user.
        # "neutral" is not embedded as a label; neutral is inferred when no
        # target reaches the relevant threshold for enough consecutive windows.
        self.labels = split_target_commands(command)
        if not self.labels:
            raise ValueError("No target activities were provided.")
        self.selected_indices = list(range(len(self.labels)))

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.output_channels = int(output_channels)
        self.block_duration = None if block_duration is None else float(block_duration)
        self.requested_blocksize = None if blocksize is None else int(blocksize)
        self.latency = latency
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.analysis_buffer_seconds = (
            float(analysis_buffer_seconds)
            if analysis_buffer_seconds is not None
            else self.window_seconds * 3.0
        )
        self.analysis_queue_size = max(1, int(analysis_queue_size))
        self.base_gain_db = float(base_gain_db)
        self.boost_gain_db = float(boost_gain_db)
        self.on_threshold = float(on_threshold)
        # If the user does not explicitly set an off threshold, use the same
        # threshold for turning on and staying on. This avoids confusing behavior
        # where boost turns off even though the visible score is still above the
        # user-selected activity threshold.
        self.off_threshold = self.on_threshold if off_threshold is None else float(off_threshold)
        if self.off_threshold > self.on_threshold:
            raise ValueError(
                f"off_threshold ({self.off_threshold}) must be <= on_threshold ({self.on_threshold}). "
                "Use the same value for both if you want gain to stay on as long as "
                "the score remains above the threshold."
            )
        self.on_windows_required = max(1, int(on_windows_required))
        self.off_windows_required = max(1, int(off_windows_required))
        self.device = device
        self.allow_mono_input_fallback = bool(allow_mono_input_fallback)

        if self.channels < 1:
            raise ValueError("channels must be >= 1.")
        if self.output_channels < 1:
            raise ValueError("output_channels must be >= 1.")
        if torch is not None and torch_num_threads is not None:
            # Keep CLAP CPU inference from starving the real-time audio callback.
            # This is a common cause of PortAudio "input overflow" warnings.
            try:
                torch.set_num_threads(max(1, int(torch_num_threads)))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(max(1, int(torch_num_threads)))
            except Exception:
                pass
        if self.analysis_buffer_seconds < self.window_seconds:
            raise ValueError(
                "analysis_buffer_seconds must be greater than or equal to window_seconds "
                f"({self.analysis_buffer_seconds} < {self.window_seconds})."
            )

        self.gain_processor = PaddleGainProcessor(sample_rate=self.sample_rate)
        self.gain_backend = getattr(self.gain_processor, "backend", "unknown")

        self.model = CLAP(version="2023", use_cuda=False)
        self.text_embeddings = to_numpy(self.model.get_text_embeddings(self.labels)).astype(np.float32)

        # For low-latency monitoring, do not force a large fixed block by default.
        # blocksize=0 lets PortAudio/host API choose the optimal callback size.
        # If the user explicitly provides --blocksize or --block-duration, use that.
        if self.requested_blocksize is not None:
            if self.requested_blocksize < 0:
                raise ValueError("blocksize must be >= 0. Use 0 for host/API default.")
            self.blocksize = self.requested_blocksize
        elif self.block_duration is not None:
            if self.block_duration <= 0:
                raise ValueError("block_duration must be > 0 when provided.")
            self.blocksize = max(1, int(round(self.sample_rate * self.block_duration)))
        else:
            self.blocksize = 0

        self.window_samples = max(1, int(round(self.sample_rate * self.window_seconds)))
        self.hop_samples = max(1, int(round(self.sample_rate * self.hop_seconds)))
        self.analysis_buffer_samples = max(
            self.window_samples,
            int(round(self.sample_rate * self.analysis_buffer_seconds)),
        )

        self.analysis_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.analysis_queue_size)
        self.state = SharedState(
            current_gain=self.base_gain_db,
            max_target_scores=[0.0 for _ in range(self.channels)],
            top_predictions_by_channel=[[] for _ in range(self.channels)],
            target_scores_by_channel=[[] for _ in range(self.channels)],
            consecutive_target_windows_by_channel=[0 for _ in range(self.channels)],
            consecutive_neutral_windows_by_channel=[0 for _ in range(self.channels)],
        )
        # Keep one analysis buffer per input channel. Shape is (frames, channels).
        self.analysis_buffer = np.zeros((0, self.channels), dtype=np.float32)
        self.samples_since_last_eval = 0
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.analysis_loop, daemon=True)

    def audio_embedding(self, mono_audio: np.ndarray) -> np.ndarray:
        mono_audio = ensure_mono_float32(mono_audio)

        # Try in-memory array first. If the installed msclap variant only accepts file paths,
        # fall back to a temporary WAV file.
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
        """Score a single channel/window against the user-defined target labels only.

        Returns:
            max_target_score: highest score among the user-provided targets.
            top_predictions: all targets sorted from highest score to lowest score.
            target_scores: all target scores in the same order the user entered them.
        """
        audio_emb = self.audio_embedding(mono_audio).astype(np.float32)
        sims_np = cosine_similarity_scores(audio_emb, self.text_embeddings)

        # Since all labels are targets, the max target score is simply the max
        # similarity among the user-specified labels.
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
        """Score each input channel independently.

        Args:
            window_audio: Array with shape (frames, channels).

        Returns:
            max_target_scores: one max user-target score per channel.
            top_predictions_by_channel: all target predictions per channel, sorted by score.
            target_scores_by_channel: all target scores per channel, in user-entered order.
        """
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
                print(
                    f"\n[analysis warning] Expected {self.channels} channel(s), "
                    f"but got block with shape {block.shape}.",
                    file=sys.stderr,
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
                print(f"\n[analysis warning] CLAP scoring failed: {exc}", file=sys.stderr)
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
                    # When boost is active, require every channel to be neutral
                    # for the requested number of consecutive analysis windows
                    # before turning boost off. This avoids turning off while one
                    # wrist microphone still hears a target activity.
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
                        self.state.consecutive_neutral_windows_by_channel = [
                            0 for _ in range(self.channels)
                        ]
                else:
                    # When boost is inactive, each input channel has its own
                    # consecutive-target counter. If any channel reaches the
                    # requested count, the shared gain turns on.
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
                        self.state.consecutive_target_windows_by_channel = [
                            0 for _ in range(self.channels)
                        ]

                self.state.current_gain = self.boost_gain_db if self.state.target_active else self.base_gain_db



    def analysis_input_from_stream(self, input_audio: np.ndarray) -> np.ndarray:
        """Return exactly self.channels logical hand channels for CLAP.

        The normal case is a real 2-channel input device, where hand1 and hand2
        are distinct physical channels. If the selected/default input device only
        opens as mono and mono fallback is enabled, duplicate the mono signal so
        the UI still shows hand1 and hand2. That fallback keeps the script alive,
        but it does not create two independent wrist-mic signals.
        """
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

    def route_input_to_output(self, input_audio: np.ndarray) -> np.ndarray:
        """Map hand-mic input channels to headphone/output channels.

        Analysis is always done on self.channels input channels. Output can be
        mono or stereo depending on the selected output device. If output has
        the same number of channels, preserve the channel mapping.
        """
        input_audio = ensure_2d_float32(input_audio)
        frames, input_channels = input_audio.shape
        output_channels = self.output_channels

        if output_channels == input_channels:
            return input_audio

        if output_channels == 1:
            return input_audio.mean(axis=1, keepdims=True)

        if input_channels == 1:
            return np.repeat(input_audio, output_channels, axis=1)

        out = np.zeros((frames, output_channels), dtype=np.float32)
        channels_to_copy = min(input_channels, output_channels)
        out[:, :channels_to_copy] = input_audio[:, :channels_to_copy]
        if output_channels > input_channels:
            mono_mix = input_audio.mean(axis=1)
            out[:, input_channels:] = mono_mix[:, None]
        return out

    def callback(self, indata, outdata, frames, time_info, status) -> None:
        # Never print directly from the real-time audio callback. Printing can
        # block, interleave with the status line, and make overflows worse.
        if status:
            with self.lock:
                key = str(status)
                self.state.audio_status_counts[key] = self.state.audio_status_counts.get(key, 0) + 1

        # Copy only for the background CLAP analysis queue. The live monitor
        # output uses the current callback's input buffer directly to minimize
        # microphone-to-headphone latency.
        analysis_block = self.analysis_input_from_stream(np.array(indata, dtype=np.float32, copy=True))

        # Keep analysis real-time: if CLAP falls behind, drop the oldest queued
        # analysis block instead of letting the queue grow stale. Audio monitoring
        # still passes through immediately.
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

        with self.lock:
            gain_db = self.state.current_gain

        # Apply gain through the Pedalboard helper, using the same Gain(gain_db=...)
        # approach as paddle_stream.py.
        out = self.route_input_to_output(indata)
        out = self.gain_processor.process(out, gain_db=gain_db)
        outdata[:] = out

    def print_status_loop(self) -> None:
        while self.state.running:
            with self.lock:
                gain = self.state.current_gain
                active = self.state.target_active
                top_predictions_by_channel = [
                    list(predictions) for predictions in self.state.top_predictions_by_channel
                ]
                target_scores_by_channel = [
                    list(scores) for scores in self.state.target_scores_by_channel
                ]
                audio_status_counts = dict(self.state.audio_status_counts)
                self.state.audio_status_counts.clear()
                dropped_analysis_blocks = self.state.dropped_analysis_blocks
                self.state.dropped_analysis_blocks = 0

            # Display one compact status line. "neutral" is not a CLAP label;
            # it is shown when this channel's best user-defined target score
            # is below the activity/on threshold. Do not print internal debounce counters.
            parts = [f"gain_db={gain:.1f} dB ({'BOOST' if active else 'base'})"]
            for channel_index in range(self.channels):
                hand_name = f"hand{channel_index + 1}"
                top_predictions = (
                    top_predictions_by_channel[channel_index]
                    if channel_index < len(top_predictions_by_channel)
                    else []
                )

                target_scores = (
                    target_scores_by_channel[channel_index]
                    if channel_index < len(target_scores_by_channel)
                    else []
                )

                if not top_predictions:
                    prediction_text = "waiting"
                else:
                    top_label, top_score = top_predictions[0]
                    if top_score >= self.on_threshold:
                        prediction_text = f"{top_label}={top_score:.3f}"
                    else:
                        prediction_text = "neutral"

                    if target_scores:
                        scores_text = ", ".join(
                            f"{label}={score:.3f}" for label, score in target_scores
                        )
                        prediction_text = f"{prediction_text} (scores: {scores_text})"

                parts.append(f"{hand_name}: {prediction_text}")

            if audio_status_counts:
                status_text = ", ".join(
                    f"{name} x{count}" for name, count in sorted(audio_status_counts.items())
                )
                parts.append(f"audio status: {status_text}")
            if dropped_analysis_blocks:
                parts.append(f"dropped analysis blocks: {dropped_analysis_blocks}")

            # Print normal newline-separated status updates. This is less fancy
            # than a single updating carriage-return line, but it avoids garbled
            # output when PortAudio/status messages happen at the same time.
            print(" | ".join(parts), flush=True)
            time.sleep(1)

    def run(self) -> None:
        print("Using user-defined target labels only:")
        for label in self.labels:
            print(f"  {label}")
        print()
        print("No predefined label dataset is used, and 'neutral' is not embedded as a CLAP label.")
        print("Neutral/base state is inferred when target scores stay below the off threshold.")
        print()
        print("Starting live CLAP activity amplifier.")
        print("Use wired headphones / a wired audio interface for lowest monitoring latency.")
        print("This script expects two input channels by default: hand1 and hand2.")
        print(
            f"Audio stream request: input_channels={self.channels}, "
            f"output_channels={self.output_channels}, "
            f"blocksize={self.blocksize} (0 = host/API default), latency={self.latency!r}"
        )
        print(
            f"Detection thresholds: on_threshold={self.on_threshold:.3f}, "
            f"off_threshold={self.off_threshold:.3f}"
        )
        print(
            f"Gain processing: base_gain_db={self.base_gain_db:.1f}, "
            f"boost_gain_db={self.boost_gain_db:.1f}, backend={self.gain_backend}"
        )
        if PADDLE_GAIN_IMPORT_ERROR is not None:
            print(
                "WARNING: paddle_stream.py could not be imported; using internal NumPy dB gain fallback. "
                f"Original import error: {PADDLE_GAIN_IMPORT_ERROR}"
            )
        print("Press Ctrl+C to stop.\n")

        self.worker.start()

        def open_stream(input_channels: int) -> sd.Stream:
            return sd.Stream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=(input_channels, self.output_channels),
                dtype="float32",
                blocksize=self.blocksize,
                latency=self.latency,
                callback=self.callback,
            )

        try:
            physical_input_channels = self.channels
            try:
                stream = open_stream(physical_input_channels)
            except sd.PortAudioError as first_exc:
                if self.allow_mono_input_fallback and self.channels == 2:
                    print(
                        "WARNING: Could not open a 2-channel input stream. "
                        "Falling back to a 1-channel physical input and duplicating it "
                        "to hand1 and hand2 for analysis. This is NOT two independent wrist mics. "
                        "Use --list-devices and --input-device to select a real stereo/two-channel input device."
                    )
                    physical_input_channels = 1
                    try:
                        stream = open_stream(physical_input_channels)
                    except sd.PortAudioError:
                        raise first_exc
                else:
                    raise

            with stream:
                actual_latency = getattr(stream, "latency", None)
                if actual_latency is not None:
                    print(f"Audio stream opened. Reported latency: {actual_latency}")
                print(
                    f"Logical hand channels: {self.channels}; "
                    f"physical input channels opened: {physical_input_channels}; "
                    f"output channels opened: {self.output_channels}"
                )
                status_thread = threading.Thread(target=self.print_status_loop, daemon=True)
                status_thread.start()
                while True:
                    time.sleep(0.5)
        except sd.PortAudioError as exc:
            print(
                "\nERROR: Could not open the requested audio stream.\n"
                f"Requested input_channels={self.channels}, output_channels={self.output_channels}.\n"
                "Use --list-devices to verify that your selected input device exposes "
                "two input channels for the two wrist microphones. If your headphones/output "
                "are mono, run with --output-channels 1 while keeping --channels 2.\n"
                f"Original PortAudio error: {exc}",
                file=sys.stderr,
            )
            raise
        except KeyboardInterrupt:
            pass
        finally:
            self.state.running = False
            print("\nStopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live CLAP prototype: use only the user-provided target labels, "
            "score each input channel independently, and boost audio when any "
            "channel detects one of the targets."
        )
    )
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "Free-form text describing one or more activities to amplify. "
            "Separate multiple activities with commas, semicolons, newlines, or 'and'. "
            "These strings become the only CLAP output labels. If omitted, you will be prompted."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help=(
            "Number of input hand-microphone channels to analyze independently. "
            "Defaults to 2 for two wrist microphones."
        ),
    )
    parser.add_argument(
        "--output-channels",
        type=int,
        default=2,
        help=(
            "Number of headphone/output channels. Defaults to 2. If your output "
            "device is mono, use --output-channels 1 while keeping --channels 2."
        ),
    )
    parser.add_argument(
        "--no-mono-input-fallback",
        action="store_true",
        help=(
            "Disable fallback that duplicates a 1-channel physical input into hand1/hand2 "
            "when a real 2-channel input device cannot be opened."
        ),
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=None,
        help=(
            "Optional fixed callback block duration in seconds. "
            "For lowest latency, leave this unset so PortAudio can choose blocksize=0. "
            "Example: 0.02 requests about a 20 ms callback block."
        ),
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=None,
        help=(
            "Optional fixed callback block size in frames. "
            "Use 0 for host/API default. If omitted and --block-duration is omitted, defaults to 0."
        ),
    )
    parser.add_argument(
        "--latency",
        type=parse_latency_arg,
        default=0.10,
        help=(
            "Requested sounddevice stream latency: 'low', 'high', 'default', "
            "or a number of seconds such as 0.02. Defaults to 0.10 seconds, "
            "which is more stable when the CPU is also running CLAP inference."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="CLAP analysis window. 2.0 s is a good starting point for live use.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="How often to update the activity score.",
    )
    parser.add_argument(
        "--analysis-buffer-seconds",
        type=float,
        default=None,
        help=(
            "How much recent audio to keep in the analysis buffer. "
            "Must be >= --window-seconds. Defaults to 3 * window_seconds."
        ),
    )
    parser.add_argument(
        "--analysis-queue-size",
        type=int,
        default=4,
        help=(
            "Maximum number of audio blocks queued for CLAP analysis. Small values "
            "keep predictions real-time by dropping stale analysis blocks."
        ),
    )
    parser.add_argument(
        "--base-gain-db",
        type=float,
        default=None,
        help="Pedalboard Gain amount, in dB, when no target activity is active. Defaults to 0 dB.",
    )
    parser.add_argument(
        "--boost-gain-db",
        type=float,
        default=None,
        help="Pedalboard Gain amount, in dB, when a target activity is active. Defaults to +12 dB.",
    )
    parser.add_argument(
        "--base-gain",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--boost-gain",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--on-threshold",
        type=float,
        default=0.20,
        help=(
            "Activity threshold. Gain turns on when any hand's max target score is "
            ">= this value. Defaults to 0.20."
        ),
    )
    parser.add_argument(
        "--off-threshold",
        type=float,
        default=None,
        help=(
            "Optional turn-off threshold. If omitted, this equals --on-threshold, "
            "so boost stays on as long as at least one hand's target score remains "
            ">= the activity threshold. Set lower than --on-threshold only if you "
            "want hysteresis."
        ),
    )
    parser.add_argument(
        "--on-windows-required",
        type=int,
        default=1,
        help=(
            "Number of consecutive analysis windows with max target score >= "
            "on_threshold required before gain turns on."
        ),
    )
    parser.add_argument(
        "--off-windows-required",
        type=int,
        default=1,
        help=(
            "Number of consecutive analysis windows with max target score < "
            "off_threshold required before gain turns off."
        ),
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help=(
            "CPU threads for PyTorch/CLAP inference. Default 1 prevents CLAP from "
            "starving the real-time audio callback, which can cause input overflow."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional sounddevice device name or index used for both input and output. "
            "Prefer --input-device and --output-device when your wrist mics and headphones are separate devices."
        ),
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Optional input device name or index for the wrist microphones.",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Optional output device name or index for the headphones.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    command = args.command
    if not command:
        command = input("What activities would you like to amplify? Separate multiple activities with commas: ").strip()
    if not command:
        print("ERROR: no target activities provided.")
        return 2

    # Use separate input/output devices when provided. This is important for
    # common setups where the wrist microphones are one device and headphones
    # are another. sounddevice accepts a tuple: (input_device, output_device).
    input_device = parse_device_arg(args.input_device)
    output_device = parse_device_arg(args.output_device)
    shared_device = parse_device_arg(args.device)
    if input_device is not None or output_device is not None:
        device_arg = (input_device, output_device)
    else:
        device_arg = shared_device

    # New gain control uses Pedalboard Gain(gain_db=...). Keep old linear
    # --base-gain/--boost-gain accepted as hidden legacy aliases and convert
    # them to dB when provided.
    base_gain_db = (
        float(args.base_gain_db)
        if args.base_gain_db is not None
        else (linear_gain_to_db(args.base_gain) if args.base_gain is not None else 0.0)
    )
    boost_gain_db = (
        float(args.boost_gain_db)
        if args.boost_gain_db is not None
        else (linear_gain_to_db(args.boost_gain) if args.boost_gain is not None else 12.0)
    )

    app = LiveCLAPAmplifier(
        command=command,
        sample_rate=args.sample_rate,
        channels=args.channels,
        output_channels=args.output_channels,
        block_duration=args.block_duration,
        blocksize=args.blocksize,
        latency=args.latency,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        analysis_buffer_seconds=args.analysis_buffer_seconds,
        analysis_queue_size=args.analysis_queue_size,
        base_gain_db=base_gain_db,
        boost_gain_db=boost_gain_db,
        on_threshold=args.on_threshold,
        off_threshold=args.off_threshold,
        on_windows_required=args.on_windows_required,
        off_windows_required=args.off_windows_required,
        device=device_arg,
        torch_num_threads=args.torch_num_threads,
        allow_mono_input_fallback=not args.no_mono_input_fallback,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
