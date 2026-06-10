#!/usr/bin/env python3
"""Tiny local web server for the CLAP amplifier.

Serves a single-page UI and drives a StreamController from paddle_stream_new.
The UI lets the user pick which activities to amplify, set a live boost gain,
choose input/output devices, and (via a hidden debug panel) tune detector
parameters.

Run:
    python3 app.py
Then open http://127.0.0.1:8765 in a browser.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paddle_stream_new import StreamController

HERE = Path(__file__).resolve().parent
INDEX_FILE = HERE / "index.html"

DEFAULT_INPUT_DEVICE = "Sonic Presence SP-15 V2.0"
DEFAULT_OUTPUT_DEVICE = "External Headphones"
DEFAULT_BOOST_GAIN_DB = 12.0

# Default settings. These mirror the StreamController / detector defaults and are
# surfaced to the UI (sidebar + debug panel) so the user can see and change them.
DEFAULTS: dict = {
    "input_device_name": DEFAULT_INPUT_DEVICE,
    "output_device_name": DEFAULT_OUTPUT_DEVICE,
    "boost_gain_db": DEFAULT_BOOST_GAIN_DB,
    "base_gain_db": 0.0,
    "window_seconds": 1.0,
    "hop_seconds": 1.0,
    "on_threshold": 0.20,
    "off_threshold": None,
    "on_windows_required": 3,
    "off_windows_required": 7,
    "input_channels": 2,
    "torch_num_threads": 1,
    "analysis_queue_size": 4,
    "allow_feedback": False,
}

# Which settings are numeric (parsed as float) vs integer vs boolean.
_FLOAT_KEYS = {"boost_gain_db", "base_gain_db", "window_seconds", "hop_seconds",
               "on_threshold", "off_threshold"}
_INT_KEYS = {"on_windows_required", "off_windows_required", "input_channels",
             "torch_num_threads", "analysis_queue_size"}
_BOOL_KEYS = {"allow_feedback"}

_lock = threading.Lock()
_controller: StreamController | None = None
_settings: dict = dict(DEFAULTS)


def _coerce(key: str, value):
    if key in _BOOL_KEYS:
        return bool(value)
    if value is None or value == "":
        return None if key == "off_threshold" else DEFAULTS[key]
    if key in _FLOAT_KEYS:
        return float(value)
    if key in _INT_KEYS:
        return int(value)
    return str(value)


def _stopped_payload() -> dict:
    return {
        "running": False,
        "gain_state": "stopped",
        "base_gain_db": _settings["base_gain_db"],
        "boost_gain_db": _settings["boost_gain_db"],
        "active": False,
        "clap_enabled": False,
        "command": None,
        "on_threshold": _settings["on_threshold"],
        "input_device_name": _settings["input_device_name"],
        "output_device_name": _settings["output_device_name"],
        "scores": None,
        "status_lines": [],
        "error": None,
    }


def _status_payload() -> dict:
    with _lock:
        controller = _controller
        settings = dict(_settings)
    payload = controller.get_status() if controller is not None else _stopped_payload()
    payload["settings"] = settings
    return payload


def _list_devices() -> dict:
    try:
        from pedalboard.io import AudioStream

        return {
            "input": list(AudioStream.input_device_names),
            "output": list(AudioStream.output_device_names),
        }
    except Exception as exc:
        return {"input": [], "output": [], "error": str(exc)}


def _start(data: dict) -> dict:
    global _controller
    activities = data.get("activities") or []
    if not isinstance(activities, list):
        activities = []
    command = ", ".join(a.strip() for a in activities if isinstance(a, str) and a.strip())

    with _lock:
        # Merge any provided settings over the current ones.
        for key in DEFAULTS:
            if key in data:
                _settings[key] = _coerce(key, data[key])

        if _controller is not None:
            _controller.stop()
            _controller = None

        controller = StreamController(
            command=command or None,
            boost_gain_db=float(_settings["boost_gain_db"]),
            base_gain_db=float(_settings["base_gain_db"]),
            input_device_name=_settings["input_device_name"],
            output_device_name=_settings["output_device_name"],
            input_channels=int(_settings["input_channels"]),
            window_seconds=float(_settings["window_seconds"]),
            hop_seconds=float(_settings["hop_seconds"]),
            on_threshold=float(_settings["on_threshold"]),
            off_threshold=_settings["off_threshold"],
            on_windows_required=int(_settings["on_windows_required"]),
            off_windows_required=int(_settings["off_windows_required"]),
            torch_num_threads=int(_settings["torch_num_threads"]),
            analysis_queue_size=int(_settings["analysis_queue_size"]),
            allow_feedback=bool(_settings["allow_feedback"]),
            enable_clap=bool(command),
        )
        controller.start()
        _controller = controller
    return _status_payload()


def _stop() -> dict:
    global _controller
    with _lock:
        if _controller is not None:
            _controller.stop()
            _controller = None
    return _status_payload()


def _set_gain(boost_gain_db: float) -> dict:
    with _lock:
        _settings["boost_gain_db"] = float(boost_gain_db)
        controller = _controller
    if controller is not None:
        controller.set_gain(float(boost_gain_db))
    return _status_payload()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                body = INDEX_FILE.read_bytes()
            except OSError:
                self._send_json({"error": "index.html not found"}, code=500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            self._send_json(_status_payload())
        elif self.path == "/devices":
            self._send_json(_list_devices())
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self) -> None:  # noqa: N802
        data = self._read_json()
        try:
            if self.path == "/start":
                self._send_json(_start(data))
            elif self.path == "/stop":
                self._send_json(_stop())
            elif self.path == "/gain":
                boost = float(data.get("boost_gain_db", DEFAULT_BOOST_GAIN_DB))
                self._send_json(_set_gain(boost))
            else:
                self._send_json({"error": "not found"}, code=404)
        except Exception as exc:  # surface backend errors to the UI
            self._send_json({"error": str(exc)}, code=500)

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Local web UI for the CLAP amplifier.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--input-device-name", default=DEFAULT_INPUT_DEVICE)
    parser.add_argument("--output-device-name", default=DEFAULT_OUTPUT_DEVICE)
    args = parser.parse_args()

    _settings["input_device_name"] = args.input_device_name
    _settings["output_device_name"] = args.output_device_name

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CLAP amplifier UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop()
        server.shutdown()
        print("Server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
