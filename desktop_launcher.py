"""Local desktop launcher for the BSLVC dashboard.

Shows a small control window immediately so users can see startup progress,
then starts the Dash server and opens the default browser. The window stays
open for the lifetime of the app and provides:

  - Animated startup status (loading modules → starting server → ready)
  - "Open in Browser" button to (re)open the tab at any time
  - "Quit" button that gracefully shuts down the server
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


# ── Runtime root ──────────────────────────────────────────────────────────────

def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        # Return the bundle root (directory containing the executable) rather
        # than sys._MEIPASS (_internal/). PyInstaller 6 moved _MEIPASS into
        # _internal/, so using sys.executable.parent keeps data files in the
        # user-visible bundle root while _internal/ stays as Dash's CSS store.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _configure_runtime_environment(root: Path) -> None:
    os.chdir(root)
    os.environ.setdefault("APP_DIR", str(root))
    os.environ.setdefault("DATA_DIR", str(root / "assets" / "data"))
    os.environ.setdefault("DATABASE_PATH", str(root / "assets" / "data" / "BSLVC_sqlite.db"))
    os.environ.setdefault(
        "ADVANCED_MAPPING_PATH",
        str(root / "assets" / "data" / "advanced_regional_mapping.csv"),
    )
    os.environ.setdefault("CACHE_DIR", str(root / "cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))
    os.environ.setdefault("ENABLE_URL_CACHE_CLEAR", "false")
    os.environ.setdefault("ASSETS_FOLDER", str(root / "assets"))


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.15)
    return False


# ── Control window ────────────────────────────────────────────────────────────

class ControlWindow:
    """Small tkinter window showing server status and quit/open controls."""

    STATUS_LOADING  = "loading"
    STATUS_STARTING = "starting"
    STATUS_READY    = "ready"
    STATUS_ERROR    = "error"

    def __init__(self, url: str) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._url = url
        self._tk = tk
        self._lock = threading.Lock()
        self._status = self.STATUS_LOADING
        self._quit_event = threading.Event()

        win = tk.Tk()
        win.title("BSLVC Dashboard")
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", self._on_quit)
        self._win = win

        pad = {"padx": 20, "pady": 5}
        title_font = tkfont.Font(family="sans-serif", size=12, weight="bold")
        tk.Label(win, text="BSLVC Dashboard", font=title_font).pack(padx=20, pady=(16, 4))

        self._lbl = tk.Label(
            win, text="", fg="#777777", width=44, wraplength=320, justify="left"
        )
        self._lbl.pack(**pad)

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=(8, 16))

        self._open_btn = tk.Button(
            btn_frame, text="Open in Browser", width=18,
            command=self._on_open_browser, state=tk.DISABLED,
        )
        self._open_btn.pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="Quit", width=10, command=self._on_quit).pack(
            side=tk.LEFT, padx=8
        )

        win.update_idletasks()
        w  = win.winfo_reqwidth()
        h  = win.winfo_reqheight()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

        self._schedule_tick()

    # ── Thread-safe state setters ─────────────────────────────────────────────

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    @property
    def quit_event(self) -> threading.Event:
        return self._quit_event

    # ── UI update loop (main thread only via after()) ─────────────────────────

    def _schedule_tick(self) -> None:
        self._win.after(500, self._tick)

    def _tick(self) -> None:
        with self._lock:
            status = self._status

        now = time.monotonic()

        if status == self.STATUS_LOADING:
            dots = "." * (int(now * 1.5) % 4)
            self._lbl.config(
                text=f"Starting Dashboard{dots}\n(This may take a few moments)",
                fg="#888888",
            )
            self._open_btn.config(state=self._tk.DISABLED)

        elif status == self.STATUS_STARTING:
            self._lbl.config(text="Starting server…", fg="#888888")
            self._open_btn.config(state=self._tk.DISABLED)

        elif status == self.STATUS_READY:
            self._lbl.config(text=f"✓  Running — {self._url}", fg="#2a7a2a")
            self._open_btn.config(state=self._tk.NORMAL)

        elif status == self.STATUS_ERROR:
            self._lbl.config(
                text="Failed to start. Check the terminal for details.", fg="#cc0000"
            )
            self._open_btn.config(state=self._tk.DISABLED)

        if not self._quit_event.is_set():
            self._schedule_tick()

    def _on_open_browser(self) -> None:
        webbrowser.open(self._url, new=1, autoraise=True)

    def _on_quit(self) -> None:
        self._quit_event.set()
        self._win.quit()

    def run(self) -> None:
        """Block the calling thread until the user quits."""
        self._win.mainloop()


# ── Server bootstrap (background thread) ─────────────────────────────────────

def _start_server(
    host: str,
    port: int,
    url: str,
    window: ControlWindow,
    http_server_ref: list,
) -> None:
    from werkzeug.serving import make_server

    try:
        window.set_status(ControlWindow.STATUS_LOADING)

        from app import app  # noqa: PLC0415  (intentional deferred import)

        window.set_status(ControlWindow.STATUS_STARTING)

        srv = make_server(host, port, app.server, threaded=True)
        http_server_ref.append(srv)

        srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        srv_thread.start()

        if not _wait_for_server(host, port):
            raise RuntimeError(
                f"Server did not respond within the timeout on {host}:{port}."
            )

        window.set_status(ControlWindow.STATUS_READY)

        if os.environ.get("NO_BROWSER", "false").strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            webbrowser.open(url, new=1, autoraise=True)

    except Exception as exc:
        window.set_status(ControlWindow.STATUS_ERROR)
        print(f"[desktop_launcher] Error: {exc}", file=sys.stderr)
        raise


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = _runtime_root()
    _configure_runtime_environment(root)

    host = os.environ.get("DASH_HOST", "127.0.0.1")
    port = int(os.environ.get("DASH_PORT", "27589"))
    if port == 0:
        port = _find_free_port(host)
    url = f"http://{host}:{port}"

    window = ControlWindow(url=url)
    http_server_ref: list = []

    server_thread = threading.Thread(
        target=_start_server,
        args=(host, port, url, window, http_server_ref),
        daemon=True,
    )
    server_thread.start()

    window.run()  # blocks on main thread until the user quits

    # Graceful shutdown after the window closes.
    if http_server_ref:
        http_server_ref[0].shutdown()
        http_server_ref[0].server_close()


if __name__ == "__main__":
    import multiprocessing
    # Required for PyInstaller + multiprocessing/diskcache background callbacks.
    # Spawned worker subprocesses re-execute the frozen binary; freeze_support()
    # detects that case and exits before reaching main(), preventing new windows.
    multiprocessing.freeze_support()
    main()