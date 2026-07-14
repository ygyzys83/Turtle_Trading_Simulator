from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Callable


APP_HOST = "127.0.0.1"
APP_PORT = 8501
_WINDOWS_CONSOLE_HANDLER = None


def _is_windows() -> bool:
    return os.name == "nt"


def app_port_is_available(host: str = APP_HOST, port: int = APP_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def streamlit_arguments(project_root: Path, port: int = APP_PORT) -> list[str]:
    return [
        "streamlit",
        "run",
        str(project_root / "turtle_trading.py"),
        "--server.port",
        str(port),
    ]


def streamlit_command(project_root: Path, port: int = APP_PORT) -> list[str]:
    """Display-ready equivalent of the in-process Streamlit command."""
    return [sys.executable, "-m", "streamlit", *streamlit_arguments(project_root, port)[1:]]


def install_immediate_console_shutdown(exit_process: Callable[[int], None] = os._exit) -> None:
    """Make one Ctrl-C or console close terminate the single UI process immediately."""
    if not _is_windows():
        return
    import ctypes

    handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

    @handler_type
    def console_handler(event: int) -> bool:
        if event in {0, 1, 2, 5, 6}:  # Ctrl-C, Ctrl-Break, console close, logoff, shutdown.
            exit_process(130)
            return True
        return False

    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(console_handler, True):
        raise ctypes.WinError()
    global _WINDOWS_CONSOLE_HANDLER
    _WINDOWS_CONSOLE_HANDLER = console_handler


def run_streamlit_in_process(arguments: list[str]) -> int:
    """Run Streamlit inside this process so no UI child can be orphaned."""
    from streamlit.web import cli as streamlit_cli

    previous_arguments = sys.argv
    sys.argv = list(arguments)
    try:
        result = streamlit_cli.main()
        return int(result or 0)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = previous_arguments


def run_app(project_root: Path | None = None) -> int:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    if not app_port_is_available():
        print(
            f"AgentLoop Trader is already using http://{APP_HOST}:{APP_PORT}. "
            "Stop that terminal session before starting another UI.",
            file=sys.stderr,
        )
        return 1

    install_immediate_console_shutdown()
    return run_streamlit_in_process(streamlit_arguments(root))


def main() -> int:
    return run_app()
