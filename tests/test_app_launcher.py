import sys
from unittest.mock import Mock

from agentloop_trader import app_launcher


def test_streamlit_command_uses_current_python_app_and_fixed_port(tmp_path):
    command = app_launcher.streamlit_command(tmp_path)

    assert command == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(tmp_path / "turtle_trading.py"),
        "--server.port",
        "8501",
    ]


def test_launcher_refuses_to_start_when_app_port_is_in_use(monkeypatch, tmp_path):
    monkeypatch.setattr(app_launcher, "app_port_is_available", lambda: False)

    result = app_launcher.run_app(tmp_path)

    assert result == 1


def test_launcher_runs_streamlit_in_the_same_process(monkeypatch, tmp_path):
    install_handler = Mock()
    run_streamlit = Mock(return_value=0)
    monkeypatch.setattr(app_launcher, "app_port_is_available", lambda: True)
    monkeypatch.setattr(app_launcher, "install_immediate_console_shutdown", install_handler)
    monkeypatch.setattr(app_launcher, "run_streamlit_in_process", run_streamlit)

    result = app_launcher.run_app(tmp_path)

    assert result == 0
    install_handler.assert_called_once_with()
    run_streamlit.assert_called_once_with([
        "streamlit", "run", str(tmp_path / "turtle_trading.py"), "--server.port", "8501",
    ])


def test_non_windows_console_shutdown_installs_nothing(monkeypatch):
    monkeypatch.setattr(app_launcher, "_is_windows", lambda: False)

    app_launcher.install_immediate_console_shutdown(Mock())

    assert app_launcher._WINDOWS_CONSOLE_HANDLER is None
