"""Runs the check scripts against the INSTALLED package (pip install -e . or pip install .).
Each script is a long list of assertions; pytest reports one result per script."""
import runpy, shutil
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).parent / "scripts"

def _run(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)               # scripts write small wav/mp4 files into the working directory
    runpy.run_path(str(SCRIPTS / name), run_name="__main__")

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

@needs_ffmpeg
def test_audio_and_muxing(tmp_path, monkeypatch):
    _run("check_audio.py", tmp_path, monkeypatch)

@needs_ffmpeg
def test_agent_checker_and_loop(tmp_path, monkeypatch):
    _run("check_agent.py", tmp_path, monkeypatch)

def test_piper_voice_if_configured(tmp_path, monkeypatch):
    _run("check_piper.py", tmp_path, monkeypatch)       # skips itself unless PIPER_BIN and PIPER_MODEL are set
