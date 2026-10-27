"""Installed examples must not require a network probe on every startup."""

from unittest.mock import Mock

import pytest

from indextts.utils import examples_downloader


@pytest.fixture
def examples(tmp_path, monkeypatch):
    monkeypatch.setattr(examples_downloader, "_EXAMPLES_DIR", str(tmp_path))
    monkeypatch.setattr(examples_downloader, "get_required_files", lambda: ["voice.wav"])
    (tmp_path / "voice.wav").write_bytes(b"existing audio")
    return tmp_path


def test_installed_examples_skip_network_probe(examples, monkeypatch):
    probe = Mock(side_effect=AssertionError("Unexpected network probe"))
    download = Mock(side_effect=AssertionError("Unexpected download"))
    monkeypatch.setattr(examples_downloader, "need_proxy", probe)
    monkeypatch.setattr(examples_downloader, "_download_file", download)

    examples_downloader.ensure_examples_available()

    probe.assert_not_called()
    download.assert_not_called()
    assert (examples / "voice.wav").read_bytes() == b"existing audio"


def test_missing_examples_still_download(examples, monkeypatch):
    monkeypatch.setattr(examples_downloader, "get_required_files", lambda: ["voice.wav", "missing.wav"])
    probe = Mock(return_value=False)
    download = Mock()
    monkeypatch.setattr(examples_downloader, "need_proxy", probe)
    monkeypatch.setattr(examples_downloader, "_download_file", download)

    examples_downloader.ensure_examples_available()

    probe.assert_called_once_with()
    download.assert_called_once_with(
        examples_downloader._HF_RAW_URL + "/examples/missing.wav",
        str(examples / "missing.wav"), min_size=100, timeout=120,
    )


def test_force_refresh_still_downloads_existing_examples(examples, monkeypatch):
    monkeypatch.setattr(examples_downloader, "need_proxy", lambda: False)
    download = Mock()
    monkeypatch.setattr(examples_downloader, "_download_file", download)

    examples_downloader.ensure_examples_available(force=True)

    assert download.call_count == 1
    assert download.call_args.args[1] == str(examples / "voice.wav")
