"""Exercise WebUI device policy without importing Gradio or loading models."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


WEBUI_PATH = Path(__file__).resolve().parents[1] / "webui.py"


def _backend(available=False, memory_gb=12, bf16=False):
    return SimpleNamespace(
        is_available=Mock(return_value=available),
        get_device_properties=Mock(
            return_value=SimpleNamespace(total_memory=memory_gb * 1024 ** 3)
        ),
        is_bf16_supported=Mock(return_value=available and bf16),
    )


def _load_policy(monkeypatch, *, cuda=None, xpu=None, fp16=False,
                 force_qwen=False, is_v25=True, device=None, no_qwen=False):
    torch = SimpleNamespace(cuda=cuda or _backend())
    if xpu is not None:
        torch.xpu = xpu
    monkeypatch.setitem(sys.modules, "torch", torch)

    # Importing webui.py parses argv and initializes the model. Execute its
    # actual policy functions and assignments without those startup effects,
    # following the existing source-based WebUI tests.
    functions = {"detect_vram_gb", "build_tts"}
    assignments = {
        "LOW_VRAM_THRESHOLD_GB", "_vram_gb", "LOW_VRAM",
        "HALF_PRECISION", "LOAD_QWEN_EMO",
    }
    tree = ast.parse(WEBUI_PATH.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in functions
        or isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in assignments
            for target in node.targets
        )
    ]
    namespace = {
        "os": os,
        "cmd_args": SimpleNamespace(
            model_dir="checkpoints", fp16=fp16, qwen_emo=force_qwen,
            deepspeed=False, cuda_kernel=False, device=device, no_qwen_emo=no_qwen,
        ),
        "IS_V25": is_v25,
        "IndexTTS2": lambda **kwargs: kwargs,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WEBUI_PATH), "exec"), namespace)
    return namespace, torch


@pytest.mark.parametrize("memory_gb", [8, 10, 12])
def test_xpu_vram_is_detected_without_cuda(monkeypatch, memory_gb):
    xpu = _backend(available=True, memory_gb=memory_gb)
    policy, torch = _load_policy(monkeypatch, xpu=xpu)

    assert policy["_vram_gb"] == memory_gb
    xpu.get_device_properties.assert_called_once_with(0)
    torch.cuda.get_device_properties.assert_not_called()


def test_xpu_fp16_flag_enables_v25_bf16(monkeypatch):
    xpu = _backend(available=True, bf16=True)
    policy, torch = _load_policy(monkeypatch, xpu=xpu, fp16=True)

    kwargs = policy["build_tts"]()

    assert kwargs["use_bf16"] is True
    assert "use_fp16" not in kwargs
    xpu.is_bf16_supported.assert_called_once_with()
    torch.cuda.is_bf16_supported.assert_not_called()


@pytest.mark.parametrize("memory_gb,low_vram", [(8, True), (10, False), (12, False)])
def test_xpu_uses_existing_low_vram_policy(monkeypatch, memory_gb, low_vram):
    policy, _ = _load_policy(
        monkeypatch, xpu=_backend(available=True, memory_gb=memory_gb, bf16=True)
    )

    kwargs = policy["build_tts"]()

    assert policy["LOW_VRAM"] is low_vram
    assert kwargs["use_bf16"] is low_vram
    assert kwargs["use_qwen_emo"] is not low_vram


def test_qwen_can_still_be_forced_on_a_low_vram_xpu(monkeypatch):
    policy, _ = _load_policy(
        monkeypatch, xpu=_backend(available=True, memory_gb=8, bf16=True),
        force_qwen=True,
    )

    assert policy["build_tts"]()["use_qwen_emo"] is True


def test_xpu_without_bf16_keeps_full_precision_fallback(monkeypatch, capsys):
    policy, _ = _load_policy(
        monkeypatch, xpu=_backend(available=True, bf16=False), fp16=True
    )

    assert policy["build_tts"]()["use_bf16"] is False
    assert "falling back to full precision" in capsys.readouterr().out


@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_remains_preferred_when_both_backends_are_available(monkeypatch, bf16):
    cuda = _backend(available=True, memory_gb=16, bf16=bf16)
    xpu = _backend(available=True, memory_gb=8, bf16=not bf16)
    policy, _ = _load_policy(monkeypatch, cuda=cuda, xpu=xpu, fp16=True)

    kwargs = policy["build_tts"]()

    assert policy["_vram_gb"] == 16
    assert kwargs["use_bf16"] is bf16
    assert kwargs["use_qwen_emo"] is True
    xpu.get_device_properties.assert_not_called()
    xpu.is_bf16_supported.assert_not_called()


@pytest.mark.parametrize("has_xpu_module", [False, True])
def test_no_accelerator_keeps_full_precision(monkeypatch, has_xpu_module):
    policy, _ = _load_policy(
        monkeypatch, xpu=_backend() if has_xpu_module else None, fp16=True
    )

    assert policy["_vram_gb"] is None
    assert policy["LOW_VRAM"] is False
    assert policy["build_tts"]()["use_bf16"] is False


def test_v2_still_receives_use_fp16(monkeypatch):
    policy, _ = _load_policy(
        monkeypatch, xpu=_backend(available=True, memory_gb=8), is_v25=False
    )

    kwargs = policy["build_tts"]()

    assert kwargs["use_fp16"] is True
    assert "use_bf16" not in kwargs


def test_precision_probe_is_skipped_when_half_precision_is_disabled(monkeypatch):
    xpu = _backend(available=True, bf16=True)
    policy, _ = _load_policy(monkeypatch, xpu=xpu)

    assert policy["build_tts"]()["use_bf16"] is False
    xpu.is_bf16_supported.assert_not_called()


def test_explicit_xpu_device_overrides_cuda_detection(monkeypatch):
    adapter = SimpleNamespace(require_b580=Mock(return_value=1))
    monkeypatch.setitem(sys.modules, "indextts.utils.webui_xpu", adapter)
    cuda = _backend(available=True, memory_gb=24)
    xpu = _backend(available=True, memory_gb=12)

    policy, _ = _load_policy(monkeypatch, cuda=cuda, xpu=xpu, device="xpu:1")

    assert policy["_vram_gb"] == 12
    adapter.require_b580.assert_called_once_with("xpu:1")
    xpu.get_device_properties.assert_called_once_with(1)
    cuda.get_device_properties.assert_not_called()


def test_explicit_xpu_factory_respects_disabled_qwen(monkeypatch):
    factory = Mock(side_effect=lambda **kwargs: kwargs)
    adapter = SimpleNamespace(require_b580=Mock(return_value=0), XpuWebTTS=factory)
    monkeypatch.setitem(sys.modules, "indextts.utils.webui_xpu", adapter)
    policy, _ = _load_policy(
        monkeypatch, xpu=_backend(available=True), device="xpu:0",
        fp16=True, force_qwen=True, no_qwen=True,
    )

    kwargs = policy["build_tts"]()

    assert kwargs["device"] == "xpu:0"
    assert kwargs["use_bf16"] is True
    assert kwargs["use_qwen_emo"] is False
    factory.assert_called_once()
