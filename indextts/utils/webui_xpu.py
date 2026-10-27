"""B580 support for the upstream WebUI; its controls and presets stay upstream."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
import uuid

import numpy as np
import soundfile as sf
import torch

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.webui_models import (
    MODEL_CATALOG, XPU_RESTART_MESSAGE, is_xpu_runtime_error, model_key,
    same_auxiliary_assets, validate_checkpoint,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/official"
LOG_DIR = ROOT / "logs"
SHUTDOWN = threading.Event()
DEVICE_POLICY = "speech_xpu_qwen_cpu_v1"

# This must run entirely in the browser: Python cannot send a final update
# after exiting. Only start checking after the shutdown request succeeds.
SHUTDOWN_STATUS_JS = """async () => {
    const retained = "已保存的音频和预设保留。再次使用请双击 open-official-web.cmd。";
    const result = (text, label) => [text, {
        __type__: "update", value: label, interactive: false
    }];
    const deadline = Date.now() + 15000;
    let disconnected = 0;
    while (Date.now() < deadline) {
        if (!navigator.onLine) break;
        try {
            await fetch(new URL("config", window.location.href), {
                cache: "no-store", signal: AbortSignal.timeout(1000)
            });
            disconnected = 0;
        } catch (error) {
            // A timeout can mean a busy server; it does not prove exit.
            disconnected = error.name === "TimeoutError" ? 0 : disconnected + 1;
            if (disconnected >= 2 && navigator.onLine) {
                return result("**服务已停止，可以关闭此页面。** " + retained, "服务已停止");
            }
        }
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    return result("**已发送关闭请求，但尚未确认后台断开。** " +
        "请运行 scripts/stop-official-web.ps1 检查并停止后台。" + retained, "关闭状态未确认");
}"""


def require_b580(device):
    if not torch.xpu.is_available():
        raise RuntimeError("B580 XPU is unavailable; no CPU fallback is allowed")
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    if "B580" not in torch.xpu.get_device_name(index).upper():
        raise RuntimeError("The selected XPU is not the requested Intel Arc B580")
    return index


class XpuWebTTS(IndexTTS2):
    def __init__(self, *, model_dir, device="xpu:0", use_bf16=True, use_qwen_emo=True):
        self.xpu_index = require_b580(device)
        if use_bf16 and not torch.xpu.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on the selected XPU")
        torch.xpu.set_device(self.xpu_index)
        torch.set_num_threads(1)
        self.generation_lock = threading.RLock()
        model_dir = Path(model_dir).resolve()
        self.auxiliary_model_dir = model_dir
        self.active_model = model_key(model_dir)
        self.restart_reason = None
        self.compatible_profiles = {self.active_model}
        qwen_dir = model_dir / "qwen0.6bemo4-merge"
        if not qwen_dir.is_dir():
            qwen_dir = ROOT / "models/IndexTTS-2.5/qwen0.6bemo4-merge"
        started = time.perf_counter()
        super().__init__(cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir),
                         device=f"xpu:{self.xpu_index}", use_bf16=use_bf16,
                         use_qwen_emo=use_qwen_emo, qwen_model_dir=str(qwen_dir),
                         use_cuda_kernel=False, use_deepspeed=False,
                         use_accel=False, use_torch_compile=False)
        torch.xpu.synchronize()
        networks = {name: getattr(self, name) for name in
                    ("gpt", "semantic_model", "semantic_codec", "s2mel", "campplus_model", "bigvgan")}
        devices = {name: sorted({str(parameter.device) for parameter in network.parameters()})
                   for name, network in networks.items()}
        if any(value != [f"xpu:{self.xpu_index}"] for value in devices.values()):
            raise RuntimeError("Some inference network parameters are not on the B580: " + str(devices))
        if self.qwen_emo is not None:
            devices["qwen_emotion"] = sorted({str(p.device) for p in self.qwen_emo.model.parameters()})
            if devices["qwen_emotion"] != ["cpu"]:
                raise RuntimeError("QwenEmotion must remain on CPU to preserve B580 VRAM")
        self.xpu_evidence = {"device": f"xpu:{self.xpu_index}", "gpu_name": torch.xpu.get_device_name(self.xpu_index),
                             "torch": torch.__version__, "use_bf16": use_bf16,
                             "model_dir": str(model_dir), "component_devices": devices,
                             "model_profile": self.active_model, "state": "ready",
                             "gpt_checkpoint": str(model_dir / "gpt.pth"),
                             "auxiliary_model_dir": str(self.auxiliary_model_dir),
                             "qwen_emotion_loaded": self.qwen_emo is not None,
                             "qwen_emotion_device": "cpu" if self.qwen_emo is not None else None,
                             "device_policy": DEVICE_POLICY,
                             "xpu_text_input_packing": "cpu",
                             "model_load_seconds": time.perf_counter() - started}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "official-xpu-model.json").write_text(
            json.dumps(self.xpu_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OFFICIAL XPU MODEL " + json.dumps(self.xpu_evidence, ensure_ascii=False), flush=True)

    def _require_ready(self):
        if self.restart_reason or self.active_model is None:
            raise RuntimeError(XPU_RESTART_MESSAGE)

    def _mark_restart_required(self, error):
        # A failed GPU copy may leave mixed weights. Never present either model
        # as usable or attempt further GPU operations in this process.
        self.restart_reason = str(error)[:2000]
        self.active_model = None
        self.xpu_evidence.update(state="restart_required", model_profile=None,
                                 restart_reason=self.restart_reason)
        (LOG_DIR / "official-xpu-model.json").write_text(
            json.dumps(self.xpu_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        for path in LOG_DIR.glob("official-web-server-*.json"):
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record.get("pid") == os.getpid():
                record.update(state="restart_required", restart_reason=self.restart_reason)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def switch_model(self, profile):
        with self.generation_lock:
            if SHUTDOWN.is_set():
                raise RuntimeError("服务正在关闭，请重新启动后再切换模型。")
            self._require_ready()
            if profile not in MODEL_CATALOG:
                raise ValueError("请选择已安装的合成模型。")
            if profile == self.active_model:
                return self.xpu_evidence
            target = MODEL_CATALOG[profile]["directory"]
            started = time.perf_counter()
            if profile not in self.compatible_profiles:
                same_auxiliary_assets(self.auxiliary_model_dir, target)
                self.compatible_profiles.add(profile)
            checkpoint = torch.load(target / "gpt.pth", map_location="cpu", weights_only=True, mmap=True)
            checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
            aliases = validate_checkpoint(self.gpt, checkpoint)
            try:
                torch.xpu.synchronize()
                self.active_model = None
                result = self.gpt.load_state_dict(checkpoint, strict=False)
                if result.unexpected_keys or sorted(result.missing_keys) != aliases:
                    raise RuntimeError("GPT weight loading did not match the validated aliases")
                torch.xpu.synchronize()
            except Exception as exc:
                self._mark_restart_required(exc)
                raise RuntimeError(XPU_RESTART_MESSAGE) from exc
            finally:
                del checkpoint
            for name in ("cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt", "cache_mel",
                         "cache_emo_cond", "cache_spk_audio_prompt", "cache_emo_audio_prompt"):
                setattr(self, name, None)
            self.gpt_path = str(target / "gpt.pth")
            self.active_model = profile
            self.xpu_evidence.update(model_profile=profile, model_dir=str(target),
                                     gpt_checkpoint=self.gpt_path,
                                     last_switch_seconds=time.perf_counter() - started)
            (LOG_DIR / "official-xpu-model.json").write_text(
                json.dumps(self.xpu_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            print("OFFICIAL MODEL SWITCH " + json.dumps(self.xpu_evidence, ensure_ascii=False), flush=True)
            return self.xpu_evidence

    def infer(self, *args, **kwargs):
        if args:
            raise TypeError("The official WebUI adapter expects named inference arguments")
        with self.generation_lock:
            if SHUTDOWN.is_set():
                raise RuntimeError("服务正在关闭，请重新启动后再生成。")
            self._require_ready()
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output = OUTPUT_DIR / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8] + ".wav")
            kwargs["output_path"] = str(output)
            if "num_beams" in kwargs:
                kwargs["num_beams"] = int(kwargs["num_beams"])
            row = {**self.xpu_evidence, "status": "running", "text": kwargs.get("text"),
                   "language": str(kwargs.get("lang", "ZH")).lower(), "output": str(output),
                   "reference": str(kwargs.get("spk_audio_prompt")),
                   "emotion_reference": kwargs.get("emo_audio_prompt"),
                   "emotion_vector": kwargs.get("emo_vector"), "emotion_text": kwargs.get("emo_text"),
                   "use_emo_text": kwargs.get("use_emo_text", False),
                   "emotion_alpha": kwargs.get("emo_alpha"), "duration_factor": kwargs.get("duration_factor", 1),
                   "parameters": {key: kwargs.get(key) for key in
                                  ("do_sample", "temperature", "top_p", "top_k", "num_beams",
                                   "repetition_penalty", "length_penalty", "max_mel_tokens", "max_text_tokens_per_segment")},
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            started = time.perf_counter()
            try:
                torch.xpu.synchronize()
                torch.xpu.reset_peak_memory_stats(self.xpu_index)
                with torch.inference_mode():
                    result = super().infer(**kwargs)
                torch.xpu.synchronize()
                row["generation_seconds"] = time.perf_counter() - started
                samples, rate = sf.read(output, dtype="float32", always_2d=True)
                if not np.isfinite(samples).all() or len(samples) / rate < 0.2 or np.max(np.abs(samples)) < 0.0001:
                    raise RuntimeError("Generated audio is invalid, too short or silent")
                row["audio"] = {"sample_rate": rate, "duration_seconds": len(samples) / rate,
                                "peak": float(np.max(np.abs(samples))),
                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
                row["xpu_peak_allocated_bytes"] = torch.xpu.max_memory_allocated(self.xpu_index)
                row["xpu_peak_reserved_bytes"] = torch.xpu.max_memory_reserved(self.xpu_index)
                row["status"] = "passed"
            except Exception as exc:
                row.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                           generation_seconds=time.perf_counter() - started)
                if is_xpu_runtime_error(exc):
                    self._mark_restart_required(exc)
                    raise RuntimeError(XPU_RESTART_MESSAGE) from exc
                raise
            finally:
                output.with_suffix(".json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                print("OFFICIAL RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
            return result


def write_server_record(port, script, model_info):
    record = {"pid": os.getpid(), "port": port, "host": "127.0.0.1", "url": f"http://127.0.0.1:{port}",
              "script": str(Path(script).resolve()), "model_dir": model_info["model_dir"],
              "qwen_emotion_loaded": model_info["qwen_emotion_loaded"], "state": "ready",
              "qwen_emotion_device": model_info["qwen_emotion_device"],
              "device_policy": model_info["device_policy"],
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (LOG_DIR / f"official-web-server-{port}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def add_shutdown_controls(demo):
    import gradio as gr

    def stop():
        if not SHUTDOWN.is_set():
            SHUTDOWN.set()
            receipt = LOG_DIR / f"official-web-shutdown-{time.time_ns()}.json"
            receipt.write_text(json.dumps({"pid": os.getpid(), "source": "browser_stop_button"}), encoding="utf-8")
            timer = threading.Timer(2.0, lambda: os._exit(0))
            timer.daemon = True
            timer.start()
        return "**正在关闭官方 WebUI 并释放显存，通常约需 2 秒。** 正在检查后台是否已断开。", gr.update(value="正在关闭…", interactive=False)

    with demo:
        gr.Markdown("B580 XPU · 结束使用后点击下方按钮关闭后台。正在生成的任务会中止，已保存音频和预设保留。")
        button = gr.Button("停止服务并释放显存", variant="stop")
        message = gr.Markdown("")
        request = button.click(stop, outputs=[message, button], queue=False, api_name=False)
        request.success(fn=None, outputs=[message, button], js=SHUTDOWN_STATUS_JS,
                        queue=False, api_name=False)
