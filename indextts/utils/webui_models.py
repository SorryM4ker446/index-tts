"""Installed local model identities and complete checkpoint validation."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_CATALOG = {
    "base": {"label": "IndexTTS 2.5（原模型）", "directory": ROOT / "models/IndexTTS-2.5"},
    "murasame": {"label": "丛雨训练模型（仅非商用）", "directory": ROOT / "models/murasame-index-tts-weights"},
}
XPU_RESTART_MESSAGE = (
    "XPU 后台已异常，需要重启服务。请重新双击 open-official-web.cmd，"
    "等待启动完成后刷新网页。"
)


def is_xpu_runtime_error(error):
    message = str(error).upper()
    return any(marker in message for marker in (
        "UR_RESULT_ERROR_", "UR BACKEND FAILED", "DEVICE LOST", "OUT OF MEMORY",
    ))


def model_selection_state(profile, restart_reason=None):
    if restart_reason or profile not in MODEL_CATALOG:
        return {"value": None, "interactive": False, "status": f"**{XPU_RESTART_MESSAGE}**"}
    return {"value": profile, "interactive": True,
            "status": f"当前模型：**{MODEL_CATALOG[profile]['label']}**"}


def installed_models():
    return [(entry["label"], key) for key, entry in MODEL_CATALOG.items()
            if (entry["directory"] / "gpt.pth").is_file() and (entry["directory"] / "config.yaml").is_file()]


def model_key(directory):
    directory = Path(directory).resolve()
    for key, entry in MODEL_CATALOG.items():
        if directory == entry["directory"].resolve():
            return key
    raise ValueError("This XPU WebUI accepts the two installed model profiles only")


def validate_checkpoint(model, checkpoint):
    """Require every unique parameter/buffer, allowing only registered aliases.

    Upstream registers the same GPT modules again for its inference wrapper.
    Those duplicate names are absent from the released checkpoint. Validate
    storage identities rather than silently skipping genuinely missing weights.
    """
    current = model.state_dict()
    unexpected = sorted(set(checkpoint) - set(current))
    wrong_shapes = [key for key in checkpoint if key in current and checkpoint[key].shape != current[key].shape]
    if unexpected or wrong_shapes:
        raise ValueError(f"Incompatible GPT checkpoint: unexpected={unexpected[:3]}, shapes={wrong_shapes[:3]}")

    def identity(tensor):
        return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))

    covered = {identity(current[key]) for key in checkpoint}
    missing = [key for key, tensor in current.items() if key not in checkpoint and identity(tensor) not in covered]
    if missing:
        raise ValueError("Incomplete GPT checkpoint: " + str(missing[:5]))
    return sorted(set(current) - set(checkpoint))


def same_auxiliary_assets(left, right):
    # These profiles differ only in GPT. Reuse the already resident auxiliary
    # networks only after checking their matching files and text frontend.
    for name in ("config.yaml", "codec.pth", "s2mel.pth", "wav2vec2bert_stats.pt",
                 "feat1.pt", "feat2.pt", "multilingual_zh_ja_yue_char_del.tiktoken"):
        def digest(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        if digest(left / name) != digest(right / name):
            raise ValueError("The installed profiles cannot share auxiliary networks: " + name)
