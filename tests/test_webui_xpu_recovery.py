"""Failed GPU operations must not leave an apparently usable model or page."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from indextts.utils.webui_models import is_xpu_runtime_error, model_selection_state
from indextts.utils.webui_xpu import XpuWebTTS


class RecoveryTests(unittest.TestCase):
    def test_empty_model_state_renders_recovery_instead_of_keyerror(self):
        for profile, reason in [(None, None), (None, "device lost"), ("base", "device lost")]:
            with self.subTest(profile=profile, reason=reason):
                state = model_selection_state(profile, reason)
                self.assertIsNone(state["value"])
                self.assertFalse(state["interactive"])
                self.assertIn("重启", state["status"])
        self.assertTrue(model_selection_state("murasame")["interactive"])

    def test_gpu_error_classification_does_not_disable_service_for_bad_input(self):
        for message in ["UR_RESULT_ERROR_OUT_OF_RESOURCES", "UR_RESULT_ERROR_DEVICE_LOST",
                        "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"]:
            self.assertTrue(is_xpu_runtime_error(RuntimeError(message)))
        self.assertFalse(is_xpu_runtime_error(ValueError("Reference audio is empty")))

    def test_fault_before_generation_is_recorded_and_requires_restart(self):
        tts = XpuWebTTS.__new__(XpuWebTTS)
        tts.active_model, tts.restart_reason = "murasame", None
        tts.generation_lock = threading.RLock()
        tts.xpu_evidence = {"model_profile": "murasame", "state": "ready"}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch("indextts.utils.webui_xpu.LOG_DIR", directory), \
                    patch("indextts.utils.webui_xpu.OUTPUT_DIR", directory / "audio"), \
                    patch("torch.xpu.synchronize", side_effect=RuntimeError("UR_RESULT_ERROR_OUT_OF_RESOURCES")):
                with self.assertRaisesRegex(RuntimeError, "重启"):
                    tts.infer(text="おかえりなさい。", lang="JA", spk_audio_prompt="reference.wav")
            record = json.loads(next((directory / "audio").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertIn("UR_RESULT_ERROR_OUT_OF_RESOURCES", record["error"])
            self.assertIsNone(tts.active_model)
            self.assertIn("OUT_OF_RESOURCES", tts.restart_reason)

    def test_partial_weight_copy_requires_restart_and_blocks_retry(self):
        tts = XpuWebTTS.__new__(XpuWebTTS)
        tts.active_model, tts.restart_reason = "base", None
        tts.generation_lock = threading.RLock()
        tts.compatible_profiles = {"base", "murasame"}
        tts.gpt = torch.nn.Linear(3, 2)
        tts.xpu_evidence = {"model_profile": "base", "state": "ready"}
        checkpoint = {name: tensor.clone() for name, tensor in tts.gpt.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            log_dir = Path(folder)
            owned = log_dir / "official-web-server-7861.json"
            other = log_dir / "official-web-server-9999.json"
            owned.write_text(json.dumps({"pid": os.getpid(), "state": "ready"}), encoding="utf-8")
            other.write_text(json.dumps({"pid": -1, "state": "ready"}), encoding="utf-8")
            with patch("indextts.utils.webui_xpu.LOG_DIR", log_dir), \
                    patch("torch.load", return_value=checkpoint), \
                    patch("torch.xpu.synchronize") as synchronize, \
                    patch.object(tts.gpt, "load_state_dict", side_effect=RuntimeError("UR_RESULT_ERROR_DEVICE_LOST")) as copy:
                with self.assertRaisesRegex(RuntimeError, "重启"):
                    tts.switch_model("murasame")
                self.assertIsNone(tts.active_model)
                self.assertEqual(json.loads(owned.read_text())['state'], "restart_required")
                self.assertEqual(json.loads(other.read_text())['state'], "ready")
                self.assertEqual(json.loads((log_dir / "official-xpu-model.json").read_text())["state"], "restart_required")
                calls = synchronize.call_count
                with self.assertRaisesRegex(RuntimeError, "重启"):
                    tts.switch_model("base")
                self.assertEqual(copy.call_count, 1)
                self.assertEqual(synchronize.call_count, calls)


if __name__ == "__main__":
    unittest.main()
