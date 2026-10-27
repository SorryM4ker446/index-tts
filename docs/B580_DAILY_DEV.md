# B580 日常开发分支

`dev/b580-xpu` 保存本机 Windows / Intel Arc B580 的日常适配。
基线为上游 `d9e41aac89fd00b3d71497fddb287b7f24613712`，不代表官方通用支持。
上游 PR #830 仍只提交 WebUI 显存与 BF16 检测；本分支包含这两项修复及下列本地改动。

## 已保存的改动

- 关闭服务后由浏览器检查后台断开并显示最终状态，避免页面一直停在“正在关闭”。
- 原模型与丛雨模型串行切换；校验辅助资源一致、GPT 权重形状与完整性，清空旧模型缓存。
  模型切换或生成遇到设备错误后标记为需要重启，避免使用部分更新的权重以及 `KeyError: None`。
- 预设保存模型和语言，加载时恢复对应模型；生成与模型切换共用锁。
- XPU 模式将 QwenEmotion 固定为 CPU/FP32，六个语音网络保留在 B580 上，降低显存压力。
- XPU 上少量整数 token 在 CPU 筛选、补齐和生成位置编号，再返回 XPU 执行嵌入与生成。
  这是本机兼容处理，尚未证实某个算子或驱动是唯一根因，也不保证任意长文本都不会 OOM。
- 示例音频全部存在时跳过联网探测。
- CUDA 不可用时查询 XPU 显存和 BF16 能力。2.5 的旧选项 `--fp16` 对应 `use_bf16`，不启用 FP16 GPT 推理。

代码范围：`webui.py`、`indextts/infer_v2_5.py`、`indextts/gpt/model_v2.py`、
`indextts/utils/examples_downloader.py`，以及新增的 `webui_models.py`、`webui_xpu.py`。

## 本机目录与启动

适配沿用仓库外层的现有工作目录，模型目录按此布局识别：

```text
indextts-b580-xpu/
  .venv/
  models/
    IndexTTS-2.5/
    murasame-index-tts-weights/    # 可选；仅非商用
  outputs/official/
  logs/
  repo/                         # 本仓库
```

原模型需要完整的官方辅助权重与文本资源。可选丛雨配置需与原模型的辅助资源一致，
并提供完整 GPT 权重。模型、参考录音、用户预设、环境、日志及生成音频均未上传。
仓库外的双击入口和启停脚本也未包含在本次提交中。

已有配置好的 XPU 环境时，在仓库目录执行：

```powershell
& ..\.venv\Scripts\python.exe -u .\webui.py --version 2.5 `
  --model_dir ..\models\IndexTTS-2.5 --device xpu:0 --fp16 `
  --qwen_emo --ui-language zh_CN --host 127.0.0.1 --port 7861
```

可用 `--no-qwen-emo` 替换 `--qwen_emo` 跳过文字情绪模型。
停止按钮中关于双击重启和 PowerShell 停止脚本的提示针对原工作目录；
仅检出本仓库时，请在启动终端结束进程，再重新执行上述命令。

日常环境使用 PyTorch/Torchaudio 2.8.0+xpu 与 Gradio 5.45.0，
不是完全照官方 CUDA 锁文件重建的环境。本次不修改依赖声明和 `uv.lock`。
默认 `uv sync` 会选择官方配置的 CUDA 后端，不适用于直接恢复此 B580 环境。

## 验证范围

本分支保留所有官方测试原文，并提交 `tests/test_webui_xpu_*.py` 和
`tests/test_examples_offline.py`，覆盖设备检测、权重完整性、设备异常恢复和离线示例检查。
这些新增检查不需要 GPU 或模型权重，随官方非 GPU 测试一起运行：

```powershell
python -m pytest -m "not gpu" -q
```

官方非 GPU 测试、新增检查与真实硬件验证应分别记录，不得把跳过的官方 GPU 测试、
历史网页验证或本地兼容处理声称为托管 CI 已通过。上游 PR #830 不包含这些新增测试。

2026-10-27 在此前隔离的锁文件基线加 XPU 后端例外环境中运行上述命令：
`207 passed, 22 deselected, 35 subtests passed`，其中官方用例 180 项、新增用例 27 项。
本轮未运行 22 项官方 GPU 测试，也未重新执行浏览器与长文本硬件验证。
