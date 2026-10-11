# v0.1.0 Experimental Preview

首个研究预览版：在 Lunar Lake NPU4000 上，通过 npunlock、Level Zero 和自定义 SHAVE 内核
实现 PyTorch `intel_npu` 设备入口，并提供冻结 DPU 模型适配器。发布状态以 GitHub 实际 tag/release 为准。

## 包含内容

- FP16/真正 FP32 Tensor、已列明的算术/归约/二维矩乘、Linear/Conv2d/BatchNorm/池化、一阶梯度与基本 SGD。
- 自定义 SHAVE 源码、汇编、源码配套 ELF 与 manifest；宿主组件源码构建脚本和 requirements。
- 独立的冻结 FP16/INT8/INT4/NF4 Linear 及 SmolLM2-135M-Instruct 适配路径，包含会话/缓冲区复用与可选缓存。
- 安装、兼容性、性能口径、许可证、来源/reference、贡献说明与离线 CI。

## 已验证范围

Windows x64、Core Ultra 7 258V、NPU `8086:643e`、驱动 `32.0.100.5540`、图编译器 `8.3`、
Python 3.12、Torch 2.14.1+cpu。新 Conda 环境的宿主源码构建与真实设备冒烟已通过。
最新发布回归见 [RELEASE_READINESS.md](RELEASE_READINESS.md)，逐项契约见 [COMPATIBILITY.md](COMPATIBILITY.md)。

## 使用前须知

普通 FP16 `torch.mm` 仍走 FP32 SHAVE 路径，通用 GEMM 较慢；冻结适配器的 DPU 性能不能套用到任意模型。
FP16 SmallResNet 仅通过同权重逐步对照，15 步测试接近收敛时梯度相对 L2 误差最高约 22.65%；
W8 完整模型质量未全部过关。未认证任意 Hugging Face 模型、
AMP/广泛优化器、异步多流、GPU+NPU 协同、长上下文和其他设备/驱动组合。
低位支持不表示原生 4-bit 算力或通用低位训练。

运行无需 OpenVINO Python/runtime，仍需 Intel NPU 驱动及图编译器。源码包不包含模型、工具链、驱动、
宿主二进制或图 blob。修改 SHAVE C 需额外 MoviTools；目前按源码安装，暂无 wheel。

从 [INSTALL.md](INSTALL.md) 开始。性能数字沿用已标明日期/条件的历史报告，非通用排名或芯片峰值。
主要基础为 [npunlock](https://github.com/hsfzxjy/npunlock)；改编与参考来源见 [REFERENCES.md](REFERENCES.md)、
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE](LICENSE)。
