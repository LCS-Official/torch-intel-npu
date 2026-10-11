# PyTorch backend for Intel NPU4000

让 Lunar Lake 的 Intel NPU 执行 PyTorch Tensor 运算、小网络训练与显式适配的模型推理。
项目基于 [npunlock](https://github.com/hsfzxjy/npunlock)，实现 `intel_npu` 设备后端、自定义 SHAVE 内核、
冻结模型的 DPU 图执行，以及缓冲区、会话和编译图复用。

**当前版本：v0.1.0 Experimental Preview，更新于 2026-10-11。**
已在本机完成独立 Conda 环境的宿主组件源码构建与真实 NPU 验证；
发布状态见 [GitHub Releases](https://github.com/LCS-Official/torch-intel-npu/releases)。
目前适合 NPU 算子与模型适配研究，尚未达到通用 CUDA/XPU 后端的兼容性。

[安装说明](INSTALL.md) · [兼容性矩阵](COMPATIBILITY.md) · [性能实测](#性能实测) · [未来工作](#未来工作) · [来源与引用](REFERENCES.md)

首发材料与剩余操作见 [发布验收](RELEASE_READINESS.md)，版本正文见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。
报告问题与贡献请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 两条执行路径

| 路径 | 使用方式 | 当前适用范围 |
|---|---|---|
| PyTorch 设备后端 | `tensor.to("intel_npu")`、`model.to("intel_npu")` | 已支持的 FP16/FP32 算子、一阶梯度和基本 SGD；主要使用自定义 SHAVE 计算 |
| 冻结模型适配器 | `QuantizedLinear`、`FrozenSmolLayer`、`FrozenSmolStack` | 显式生成融合图，用 DPU 执行矩乘；用于 FP16 和部分低位推理 |

**普通 FP16 `torch.mm` 目前仍通过 FP32 SHAVE 路径计算并转回 FP16，不会自动使用适配器的 DPU FP16 矩乘。**
因此，下文模型/DPU 性能不能套用到任意 `.to("intel_npu")` 模型。冻结适配器也不提供 LLM 训练。

默认运行不需要 OpenVINO Python/runtime 包，仍依赖 Intel NPU 系统驱动及其图编译器。
设备通过 PyTorch PrivateUse1 注册为 `intel_npu:0`，没有合并进 `torch.xpu`。

## 快速开始

已验证平台为 **Windows x64 / Core Ultra 7 258V / NPU `8086:643e`**，
驱动 `32.0.100.5540`、图编译器 `8.3`、Python 3.12.15、Torch 2.14.1+cpu、NumPy 2.5.3。
目前只对这一设备/驱动组合完成验证；准备 MSVC x64 与 Windows SDK 后，在源码根目录运行：

```powershell
conda create -p .\.conda\npu --no-default-packages --override-channels -c conda-forge python=3.12 pip -y
.\.conda\npu\python.exe -m pip install --no-cache-dir -r requirements-build.txt
# 在 x64 Visual Studio Developer PowerShell 中构建
.\build_npu.ps1 -Python .\.conda\npu\python.exe
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\tools\smoke_install.py
```

独立 MSVC 工具链可通过 `-Toolchain` 指定，完整步骤见 [INSTALL.md](INSTALL.md)。
源码构建生成宿主 DLL、worker、session 接口及 Torch PYD；默认使用随源码提供、带哈希校验的 SHAVE ELF。
修改 SHAVE C 另需外部 MoviTools。当前没有可直接 `pip install` 的项目 wheel。

| 依赖清单 | 直接依赖 | 用途 |
|---|---|---|
| [requirements.txt](requirements.txt) | Torch CPU wheel、NumPy | 核心运算与已支持的训练 |
| [requirements-models.txt](requirements-models.txt) | 核心 + safetensors、Transformers | 检查点、tokenizer 与模型示例 |
| [requirements-build.txt](requirements-build.txt) | 核心 + CMake、Ninja | 从源码构建；构建后可卸载这两个工具 |

以下保存为 `my_test.py`，用同一 `run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\my_test.py` 入口运行：

```python
import torch
import torch_intel_npu  # 在首次 backward 前导入，注册 intel_npu:0

x = torch.arange(17, dtype=torch.float32).to("intel_npu")
print((x + 70000).cpu())

model = torch.nn.Linear(4, 3).to("intel_npu")
inputs = torch.randn(2, 4).to("intel_npu")  # 随机数在 CPU 生成
target = torch.zeros(2, 3).to("intel_npu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(inputs), target)
loss.backward()
optimizer.step()
print(loss.detach().cpu())
```

## 当前能力

| 功能 | 已实现与验证范围 |
|---|---|
| NPU Tensor | FP16/FP32 存储、CPU/NPU 复制、部分密集转置与布局处理；Level Zero shared allocation |
| 算子与训练 | 加减乘除、部分一元运算/归约、二维矩乘、Linear、Conv2d、BatchNorm、ReLU、池化、MSE；有限参数组合的一阶 autograd / 基本 SGD |
| 真正 FP32 | SHAVE 上的 FP32 算术、补偿归约与矩乘；未用 FP16/TF32 替代这条计算路径，GEMM 性能仍有限 |
| FP16 模型推理 | SmolLM2-135M-Instruct 的完整 30 层 decoder、prefill、KV cache、decode；decoder/KV 在 NPU，embedding、最终 norm、lm_head、采样在 CPU |
| 低位推理 | 显式 `QuantizedLinear` 的 INT8、INT4、NF4 权重与分组实验；W8A16 / W8A8 图路径。INT4/NF4 支持不等于原生 4-bit 乘加，也不等于通用低位 Tensor |
| 运行时优化 | 图融合、原生会话复用、链式提交、缓冲区复用、可选磁盘图缓存；部分 FP16 prefill 缓存命中可跳过 IR 重建 |
| 安装基线 | 新 Conda 环境从源码构建；44 项离线测试、两套原生 session 回归、真实 NPU 运算/训练/单层模型冒烟通过 |

完整能力以[算子/参数/梯度兼容性矩阵](COMPATIBILITY.md)及对应测试为准；表中的模块名称不代表所有参数组合都受支持。
最近的安装验证见 [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)，历史算子/训练细节见 [TORCH_NPU_README.md](TORCH_NPU_README.md)。

### 已知边界

- 设备 Tensor 当前支持 rank 0–4、1–1,048,576 元素；中间工作区还有约束。这是软件边界，不是机器的物理内存上限。
- SmolLM 适配器固定 64-position KV 容量，batch 上限 8，仍受形状与工作区限制；不能直接运行任意 Hugging Face 模型或长上下文。
- 纯 FP16 深层训练仍有数值限制；AMP、二阶梯度、跨 CPU/NPU 可微复制、广泛优化器及任意偏移/重叠视图尚未覆盖。
- 通用 BF16/整数设备 Tensor、设备 RNG、异步多流和 GPU+NPU 协同尚未实现；不支持的算子报错，不静默回退 CPU 算术。
- W8 完整模型校准尚未全部通过既有质量门槛；INT4/NF4 仍以 Linear/小网络实验为主，不能宣称完整低位 LLM 已适配。当前模型评估优先显式选择 FP16。
- CPU/GPU/NPU 共用物理内存不代表零搬运、零同步；统一内存不能消除驱动内部 DMA 和精度/布局转换。

## 性能实测

以下是已保存报告中的历史测量，**不是芯片峰值或通用模型排名**；本次首页整理未重跑性能测试。
各表保留独立的日期、执行路径和计时口径。

### 模型生成：热请求

2026-10-10，Core Ultra 7 258V / Arc 140V / NPU4000，SmolLM2-135M-Instruct 完整 30 层，
固定 24-token 提示词、greedy 生成 32 token；额外预热 2 次，正式 7 次取中位数。
下表单位为 **整个 batch 合计 token/s**，包含 CPU 两端、每步 token 返回主机及同步，排除加载和首次编译。

| 路线 | batch 1 | batch 4 | batch 8 |
|---|---:|---:|---:|
| CPU FP16，8 线程 | 30.668 | 69.064 | 108.509 |
| Arc 140V FP16，eager | 25.446 | 105.455 | 207.850 |
| Arc 140V FP16，SDPA | 32.325 | 122.545 | 196.090 |
| NPU FP16 decoder + CPU 两端 | 72.925 | 167.394 | 288.046 |

batch 8 的 NPU 288.046 token/s 对应每请求约 36.01 token/s，不能当作单请求速度。
这些 NPU 配置与 CPU FP16 参考的最大 logits 相对 L2 误差约 0.74%–1.12%，固定测试中的生成 token 一致；
这不代替完整语言模型质量评测。

CPU/GPU 使用 Transformers 模型，NPU 使用专门融合的适配器，设备放置和 Torch 环境也有差异；
桌面负载与频率未锁定。GPU batch 8 配对复核存在明显波动，未据此承诺稳定排名或速度比。
完整条件、原始数值摘要和质量门槛见 [模型对照报告](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)。

### 首 token：已有缓存的新进程

2026-10-11，同一模型、FP16、batch 1、24-token 提示词，使用相同 30 份 prefill 编译图，
各 3 个新进程交替对照。磁盘缓存均已建立，区别是是否提前命中并跳过 IR 重建。

| 计时口径 | 完整 IR 路径 | 提前缓存命中 | 中位耗时降低 |
|---|---:|---:|---:|
| 模型准备后到首 token | 2.009 秒 | 0.967 秒 | 51.8% |
| Python 脚本入口到首 token | 6.795 秒 | 5.491 秒 | 19.2% |

第二行包含库导入、初始化及模型准备，不含 PowerShell/Python 解释器启动。
**0.967 秒依赖已有磁盘缓存**：同轮无缓存首次 prefill 编译样本约 28.56 秒，不能把缓存命中收益用于首次安装体验。
本次改动没有证明热吞吐提升。详见 [首 token 优化报告](npunlock-probe/research/results/startup-ir-cache/REPORT.md)。

图磁盘缓存默认关闭，需显式设置 `NPUNLOCK_GRAPH_CACHE`；每目录容量上限 512 MiB。
删除缓存后下次需要重新编译，运行命令与空间约定见 [安装说明](INSTALL.md) 和 [空间记录](WORKSPACE_STORAGE.md)。

### 矩乘：指定形状的有效吞吐

2026-10-05，逻辑操作数为 `2*M*K*N`，形状均为 `1024×1024 @ 1024×1024`，冻结权重、FP16 输出。
输入/输出预分配，排除编译和显式上传/下载，包含驱动调用、图内 DMA 与 fence；31 次取中位数。

| 执行路径 | 中位耗时 | 有效吞吐 |
|---|---:|---:|
| DPU FP16 | 0.6733 ms | 3.18949 TFLOPS |
| DPU W8A16 | 0.6308 ms | 3.40438 TOp/s |
| DPU W8A8 | 0.5462 ms | 3.93168 TOPS |

W8A16 只有权重量化，不能把该项称为全 INT8 TOPS；W8A8 已检查实际 8-bit DPU 操作数。
同轮普通 FP32 `torch.mm` 的 256 方阵为 614.8185 ms、约 0.05458 GFLOPS，走 SHAVE 分块与合并路径，
与上述 DPU 路径的精度、形状和计时范围不同。它提示当前通用 FP32 GEMM 仍很慢，不能用于推断整个芯片的 FP32 峰值。
详见 [矩乘吞吐报告](npunlock-probe/research/results/throughput-20261005/REPORT.md)。

## 未来工作

以下是按当前证据排序的方向，不是已实现功能或交付日期承诺。

| 优先级 | 方向 | 下一步与验收依据 |
|---|---|---|
| 近期 | 扩大发布验证 | 本机 Quick/非 Quick 结果见 [发布验收](RELEASE_READINESS.md)；补充长时间压力回归与另一台干净 Windows 安装验证 |
| 近期 | 首 token 与运行时开销 | 剖析缓存命中的 ELF/DPU 重复解析、库导入与模型重复准备；保持同等校验，做交替 A/B 与输出一致性检查 |
| 近期 | 模型覆盖与质量 | 扩展提示词、batch、KV 边界和数值测试；再扩展上下文与模型，W8/INT4/NF4 必须分别通过质量门槛 |
| 中期 | 实际模型热点 | 重新测 RMSNorm/SHAVE 的指令与寄存器开销，处理大 batch 的缓冲区复用；逐项验证端到端收益 |
| 中期 | 通用算子与训练 | 按实际模型缺失算子补齐派发与梯度，改善通用 FP32 GEMM；明确精度语义后再评估 AMP 与更多优化器 |
| 长期 | GPU+NPU 协同与分发 | 研究分层/流水线调度、内存和同步成本，再探索自动图分区；改善安装打包，扩大设备/驱动验证范围 |

稳定 v1.0 需要明确且可复现的兼容性、正确性、安装与性能基线，以及更广的模型/环境验证；
当前的单机实验结果尚不足以承诺通用训练或 GPU+NPU 加速。
优化候选只有通过数值检查并取得实测收益才进入默认路径，例如
[四行 SIMD 归约候选](npunlock-probe/research/results/shave-fp32-pipeline/REPORT.md)虽数值通过但更慢，未启用。

## 源码、文档与复现

| 路径 / 文档 | 内容 |
|---|---|
| `torch_intel_npu.py`、`torch_npu_*.py` | 设备后端、运行时、量化、模型适配器与示例 |
| `npunlock/` | 修改后的基础库；上游基准与差异见 [UPSTREAM.md](UPSTREAM.md) |
| `npunlock/tests/fixtures/npu4000/` | 自有 SHAVE C、汇编、ELF 及 manifest |
| `npunlock-probe/torch-device/backend.cpp` | PrivateUse1 原生扩展 |
| `npunlock-probe/research/` | 诊断、构建、优化实验源码与 Markdown 报告 |
| [INSTALL.md](INSTALL.md) / [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md) | 安装命令、依赖与新环境验证范围 |
| [COMPATIBILITY.md](COMPATIBILITY.md) | 算子、精度、梯度、布局、模型支持与发布回归基线 |
| [TORCH_NPU_README.md](TORCH_NPU_README.md) | 历史详细进展；旧路径/实验链接可能不再有效 |
| [SOURCE_RELEASE_REPORT.md](SOURCE_RELEASE_REPORT.md) | 精简源码、来源整理、快照验证与导出边界 |

源码快照约 6.1 MiB，逐文件 SHA-256 记录在 `SOURCE_MANIFEST.json`。
模型、环境、SDK/工具链、宿主二进制及驱动生成的 `.blob` 不打包；完整夹具回归需要另备图数据。
历史原始数组、图缓存和计时 JSON 已按存储策略清理，报告保留测量摘要；复测需要重新生成实验输入/输出。

<details>
<summary>离线测试与源码快照校验</summary>

安装核心依赖后可运行无需硬件的 45 项基本回归：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
python -B -m unittest test_torch_npu_geometry test_torch_npu_lowbit
python -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
python -B tools/test_source_release.py
```

首次构建前，在导出目录运行 `python -B tools/export_source_release.py --verify .` 校验文件集与哈希。
构建产生本地文件后，严格快照校验会报告额外文件。开发工作区含上游 Git 元数据时可重新导出：

```powershell
python -B tools/export_source_release.py --output source-release-next
python -B tools/export_source_release.py --verify source-release-next
```

导出器拒绝覆盖已有目录，快照不包含嵌套 `.git`。历史研究 audit 可能依赖已清理的数据，不属于这 45 项回归。

</details>

## 许可证与引用

新增代码采用 Apache-2.0，与 npunlock 上游一致；第三方改编部分保留对应许可和版权声明。
**npunlock 是直接基础**；OpenVINO 的部分公开实现与 Intel NPU compiler/ELF/Level Zero 源码也有明确引用关系，
详见 [REFERENCES.md](REFERENCES.md)、[references.bib](references.bib)、[NOTICE](NOTICE)、
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE](LICENSE)。
引用本项目时请记录实际版本/提交并同时标注被采用的上游工作；文献列表不能替代许可证声明。
