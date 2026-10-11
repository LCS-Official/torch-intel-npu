# 新环境源码安装与依赖精简验证

日期：2026-10-11。结论：**在本机新建 Conda 环境中，从独立导出的源码重建宿主组件，真实 NPU 运算、训练和单层模型检查通过。**
这完成了首发前的本机隔离环境安装检查，项目仍定位为 v0.1.0 Experimental Preview；没有创建或发布 GitHub release。

## 环境与隔离范围

| 项目 | 实测值 |
|---|---|
| 系统 / CPU | Windows x64 / Core Ultra 7 258V |
| NPU / 驱动 / 图编译器 | `8086:643e` / `32.0.100.5540` / `8.3` |
| 新环境 | `.conda/release-validation`，由 conda-forge 新建，未克隆旧环境 |
| Python / Torch / NumPy | 3.12.15 / 2.14.1+cpu / 2.5.3 |
| C/C++ 编译器 | MSVC 19.44.35229，工具目录版本 14.44.35207 |
| Windows SDK | 10.0.26100.0 |
| 构建工具 | CMake 4.4.4 / Ninja 1.13.2 |
| 可选模型包 | safetensors 0.8.0 / Transformers 5.18.0 |

清空外部 `PYTHONPATH` 并禁用 user-site，实际导入 `source-release-validation/` 中的 Torch 后端和 npunlock 源码。
未安装另一份 npunlock wheel，也未复制旧环境的 DLL、worker 或 PYD。
Torch 通过 [官方 CPU wheel 源](https://download.pytorch.org/whl/cpu/torch/)安装；`+cpu` 指宿主框架包，NPU 执行由本项目注册的 `intel_npu` 后端提供。
安装核心依赖时，OpenVINO、safetensors、Transformers 均未安装；之后才安装可选模型清单。

本轮复用了本机系统驱动、图编译器和已有 MSVC/SDK 工具链，因此**不等同于新装 Windows 机器验证**。
原 `.conda/npunlock` 环境及原生运行文件未被替换。

## 依赖与构建入口

| 文件 | 固定的直接依赖 | 作用 |
|---|---|---|
| [requirements.txt](requirements.txt) | Torch 2.14.1+cpu、NumPy 2.5.3 | 核心运算与已支持的训练 |
| [requirements-models.txt](requirements-models.txt) | 核心 + safetensors 0.8.0、Transformers 5.18.0 | 检查点、tokenizer 和参考模型示例 |
| [requirements-build.txt](requirements-build.txt) | 核心 + CMake 4.4.4、Ninja 1.13.2 | 从源码构建宿主组件 |

“两个核心依赖”是直接依赖数，不代表环境只有两个包。未把旧研究环境的完整 pip freeze 当成安装要求。
以上清单不安装 OpenVINO、torchvision、torchaudio 或 CUDA/XPU 软件包；系统驱动与 C/C++ 工具链单独准备。

[build_npu.ps1](build_npu.ps1) 用 MSVC 构建 npunlock DLL、worker，运行两个 session mock-driver 回归，
再构建 Torch PrivateUse1 PYD。完整 npunlock DLL 同时导出 session API，复制为兼容的 `npunlock_session.dll`，无需额外 MinGW。
[build_torch_npu_device.ps1](build_torch_npu_device.ps1) 从所选 Python/Torch 查询 include、库名、版本，
[run_torch_npu.ps1](run_torch_npu.ps1) 可显式传入 `-Python` 与 native runtime 路径。
完整安装命令见 [INSTALL.md](INSTALL.md)。

## 验证结果

| 检查 | 结果 |
|---|---|
| 新环境离线测试 | 44 项通过：卷积几何 3、lowbit/校准 5、上游 Python API/mock 32、导出器 4 |
| 原生 session 回归 | 两套通过：复用、绑定、失败隔离，以及链式执行的成功、拒绝和失败恢复路径；Release 构建保留测试断言 |
| 核心真实 NPU 冒烟 | 73 次 NPU 调用；严格 FP32 加法、FP16 矩乘、Linear/Conv2d 前向、反向、SGD 参数更新通过 |
| 可选模型冒烟 | 共 77 次 NPU 调用；本地 SmolLM2-135M-Instruct tokenizer/权重加载、一层 FP16 prefill/decode、KV 长度与有限输出检查通过 |
| 移除构建工具后运行 | 卸载 CMake、Ninja 后，以上核心 + 模型检查再次通过，77 次 NPU 调用 |
| 依赖一致性 | 核心构建环境、加入模型包后及卸载构建工具后的 `pip check` 均通过 |
| 退出状态 | live/cache/quarantine/session 资源计数全部归零，`context_failed=false`，未导入 OpenVINO |

真实设备检查源码为 [tools/smoke_install.py](tools/smoke_install.py)。FP32 加法使用 17 元素尾部与 70000 偏移，
避免把 FP16 范围内的结果当作 FP32 验证；矩乘使用 FP16 4×4 恒等矩阵对照。
训练检查把 loss、梯度及一次 SGD 后的参数与 CPU 比较，容差为 `rtol=5e-5, atol=1e-6`。
需要梯度的结果搬回 CPU 检查时用 `detach().cpu()`，不声称支持跨设备自动微分复制。

模型检查加载真实 checkpoint，但输入为小型随机 hidden states，仅验证安装路径与一层执行。
本轮没有重新评估完整模型 logits 质量、长上下文、吞吐或首 token 性能，也没有验证其他 Python 次版本和其他 NPU。

## 源码构建证据

以下是本轮构建输出的 SHA-256；Windows 链接产物不保证重复构建字节一致，哈希用于识别本次产物。

```text
npunlock.dll / npunlock_session.dll
3bb3686c03ae6146022bdbc753c744ba301cf10562129ec770cdff25537ac349
npunlock_worker.exe
b08a115b3c661aef1e9c5cf0d2db25161c92e6d07dc28e12ec03f62d955b1da0
_torch_intel_npu.pyd
42b168913cc1c933ebafb95029fe9198e1de7d8dea1cd6985a10d81f14355ee7
backend.cpp（计算源码未修改）
d6acf5d765b1024ef783e4370f30064e7c67aa4a26f7a10e9eba8c0864fe6d09
```

SHAVE 运算使用随源码保留、带源码/汇编/ELF manifest 校验的现有自有内核。
**本轮没有用 MoviTools 重新编译 SHAVE C**；修改这些内核仍需外部工具链，历史 SHAVE 构建器的开发路径尚未全部通用化。
驱动图 `.blob` 夹具未打包，44 项离线测试和本轮冒烟不能代替完整设备测试。

## 本轮解析到的 Python 包

除去已卸载的 CMake/Ninja，下列 37 个发行包是本次环境实际状态；记录传递依赖以便追溯，requirements 只固定直接依赖。
这是验证记录，不宣称其他时间解析出的传递依赖也已验证。

```text
annotated-doc==0.0.5       anyio==4.15.1              click==8.5.0
colorama==0.4.6           filelock==4.1.0            fsspec==2026.9.0
h11==0.16.0              hf-xet==1.7.0              httpcore2==2.13.1
httpx2==2.13.1            huggingface_hub==2.2.0     idna==3.20
Jinja2==3.1.6            markdown-it-py==4.2.0      MarkupSafe==3.0.4
mdurl==0.1.2             mpmath==1.3.0              networkx==3.7
numpy==2.5.3             packaging==26.3            pip==26.2.1
Pygments==2.21.0          PyYAML==6.0.3              regex==2026.9.29
rich==15.0.0             safetensors==0.8.0         setuptools==84.0.0
shellingham==1.5.4       sympy==1.14.0              tokenizers==0.23.3
torch==2.14.1+cpu         tqdm==4.70.1               transformers==5.18.0
truststore==0.10.4       typer==0.27.3              typing_extensions==4.16.0
wheel==0.48.0
```

## 空间与交付

核心 + 构建工具的新环境逻辑大小约 893.72 MiB；加入模型依赖并移除构建工具后约 944.60 MiB。
这是文件长度之和，包含 Conda 的 Python 和基础组件，不是 NTFS 硬链接/压缩去重后的物理占用。
安装使用 `pip --no-cache-dir`，验证关闭 NPU 图磁盘缓存，并使用已有本地模型，没有新增模型下载。

验证后已删除本轮临时环境和临时构建目录，仅保留源码、requirements、安装说明、报告与干净的 `source-release/` 快照。
原工作环境、当前可用内核、必要工具链及测试模型继续保留。
