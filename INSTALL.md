# Windows NPU4000 源码安装

已验证平台：Windows x64、Core Ultra 7 258V / Intel NPU `8086:643e`、驱动 `32.0.100.5540`、图编译器 `8.3`。
选用算子/模型前请核对 [COMPATIBILITY.md](COMPATIBILITY.md) 中的参数、梯度和布局限制。
Python 验证版本为 3.12.15，Torch 为 2.14.1+cpu，NumPy 为 2.5.3。

## 安装 Python 依赖

在源码根目录创建独立环境：

```powershell
conda create -p .\.conda\npu --no-default-packages --override-channels -c conda-forge python=3.12 pip -y
.\.conda\npu\python.exe -m pip install --no-cache-dir -r requirements-build.txt
```

三份依赖清单的用途：

| 清单 | 直接安装项 | 用途 |
|---|---|---|
| `requirements.txt` | Torch CPU、NumPy | 普通 `intel_npu` Tensor 运算和已支持的训练 |
| `requirements-models.txt` | 核心清单 + safetensors、Transformers | SmolLM 检查点、tokenizer 和参考模型 demo |
| `requirements-build.txt` | 核心清单 + CMake、Ninja | 从源码构建宿主 DLL、worker 与 Torch PYD |

清单固定了本轮验证的直接依赖版本；Torch / Transformers 的传递依赖仍由 pip 解析，
本轮实际版本记录在 [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)。
源码使用随项目提供的 npunlock 修改版，由运行入口设置 Python 路径；requirements 不另行下载 npunlock 包。
OpenVINO、torchvision、torchaudio、GPU/CUDA/XPU 软件包不在这些清单中。
只使用 `FrozenSmolStack` 检查点 API、自备输入与 tokenizer 时，其新增 Python 依赖只有 safetensors；
完整模型 demo 依赖上表的模型清单。

## 从源码构建

准备 MSVC x64 C/C++ 编译器和 Windows SDK。在 **x64 Visual Studio Developer PowerShell** 中运行：

```powershell
.\build_npu.ps1 -Python .\.conda\npu\python.exe
```

已有独立工具链目录时，可显式指定（目录下包含 `VC/Tools/MSVC` 与 `Windows Kits/10`）：

```powershell
.\build_npu.ps1 -Python .\.conda\npu\python.exe -Toolchain 'D:\your-msvc-root'
```

入口使用 CMake/Ninja 重建 npunlock DLL 和隔离 worker，并运行两个 session mock-driver 回归。
同一个源代码构建的完整 DLL 提供 session API，按当前运行时约定放到
`npunlock-probe/session-runtime/npunlock_session.dll`；这条构建路线只使用 MSVC。
随后依据所选 Python/Torch 的实际 include、库名和版本构建 `_torch_intel_npu.pyd`，写入源码/二进制哈希清单。
只支持与构建时一致的 Torch 版本；换 Torch 或修改相关原生源码后重跑构建入口。

默认运行使用随附的 SHAVE ELF，并校验配套源码和 manifest 哈希。
修改 SHAVE C 代码另需外部 MoviTools；本轮未把历史 SHAVE 构建脚本的所有开发路径通用化。
Intel NPU 驱动及其图编译器由系统提供，不属于 pip requirements。

## 运行安装检查和自己的脚本

```powershell
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\tools\smoke_install.py
```

该检查实际执行 FP32 尾部加法、FP16 矩乘、Linear / Conv2d 的 FP32 前向、反向和 SGD，
与 CPU 对照，再检查资源释放。它不跑完整模型性能 benchmark。

自己的脚本也通过同一入口运行：

```powershell
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\my_test.py
```

示例 `my_test.py`：

```python
import torch
import torch_intel_npu  # 注册 intel_npu:0

x = torch.arange(17, dtype=torch.float32).to('intel_npu')
print((x + 70000).cpu())
```

需要梯度的结果和参数搬回 CPU 作检查时，使用 `tensor.detach().cpu()`；当前不支持跨 CPU/NPU 的自动微分复制。
运行入口优先选择当前源码构建的 native runtime；`-NativeDirectory` 可显式指定其目录。
原开发工作区保留的 `.conda/npunlock` 默认行为仍可用；验证其他环境时显式传入 `-Python`。

## 可选模型检查

```powershell
.\.conda\npu\python.exe -m pip install --no-cache-dir -r requirements-models.txt
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\tools\smoke_install.py -ScriptArguments '--model','D:\models\SmolLM2-135M-Instruct'
```

模型目录需有本地 config、tokenizer 和 `model.safetensors`。检查不会联网下载模型；
实际加载真实权重，运行一层 FP16 prefill/decode 并检查 KV 长度和有限输出。
这是安装冒烟检查，不代替完整模型的 logits 质量、吞吐和长上下文验证。

## 空间与检查

构建完成后可移除两个 Python 构建工具，保留核心/模型运行依赖：

```powershell
.\.conda\npu\python.exe -m pip uninstall -y cmake ninja
.\.conda\npu\python.exe -m pip check
```

后续需重编时重新安装 `requirements-build.txt`。模型与图缓存按需单独管理，默认图磁盘缓存关闭。
源码快照首次构建前可以运行 `python tools/export_source_release.py --verify .`；
构建后会出现本地输出，严格快照校验会报告额外文件，不应通过篡改源码 manifest 忽略它们。
完整离线夹具回归仍需要未打包的驱动图数据，详见 [README.md](README.md)。
