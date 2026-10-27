# Windows NPU4000 源码安装

v0.2.0 Experimental Preview：从v0.1.0升级请按下文build_npu.ps1重新构建宿主组件；不要只复制新Python文件后继续使用旧DLL/PYD。以下Sxx是阶段性更新记录，完整版本应以本版源码及兼容性矩阵为准。

本地S45需同步torch_intel_npu.py、torch_npu_smollm.py和torch_npu_graph_cache.py后重启Python。最终改动只涉及Python缓存管理，无新增依赖或原生接口，不要求重建ELF/DLL/PYD；诊断用原生修改已撤回。默认预算仍512MiB，必要时可在下一次调用前用torch.intel_npu.set_graph_cache_budget(768*1024**2)增大，不能缩至已缓存字节数以下或在分组计划构建中修改。[缓存回避措施、代价及剩余根因](npunlock-probe/research/results/smollm-cache-lifetime/REPORT.md)与[低位质量诊断](npunlock-probe/research/results/w8-guarded-quality/REPORT.md)分别列明；不需要开启磁盘缓存才能使用图族保护。

本地S44的可选W8残差修正需同步新增torch_npu_lowbit_correction.py和更新的torch_npu_smollm.py，再重启Python；不需要新增依赖或重建ELF/DLL/PYD。校准需已有可选模型依赖与本地SmolLM2 checkpoint，因子必须配合同次准备的W8权重，默认不启用。用法、完整复现、质量边界与FP16控制组缓存预算见[S44报告](npunlock-probe/research/results/w8-lowrank-quality/REPORT.md)。

本地S42需同步torch_npu_flash_attention.py、torch_npu_nested.py及相关派发实现、resident_flash_attention.c、两枚flash ELF/汇编和更新的manifest，然后重启Python。已有S38–S41 DLL/PYD无需重建，核心依赖不增加。使用现有Movi工具链可运行npunlock-probe/research/build_flash_attention_kernels.py；默认source build使用核验后的ELF。显式SDPA选择、Nested推理范围、复现及性能限制见[S42报告](npunlock-probe/research/results/flash-nested/REPORT.md)。

本地S43继续更新flash Python实现、resident_flash_attention.c、两枚flash ELF/汇编及manifest，加入SHAVE热路径和自动分块；仍不需要新依赖或DLL/PYD重建。应作为同一组文件同步并重启Python，避免使用已缓存的旧图。预热性能与容量规则见[S43报告](npunlock-probe/research/results/flash-optimization/REPORT.md)。

本地S41扩展Attention/HF模型迁移，只需同步Python实现（包括新增torch_npu_sequence.py）后重启Python；已有S38–S40原生组件无需重建。HF复现使用现有可选requirements-models.txt，核心依赖不增加。共享参数转换步骤、两类小模型配置与完整命令见[S41报告](npunlock-probe/research/results/attention-migration/REPORT.md)。

本地S40新增标准`torch.amp.GradScaler("intel_npu")`构造适配，保留旧专用入口；已有S38/S39环境同步Python源码并重启Python即可，无需改PyTorch安装、增加依赖或重建原生组件。FP32主参数及显式foreach=False/fused=False等范围见[S40报告](npunlock-probe/research/results/amp-standard/REPORT.md)。

本地S39为Python层Attention边界与随机接口更新：已有S38环境同步Python源码后重新启动Python即可，无新pip依赖或DLL/PYD/ELF构建。旧环境仍须满足下述S38组件要求。用法、范围及复现见[S39报告](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md)。

本地S38按实际字节大小扩容存储，并新增密集MaxPool内核。已有环境需执行`./build_torch_npu_device.ps1`重建Torch PYD，重新启动Python会话，并同步Python实现、新pool C/ELF/汇编、更新的selection-u32和manifest。session DLL与原62枚内核不变，无新增pip运行依赖。源码附带核验ELF；开发工作区重建内核可执行build_selection_kernels.py与build_pool_dense_kernel.py。容量和真机证据见[S38报告](npunlock-probe/research/results/storage-shape/REPORT.md)。

本地S37新增5,740字节的selection-backward-u32 ELF以解除旧2²⁴单平面元数据限制。更新时需同步Python派发、resident_selection.c、新ELF/汇编及resident manifest；原62枚ELF及S32 DLL/PYD无须重建，没有新增pip依赖。开发工作区如需重建，可用现有Movi工具链执行npunlock-probe/research/build_selection_kernels.py；源码发布自带经指纹校验的ELF。验收范围见[S37报告](npunlock-probe/research/results/selection-capacity/REPORT.md)。

本地S36扩展普通MaxPool/极值容量、规则平面反向及页对齐窗口读取，只修改Python实现与检查；已有S32原生组件和62枚ELF无需重建或新增依赖。真机范围和复现见[S36报告](npunlock-probe/research/results/pool-capacity/REPORT.md)。

本地S35优化普通Conv2d窗口工作区和逐块结果组装，增加5项离线检查及真机对照；已有S32原生组件和62枚ELF无需重建或新增依赖。基线、实际内存收益及复现见[S35报告](npunlock-probe/research/results/conv-capacity/REPORT.md)。

本地S34新增Embedding列分块与排序/topk总量边界扩展，仅修改Python宿主和检查；已有S32原生组件和62枚ELF无需重建或新增依赖。具体实测、剩余边界及复现命令见[S34报告](npunlock-probe/research/results/index-capacity/REPORT.md)。

本地S33只修改Python图生命周期和检查：提供工厂的冻结图可重建、字节预算512MiB、超预算可重建图临时执行。已有S32 DLL/PYD及原62枚ELF时无需重建或安装新依赖；从更早版本升级仍按下方S32步骤重建。范围见[S33报告](npunlock-probe/research/results/frozen-capacity/REPORT.md)。

本地S32把单allocation扩到1GiB，并加入普通算子图LRU。更新时依次运行`build_npu_session.ps1`、`build_torch_npu_device.ps1`；两者已在本机重建。无新pip依赖或SHAVE ELF。FP16/BF16逻辑Tensor最多512MiB，冻结自定义图仍pinned，范围见[S32报告](npunlock-probe/research/results/graph-capacity/REPORT.md)。下文S30的512MiB属于历史版本。

本地S31仅修改Python分块、容量检查、autograd和测试，复用S30的session DLL/PYD及原62枚ELF，无需额外构建或依赖；损失与专用工作区范围见[报告](npunlock-probe/research/results/operator-capacity/REPORT.md)。从更早版本更新仍须按下方S30说明重建宿主组件。

本地S30新增带范围校验的native绑定并把单allocation扩至512MiB。升级本地代码时须依次运行`build_npu_session.ps1`和`build_torch_npu_device.ps1`，保证DLL/PYD与Python源码一致；不新增pip依赖或SHAVE ELF。容量和训练实测见[S30报告](npunlock-probe/research/results/memory-scale/REPORT.md)。总训练占用由模型参数、梯度、优化器及工作区决定，512MiB是单次分配上限。

本地S29损失函数扩展复用已有62枚resident ELF和S26宿主组件；无需新增pip依赖、重建SHAVE或DLL/PYD，验收入口与契约见[S29报告](npunlock-probe/research/results/loss-extended/REPORT.md)。

本地S28的SDPA/GQA扩展只修改Python派发与autograd，复用S27的62枚resident ELF；已有S26/S27宿主组件时无需重建DLL/PYD或安装依赖，见[S28验收](npunlock-probe/research/results/sdpa-extended/REPORT.md)。

本地S27提供62个已校验resident ELF，新增滚动补偿累加的`stream-sum-f32`；配套二进制可直接使用，不新增pip依赖，也无需再次重建S26的session DLL/PYD。修改新SHAVE C源码时，需要MoviTools并运行`npunlock-probe/research/build_stream_sum_kernel.py`；完整`build_resident_kernels.py`也已包含此步骤，见[S27验收](npunlock-probe/research/results/matmul-workspace/REPORT.md)。S25内核源码修改时仍用`npunlock-probe/research/build_compat_kernels.py`重建。

从S25或更早checkout升级时，S26的256MiB/rank64存储变更仍要求依次运行`build_npu_session.ps1`、`build_torch_npu_device.ps1`；从零构建使用下文`build_npu.ps1`，详见[S26验收](npunlock-probe/research/results/capacity-model/REPORT.md)。

已验证平台：Windows x64、Core Ultra 7 258V / Intel NPU `8086:643e`、驱动 `32.0.100.5540`、图编译器 `8.3`。
选用算子/模型前请核对 [COMPATIBILITY.md](COMPATIBILITY.md) 中的参数、梯度和布局限制。
Python使用3.12，本轮源码发布验证为3.12.14，历史新环境验证为3.12.15；Torch为2.14.1+cpu，NumPy为2.5.3。

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
只使用 `FrozenSmolStack` / 本地开发版 `FrozenSmolIO` 检查点 API、自备输入与 tokenizer 时，其新增 Python 依赖只有 safetensors；
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

本地开发版的混合投影与缓存身份选项有10项离线回归，轻量CPU入口另有5项，需已安装safetensors，无需模型或NPU：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
.\.conda\npu\python.exe -B test_torch_npu_smollm_policy.py
.\.conda\npu\python.exe -B test_torch_npu_smollm_io.py
```

该参数及测试已纳入v0.2.0；数值范围和复现步骤见[混合精度质量报告](npunlock-probe/research/results/w8-projection-mixed/REPORT.md)。

本地INT64/默认BatchNorm扩展需使用当前源码重新执行 `build_torch_npu_device.ps1`。
新增SHAVE ELF已有配套C源码和指纹；仅修改/重建该ELF才需要既有外部Movi工具链。
不需新增Python依赖，验收入口为：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
.\.conda\npu\python.exe -B test_torch_npu_int64.py
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\torch_npu_batchnorm_demo.py -ScriptArguments '--case','integer'
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\torch_npu_batchnorm_demo.py -ScriptArguments '--case','models'
```

离线3项检查防止编译器把INT64两个字拆开；真实NPU检查覆盖存储、计数和六种BatchNorm训练组合。
具体范围见[INT64与BatchNorm验收](npunlock-probe/research/results/int64-batchnorm/REPORT.md)，已纳入v0.2.0。

本地分类扩展复用上述INT64设备扩展和现有Python依赖，新增四份有源码/指纹的SHAVE ELF。无需再次重编C++或安装包：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
.\.conda\npu\python.exe -B test_torch_npu_classification.py
foreach ($case in @('logsoftmax','nll','contracts','training')) {
    .\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\torch_npu_classification_demo.py -ScriptArguments @('--case',$case)
}
```

修改分类C源码后，使用现有Movi环境运行`npunlock-probe/research/build_classification_kernels.py`；
完整`build_resident_kernels.py`也包含这些内核。默认运行直接使用配套ELF，不依赖OpenVINO runtime。
范围及复现后的报告审计入口见[分类训练报告](npunlock-probe/research/results/classification/REPORT.md)，仍为本地开发版。

本地AdamW复用既有C++扩展和依赖，新增有源码/指纹的sqrt/maximum SHAVE ELF；无需安装包或重建C++。
运行算子/状态与整网对照：

```powershell
foreach ($case in @('primitives','optimizer','contracts')) {
    .\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\torch_npu_adamw_demo.py -ScriptArguments @('--case',$case)
}
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\torch_npu_classification_demo.py -ScriptArguments @('--case','training','--optimizer','adamw','--no-bn-bias')
```

整网命令仍可能因纯FP16轨迹数值检查失败而返回非零，不能等同于全部模型已认证；FP32结果及失败位置见[AdamW报告](npunlock-probe/research/results/adamw/REPORT.md)。
修改新增C源码后用已有Movi环境运行`build_optimizer_kernels.py`；完整`build_resident_kernels.py`也包括它。

本地LayerNorm/GELU扩展复用上述环境与C++扩展，新增四份GELU SHAVE ELF，无需新依赖或重编C++：

```powershell
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\torch_npu_norm_activation_demo.py
```

该入口的三组独立LayerNorm检查现已通过，包括有序统计修复后的FP16输入/FP32 affine；
旧范围与训练回归见[LayerNorm/GELU报告](npunlock-probe/research/results/norm-activation/REPORT.md)，最新范围见[混合统计修复报告](npunlock-probe/research/results/layernorm-mixed/REPORT.md)。
修改GELU C后使用已有Movi环境运行`npunlock-probe/research/build_gelu_kernels.py`；完整resident构建也包含这些内核。

混合LayerNorm新增配套有源码/指纹的统计与融合验证ELF，无需新增包或重编C++：

```powershell
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\npunlock-probe\research\layernorm_mixed_probe.py
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\npunlock-probe\research\layernorm_mixed_acceptance.py -ScriptArguments @('--case','stress')
```

修改统计C后用已有Movi环境执行`build_layernorm_kernel.py`；完整resident构建已包含它。
后续Tensor `add_`的alpha转换修复后，混合整网`--case training --fp32-loss`的none/tanh两组四步检查均通过；
`--case training`的两组FP16损失对照仍返回非零，不等同于通用AMP支持。
更新规则与逐层诊断见[混合训练更新报告](npunlock-probe/research/results/mixed-training-rounding/REPORT.md)。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\add_alpha_regression.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_acceptance.py -ScriptArguments @('--case','training','--fp32-loss')
```

本次只修改Python设备分派中的标量转换，复用已有NPU算术/转换内核，无需重编ELF或C++扩展。

后续精度与稳定性探针复用同一环境，不增加运行依赖：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\classification_precision_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_stability_probe.py
```

第一项的10组FP64对照通过；第二项完成两组32步固定样本学习，但因第6/第5步开始的严格CPU轨迹偏差返回非零。
有限学习不替代严格验收，原阈值未放宽，详见[精度与32步边界报告](npunlock-probe/research/results/classification-precision/REPORT.md)。

原第5/6步偏差的局部算子/SGD状态重放，以及独立合成验证集入口：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py -ScriptArguments @('--steps','6')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py -ScriptArguments @('--steps','6','--unfused-reference')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_validation_probe.py
```

`--unfused-reference`仅改变诊断中的CPU更新舍入顺序，NPU仍使用原标准SGD，不是新的生产后端模式。
六步探针返回零表示完成预期差异重放，不能将其视为标准CPU轨迹通过；独立验证集按严格差异返回非零。
范围及样本见[SGD舍入归因报告](npunlock-probe/research/results/mixed-step-diagnosis/REPORT.md)。

本地新增softmax/bmm与小型attention训练检查，无新增Python依赖或宿主扩展重编需求；源码中附带两个新SHAVE ELF及哈希。
修改其C源码时，外部MoviTools构建入口为`npunlock-probe/research/build_softmax_kernels.py`；完整resident构建脚本也会调用它。

```powershell
foreach ($case in @('softmax','bmm','bmm-large','contracts','attention','training')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_attention_demo.py -ScriptArguments @('--case',$case)
}
.\.conda\npunlock\python.exe -B npunlock-probe/research/attention_primitives_report.py
```

报告审计还需重新生成既有分类logsoftmax/nll/contracts及迁移boundaries的原始结果，完整命令见[报告](npunlock-probe/research/results/attention-primitives/REPORT.md)。

本地高维matmul/SDPA及可微clone/contiguous复用同一环境，默认无需OpenVINO runtime或宿主扩展重编。
新增的safe/causal softmax ELF随源码提供，修改C时用外部MoviTools执行`npunlock-probe/research/build_sdpa_softmax_kernel.py`。

```powershell
foreach ($case in @('matmul','matmul-large','sdpa','sdpa-masks','layouts','contracts','training')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments @('--case',$case)
}
```

审计脚本和设备/原算子回归命令见[SDPA报告](npunlock-probe/research/results/sdpa/REPORT.md)。前面的旧轮次报告保留历史结果，其审计脚本依赖相应版本/原始数据；当前版本按新报告验收。

## 空间与检查

S23新增稳定排序ELF/汇编、`torch_npu_vision.py`、`torch_npu_sort.py`及配套几何/反向；无新pip依赖或C++/PYD重建。
默认运行使用配套ELF。修改排序C后用已有MoviTools运行`npunlock-probe/research/build_sort_kernel.py`；
完整resident构建入口已补上索引、行极值、选择和排序构建步骤，避免重建时遗漏这些manifest条目。
可在原环境串行复现：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = ''
foreach ($precision in @('float16','float32')) {
  foreach ($case in @('sort','extrema','wide','interpolation','normalization','transpose','contracts','tiled','vision-training','training')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_vision_sort_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
  }
}
foreach ($case in @('vision-amp','amp')) {
  .\run_torch_npu.ps1 -Script .\torch_npu_vision_sort_demo.py -ScriptArguments @('--case',$case,'--precision','float16')
}
```

`training/amp`含排序离散路由；遇到严格轨迹边界时输出`passed_with_limitations`并保存原失败及固定路由诊断，
不把该诊断等同于严格整网通过。`vision-training/vision-amp`是单独的无排序视觉训练控制。
AMP始终使用FP32主参数/FP16 autocast；`--precision float16`选择半精度模型的比较阈值，
额外的`vision-amp --precision float32`运行统计严格探针有已记录的约1.46e-5偏差，不计为严格通过。
边界、容量、指纹和清理记录见[视觉/排序报告](npunlock-probe/research/results/vision-sort/REPORT.md)。
脚本临时JSON按项目存储策略在汇总后清理；不需要图表或模型下载。

S22新增三枚选择/反向ELF/汇编及manifest，无新pip依赖或C++/PYD重建。只在修改C时用已有MoviTools运行`npunlock-probe/research/build_selection_kernels.py`；普通运行直接使用有源码/指纹的ELF。

```powershell
foreach ($precision in @('float16','float32')) {
    foreach ($case in @('relu','topk','pools','edges','contracts','training','tiled')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_pool_topk_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
.\run_torch_npu.ps1 -Script .\torch_npu_pool_topk_demo.py -ScriptArguments @('--case','amp')
```

使用标准`nn.MaxPool1d/2d`、`nn.ReLU(inplace=True)`和`torch.topk`，无需模型专属调用；仍须满足[报告](npunlock-probe/research/results/pool-topk/REPORT.md)中的存储/可微view/k/工作量边界。

S21新增两枚行极值ELF/汇编及manifest，随源码分发；无需新pip依赖或重建C++/PYD。修改C源码时，使用已有MoviTools运行`npunlock-probe/research/build_row_extrema_kernels.py`。专项可串行复现：

```powershell
foreach ($precision in @('float16','float32')) {
    foreach ($case in @('reductions','norms','edges','clipping','contracts','tiled','training')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_reduction_norm_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
.\run_torch_npu.ps1 -Script .\torch_npu_reduction_norm_demo.py -ScriptArguments @('--case','amp')
```

实际训练使用标准`torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=False)`；参数/梯度dtype和布局边界、非有限异常控制及验收见[S21报告](npunlock-probe/research/results/reduction-norm/REPORT.md)。

S20新增有源码/指纹的索引ELF及manifest，随源码分发；正常运行不需要新依赖或C++/PYD重建。修改索引C源码后，使用已有MoviTools环境运行`npunlock-probe/research/build_index_kernels.py`重新生成三个索引内核及指纹。硬件入口为：

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_index_pool_demo.py -ScriptArguments @('--case','indexed','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_index_pool_demo.py -ScriptArguments @('--case','cnn_amp')
```

平均池化、限幅和其他索引/精度的复现及边界见[报告](npunlock-probe/research/results/index-pool/REPORT.md)。

S19的分组/深度Conv2d、Conv1d、任意轴softmax、多维硬标签损失、label smoothing和rsub更新只涉及Python派发与autograd，不需要重编PYD/SHAVE或安装新依赖。复现入口为：

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_portability_demo.py -ScriptArguments @('--case','conv1d','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_portability_demo.py -ScriptArguments @('--case','transformer-amp')
```

其余精度/边界及普通分割网络检查见[完整验收报告](npunlock-probe/research/results/portability/REPORT.md)；AMP现可使用S40标准GradScaler入口，仍受FP32主参数及普通单Tensor优化器等限制，未声明任意模型或长期收敛。

本地S17增加C++ autocast策略，更新旧本地版本须重编设备扩展：

```powershell
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_amp_demo.py
```

使用标准torch.autocast("intel_npu",dtype=torch.float16)，模型保留FP32；S40起可选torch.amp.GradScaler("intel_npu")，旧torch.intel_npu.amp.GradScaler()仍保留，普通优化器显式foreach=False/fused=False。无新增pip依赖或SHAVE编译要求，范围及卷积数值边界见[AMP报告](npunlock-probe/research/results/amp/REPORT.md)。

S18卷积归因与四步训练验收可直接使用同一环境，不需要再次构建：

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_amp_conv_demo.py -ScriptArguments @('--steps','4')
```

该入口串行运行局部检查、SiLU小网络和三卷积残差网络，明确区分原生CPU Half与FP32累加控制；结果范围见[S18报告](npunlock-probe/research/results/amp-conv/REPORT.md)。

本地S16更新新增C++ Generator，旧PYD不能直接搭配新Python派发。保留同一环境，运行`.\build_torch_npu_device.ps1`重编设备扩展；新源码完整安装的build_npu.ps1已包含此步骤。两个新增SHAVE ELF随源码提供，修改其C源码时才需要外部MoviTools与`npunlock-probe/research/build_discrete_rng_kernels.py`。没有新增pip依赖。

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_discrete_demo.py
```

该入口串行运行INT64随机数、FP16/FP32采样、独立Generator、拒绝契约及训练/checkpoint；设备需支持项目的既有驱动组合。范围见[S16报告](npunlock-probe/research/results/discrete-rng/REPORT.md)。

构建完成后可移除两个 Python 构建工具，保留核心/模型运行依赖：

```powershell
.\.conda\npu\python.exe -m pip uninstall -y cmake ninja
.\.conda\npu\python.exe -m pip check
```

后续需重编时重新安装 `requirements-build.txt`。模型与图缓存按需单独管理，默认图磁盘缓存关闭。
源码快照首次构建前可以运行 `python tools/export_source_release.py --verify .`；
构建后会出现本地输出，严格快照校验会报告额外文件，不应通过篡改源码 manifest 忽略它们。
完整离线夹具回归仍需要未打包的驱动图数据，详见 [README.md](README.md)。


S24扩大五维/空Tensor存储，旧本地PYD必须重编一次；无需新增pip包：

```powershell
.\build_torch_npu_device.ps1
foreach ($precision in @('float32','float16')) {
  foreach ($case in @('foundation','integer','statistics','images','pooling','volume','grid','training','warp','contracts','outputs')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_vision_sort_extended_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
  }
}
```

新增两个源码SHAVE内核与动态读取内核随ELF/汇编/manifest提供；仅改内核源码时需已有MoviTools及`build_sort_kernel.py`、`build_selection_kernels.py`、`build_order_stats_kernel.py`、`build_grid_kernel.py`。完整`build_resident_kernels.py`已包含这些步骤。动态grid/indices/q留在NPU，仅回读4字节校验标志。FP16抗锯齿、3D均值池化和grid_sample因本机CPU无原生Half实现，以相同Half数据的FP32计算再转Half作为独立参考。容量、数值边界和完整范围见[S24报告](npunlock-probe/research/results/vision-sort-extended/REPORT.md)。
