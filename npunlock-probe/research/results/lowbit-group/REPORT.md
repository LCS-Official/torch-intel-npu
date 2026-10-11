# Grouped lowbit graph optimization — 2026-10-04

已完成两项低位图优化：默认分组 INT4 小 batch 的补零、矩阵乘法和截取合入单图；另提供默认关闭的 `compact_groups=True` 分组压缩实验。继续直接调用 Intel 驱动编译器和 Level Zero，不增加 OpenVINO runtime 或其他运行时依赖。设备为 Lunar Lake 8086:643e，驱动 32.0.100.5540，compiler 8.3。

## 默认路径：减少同步提交

旧包装层为避免分组 INT4 小 M 的错算，每次生成零 → 拼接 → MatMul → 截取，共四次同步图提交。现在图的输入和输出仍是实际 batch，只在图内补到至少 32 行后计算和截取；外部连续输入只提交一次，密集转置仍额外提交一次物化。

物理输入/输出仍须各不超过 1048576 个元素，超界在物化、图提交前拒绝。没有放大分配、Gather 或 SHAVE 限制，没有改变 FP16/FP32 常规 ATen 和训练实现。该路径继续保守绕过 compiler8.3 的形状相关错算，没有修复驱动内部 pass。

## 显式分组压缩实验

```python
import torch
import torch_intel_npu
from torch_npu_lowbit import QuantizedLinear

source = torch.nn.Linear(1024, 1024).eval()
normal = QuantizedLinear.from_float(source, format="int4", group_size=128)
compact = QuantizedLinear.from_float(source, format="int4", group_size=128,
                                     compact_groups=True)
x = torch.randn(1, 1024).half().to("intel_npu")
with torch.no_grad():
    y = normal(x)
    z = compact(x)
```

原图的缩放张量同时跨 N 与 group 两个维度；`compact_groups=True` 将每组权重重新打包为 (N,group_size) 的压缩 Const 和单轴 scale，切分输入列后逐组 MatMul，再用平衡 FP16 加法树合并。全部在一个设备图中执行，可绕过分组 INT4 小 M 的原问题。参数必须为 bool，最多 32 组；奇数个组、奇数 group_size 和尾 nibble 均有覆盖。每组编译常量区有对齐与元数据开销，小权重不一定节省空间。

Intel [公开的缩放合并源码](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/dialect/IE/transforms/passes/consolidate_weights_dequantization.cpp)区分可表示为单轴静态量化的缩放与多轴分组；[组展开源码](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/dialect/IE/transforms/passes/unroll_group_quantize.cpp)包含形状、消费者及小 batch 的判断。这些是图改写的机制线索；未证明本机 compiler8.3 与此公开版本相同，也未精确定位旧错算所在 pass。五份下载成功的源文件、版本与 SHA256 保存在 `sources/manifest.json`；另一个尝试的头文件路径为 404。

仅把分组 Const 扁平化成 (N*groups,group_size) 并重排 scale，可令原小样例算对，但大矩阵仍展开常量。因此未接入此尝试。原始组展开实验采用顺序加法；正式压缩选项改为平衡树，减少部分和舍入深度。

## 数值边界：压缩选项保持默认关闭

压缩选项在组间物化 FP16 部分和，不能保证与完整矩阵乘法相同。对独立 FP64 缩放码参考，本轮归一化随机输入中压缩路径相对 L2 最大约 0.0589%，完整路径约 0.02%–0.03%；另外检查逐输出绝对乘积舍入预算。量化误差不参与此计算正确性门槛。

全部 16 个编码、两组缩放以及 0.125/1/16 的严格位比较共六项通过；奇数分组零值两项通过。但有限的完整结果并不意味着组部分和可表示：

| 格式及输入 | 权重两组 | 完整路径 | 压缩路径 |
|---|---|---:|---:|
| INT4，x=1024，K=32，每组16项 | +7；-3×2 | 16384 | 0 |
| NF4，x=4096，K=32，每组16项 | +1；-1×0.5 | 32768 | 32736 |

这些反例与本机 DPU 的 FP16 部分和截断相容，具体内部截断位置尚未通过寄存器追踪确认。初始测试预期 IEEE Inf/NaN，实际表现不同；已修正诊断预期，并保留失败记录。压缩选项只适用于确认各组部分和范围并验证模型质量的实验，不能默认替代完整路径。仍未实现低位训练、动态量化或完整 LLM。

## 同进程性能与常量

以下使用同一进程、相同权重和常驻输入，对旧四提交包装层、默认单图和压缩选项各预热三次，然后 21 轮轮换及反向顺序采样。计时含 Python、分配和同步；不含量化准备、上传、冷编译与验证下载。预热测量内无显式主机复制和新编译。桌面负载未控制，不代表稳定平台极限，也未测能耗。

| 格式、M/K/N、分组 | 旧包装层 ms | 单图默认 ms | 压缩选项 ms |
|---|---:|---:|---:|
| INT4，1/64/32，32 | 1.251 | 0.322 | 0.320 |
| INT4，1/1024/1024，128 | 1.415 | 0.369 | 0.372 |
| INT4，128/1024/1024，128 | 0.560 | 0.547 | 0.580 |
| NF4，1/64/32，32 | 0.371 | 0.331 | 0.352 |
| NF4，1/1024/1024，128 | 0.343 | 0.349 | 0.354 |
| NF4，128/1024/1024，128 | 0.464 | 0.485 | 0.500 |

主要延迟收益来自将四次提交降为一次，NF4 和 M≥32 原来就只有一次，不能宣称同样提速。1024x1024 group128 的正式压缩图常量区为 655360 bytes（640 KiB）；相应完整 INT4 约 1.86 MiB、NF4 约 1.99 MiB（batch128）。图包含 DPU，没有 SHAVE invocation。常量区约为原来的三分之一，但这不是整个驱动运行时峰值内存，压缩选项的速度也未优于默认单图。

## 验证与复现

当前生产源码通过原 20 组低位数值检查、dtype/形状/autograd 拒绝检查、两种默认 MLP；三项离线 codec/IR/物理边界检查通过。同进程的旧/默认/压缩 Linear 对照 42 项、两种压缩 MLP、六项严格编码、两项零值及物理超界拒绝通过；两项大幅部分和截断反例按已知限制记录，不能计为数值等价性成功。

所有最终通过进程关闭时 live buffers/bytes 和 cached sessions 为零，context_failed=false。FP16/FP32 后端、矩形矩阵模块、npu4000 载体源码与上一轮哈希相同。未重复运行未修改的原 33 项设备回归和 49 项 npunlock 测试；它们上一轮的通过记录仍保留。汇总、实际调用计数和阶段来源见 `summary.json`，数值、计时与日志见 `numeric/guards/mlp/integration/edges.json/.log`。

另修复旧诊断程序对 Convert 名称 `decode` 的硬编码，改为识别 decompression 元数据。12 个有/无标记图均完成编译执行和清理；这是诊断兼容性检查，没有把其非逐位结果计为新增严格精度成功。

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script test_torch_npu_lowbit.py
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','numeric'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','guards'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','mlp'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','integration'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','edges'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','diagnose'
# 原始图改写诊断；包含预期错算，状态为 completed_diagnostic。
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','group'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_group_demo.py -ScriptArguments '--case','pad'
```

`before/` 保存上一轮实现及文档；旧报告 `../lowbit/REPORT.md` 保持作为当时结果。初始探针输出命名、对照计数、NumPy 参考 API 及溢出预期错误的报告/源码快照均保留，未并入最终通过计数。该阶段没有修改 C++、DLL 或安装依赖。

下一步可开始 INT8 矩阵路线，同时在实际模型中验证分组缩放的质量；当前压缩选项的部分和范围问题应作为明确约束继续研究。
