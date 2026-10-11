# Lunar Lake INT8 / W8A8：实现与本机验证

2026-10-04；Core Ultra 7 258V，NPU 8086:643e，驱动 32.0.100.5540，编译器 8.3。
Python 3.12.14 / Torch 2.14.1+cpu / NumPy 2.5.3。结论为 `passed_with_limitations`：
已实现显式 INT8 Linear 推理，包括真正的 8-bit 激活与权重矩阵任务，未实现低位训练或通用 int8 Tensor。

`torch_npu_lowbit.py` 在原接口上增加 `format='int8'`，权重数据为有符号字节，量化器生成
[-127,127]，手工编码可覆盖 [-128,127]；缩放仍为 FP16。默认 W8A16 可逐通道或分组；
显式 `activation_scale` 启用逐通道 W8A8。运行时输入/输出是 FP16 NPU Tensor，
激活量化、矩阵与偏置在驱动图内执行，无 CPU 数学回退。未改 C++/DLL、普通 FP16/FP32
设备派发或 npunlock 符号层，没有增加 OpenVINO runtime 或其他运行依赖。

```python
import torch
import torch_intel_npu
from torch_npu_lowbit import QuantizedLinear

source = torch.nn.Linear(1024, 1024).eval()  # CPU 准备，冻结权重
layer = QuantizedLinear.from_float(source, format='int8', activation_scale=1/128)
x = (torch.randn(512, 1024)/8).half().to('intel_npu')
with torch.no_grad():
    y = layer(x)  # FP16 intel_npu 输出
```

省略 activation_scale 即 W8A16。每个输入、输出和权重仍最多 1,048,576 元素；连续输入
一次图提交，完整密集转置输入多一次 DMA materialize。首次按 batch/权重编译，热调用不
重新序列化权重。复制计数仅覆盖显式张量复制，不包含编译器、驱动内部 DMA。

## 实际 DPU 格式证据

对照 FP16 图的非逐元素任务为 amode=0/wmode=0；逐通道 W8A16 为 0/2；W8A8 为 1/1，
即 U8 激活×U8 权重，以零点 128 表示有符号码。前置 FP16 逐元素任务负责激活量化，
整个 Linear 仍是一次图提交。W8A8 加载前检查所有非逐元素 DPU invariant，若激活或
权重不是 8-bit 就拒绝；不会将仅有 INT8 常量的 FP16 MatMul 宣称为 W8A8。
这证明 8-bit 操作数矩阵执行；没有对外提供 bit-exact INT32 累加输出。

解析使用 352 字节 invariant、tensor_mode 偏移 0x34、amode/wmode 的 4-bit 字段，
并检查文件与 section 边界。参照 Intel 固定提交的
[格式枚举](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nce_hw_40xx.h)、
[寄存器布局](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/tblgen/vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.td)、
[固件结构大小](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nnrt_api_40xx.h)
以及 [INT8 权重 MatMul 图模式](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/tests/functional/subgraph_tests/quantized_matmul.cpp)。
这些公开源码不是驱动 8.3 的逐字版本证明；实际 profile 限定与本机 FP16/INT8 对照
共同验证了这里使用的 ABI。12 个源码快照与哈希保存在 sources/manifest.json。

## 验证结果

| 检查 | 结果 |
|---|---|
| INT8 数值 | 25 组通过，含全部 256 个权重码、三种权重 scale、奇数 K、分组/compact、零权重、偏置/转置 |
| 激活量化 | 全部 256 个激活码、半步 ties-to-even、饱和、K=4096 大整数点积与抵消通过 |
| 激活 scale 边界 | 2^-24、2^-14、256 三组逐位通过 |
| INT8 MLP | W8A16/W8A8 的 64→32→7 ReLU、batch7 均通过 |
| 边界拒绝 | 5 种非法输入、缓存满拒绝；拒绝时无图提交，恢复后正常执行 |
| 大矩阵数值 | 1024×1024 权重，batch 1/32/128/512，两种 INT8 路线的 8 组检查通过 |
| 旧路径回归 | INT4/NF4 的 20 组数值与两个 MLP 通过 |
| 离线 | test_torch_npu_lowbit.py 四个 codec/IR/guard 检查通过 |

25 组 INT8 数值、3 边界、2 MLP、guard、8 大矩阵，以及旧路径回归共 465 次图提交，
各进程 shutdown 后 live_buffers/live_buffer_bytes/cached_sessions 为零，context_failed=false。
两个额外诊断进程不计为数值通过项。原始 JSON、日志、NPZ 与探针图保存在本目录。

W8A8 参考采用独立 NumPy INT64 点积：激活 rint(x/step) 饱和到 [-128,127]，
与原 INT8 权重码相乘，再应用 step 和有效逐通道 scale。有效权重 scale 为
(FP16(127*s)-FP16(-128*s))/255，整数零点对齐到 128。浮点 FQ 公式的零点可能位于
两个整数码之间，因此不能直接把它当作硬件量化参考。1024 行 one-hot 探针比较了原
scale、FP16 解码、浮点 FQ 和对齐零点参考；后者最贴近实测。该对齐规则是源码布局与
本机数值的推断，不能推广为任意驱动上的 FQ 逐位语义。

W8A16 的独立 FP64 参考使用整数码×scale，不额外先把每个重建权重舍入到 FP16，
因为编译器可能融合缩放。随机逐项门槛保持 rtol=.003/atol=.0003，整体 relative L2<.1%；
编码/量化边界要求最终 FP16 位模式一致。大矩阵 W8A8 relative L2 为约 .0201–.0209%，
W8A16 约 .0224–.0229%。K4096 的 exact 指最终 FP16 一致，其未舍入参考 4032.25
舍入为 4032，记录的绝对差 .25 是 FP16 输出舍入。

MLP 与量化算术参考最大绝对差均为 .000122；相对原 CPU 浮点模型的输出 relative L2
分别为 .177%（W8A16）与 .845%（W8A8）。这是一个合成小网络的输出差异，不是任务准确率。

## 常驻性能

每个形状同进程预热 3 次，21 轮轮换且交错反向顺序；矩阵权重为 1024×1024，输入与
原始权重为随机正态/8。CPU 为 Torch FP16、8 线程；NPU FP16 对照使用解码后 FP16
权重；W8A8 会改变输入精度，不能视为完全相同数值任务。计时包含 Python、分配、提交和
同步，排除首次编译、初始上传与验证下载。桌面负载、频率、功耗未控制。

| batch | W8A16 ms | W8A8 ms | NPU FP16 ms | CPU FP16 ms |
|---:|---:|---:|---:|---:|
| 1 | .5252 | .5190 | .4544 | .1424 |
| 32 | .5991 | .5777 | .5148 | .2819 |
| 128 | .8525 | .6627 | .6107 | .7125 |
| 512 | 1.2501 | .8240 | .9625 | 2.1354 |

batch512 W8A8 约为 NPU FP16 的 1.17×速度、CPU FP16 的 2.59×速度；小 batch CPU
仍更快。INT8 图常量区为 1,064,960 字节（1040 KiB），FP16 对照为 2,113,536
（2064 KiB），约减 49.6%。这不是整个进程峰值内存；FP16 IO、驱动工作区与图缓存另计。
没有 GPU 对照、真实 LLM 质量/吞吐或 NPU+GPU 协同测量。

## 限制与调试记录

W8A8 仅逐通道权重，不能启用 compact_groups；激活 scale 必须为 2 的幂，范围
2^-24..256，由用户用校准数据选择。1/128 只覆盖 [-1,.9921875]，越界会饱和；
输入应为有限值，未认证 NaN 行为。任意激活 scale=.0037 已复现阈值漂移，当前明确
拒绝，失败证据为 numeric-arbitrary-scale-failure.*。权重范围端点必须保持有限 FP16。
激活 scale 和权重都是只读公开属性，编译图不会因外部参数修改而悄悄使用旧常量。

最初使用浮点 FQ 参考导致大矩阵检查失败；one-hot 诊断后改为整数零点参考，保留
performance-fq-reference-failure.*。另一失败来自先舍入 FP16 权重的参考，改用独立
scaled-code FP64 参考后通过，保留 performance-fp16-weight-reference-failure.*。
原始输出 FQ 额外探针有 .125 误差，仅为 diagnostic，未加入生产接口。
NumPy conda MKL 的 libomp 与 Torch libiomp5 冲突发生在 CPU NumPy 浮点 BLAS 参考；
参考改用 INT64/non-BLAS einsum，未启用 KMP_DUPLICATE_LIB_OK，未修改环境 DLL。

普通 int8 设备存储/ATen、低位梯度、动态校准、QAT/QLoRA、checkpoint 导入、普通
state_dict 低位权重保存与 LLM 支持都未实现。W8A16 compact 继承原 FP16 部分和舍入/
溢出截断限制，仍需显式选用。本实现不会自动替换普通 nn.Linear 或改变 FP32 契约。

复现：`run_torch_npu.ps1 -Script torch_npu_int8_demo.py -ScriptArguments '--case','numeric'`；
case 还有 edges、mlp、guards、performance、lowbit-numeric、lowbit-mlp。
生产源码快照见 implementation.py；各阶段执行源指纹、环境与总计见 summary.json。
下一步宜加入一个小 Transformer 的校准与精度检查，再评估 attention/MLP 图融合；
现有结果尚不能证明 LLM 单 token 解码会更快。
