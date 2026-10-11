# INT4 / NF4 weight-only DPU inference — 2026-10-04

本机 Lunar Lake NPU 8086:643e、驱动 32.0.100.5540、编译器 8.3 已跑通显式 `QuantizedLinear`：冻结的 INT4/NF4 权重，FP16 激活与输出，通过驱动直接编译和执行包含 DPU 的图。没有增加 OpenVINO runtime 或 bitsandbytes 依赖。当前是量化 Linear/MLP 推理实验，尚未实现通用低位 Tensor、低位训练或完整 LLM。

## 接入方式

```python
import torch
import torch_intel_npu
from torch_npu_lowbit import QuantizedLinear

source = torch.nn.Linear(256, 256).eval()  # CPU 上准备权重
layer = QuantizedLinear.from_float(source, format="nf4")  # 或 "int4"
x = torch.randn(16, 256).half().to("intel_npu")
with torch.no_grad():
    y = layer(x)
print(y.device, y.dtype)
```

默认每个输出通道一个 FP16 缩放值；`group_size=32/64/128` 等整除 K 的大小可沿输入维度分组。INT4 为有符号二进制补码，量化准备使用对称 -7..7，解码支持全部 -8..7；NF4 为固定 16 项码表。每字节两个码，低 nibble 对应先出现的元素。NF4 数值依据 [Intel 的公开码表](https://github.com/openvinotoolkit/openvino/blob/master/src/core/src/type/nf4.cpp)，不是通用小型浮点格式。

`PackedWeight` 保存不可变的形状、格式、码和缩放字节。CPU 量化仅发生在准备阶段。首次调用生成压缩 Const → Convert → scale Multiply → Reshape → MatMul/bias 的 IR，再由驱动编译；权重与形状指纹用于复用图会话。预热调用缓存指纹，不重新序列化或扫描完整权重。运行时输入、输出和必要的搬运均使用已有 NPU 存储和操作，没有 CPU 输出算术回退。

新增 `npunlock.constant_packed` 和符号 `i4/u4/nf4` 序列化支持，严格检查字节数和奇数元素的空闲高 nibble。它不代表 PyTorch 设备已支持这些整数 dtype；底层 Torch allocator 和图 IO 仍为 FP16/FP32。低位接口要求二维 FP16 激活，拒绝活动 autograd、错误 dtype/形状及不支持的设备/编译器 profile；编译图必须有非空 DPU section。

普通 FP16/FP32 ATen、真正 FP32 SHAVE 语义和默认方阵分流没有改变。低位模块没有可求梯度的权重 Parameter，也未实现普通 state_dict 权重保存、HF/GGUF/bitsandbytes checkpoint 布局导入、零点、双重量化或 QLoRA。现阶段从原 CPU Linear 重新准备冻结权重。

## 找到的问题及处理

**分组 INT4 的形状相关错算。** 直接编译 `(M,K,N)=(5,64,32)`、group32 时，159/160 个输出未通过检查，最大差约 0.428；重复运行及单热输入均复现。改变 tile 数为 1/2/4 仍有约 0.569 的误差，普通解码权重 FP16 图正常。更大 batch 和其他组大小可正常运行，内部具体编译 pass 尚未定位，不能据此认定所有分组图都有问题。

包装层保守地将所有分组 INT4 的 M<32 用已有设备 DMA 补零到 32，DPU 完成后截取有效行。额外物理输入/输出必须仍满足存储上限。原失败形状及分组、偏置、转置检查通过；典型预热调用从 1 次图提交增加为 4 次，转置另需一次。这里修复的是包装层行为，没有修复驱动编译器本身。NF4 在已查案例中无需此补零。

**FP16 融合舍入。** 大矩阵上，压缩图与“先把每个解码权重舍入到 FP16，再做矩阵乘法”的基线不必逐位一致。初始固定点值容差使 131072 个输出中的 14 个接近零的值未通过，最大差约 0.000732。独立 FP64 缩放码参考显示相对 L2 误差约 0.028%，与解码/缩放融合的舍入差异相容；具体融合 pass 未定位。

最终大矩阵检查同时要求：对独立 FP64 缩放码参考的相对 L2 <0.1%，以及相对解码 FP16 基线的逐输出差落在实际权重舍入差预算内。预算包含 `abs(X) @ abs(W_real-W_fp16).T`、FP16 输出舍入余量和小绝对余量；不以量化本身约 10% 的误差掩盖计算错误。保留全部 16 个编码和幂二缩放的严格逐位检查，以及小矩阵原容差。没有放宽既有真正 FP32 的测试。

## 设备压缩效果

逐输出通道的 1024x1024 权重，packed 数据为 524288 bytes，FP16 scale 为 2048 bytes。已编译常量区为 540672 bytes（528 KiB），匹配的解码权重 FP16 图为 1872896–2031616 bytes（约 1.79–1.94 MiB），这些图包含 DPU、没有 SHAVE invocation。

但 group128 图的常量区可扩展为 1949696/2085888 bytes，与相应 FP16 基线相同。因此只能说输入 IR 支持 4-bit 分组权重，不能宣称当前分组设备图保留了四倍压缩。常量区尺寸也不是整个运行时峰值内存；C++ allocator 计数不覆盖驱动图常量等内部资源。

随机正态权重的相对 L2 量化误差：逐通道 INT4 约 12.7%–14.3%，NF4 约 9.9%–10.4%；group128 INT4 约 11.7%，NF4 约 9.6%。这些是合成权重误差，不是模型质量、困惑度或下游准确率评估。量化误差与执行误差分别记录。

## 同轮性能

常驻输入，排除准备、上传、冷编译与验证下载；每路径预热 3 次，15 轮交错执行，包含 Python 调用、分配、同步和补零/截取。下表为中位数 ms；FP16 基线分别使用该格式解码后的权重。CPU 为 8 线程 Torch FP16 API，桌面负载未控制，不能用于确定平台极限或能耗。

| M/K/N | 分组 | INT4 | NF4 | FP16-INT4 参考 | FP16-NF4 参考 | CPU |
|---|---|---:|---:|---:|---:|---:|
| 1/256/256 | 逐通道 | 0.485 | 0.387 | 0.437 | 0.372 | 0.138 |
| 1/256/256 | 128 | 1.782 | 0.436 | 0.419 | 0.379 | 0.106 |
| 128/256/256 | 逐通道 | 0.397 | 0.384 | 0.371 | 0.365 | 0.117 |
| 128/256/256 | 128 | 0.459 | 0.433 | 0.381 | 0.444 | 0.109 |
| 1/1024/1024 | 逐通道 | 0.530 | 0.428 | 0.415 | 0.400 | 0.183 |
| 1/1024/1024 | 128 | 2.038 | 0.495 | 0.449 | 0.439 | 0.153 |
| 128/1024/1024 | 逐通道 | 0.617 | 0.612 | 0.618 | 0.510 | 0.607 |
| 128/1024/1024 | 128 | 0.794 | 0.567 | 0.637 | 0.609 | 1.006 |

预热采样期间无测得的显式主机上传/下载或新编译，计数器不覆盖驱动 DMA。通常每次调用 1 个图，分组 INT4 小 batch 因绕法为 4 个图。低位格式没有稳定跑赢普通 FP16 DPU；较大 batch 可接近 CPU，但目前没有充分证据宣布一般优势。这条 FP16 量化推理路线也不能与旧 FP32 补偿训练路线作同精度速度比较。尚未测能耗、完整 Transformer 或 LLM prefill/decode。

## 检查、记录和复现

20 组数值检查通过，覆盖全部 16 个编码、三种幂二缩放、奇数维度/尾码、group32/64/128、偏置、密集转置和全零权重。INT4/NF4 两种 64→32→7 ReLU MLP、batch7 推理通过 CPU 解码权重参考；预热各重复 3 次，无测得的复制/编译，最大差分别 6.10e-5/1.22e-4。8 组性能配置的 16 个格式输出均通过独立参考及舍入预算，最大相对 L2 约 0.02945%。活动梯度、dtype 和形状拒绝检查通过。

49 项 npunlock 离线测试及 2 项 codec/IR 测试通过；原 33 项设备回归通过。最终 numeric/performance/guards/mlp 四个进程共 859 次 NPU 提交，设备回归 4160 次；均 `context_failed=false`，关闭后 live buffers/bytes、cached sessions 为零。历史失败报告保留，不并入最终通过计数。Git 的只读 diff 检查因自动审批服务用量限制未执行；独立源码语法/空白检查记录在 summary 中。

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script torch_npu_lowbit_demo.py -ScriptArguments '--case','numeric'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_demo.py -ScriptArguments '--case','mlp'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_demo.py -ScriptArguments '--case','guards'
.\run_torch_npu.ps1 -Script torch_npu_lowbit_demo.py -ScriptArguments '--case','performance'
.\run_torch_npu.ps1 -Script test_torch_npu_lowbit.py
.\.conda\npunlock\python.exe -m unittest discover -s npunlock/tests/python -p test_*.py
.\run_torch_npu.ps1 -Device -Quick
```

`numeric/performance/guards/mlp.json`、对应 `.log`、`device-regression.*` 和 `summary.json` 保存通过记录、源码哈希、采样及关闭状态。`grouped-first-failure.*`、`diagnose*.json/log/npz` 保存错算及形状/tile 对照；`performance-first-pointwise-failure.*`、`performance-precision-diagnostic.*`、`precision-*.npz` 保存舍入诊断。`module-first-probe.py` 和各 `demo-*.py` 是测试阶段源码快照；`../../lowbit-before/` 是本轮前代码。旧小 batch 原始诊断曾覆盖同名日志，原始数值失败和其后对照保留。

下一步优先解决分组权重的编译展开和小 batch 绕法成本，随后接入 INT8 矩阵计算；以硬件可执行的格式与实际模型质量决定后续精度支持，不需要逐一填满每种位宽。
