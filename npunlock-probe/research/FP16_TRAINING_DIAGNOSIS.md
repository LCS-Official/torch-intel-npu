# FP16 残差训练：诊断、修复与剩余数值限制

本机：Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动 32.0.100.5540，
编译器 8.3，Torch 2.14.1+cpu。实验使用本地 SHAVE 设备后端，不安装 OpenVINO runtime。

结论：已修复原第 6 步超出逐项容差的累加精度问题；不能据此保证任意 FP16 深层网络
与 CPU 训练完全一致。接近收敛时，前向舍入和 ReLU 边界仍会放大相对梯度差异。

## 原始失败怎样发生

保存原运行 step=5 的 CPU 参考权重与输入。CPU 参考每步从实际 NPU 权重重新开始，
因此检查点的权重完全相同。逐层记录前向输入/输出、输入梯度种子和 ReLU 开关：

1. 最早差异在 `layers.3.branch.3` 卷积：1,024 个输出中 3 个不同，最大差 0.000244140625。
2. 差异经过 BatchNorm、卷积和残差相加，末尾 `layers.6.activation` 有一个开关不同：
   CPU 输入为 0，NPU 为 0.0009765625；ReLU 在零处梯度为 0，在正数处为 1。
3. 在诊断中只抑制这个开关（不改变 NPU 权重或前向值），首层梯度 relative L2 从
   6.25% 降到 0.53%，确认它是该检查点反向差异的主要放大因素。

CPU 导出的开关只用于因果实验；真实后端始终根据自己的输入计算 ReLU 导数。
还逐个重放卷积、BatchNorm、池化和 Linear，给 CPU/NPU 完全相同的输入和梯度种子，
并加入 CPU FP64 后舍入到目标 dtype 的参考。该检查点未发现能够解释数个百分点偏差的
单算子导数公式错误；FP16 卷积反向的一些 NPU 结果反而更接近 FP64 参考。

## 实际改动

`npunlock/tests/fixtures/npu4000/resident_reduce.c` 的 sum/mean 使用 FP32 Neumaier
补偿累加，保留 subnormal-safe `add32/div32`。每次累加记录丢失的低位，最终再补入，
减少相消误差。Inf/NaN 路径单独保留普通求和语义。已重建 sum/mean ELF、汇编和 manifest。
矩阵乘法、卷积和 BatchNorm 直接受益，不改变公开调用方式、Tensor dtype 或设备分配。

| 原检查点 | 修复前 | 补偿累加后 |
|---|---:|---:|
| 最早卷积不同元素 | 3 / 1,024 | 0 / 1,024 |
| 末尾残差 ReLU 开关差异 | 1 | 0 |
| 首层梯度最大绝对差 | 0.0012779236 | 0.0000305176 |
| 首层梯度 relative L2 | 6.25% | 0.174% |

梯度检查恢复为全部 15 步，FP16 容差仍为 rtol=0.004、atol=0.0005，没有放宽。
模型工厂从测试函数中移出；残差的函数式 ReLU 改成等价 nn.ReLU，便于注册诊断 hook。
CPU 数学、权重下载和记录 hook 都是验证工作；真实 NPU 前向/反向/SGD 未加入 CPU 回退。

## 为什么仍保留限制标记

完整四组模型/精度各 15 步通过原逐项检查，FP16 SmallResNet 损失
0.258057 → 2.74181e-6。但后期整体梯度 relative L2 最高仍为 22.6%，
最低 cosine 为 0.97406。梯度较小时，绝对容差通过不足以证明相对误差较小。
FP16 参考仍每步重设到 NPU 权重，不认证独立 FP16 训练轨迹与 CPU 一致。
报告为 `passed_with_limitations`，与 FP32 独立训练对照区分。

另从原 step=5 检查点分别继续训练 15 步，比较不缩放与 1024 倍静态损失缩放。
缩放后梯度再在 NPU 上还原，热步骤无主机载荷复制；这不是完整 AMP/GradScaler 实现。
缩放不能消除所有后期差异（该实验最高 relative L2 27.2%）。两种缩放的参数轨迹会分叉，
这个数不能与完整四模型测试的 22.6% 当成同一个轨迹的直接性能/精度比较。

后期检查点的输出已非常接近 FP16 目标 0.199951171875。例如一个位置 CPU 输出恰好等于
目标，而 NPU 输出为 0.2001953125；前向差只有 0.000244140625，MSE 梯度却会从零变成
非零。此时 Linear 输出传入的梯度 relative L2 已相差约 20%，随后再叠加一个残差 ReLU
开关差异。说明损失缩放不能恢复前向已经舍入掉的信息。

直接重放后期首层卷积：NPU 与 CPU FP16 有 2 / 1,024 个元素不同，NPU 全部元素与
CPU FP64 计算后转 FP16 的参考逐位一致；CPU FP16 的那两个元素偏离该参考。
因此剩余跨设备差异不能一概视为 NPU 内核算错，也不宜通过模仿 CPU 的舍入来消除。
该结果只证明这个卷积实例，不保证所有模型/输入正确舍入。

当前复杂模型的严格 CPU 数值对照仍建议 FP32。后续更实用的方向是显式混合精度，
保留归一化、残差/关键激活、输出头与损失的 FP32 精度；本次未实现 autocast、FP32
主权重或完整 GradScaler。补偿累加增加了 FP32 加法成本，未做受控性能比较。

## 验证与复现

- 扩展模型：9 组，31,938 次 NPU 提交，四组各 15 步；逐项梯度/更新通过，数值限制如上。
- 原训练回归：11 组，10,548 次提交；通过。
- 完整设备推理回归：33 组，19,476 次提交；通过。
- 相消、FP16/FP32 GEMM、Inf/NaN、FP32 非规格化数：6 项硬件回归通过。
- 42 项离线 Python 检查通过。三组完整测试 shutdown 后 live_buffers=0，context_failed=false。
- 扩展模型热训练步骤没有主机载荷复制或新图编译，验证下载另计。

```powershell
.\run_torch_npu.ps1 -Models
.\run_torch_npu.ps1 -Training
.\run_torch_npu.ps1 -Device
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_compensated_reduce.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/diagnose_fp16_training.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_fp16_loss_scaling.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_fp16_late.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/inspect_fp16_rounding.py
```

`diagnose_fp16_training.py` 保留原检查点；默认重放当前内核，不会重新生成修复前的
报告。因果开关试验的旧结果只适用于当时的原内核与检查点。

报告均位于 `npunlock-probe/research/results/`：

- `fp16-residual-diagnosis-before-fix.json`、`fp16-relu-boundary-intervention.json`：原失败及因果干预。
- `fp16-isolated-layers.json`：修复前相同输入/梯度的单算子与 FP64 对照。
- `fp16-residual-diagnosis.json`：补偿累加后原失败检查点重放。
- `torch-npu-models-compensated-1.json`：完整扩展模型及每步 relative L2/cosine。
- `fp16-loss-scaling.json`、`fp16-late-diagnosis.json`、`fp16-late-stem-rounding.json`：剩余差异实验。
- `torch-npu-training-compensated-pass.json`、`torch-npu-device-compensated-pass.json`：原功能回归。
- `compensated-reduce-regression.json`、`python-compensated-reduce-regression.log`：硬件边界与离线检查。
