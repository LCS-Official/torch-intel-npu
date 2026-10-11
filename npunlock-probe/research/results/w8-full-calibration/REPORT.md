# 全 30 层 W8 激活校准与质量诊断

2026-10-07，Core Ultra 7 258V / Lunar Lake，8086:643e，驱动 1005540，编译器 8.3。

已将离线激活校准扩展到 SmolLM2-135M-Instruct 的全部 30 个 decoder layers、210 个矩阵。
生成的完整 INT8 权重 map 可传入现有 `FrozenSmolStack.from_checkpoint(..., packed_weights=...)`。
当前候选改善了平均误差，但没有全部通过原质量门槛，**不推荐为通用默认权重**。
运行和离线审计通过，与“模型质量全部通过”是两个不同结论。

## 实现与校准范围

- 新入口为 `npunlock-probe/research/w8_full_calibration.py`，复用公开 `calibrate_int8`、原有提示和误差函数。
- 12 条校准提示，共 349 个 token；采集全部层的 norm1、attention、norm2、gated 输入。
- 保持原有 FP64 GPTQ 求解、1% damping、128-column block、激活排序、原 FP16 行 scale 和逐行重建误差回退。
- 210 个矩阵均满足校准样本上逐行 SSE 不增加。累计 Linear 重建 SSE 从 509366.6164 降为 1155.9094，约为原来的 0.227%。这不是独立模型误差。
- 前四层 56 个 codes/scales 数组与上一阶段逐位一致；全层准备用时约 41.14 秒。
- 校准在 CPU 离线进行，推理保持 W8A16；没有新增 OpenVINO 依赖或修改生产内核。

完整 map 包含 420 个 codes/scales 数组；权重文件 SHA256：
`6f337619bbf792852716846a8227caa5a4a6947a063cf476dc041ccb48657469`。
`prepare.json` 绑定当前源码、内核、运行库、校准输入与 checkpoint，加载继续严格核对这些指纹。
旧阶段的权重和结果保留原样。

## 全层独立验证

所有测试的 30 个 decoder layers 均在 NPU 执行，embedding、final norm、LM head 使用显式 CPU 参考。
FP16、原始 W8、校准 W8 使用相同输入。质量门槛保持 logits 相对 L2 < 0.03、next-token KL < 0.02。

16 条独立提示不参与校准。逐 token 送入提示以复用静态 decode 图，另对 bird/math 做完整 prefill 对照，共 18 项。
连续测试采用三条故事各 32 个位置，以及 32-token 重复前缀的 33 个位置，共 129 项，最后覆盖缓存长度 64。
连续输入采用相同的 CPU greedy continuation，不在 EOS 停止；这里的 next-token 一致率不代表自由生成整段文本一致。

| 测试 / 权重 | 最大 logits 相对 L2 | 最大 KL | 通过质量门槛 | next-token 一致 |
| --- | ---: | ---: | ---: | ---: |
| 独立提示 FP16 | 1.0100% | 0.00018176 | 18/18 | 18/18 |
| 独立提示原始 W8 | 10.2794% | 0.00912723 | 8/18 | 18/18 |
| 独立提示校准 W8 | 3.2710% | 0.00340916 | 17/18 | 18/18 |
| 连续测试 FP16 | 1.1898% | 0.00008132 | 129/129 | 129/129 |
| 连续测试原始 W8 | 15.9950% | 0.01023246 | 34/129 | 127/129 |
| 连续测试校准 W8 | 13.4642% | 0.01132750 | 76/129 | 127/129 |

连续测试的平均相对 L2 为 FP16 0.2880%、原始 W8 6.0981%、校准 W8 3.4545%。
校准整体有改善，但存在比原始 W8 更差的位置，不能据平均值启用默认。
最差独立提示为 fox；完整 prefill 诊断的校准误差为 3.6403%，也未过 3% 门槛。

## 剩余误差的定位

对 fox 的三组权重各检查 30 层局部 prefill，共 90 项。CPU 局部参考使用该层实际 NPU 输入，
W8 参考保留驱动 MatMul 中未预先舍入的 `INT8 code × FP16 scale` 语义。
最大局部相对 L2 为 0.00081316，最大缩放误差为 0.00716505；均通过原执行门槛 0.003 / 0.01。
这些是局部 prefill 数值核对，不能当作全部 decode 位置的逐层误差检查。

另外，`w8_full_cpu_diagnostic.py` 在 CPU 上重放各 continuation 的最差校准位置。
它不导入 `torch_intel_npu`，不提交 NPU 图；从保存的 tokens 重建完整前缀，
使用同一份 W8 权重和 staged arithmetic 参考。结果仍出现大误差：

| 位置 | NPU 校准 W8 相对 L2 | CPU 同权重相对 L2 |
| --- | ---: | ---: |
| bird/10 | 10.9220% | 10.3564% |
| garden/4 | 7.9945% | 8.5777% |
| cat/15 | 13.4642% | 13.2531% |
| repeat32/12 | 10.9596% | 11.0541% |

CPU FP16 完整前缀相对其 cached baseline 的误差最大为 0.9202%，小于上述 W8 偏差。
因此，主要问题来自量化后的模型；当前证据不支持把 10% 以上的误差归因于 Select 或 SHAVE 内核。
提示阶段校准 SSE 很低，生成位置仍有较大误差，说明必须继续检查校准数据的代表性。
当前只有 349 条激活观测，少于各矩阵 576/1536 的输入维度，而且没有覆盖回答/续写。

下一步优先扩充到约 2K～4K 个代表性 token，包含回答与续写，再用相同独立提示和 continuation 复验。
若仍有少数层敏感，再评估逐层校准或对这些层保留 FP16；本轮尚未实施这些方案。

## 审计与资源

`audit.json`：52,830 次成功 NPU 调用，447 项质量指标复算、90 项局部执行误差复算。
另有 CPU 复现的 12 项 staged 质量和 120 项层漂移复算，包含当前预测 token 的漂移。
所有 NPU 运行结束后 context 健康，live / cached / quarantined buffer bytes 为零。
warm repeat 均保持每层一次提交、无新增编译或显式主机载荷复制，输出及每层 K/V/position 逐位一致。
旧 prefill seed 在 decode 后保持不变。180 份 decode 图证据中，120 份 W8 图具有 INT8 权重 / FP16 激活 DPU task，
60 份 FP16 图没有 INT8 矩阵 task；均核对了 RMSNorm/Select 补丁 SHA 链及 Select ELF。
18 份当前源码和运行库快照保存在 `sources/`，最终核对见 `final-verification.json`。
本轮未测量模型吞吐或芯片峰值算力。

## 复现

仓库根目录，现有 `.conda/npunlock` 环境。命令写入本阶段固定目录，重跑前先保留已有结果。
NPU 测试串行执行；CPU-only 诊断不使用 NPU。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_full_calibration.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_full_calibration.py -ScriptArguments '--case','holdout'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_full_calibration.py -ScriptArguments '--case','decode'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_full_calibration.py -ScriptArguments '--case','diagnostic'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_full_cpu_diagnostic.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_full_calibration.py -ScriptArguments '--case','audit'
```

各运行 `status=passed` 表示测试执行/核对完成；质量结论以 `quality_gate_passed`、
`candidate_all_quality_gates_passed`、最终 `candidate_recommended` 为准。本轮推荐标记为 `false`。
