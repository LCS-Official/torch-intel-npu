# W8 提示与续写联合校准

2026-10-07，Lunar Lake 8086:643e，驱动 1005540、编译器 8.3。

将校准扩展为 40 条 CPU FP16 轨迹、2,560 个 token，覆盖全部 30 层 / 210 个矩阵。
生成阶段的平均误差和通过位置数进一步改善，但最差位置仍有大误差，且一条独立提示退化。
**当前候选仍不推荐默认启用。** 继续保留原来的 logits 相对 L2 < 0.03、next-token KL < 0.02 门槛。

## 实现与数据

`w8_continuation_calibration.py` 复用已有 `calibrate_int8`、全层 NPU 评估和质量审计函数。
沿用原来的 12 条提示，新增 18 条问答/代码/翻译提示及 10 条故事提示，共 40 条，均与验证提示不重合。
每条以完整提示 prefill，再由 CPU 原模型 greedy decode 到 64 个实际消费 token；与既有测试一样不在 EOS 停止。
共采集 1,170 个提示 token、1,390 个生成 token。所有输入、续写文本、token IDs 保存在 `prepare.json`。

每层的 norm1、attention、norm2、gated 均有 2,560 条激活观测。
最大激活/投影结果为 2560×1536 = 3,932,160 个元素，满足现有 4×1,048,576 的校准存储限制。
没有扩大运行时 tensor 或上下文上限，NPU 仍最多 64 个缓存位置。

保持原有 FP64 求解、1% damping、128-column block、原 FP16 行 scale、激活排序和逐行 SSE 回退。
210 个矩阵的每行重建 SSE 均不增加，累计从 1772095.1032 降至 27335.7945，约为未校准的 1.54%。
该训练集指标不能与上一阶段不同样本集的 SSE 直接比较，也不等同于模型质量。
离线准备用时约 121.59 秒；权重 SHA256：
`0cacf8f890a31f57cea77c5facdee77ee6a5a4905a987f31947afe99fe4447cb`。

生产内核、运行库、校准算法及旧实验数据均保留原样；新增 profile 继续严格绑定源码、权重、checkpoint 和输入。
本阶段另外绑定了模型 config、tokenizer JSON 和 tokenizer config，没有新增依赖。

## 全层 NPU 验证

30 个 decoder layers 全部在 NPU，embedding、final norm、LM head 使用显式 CPU 参考。
这仍是冻结推理测试。复用原来的 16 条独立提示加 2 项完整 prefill 对照，共 18 项；
连续测试为三条故事各 32 个位置，以及重复前缀的 33 个位置，共 129 项，覆盖缓存长度 64。
采用相同 CPU greedy continuation，以比较同一位置和输入。

| 指标 | 上一轮：349 个提示 token | 本轮：2,560 个提示/生成 token |
| --- | ---: | ---: |
| 独立提示最大 logits 相对 L2 | 3.2710% | 5.0802% |
| 独立提示质量通过 | 17/18 | 17/18 |
| 连续测试平均相对 L2 | 3.4545% | 2.5143% |
| 连续测试最大相对 L2 | 13.4642% | 11.8351% |
| 连续测试最大 KL | 0.01132750 | 0.02154210 |
| 连续测试质量通过 | 76/129 | 100/129 |
| 连续测试 next-token 一致 | 127/129 | 129/129 |

本轮独立提示最大 KL 为 0.00230086，next-token 一致 18/18。
letter 提示从上一轮相对 L2 约 1.29% 退化到 5.08%；完整 prefill 诊断为 4.97%，同样未通过。
连续测试最大 L2 位于 bird/15；最大 KL 位于 cat/29，同时超过原 KL 门槛。
因此，next-token 一致并不能替代分布和数值质量门槛，也不能据此宣称任意自由生成文本等价。

FP16 对照仍为独立提示 18/18、连续位置 129/129 通过，连续最大相对 L2 为 1.1898%。
本轮与上一阶段的 CPU 输入/参考共 167 项数组逐位一致；FP16 / 原始 W8 logits 共 294 项逐位一致。
因此，候选间的变化来自校准权重，控制组的输入与执行结果没有改变。

## 误差复现与敏感性诊断

letter 上三组权重各 30 层局部 prefill 数值检查，共 90 项通过。
最大局部相对 L2 为 0.00079105、最大缩放误差为 0.00599932，低于原执行门槛 0.003 / 0.01。
这是局部 prefill 检查，不代表已经逐层检查所有 decode 位置。

CPU 使用同一组量化权重和 staged arithmetic，仍复现以下误差：

| 位置 | NPU 相对 L2 | CPU 同权重相对 L2 |
| --- | ---: | ---: |
| bird/15 | 11.8351% | 11.8044% |
| garden/14 | 4.4690% | 4.1530% |
| cat/11 | 11.1299% | 11.5260% |
| repeat32/21 | 10.6072% | 10.6668% |

大误差在 CPU 上也存在，主要问题仍在量化后的模型。增加这批样本改善了总体表现，却没有消除尖峰。

另用 `w8_tail_precision_probe.py` 在 CPU 参考中比较末尾若干层保留 FP16。
复用 `reference_cpu(quantized=False)`，检查上述四个位置和 letter，共 5 个回归案例；没有修改生产策略。

| 末尾 FP16 层数 | 五案例最大相对 L2 | 质量通过 |
| ---: | ---: | ---: |
| 0 | 11.8044% | 0/5 |
| 1 | 9.9396% | 0/5 |
| 2 | 11.6855% | 1/5 |
| 4 | 12.2927% | 1/5 |
| 8 | 10.4583% | 2/5 |
| 30 | 0.9419% | 5/5 |

保留更多末端 FP16 层不保证单调改善，不能只按层号决定精度。
这些是现有失败案例上的 CPU 敏感性诊断，不是对混合精度 NPU 模型的验证，也没有据此选定部署配置。
下一步应检查逐层输入漂移、校准造成的权重偏移和求解约束，再用独立数据验证方案；
不宜继续只加同类样本或直接指定末尾若干层 FP16。

## 审计与复现

`audit.json` 通过：52,830 次成功 NPU 调用，459 项质量指标复算、90 项局部执行复算，
180 份 decode 图精度/补丁链证据、461 项历史控制数组逐位核对。
CPU 误差复现另核对 120 项层漂移；末端精度诊断自检 30 项指标。
末端诊断的全 W8 输出与另外两个 CPU 参考入口共 5 项 logits 数组逐位一致；24 份快照与当前源文件指纹核对通过。
各硬件进程最终 context 健康，live / cached / quarantined buffer bytes 均为零。
warm repeat 均为每层一次提交，无新增编译或显式主机载荷复制，输出及 KV/position 逐位一致，旧 seed 保持不变。
本轮没有测量模型吞吐或峰值算力。`status=passed` 表示执行/审计完成；`candidate_recommended=false` 表示质量尚未全部通过。

仓库根目录、现有 `.conda/npunlock` 环境；NPU 测试串行。命令写入本阶段固定目录，重跑前保留已有结果。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_continuation_calibration.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_continuation_calibration.py -ScriptArguments '--case','holdout'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_continuation_calibration.py -ScriptArguments '--case','decode'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_continuation_calibration.py -ScriptArguments '--case','diagnostic'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_continuation_calibration.py -ScriptArguments '--case','cpu'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_tail_precision_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_continuation_calibration.py -ScriptArguments '--case','audit'
```

源文件/运行库/模型配置快照位于 `sources/`，完整指纹见 `prepare.json`、`snapshots.json` 和 `final-verification.json`。
