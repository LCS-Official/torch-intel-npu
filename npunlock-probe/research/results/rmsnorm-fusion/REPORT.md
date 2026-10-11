# 图内 RMSNorm → attention / FFN 投影融合

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

已接入冻结 SmolLM2 层的单 token B2..B8 路径，包括 forward、单 token prefill 与 decode。每层暖 decode 从 4 图降到 2 图：FP32 SHAVE RMSNorm 与 attention 的 DPU 投影组成一图，第二次 RMSNorm 与 FFN 的 DPU 投影组成另一图。公开调用方法保持不变，暖分配仍为 5 次/层。

归一化保留同一生产内核、FP32 Neumaier 补偿求和、epsilon 与两次 FP16 舍入；DPU 权重支持 FP16/W8A16。没有重建内核、DLL/PYD，也未更改通用严格 FP32 算子。运行不需要 OpenVINO Python/runtime 包，仍使用 Intel 驱动图编译器。

## 融合范围与安全边界

20 组编译布局探测显示：B1 默认四 tile 将 576 元素拆成 80/64 元素片段；一 tile 仍拆成两段 288。RMSNorm 必须看到完整一行，因此 B1 和多 token 前缀继续使用原路径。B2..B8 全图的 attention-prefill、attention-decode、FFN 共 42 组验证通过，全部与分图基线逐位相同。编译器可为不同 DPU tile 复制完整行的归一化任务，不能把任务数量等同于唯一行数。

只替换具有固定 Power 代码指纹的 SHAVE 文本窗口，并验证完整行布局、FP16 静态 CMX memref、输入/输出不重叠、范围引用与代码大小。42 对原始/修改 blob 的代码窗口以外字节完全相同：未修改 DMA、barrier、buffer 或任务调度。该实验接口仅支持已验证的设备/编译器与载体指纹，不是任意图融合器。

首次批量 decode 复用 prefill 已驻留的基础 gamma，在 NPU 上广播新的行数，不重新上传 CPU 权重。首次编译/广播仍有准备成本；预热后显式上传/下载/编译为零。计数不包含编译器内部 DMA，不能称作零 IO。

## 同模型、同输入、同缓存对照

同一实例仅切换融合开关；两种策略共用输入、gamma、权重和原 KV，所有输出/KV 位模式一致。41 轮 ABBA/BAAB，每策略 82 样本，计时排除开关切换与冷准备。

| 层数 / B | 模式 | 分图 ms | 融合 ms | 中位延迟降幅 | 融合更快轮数 |
|---|---|---:|---:|---:|---:|
| 4 / 1 | fp16 | 10.087 | 10.174 | -0.86% | 20/41 |
| 4 / 1 | w8a16 | 9.799 | 10.068 | -2.75% | 20/41 |
| 4 / 2 | fp16 | 10.596 | 7.020 | 33.75% | 41/41 |
| 4 / 2 | w8a16 | 9.870 | 6.531 | 33.83% | 41/41 |
| 4 / 4 | fp16 | 10.787 | 6.146 | 43.02% | 41/41 |
| 4 / 4 | w8a16 | 10.559 | 6.217 | 41.12% | 41/41 |
| 1 / 8 | fp16 | 2.998 | 2.218 | 26.00% | 40/41 |
| 1 / 8 | w8a16 | 3.032 | 2.069 | 31.76% | 40/41 |

B1 是未改变执行路径的对照，其差异反映波动。结果为暖模型片段计时，包含绑定、提交、设备执行与 fence 同步；未锁定桌面负载、频率或内存状态。

## 独立实例与 CPU 对照

每组 21 轮轮转各路径；CPU 为 FP16、8 线程。以下均为 decode ms，完整原始样本及 full-prefix 时间在 performance.json。该比较与上面的同实例对照有不同分配/缓存条件，均保留实测结果。

| 层数 / B | FP16 分图→融合 | W8A16 分图→融合 | CPU FP16 |
|---|---:|---:|---:|
| 1 / 1 | 2.943 → 4.042 | 2.876 → 3.178 | 1.462 |
| 1 / 4 | 3.963 → 2.601 | 3.432 → 2.314 | 2.502 |
| 2 / 1 | 6.092 → 7.789 | 5.806 → 6.977 | 2.407 |
| 2 / 4 | 6.191 → 4.268 | 6.490 → 3.869 | 3.846 |
| 4 / 1 | 11.969 → 13.987 | 11.838 → 12.999 | 3.768 |
| 4 / 4 | 12.936 → 8.169 | 12.582 → 7.333 | 6.197 |

这些冻结层片段时间不能直接换算为完整 LLM 生成速度，也未测功耗。模型支持范围仍是 B≤8、相同前缀长度、总位置≤64；不增加训练、padding mask 或完整模型支持。

## 融合后阶段比例

比例来自 21 轮 traced 平均值。归一化已包含在对应 attention/FFN 阶段中，不能把旧独立 RMSNorm bucket 的零值理解为归一化耗时为零。

| 层数 / B | 模式 | norm + attention | norm + FFN | 原生调用数 |
|---|---|---:|---:|---:|
| 4 / 4 | fp16 | 52.7% | 30.1% | 8 |
| 4 / 4 | w8a16 | 52.8% | 29.3% | 8 |
| 1 / 8 | fp16 | 56.0% | 28.8% | 2 |
| 1 / 8 | w8a16 | 56.2% | 28.0% | 2 |

## 验证及追溯

102 条真实权重数值记录、336 个固定输入 K/V 对照、252 次连续位置解码通过。24 条新旧 cached decode、8 条普通 forward 与 8 组混合生成保留逐位/生成 token 比较。执行最大相对 L2 0.2003%，scaled error 0.6472%；原执行门槛 0.3% / 1% 与量化质量门槛 3% 保留。

16 组样本隔离检查覆盖 B2..B8、单 token prefill、不同样本独立 B1 对照、扰动隔离、旧 KV 与公开输出保留、冷解码零主机复制。不同 batch 形状的结果采用原数值门槛，不保证逐位一致。两线程的 6 次 warm decode 共 12 图，输出及 K/V/position 独立，值一致；NPU 锁仍串行执行，该检查不证明并行加速。

9 项离线融合保护检查、原有 16 项 ELF/载体检查与通用 FP16/FP32 快速回归通过；后者为 105 项、435 次 NPU 运算，涵盖特殊值、归约、矩乘、autograd、torch.compile 与 11 类拒绝行为。

| 成功进程/用例 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
| probe | 8 | 0 | 136 |
| numeric | 40 | 0 | 1872 |
| boundary | 14 | 0 | 592 |
| guards | 23 | 0 | 119 |
| positions | 44 | 0 | 2600 |
| hybrid | 8 | 0 | 816 |
| paired | 24 | 0 | 472 |
| regression | 8 | 0 | 88 |
| profile | 0 | 8 | 2900 |
| performance | 0 | 6 | 7532 |
| layout-probe | 20 | 0 | 0 |
| model-probe | 42 | 0 | 234 |
| ownership | 17 | 0 | 942 |
| control | 8 | 8 | 14640 |

上述成功设备进程共 32,943 次记录内 NPU 调用，全部健康关闭、存活分配归零；通用快速回归另计 435 次。两次开发阶段失败单独保存在 failures/：静态 invocationCount 为 0 的校验过严，以及冷 decode 重复上传 gamma，均已修复，未放宽数值或无主机复制门槛。初步性能样本保存在 control-preliminary.*，最终结论使用 control.json。

模型 safetensors SHA：`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。sources、kernel manifest、before/after/diff、原始/修改 blob 与审计结果保存在本目录。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_fusion_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_fusion_guards.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_fused_model_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_fusion_ownership.py
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','rmsnorm-fusion','--case','numeric'
.\run_torch_npu.ps1 -Script .\torch_npu_norm_fusion_benchmark.py
.\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_rmsnorm_fusion.py
```
