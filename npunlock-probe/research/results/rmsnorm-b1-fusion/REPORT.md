# B1 单序列：完整行归一化与投影融合

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

冻结 SmolLM2 层的单 token B1 forward/prefill/decode 已默认融合，每层暖 decode 从 4 次提交降到 2 次；四层从 16 次降到 8 次。B2..B8 延续上一阶段的融合，多 token 前缀保留原路径。每层暖分配仍为 5 次，公开 API、输入输出 batch 与 KV 大小保持原样。

## 实现与布局验证

仅在图内归一化入口将 1×576 输入复制为 2×576，执行原有 FP32 SHAVE RMSNorm，再切回第一行，送入原有 B1 投影。gamma 使用已有的两行驻留广播参数。第二行增加归一化与相应调度/数据移动工作，不执行第二份模型 attention/FFN 或 KV 更新，也不暴露额外输出；这不是多用户 batch。

10 组 compile-only 探测对照了原始行、只复制归一化输入、保留 norm 输出、复制完整投影及保留投影输出，并比较默认四 tile 与一 tile。原始 B1 分别拆成 80/64 或 288 元素片段；只复制归一化输入并切回一行，就能保留完整 576 元素，无需辅助输出或完整模型复制。

真实 FP16/W8A16 的 attention-forward、attention-prefill、attention-decode、FFN 共 8 组全部逐位一致；每组非 elementwise DPU invariant 数量与原 B1 分图相同。公开图中投影与动态注意力保持 B1，代码窗口以外的 blob 字节逐位不变；完整行、CMX 边界、精度和别名校验继续使用上一阶段的接口。

没有重建 SHAVE ELF 或 native DLL/PYD，未更改 FP32 Neumaier 补偿归约、epsilon、舍入顺序或通用严格 FP32 算子。运行仍无需 OpenVINO Python/runtime，仍使用 Intel 驱动图编译器。

## 同实例 B1 性能对照

同一模型、输入、gamma 与原 KV，仅切换 B1 融合开关。41 轮 ABBA/BAAB，每策略 82 样本，所有输出/KV 位模式一致；暖显式上传/下载/编译为零，两路径分配均为 5 次/层。

| 层数 / 前缀位置 | 模式 | 原分图 ms | B1 融合 ms | 中位延迟降幅 | 更快轮数 |
|---|---|---:|---:|---:|---:|
| 1 / 1 | fp16 | 2.951 | 1.980 | 32.91% | 41/41 |
| 1 / 1 | w8a16 | 2.907 | 1.925 | 33.78% | 41/41 |
| 1 / 32 | fp16 | 2.908 | 1.976 | 32.05% | 41/41 |
| 1 / 32 | w8a16 | 3.295 | 2.273 | 31.03% | 41/41 |
| 2 / 32 | fp16 | 5.324 | 3.720 | 30.12% | 41/41 |
| 2 / 32 | w8a16 | 5.478 | 3.840 | 29.89% | 41/41 |
| 4 / 63 | fp16 | 9.315 | 6.356 | 31.76% | 41/41 |
| 4 / 63 | w8a16 | 10.213 | 6.917 | 32.27% | 41/41 |

计时包含绑定、提交、设备工作及 fence，排除初始准备、编译和显式传输；内部 DMA 未计入复制计数。未锁定桌面负载、频率或内存状态，初步样本保存在 control-preliminary.*，最终结论使用 control.json。

## 独立实例及 CPU 对照

每组 21 轮轮转各路径，CPU 为 FP16、8 线程；下表为 decode ms。B4 的旧/新版本均执行上一阶段的批量融合，是未改变该路径的对照。B2..B8 的 56 组 FP16/W8A16 attention-forward/prefill/decode/FFN IR XML 与权重字节完全相同；部分未改变路径的计时仍出现变慢，说明不能据此归因于 B1 修改。独立实例与同实例具有不同的分配/缓存条件，保留全部测量结果。

| 层数 / B | FP16 旧→新 | W8A16 旧→新 | CPU FP16 |
|---|---:|---:|---:|
| 1 / 1 | 4.380 → 3.411 | 4.429 → 2.340 | 1.395 |
| 1 / 4 | 2.749 → 3.113 | 2.578 → 2.663 | 2.694 |
| 2 / 1 | 9.114 → 6.190 | 8.128 → 5.886 | 4.513 |
| 2 / 4 | 4.662 → 4.859 | 4.560 → 4.456 | 4.555 |
| 4 / 1 | 14.366 → 9.822 | 13.730 → 9.069 | 5.707 |
| 4 / 4 | 10.536 → 11.064 | 10.071 → 9.820 | 8.670 |

以上为冻结层片段时间，不能直接换算为完整 LLM 生成速度，也未测功耗。支持范围仍是 B≤8、相同前缀长度、总位置≤64；不增加训练、padding mask 或完整模型支持。

## 阶段比例

21 轮 traced 平均值，归一化包含在 attention/FFN 阶段；旧独立 RMSNorm bucket 为零不表示归一化没有成本。

| 层数 / B | 模式 | norm + attention | norm + FFN | 原生调用数 |
|---|---|---:|---:|---:|
| 1 / 1 | fp16 | 48.3% | 31.3% | 2 |
| 1 / 1 | w8a16 | 52.3% | 29.5% | 2 |
| 4 / 1 | fp16 | 51.1% | 29.0% | 8 |
| 4 / 1 | w8a16 | 49.6% | 29.5% | 8 |
| 4 / 4 | fp16 | 51.6% | 29.6% | 8 |
| 4 / 4 | w8a16 | 52.1% | 30.5% | 8 |
| 1 / 8 | fp16 | 53.2% | 28.4% | 2 |
| 1 / 8 | w8a16 | 51.5% | 31.1% | 2 |

## 正确性、快照与追溯

102 条真实权重数值记录、336 个固定输入 K/V 对照、252 次连续位置解码通过。24 条新旧 cached decode、8 条普通 forward 与 8 组混合生成通过逐位/token 检查。最大执行相对 L2 0.2003%，scaled error 0.6472%；原门槛 0.3% / 1% 与量化质量门槛 3% 保留。

14 组隔离检查覆盖 B1/B2/B3/B8、前缀 1/7/32/63、冷 decode 零主机复制、输入扰动、旧 KV 与保留输出。一个实例按 B1→B2→B1→B8→B3→B1 切换，B1/B2 共用 gamma 地址，6 组输出/KV 快照保持不变。两个调用者对同一 B1 缓存执行 6 次 warm decode，共 12 图，公开输出及 K/V/position 各自独立、位模式一致；执行仍由 NPU 锁串行化。

未修改底层载体/精度 helper，上一阶段的 9 项融合保护、16 项 ELF/载体与通用快速精度回归证据保留在 rmsnorm-fusion/。本轮设备回归不放宽输入保护、数值或无主机复制门槛。

| 成功进程/用例 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
| probe | 8 | 0 | 104 |
| numeric | 40 | 0 | 1464 |
| boundary | 14 | 0 | 560 |
| guards | 23 | 0 | 103 |
| positions | 44 | 0 | 1320 |
| hybrid | 8 | 0 | 528 |
| paired | 24 | 0 | 388 |
| regression | 8 | 0 | 88 |
| profile | 0 | 8 | 2000 |
| performance | 0 | 6 | 6272 |
| layout-probe | 10 | 0 | 0 |
| model-probe | 8 | 0 | 40 |
| ownership | 16 | 0 | 506 |
| control | 8 | 8 | 8184 |

成功设备进程共 21,557 次记录内 NPU 调用，全部健康关闭、存活分配归零。首次探测脚本误用一行 gamma 绑定两行图参数，被 native 缓冲区尺寸校验拒绝；修正脚本绑定后通过，失败记录在 failures/probe-gamma-binding/，未放宽保护。

模型 safetensors SHA：`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。完整 sources、kernel manifest、before/after/diff 与原始/修改 blob 在本目录。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_b1_layout_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_b1_model_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_b1_ownership.py
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','rmsnorm-b1-fusion','--case','numeric'
.\run_torch_npu.ps1 -Script .\torch_npu_b1_fusion_benchmark.py
.\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_rmsnorm_b1.py
```
