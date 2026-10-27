# SmolLM启动：精简CPU模型与重复权重准备

2026-10-16，Windows / Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。

新增可复用的 `FrozenSmolIO`：只加载CPU embedding、最终RMSNorm和共享输出权重，替代NPU推理时未使用的完整CPU decoder。另将混合50%的研究入口改为既有公开构造参数一次准备，省去30层的第二次构建。模型计算、BF16→FP16转换检查、量化权重及DPU/SHAVE/原生运行时未改变。

## 真实NPU：已有缓存的新进程

| 配置 | 原准备 ms | 新准备 ms | 原脚本入口到首token s | 新入口到首token s | 入口耗时降低 |
|---|---:|---:|---:|---:|---:|
| FP16 | 2570.110 | 2217.167 | 5.707 | 5.326 | 6.68% |
| 校准W8 | 3272.592 | 2861.260 | 6.169 | 5.702 | 7.57% |
| 混合50% | 3191.717 | 2600.444 | 5.917 | 5.280 | 10.76% |
| 逐层37.5% | 3476.111 | 3133.434 | 6.515 | 6.127 | 5.96% |

“准备”包括CPU模型/轻量入口、tokenizer、校准权重校验与解包、NPU层构造及末尾GC；脚本入口另包括导入和NPU初始化。没有把准备挪到入口计时之外。

模型准备后到首token（每种3次中位数）：

| 配置 | 原 ms | 新 ms |
|---|---:|---:|
| FP16 | 980.489 | 972.993 |
| 校准W8 | 762.341 | 762.599 |
| 混合50% | 827.990 | 809.663 |
| 逐层37.5% | 837.275 | 833.755 |

这轮收益来自CPU准备和模型构造；没有改善NPU芯片算力或认证热decode吞吐提升。准备后的首token存在桌面负载波动，应按原始配对数据阅读。

## 方法

- 完整30层SmolLM2-135M-Instruct、batch1、24-token固定提示，decoder与KV在NPU；CPU两端始终显式。Torch CPU8线程、interop1线程。沿用原来的完整24位置CPU输出头计算。
- 四种精度保持固定：FP16、全部校准W8、全部Up/Down保留FP16、先前45项逐层保护。同一份逐行GPTQ校准权重、codes/scales与保护项未调整；权重NPZ SHA256 `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62`。
- 每种精度先填充60份prefill/decode缓存，再做3对全新进程，顺序before→after、after→before、before→after。相同缓存目录、相同编译blob；首请求30次命中、0次编译、30次跳过IR。
- before保持完整CPU Transformers模型；混合50%还保留旧研究入口“先建30层再替换30层”的构造方式。after使用FrozenSmolIO及单次全局混合构造；逐层37.5%仍用原研究组合，没有新增公开逐层策略API。
- 每进程首请求后预热2次、记录7次热prefill。各精度填充进程和第一对进程另运行完整greedy32-token生成，忽略EOS，最大消费55个位置；生成在启动计时与内存快照之后。
- 输出/状态hash及数值检查在计时区间外。未锁定频率或桌面负载，未清空系统文件页缓存；三对样本只代表本机此次条件。无缓存编译、其它模型、长上下文、训练和GPU协同不在范围内。

## 独立CPU分段

独立CPU profile排除了profile前已完成的库导入；与上面的入口计时口径不同，不能将两表直接相减。hook为inclusive，不能与分段相加。

| 配置 | 完整CPU模型 ms | tokenizer ms | 校准权重校验/解包 ms | stack ms | 层构造次数 |
|---|---:|---:|---:|---:|---:|
| FP16 | 418.259 | 88.766 | 0.001 | 376.910 | 30 |
| 校准W8 | 433.863 | 81.972 | 690.746 | 341.033 | 30 |
| 混合50% | 424.399 | 85.809 | 687.826 | 620.770 | 60 |
| 逐层37.5% | 417.366 | 85.351 | 691.434 | 615.919 | 58 |

CPU校准权重加载约0.69秒仍保留完整源文件/模型SHA检查、NPZ解压与PackedWeight验证。当前checkpoint是BF16；NPU层的原BF16→FP32→FP16准备及溢出/非有限检查保持原样。

## 实际新进程的准备分段

| 配置/路径 | CPU入口与tokenizer ms | 校准权重校验/解包 ms | stack ms | 总准备 ms |
|---|---:|---:|---:|---:|
| FP16/before | 2034.969 | 0.003 | 425.408 | 2570.110 |
| FP16/after | 1699.684 | 0.003 | 411.624 | 2217.167 |
| 校准W8/before | 2074.442 | 705.503 | 385.105 | 3272.592 |
| 校准W8/after | 1658.118 | 729.404 | 358.409 | 2861.260 |
| 混合50%/before | 1814.699 | 652.148 | 607.946 | 3191.717 |
| 混合50%/after | 1512.820 | 670.149 | 324.581 | 2600.444 |
| 逐层37.5%/before | 1999.921 | 714.378 | 642.226 | 3476.111 |
| 逐层37.5%/after | 1666.042 | 691.475 | 656.704 | 3133.434 |

分项中位数不必相加为总中位数；总准备额外包括GC。CPU入口分项还包含首次所需的Transformers类导入。

## 进程内存

| 配置 | 原准备后私有提交 MiB | 新准备后私有提交 MiB | 原启动阶段峰值提交 MiB | 新启动阶段峰值提交 MiB |
|---|---:|---:|---:|---:|
| FP16 | 942.812 | 774.754 | 1611.230 | 1442.258 |
| 校准W8 | 1047.668 | 880.012 | 1407.137 | 1237.727 |
| 混合50% | 1089.961 | 879.758 | 1592.676 | 1391.023 |
| 逐层37.5% | 1056.832 | 889.203 | 1521.992 | 1352.844 |

Windows GetProcessMemoryInfo的PrivateUsage/PeakPagefileUsage为进程私有提交/截至快照的峰值提交，不是已写入pagefile的字节或NPU独占显存。启动峰值在后续32-token生成之前取样；包含导入、加载、缓存图与NPU会话、prefill及区间外hash验证。未把驱动/固件的所有内存归入此数。

## 正确性与资源

- FrozenSmolIO只依赖Torch与safetensors，不导入Transformers、npunlock或NPU运行时；加载两份CPU权重并生成自有FP16快照，输出矩阵复用embedding，不额外复制输出权重。tokenizer仍由调用者提供。
- 5项新离线测试、10项既有模型/保护策略检查、5项source-export检查通过，共20项。新测试覆盖权重所有权、tied矩阵、checkpoint、dtype/shape/设备/训练模式、非法token ID、非有限值及FP16溢出拒绝。
- 独立构造计数确认混合50%从60次层构造降至30次；所有层原FP16字节、PackedWeight、保护策略与RoPE字节一致。torch_npu_smollm.py、图缓存实现与原生二进制指纹未变，可复用上一轮图缓存。
- 实际checkpoint下4种embedding形状与12种输出头样本（random/zero/最大有限FP16，含batch8/长度64）与官方FP16 Transformers输出逐位一致；不是只比较近似误差。
- 每个精度的相同prefill blob、embedding、hidden、30层K/V/position共91个Tensor、首token logits与argmax跨新旧入口逐位一致；每进程的7次热样本也逐位一致。
- 12次完整32-token生成的每步logits、全部生成token及最终91项状态逐位一致（每种精度：填充、before-1、after-1）。生成使用各自greedy路径，未强制喂参考token。此验证不改变低位模型质量门槛。
- 各进程的prefill图验证原生DPU模式；W8/混合每层包含amode0/wmode2整数权重矩阵任务，FP16不含整数权重任务。热prefill没有新会话或驱动allocation。
- 本轮共 **22,320次真实NPU调用**；所有进程关闭后live/cache/quarantine/session归零，context_failed=False。

## 逐对原始启动数据

| 配置/进程 | 准备 ms | 模型准备后首token ms | 脚本入口到首token ms |
|---|---:|---:|---:|
| fp16-before-1 | 2541.397 | 980.912 | 5707.134 |
| fp16-after-1 | 2269.896 | 1017.766 | 5366.806 |
| fp16-after-2 | 2159.035 | 972.993 | 5189.068 |
| fp16-before-2 | 2570.110 | 980.489 | 5576.585 |
| fp16-before-3 | 2596.855 | 963.593 | 5805.394 |
| fp16-after-3 | 2217.167 | 969.086 | 5325.688 |
| w8-before-1 | 3238.114 | 758.446 | 6117.517 |
| w8-after-1 | 2885.715 | 752.374 | 5702.089 |
| w8-after-2 | 2775.685 | 762.599 | 5589.302 |
| w8-before-2 | 3272.592 | 769.583 | 6169.008 |
| w8-before-3 | 3287.324 | 762.341 | 6248.914 |
| w8-after-3 | 2861.260 | 793.604 | 5732.102 |
| mixed-before-1 | 3174.265 | 827.990 | 5999.404 |
| mixed-after-1 | 2569.994 | 807.510 | 5236.056 |
| mixed-after-2 | 2600.444 | 818.869 | 5280.340 |
| mixed-before-2 | 3191.717 | 809.277 | 5875.613 |
| mixed-before-3 | 3201.658 | 830.489 | 5917.065 |
| mixed-after-3 | 2629.573 | 809.663 | 5290.375 |
| selected-before-1 | 3469.075 | 837.275 | 6515.060 |
| selected-after-1 | 3078.831 | 870.227 | 6127.012 |
| selected-after-2 | 3133.434 | 822.063 | 6032.054 |
| selected-before-2 | 3506.507 | 824.784 | 6489.542 |
| selected-before-3 | 3476.111 | 856.279 | 6544.062 |
| selected-after-3 | 3199.422 | 833.755 | 6297.667 |

## 使用与复现

轻量入口是可复用的本地开发API；需要显式替换完整CPU模型的两端计算。既有完整CPU参考/校准脚本保留其参考模型，不会被自动替换。当前仅支持canonical tied SmolLM、D576、eps1e-5，不支持独立lm_head权重、训练或通用模型导入；默认NPU精度与图缓存开关不变。

```python
from torch_npu_smollm_io import FrozenSmolIO
io = FrozenSmolIO.from_checkpoint(model_directory)
x = io.embed_tokens(cpu_token_ids).to("intel_npu")
hidden, kv = stack.prefill(x)
logits = io.logits(hidden.cpu())[:, -1].float()
```

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:NPUNLOCK_GRAPH_CACHE=$null
.\run_torch_npu.ps1 -Script .\test_torch_npu_smollm_io.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_startup_prepare.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_startup_prepare.py -ScriptArguments '--case','profile'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_startup_prepare.py -ScriptArguments '--case','io-check'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_startup_prepare.py -ScriptArguments '--case','study'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_startup_prepare_report.py
```

本轮保留的profile在入口修改前采集，mixed为60次层构造；当前代码重新profile时会直接构建30层，数据随实现变化。正式study包含独立legacy_stack对照，不依赖删除校验、改变精度或重新编译不同计算图。

## 剩余方向

校准权重校验/解包和导入仍是启动成本；后续可评估不改变codes/scales的高效存储与解包方式，完整性检查保留。稳态NPU性能仍需单独研究图提交与设备任务，不把本轮CPU准备收益套用到tokens/s或TOPS。

## 指纹

| 对象 | SHA256 |
|---|---|
| `torch_npu_smollm.py` | `a6a97f8a2d433eaf85671c98678215aee6f3af93443d9085d4edb486400d676c` |
| `torch_npu_graph_cache.py` | `2cb5adabe7f4e380f3fda108843f2e6f031f135257a93bf87ed279090c81591c` |
| `torch_npu_smollm_io.py` | `11ac8ef7a32ca7fb6a3937c1d69db009f4dd72e04ac8e43eaa746a0b2cf31903` |
| `torch_npu_lowbit.py` | `e9f5b12b09275f73933ef1ab325346776c75a905d48486e3aad5256d08a46f21` |
| `torch_intel_npu.py` | `bd147eb2adbf49d092084ae4f1de767b395424016e9f8862eb850ab571b55cce` |
| `npunlock-probe/research/w8_layer_sensitivity.py` | `22bd14eec19de798a17a34fa1e0bbc6218f6f66883b9a18d916435bd167bcae5` |
| `npunlock-probe/research/smollm_startup_prepare.py` | `7a0a07dc046d41229850a173f4b1849342e3186b297f06eb6cf57b6f9ba76dd2` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `3325f761ea55485271e26ecc3667b820f428d31f665e4c31fb916e8c30aefc8b` |
| `test_torch_npu_smollm_io.py` | `1c83d5861a4649c73d3c75a47b1da13808a828a4d4fb6591f7ce1b2bf6ddfb2e` |
| `npunlock-probe/research/smollm_startup_prepare_report.py` | `4a22ccb95ab1d7b4b2e85965556669790af49908867bc89afaa9e438f40efbe3` |

校准文件与实验清单清理前指纹：

| 材料 | SHA256 |
|---|---|
| `weights.npz` | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| `prepare.json` | `2f1addda2a3dad1290c8621f429b83ed5057bac7421599f6f61005f17d4bc51a` |
| `selection.json` | `649f68afe9b8397c78f94ddea456fdccc753f2927597a1d3c5811bb9e40e9872` |
| `study.json` | `a7e70f5b11f546cd51ad55db138dba6bb43d66d519f1bbf633c16538669180a1` |
| `io-check.json` | `e45684f051d6fb855888dd65817eea296d35c8a5a83ba39cc53fc3212f316318` |
| `construction-check.json` | `16fb48cfd64c7dfe95584a278d1011551a0ec8a1d9746d08cc4a1b65d18c9fae` |

## 交付与清理

验收后按用户约定清理39份临时校准权重、原始计时/验收JSON及冗余适配器副本，共99,073,331字节（94.48MiB）；保留新增CPU入口、测试、研究源码和本报告。
系统临时目录的240份图缓存、1,386,207,776字节（1321.99MiB）已逐精度删除，没有安装新依赖、下载模型、生成图片或保留持久图缓存，未推送GitHub。清理后重算审计需先重新prepare和执行测量。
