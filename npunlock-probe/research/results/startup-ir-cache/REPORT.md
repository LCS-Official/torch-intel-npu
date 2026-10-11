# NPU 冷首 token：缓存命中跳过 IR 重建

2026-10-11，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动32.0.100.5540，编译器8.3，Torch2.14.1+cpu。

已接入 FP16 整层 prefill 的提前缓存命中。相同30份编译blob、新进程交替对照，冷首token中位 **2.009→0.967秒**，降低 **51.8%**。
Python脚本入口到首token中位 **6.795→5.491秒**，降低 **19.2%**。

## 测量定义

- SmolLM2-135M-Instruct，FP16，完整30层NPU decoder，batch1、24-token固定提示词。embedding、最终norm、lm_head与argmax在CPU。
- 冷首token从模型准备完成后的embedding到首token可用，包含上传、首次图准备、NPU执行、下载和CPU完整prefix输出头。
- 脚本入口计时另外包含Torch/Transformers导入、NPU初始化和模型/tokenizer准备；不含PowerShell及Python解释器启动。未把图准备挪到计时外。
- 每次均为新进程；随后预热2次、记录7次热请求。未清空Windows文件系统页缓存，未锁定桌面后台负载或频率。
- 主对照顺序为full1、fast1、full2、fast2、full3、fast3。所有主对照源码及hook一致；full模式透明包装graph入口以关闭提前命中，仍执行原来的完整建图路径。
- 对照两目录使用相同blob/patch reports，逐项SHA验证。prepare_ir_comparison.py只重新关联缓存键，未修改blob或绕过运行时验证。

## 主对照

| 模式/进程 | 冷首token ms | 脚本入口到首token ms | 热首token中位 ms | 缓存命中/编译/跳过IR |
|---|---:|---:|---:|---|
| full-1 | 1868.492 | 6754.792 | 29.239 | 30/0/0 |
| fast-1 | 961.182 | 5478.542 | 34.361 | 30/0/30 |
| full-2 | 2040.430 | 6794.627 | 32.849 | 30/0/0 |
| fast-2 | 967.466 | 5490.508 | 26.455 | 30/0/30 |
| full-3 | 2008.652 | 6839.959 | 39.658 | 30/0/0 |
| fast-3 | 974.311 | 5947.831 | 30.471 | 30/0/30 |

热请求仍存在桌面负载波动，本次不宣称热吞吐提升。NPU计算blob与全部调用数量不变。

## 实现与安全边界

- 仅对未替换graph/serializer/compiler的FrozenSmolLayer、FP16、整层fused prefill提前查缓存。decode、W8A16、诊断分段和研究替换图沿用完整建图路径。
- 键包含七组projection的实际FP16字节与形状、RoPE字节、RMSNorm内核、输入输出契约，以及模型图/serializer源码、缓存实现、NPU4000编译实现、驱动、设备与编译器profile。
- 两组gamma是运行时输入，不嵌入blob；不同gamma可以复用同一图，仍分别上传和绑定。
- 命中后用固定输入输出名称、dtype和shape构造轻量绑定元数据；它不是模型计算图，不会送入编译器。实际计算仍由原blob执行。
- 未命中或文件损坏时延迟构造完整图，核对实际I/O契约后编译。原有ZIP成员/体积、profile、SHA、patch报告及ELF/DPU检查全部保留；native session失败不会盲目重试。
- 磁盘缓存仍需NPUNLOCK_GRAPH_CACHE显式启用，默认关闭。每个缓存目录上限512MiB、预留256MiB；不自动删除旧条目。新键与旧键不同，旧缓存需要重建，旧条目占用仍计入上限。
- 30图缓存大小 216.894MiB；I/O契约比旧存储格式仅增加 11.367KiB。

## 正确性与回归

- 共18个完整30层新进程通过。所有hidden/logits跨进程逐字节相同，首token相同（ID 6403）；每进程热请求也逐位一致。
- 与CPU FP16参考首tokenlogits相对L2误差 0.484002856%，argmax一致，原3%质量门槛保留。
- 每次冷请求均90次NPU调用、31次session创建、184次driver allocation、62次显式上传和1次下载；不包含驱动内部DMA计数。
- 19项既有缓存检查、19项提前缓存检查、21项既有输入/训练/存储保护检查通过。
- 单层硬件检查覆盖(1,1,576)、(1,24,576)、(4,1,576)、(8,64,576)，prefill、允许长度的后继decode及原KV不可变；新进程10图全部命中，0编译，4图跳过IR。
- 新进程额外验证两组gamma同时改变时，新旧路径hidden/KV逐位一致，结果随gamma改变，新增session为0。共8项硬件对照通过。
- 故意改坏1项I/O契约：29命中、1拒绝、1编译、1写回、29跳过IR，冷首token1854.820ms；输出仍逐位相同。
- 所有硬件进程退出时live/cached/quarantined buffer与session归零，context_failed=false。

## 补充样本与限制

| 样本 | 冷首token ms | 脚本入口到首token ms |
|---|---:|---:|
| baseline-fill | 26444.151 | 31177.289 |
| baseline-1 | 2249.361 | 6897.203 |
| baseline-2 | 1841.841 | 6483.355 |
| baseline-3 | 1841.649 | 6366.188 |
| base-1 | 1876.541 | 7163.657 |
| base-2 | 2784.154 | 20730.470 |
| base-3 | 2346.993 | 11551.429 |
| candidate-fill | 28561.250 | 33555.552 |
| candidate-1 | 1470.815 | 7552.073 |
| candidate-2 | 954.128 | 5604.088 |
| candidate-3 | 968.264 | 5628.866 |
| candidate-corrupt | 1854.820 | 6639.412 |

base-2/3存在明显的导入/上传/其它阶段波动，未将其混入主交替对照。没有验证波动原因。
candidate-fill仍需首次完整编译，约28.56秒；本轮改善的是已有磁盘缓存的新进程请求，不能把该收益用于无缓存启动。
没有重测完整32-token生成、其它LLM、GPU协同或训练；未新增OpenVINO依赖。

## 剩余开销与下一步

初始完整图构造/序列化30层共785.827ms；主对照fast的轻量identity约77ms。还避免了为旧键拼接和散列大IR权重缓冲。
各hook为包含式时间，存在嵌套；下面不能相加为总耗时。

| fast路径hook | 三进程中位 ms |
|---|---:|
| prefill_identity | 77.204 |
| cache_read | 174.469 |
| inspect_blob | 294.775 |
| session_create | 109.756 |
| norm_parameter | 99.521 |
| layer_run_many | 178.585 |

下一步优先研究ELF/DPU证据检查中的重复解析，目标是在保留相同验证的条件下减少约0.29秒检查开销。
若目标改成整个程序启动时间，则Torch/Transformers导入及CPU模型/适配器重复准备也需要单独测量；这部分目前仍约4.5秒。

## 复现与保留

运行入口为run_torch_npu.ps1 -Script npunlock-probe/research/startup_ir_cache.py -ScriptArguments --label,<名称>；启用的缓存目录通过NPUNLOCK_GRAPH_CACHE传入。
prefill_cache_checks.py和graph_cache_checks.py不调用硬件；prefill_cache_hardware.py先--phase fill再用新进程--phase load。
主A/B需要先填充candidate-cache，再运行prepare_ir_comparison.py；full模式传--full-ir并使用paired-full-ir-cache，fast模式使用candidate-cache。修改前baseline-cache存在时会额外验证旧blob一致性。重新复算全部历史样本需按before源码分别生成，不能直接读取已清理的数据。
按用户约定，验收后删除本轮NPC/NPZ/原始JSON/日志，仅保留源码、before文字源码和本报告。清空缓存后，下次首次运行需要重新编译；实际应用要跨进程保留缓存才能获得命中收益。

## 验收源码指纹

清理记录：已删除143个临时文件，760,057,299字节（724.85MiB），无失败项。清理后原生文件校验、严格FP32算术、FP16矩乘、一层prefill/decode与资源回收检查通过，共14次NPU调用。

| 文件 | SHA256 |
|---|---|
| torch_npu_smollm.py | a9ce3949141ebd3fae258a73578d9f13ed7bd0da4e0d3809a97ca0c46296c869 |
| torch_npu_graph_cache.py | 62e2932b35597b703e56c760297eb49e474bf41fbe13a7c8f93072e82d397ce6 |
| npunlock\python\npunlock\npu4000.py | bacdcc0d857827c10de0009cdb60bb4cceef976ce3d8ce8133e57364b70649cd |
| npunlock-probe\research\startup_ir_cache.py | 759490ff87280e7ef829a43221771eb2f7cd6eabd612cb2c6bd92058464ed90a |
| npunlock-probe\research\prefill_cache_checks.py | eb1212a56fd36a55424b713a0d9b8a37bcf44359f234e507fc79b6c863432c10 |
| npunlock-probe\research\prefill_cache_hardware.py | 643e094402c2cc9919095d14da8f921790c3ca47a76d9f46b775685c98c4a079 |
| npunlock-probe\research\prepare_ir_comparison.py | 8018c48ef24b28ca3eafdcf6a12a8c0cf212ad5be204ddbf96dac99c0c1e30a2 |
| npunlock-probe\research\corrupt_prefill_entry.py | dd1f06bb0c0af83ff24b09fb8231062d1c7dbd7d2579c6bdb21fc384e6498e6a |
