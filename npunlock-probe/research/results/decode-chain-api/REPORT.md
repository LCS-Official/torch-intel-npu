# 同步分组提交：正式接口接入

2026-10-08～09，Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动 `32.0.100.5540`、编译器 8.3、Torch `2.14.1+cpu`。

## 改变了什么

已将上一轮实验整理为正式 C API `graphinfer_session_infer_chain`、C++ `run_chain`，
并接入 `FrozenSmolStack.decode`。模型适配器显式准备并提交计划，移除了研究原型依赖的全局函数替换。
每组 1～4 个图属于同一上下文，图之间使用执行 / 内存屏障，最后同步等待。
所有输入、中间值、输出及会话在调用期间保持引用；每次重新绑定当前地址，完成后清空命令列表并重置 fence。

这一轮属于**运行时调度优化**：减少 Python / C++ 往返、队列提交和逐层同步。
算子内核与数值计算方式保持原样，已有 RMSNorm、KV Select 的源码 / ELF 指纹未改变。
有依赖的层仍按顺序计算，没有把四层变成四路并行。

普通调用自动在 B1 FP16、全层融合 RMSNorm 条件下启用四层分组：

```python
hidden, next_cache = stack.decode(token, cache)
# 显式选择，输出与 KV 仍为独立快照。
hidden, next_cache = stack.decode(token, cache, group_size=4)
# 保留普通逐层执行作对照。
hidden, next_cache = stack.decode(token, cache, group_size=1)
```

`group_size` 允许整数 1、2、4。B4 / B8 和 W8A16 可显式使用分组，但默认仍为逐层路径；
它们本轮只完成七层验证，未据此扩大全部 30 层默认策略。
未融合整层的适配器继续用原路径，显式要求不兼容的分组会在执行前报错。
prefill、forward 和普通 ATen 训练继续使用原有策略。

## 性能

不带诊断计数器的正式 DLL / PYD，同一进程交错比较 `group_size=1` 和 `4`。
每项各预热三次，ABBA / BAAB 交替；七层各 15 轮，30 层 21 轮，每轮每种策略两个样本。
时间包含输出分配、计划构建、绑定、执行、等待、返回和输出释放。

| 模式 / batch / 层数 | 逐层，ms | 四层分组，ms | 中位数耗时减少 | 更快轮数 |
|---|---:|---:|---:|---:|
| FP16 / B1 / 7 | 4.804 | 3.738 | 22.2% | 15 / 15 |
| FP16 / B4 / 7 | 4.850 | 3.794 | 21.8% | 15 / 15 |
| FP16 / B8 / 7 | 5.065 | 4.118 | 18.7% | 15 / 15 |
| W8A16 / B1 / 7 | 4.792 | 3.407 | 28.9% | 15 / 15 |
| W8A16 / B4 / 7 | 5.529 | 3.979 | 28.0% | 15 / 15 |
| W8A16 / B8 / 7 | 4.981 | 3.913 | 21.4% | 15 / 15 |
| FP16 / B1 / 30 | **18.200** | **13.580** | **25.4%** | **21 / 21** |

七层使用前缀 8，30 层使用前缀 32；固定缓存容量 64，驻留输入和权重。
七层 W8 的波动不能用于推断不同 batch 的吞吐排序。
所有计时区间内新增驱动分配、图会话创建和显式主机载荷复制为零；复制计数不覆盖驱动内部 DMA。
计时不包括 CPU embedding、最终 norm、lm_head、采样、首次编译及权重准备。
这是 decoder 延迟对照，不是完整生成速度或芯片算力；桌面背景负载未固定。
与上一轮原型的绝对耗时来自不同运行时段，不作为直接加速比。

## 冷启动、校验与失败处理

公开 C 入口限制每组最多四个不同图，在执行前校验整组的结构大小、上下文、活动状态、
参数覆盖、缓冲区拥有者、精确字节大小与别名关系。后图可以读前图的输出；
后图输出不能覆盖前图输入 / 输出，因而拒绝前向引用和覆盖型依赖。
首次调用或重建命令列表后，冷组自动逐图初始化并完成计算；预热后的组才合并提交。
提交 / 等待失败时沿用上下文隔离，保留可能仍在使用的设备资源，后续访问报错。

`npu.stats()` 新增：

- `chain_groups`：成功完成的原生链式调用组数。
- `chain_graphs`：这些成功组内的图数，亦包含于原有 `npu_calls`。

这两个字段不是底层队列计数。冷组可能顺序提交多个图；预热后，一组对应一次提交与等待。
部分组失败时，之前成功的组可能已经完成；接口不提供回滚。

## 验证

- 原生模拟驱动：23 种情况，含成功、两种冷 / 混合状态、11 种拒绝、9 个失败阶段，以及安全清理后再执行。
- 真实设备接口：20 项检查，FP16 / FP32、七图链尾段、分组 1 / 2 / 4、非法参数与别名、旧输出寿命、四线程各 20 次调用。
- 带主机计数器另跑相同检查；七个图的提交 / 等待分别为 7、4、2 次，追加图始终 7 次。
- 六组七层配置及一组 30 层配置：冷 decode、分组 2 / 4、默认选择与原路径均逐位一致，包含 hidden、各层 K / V 和 position。
- 每组独立推进四步，seed 保持不变；六组七层另外做真实 63-token prefill，decode 至 64，再拒绝第 65 个 token。
- 30 层结果与上一轮保存的旧二进制输出及 seed 共 181 个数组逐位一致。
- 默认分组模型的 20 项拒绝 / inference_mode 检查通过。
- 原有设备快速回归 33 组，以及 FP16 / FP32 梯度、MLP、卷积、TinyResNet 训练快速回归 11 组通过。

现有主机及任务剖析工具也已适配新入口，避免默认分组后漏掉计数 / 每图 profiling query。
原始样本、失败阶段和最终状态可在以下记录中复查；最终资源状态由离线审计统一核对。

## 证据与复现

[models.json](models.json) 保存原始性能样本、逐组正确性和默认策略记录，
`models-outputs.npz` 保存压缩后的最终输出 / KV / position，未复制模型编译大 blob。
[checks.json](checks.json)、[checks-profile.json](checks-profile.json)、[model-guards.json](model-guards.json)
记录接口、计数与拒绝检查；[device.json](device.json)、[training.json](training.json) 记录现有回归。
[audit.json](audit.json) 核对源码 / 二进制指纹、样本统计、保存的数组和各进程结束状态。
`before/` 保留修改前的正式实现和二进制；原生构建脚本自动运行会话及链式故障测试。

```powershell
.\build_npu_session.ps1
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_api.py -ScriptArguments '--case','checks'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_api.py -ScriptArguments '--case','models'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_api.py -ScriptArguments '--case','profile'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_regressions.py -ScriptArguments '--case','device'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_regressions.py -ScriptArguments '--case','training'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_regressions.py -ScriptArguments '--case','guards'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_regressions.py -ScriptArguments '--case','task-build'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_regressions.py -ScriptArguments '--case','task-probe'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_api_audit.py
```

以上硬件命令串行执行。构建正式扩展需要本地 MSVC 工具链；运行未增加 OpenVINO runtime 依赖。

## 范围

这是冻结 SmolLM 适配器和共享执行层的正式优化，没有实现通用 Torch 图捕获、异步 streams、
训练图分组、GPU / NPU 联合执行或全模型自动适配。
后续可先扩大 30 层的 batch / W8 覆盖，再决定哪些配置默认启用；然后继续用任务剖析寻找设备侧热点。
