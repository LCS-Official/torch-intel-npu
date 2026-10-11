# 全 30 层实际生成剖析与输出准备优化

2026-10-09，Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动 `32.0.100.5540`、编译器 8.3、Torch `2.14.1+cpu`。

## 测量范围

SmolLM2-135M-Instruct，B1，24-token 聊天提示，固定贪心生成 32 个新 token，不提前停止 EOS。
**全部 30 个 decoder layers 在 NPU**；embedding、最终 norm、lm_head 和 argmax 明确使用 CPU FP16。
后续输入由模型自己的输出决定，实际消费的 KV 长度为 55，返回 token 总长度为 56。
这与之前固定驻留 token、只测 decoder 的实验不同；也与历史 4 层 NPU / 26 层 CPU 的混合生成不同。

时间覆盖 embedding、输入上传、NPU decoder、结果下载、CPU norm / 输出头、argmax 与 token 整理。
加载、分词、首次编译和预热不计入；阶段计时采用普通 CPU 时钟，不加载诊断 DLL。
先分别预热，再做 7 轮 ABBA / BAAB 交错，每轮每种策略两个完整生成样本。
所有计时样本的 32 步 logits 和生成 token 与本模式锚点逐位一致。

## 分组执行在完整生成中的收益

此轮使用优化前的输出准备代码。每种策略 14 个完整生成样本，中位数如下。

| 模式 | 逐层总耗时，ms | 四层分组总耗时，ms | 逐层 tokens/s | 分组 tokens/s | 更快轮数 |
|---|---:|---:|---:|---:|---:|
| FP16 | 781.477 | 588.478 | 40.95 | 54.38 | 7 / 7 |
| W8A16 | 689.835 | 546.539 | 46.39 | 58.55 | 7 / 7 |

这里的 tokens/s 是 32 / 完整生成秒数，包含预填充并产生第一个 token。
另算排除第一步的 31 步 decode，四层分组分别约 55.29 / 59.72 tokens/s。
暖首 token 中位数为 FP16 28.01 ms、W8A16 27.47 ms。

## 主要开销

四层分组下，先求每次生成的 31 步平均阶段耗时，再对 14 个样本取中位数：

| 阶段，每步 ms | FP16 | W8A16 |
|---|---:|---:|
| CPU embedding | 0.076 | 0.074 |
| 显式上传 | 0.155 | 0.141 |
| decoder，含 Python 准备与原生执行 | **15.362** | **14.230** |
| 显式下载 | 0.222 | 0.209 |
| CPU 最终 norm | 0.176 | 0.171 |
| CPU lm_head | 1.849 | 1.626 |
| argmax | 0.213 | 0.208 |
| token 整理 | 0.033 | 0.033 |

按上述阶段之和估算，decoder 占约 85%；显式上传 / 下载合计约 2%，CPU lm_head 约 10%。
各阶段分别取中位数，不能与总延迟中位数精确相加。
显式复制计数不覆盖驱动内部 DMA，因此不能据此排除设备内部访存或带宽瓶颈。

另用 30 次驻留 FP16 decode 做 Python 调用剖析：未启用 cProfile 平均 16.87 ms，
启用后平均 19.49 ms，原生 `run_chain` 累计时间折合每次 12.83 ms。
原生时间包含绑定、提交、设备执行和等待；不能当作纯 DPU / SHAVE 计算时间。
Python 热点包括 `_validate`、输出别名检查生成器和输出分配。
剖析期间共 58,560 次 `untyped_storage` 调用；该诊断自身有明显开销，不能用其绝对时间计算正式加速比。

## 已接入的优化

共享运行时 `torch_intel_npu._prepare_many` 的新输出来自独立的 `torch.empty` 分配。
这些存储不会与仍然存活的输入或其他新输出重叠，无需再次逐个比较别名。
生产改动只有两处条件和说明注释：

- 新分配输出继续校验形状、精度、布局、完整分配等要求，省去重复的输入 / 输出及输出 / 输出别名比较。
- 显式传入 `outs` 时继续执行原有全部别名检查；C++ / C 原生绑定校验也保留。

这项改动自动作用于多输出运行时和模型分组计划，无需用户选择开关。
native DLL / PYD、模型计算代码、RMSNorm / KV Select 内核与精度策略未改变。

同一进程内切换冻结的原输出准备函数与当前函数，两种变体均使用默认四层分组。
另外 7 轮 ABBA / BAAB 的完整生成对照结果：

| 模式 | 优化前，ms | 优化后，ms | 前 tokens/s | 后 tokens/s | 延迟减少 | 更快轮数 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 564.025 | 536.938 | 56.74 | **59.60** | 4.8% | 7 / 7 |
| W8A16 | 542.769 | 520.258 | 58.96 | **61.51** | 4.1% | 7 / 7 |

优化后排除首步的 decode 速度约 60.62 / 62.84 tokens/s。
该表与上一轮分组表来自不同运行时段，不能跨表计算累积加速比。
后台视频、桌面负载和频率未固定；未比较 CPU-only / GPU-only 完整生成或其他模型、batch、长序列。

## 精度与未采用的候选

FP16 首步 logits 相对官方 CPU FP16 的 L2 误差为 **0.48%**，通过原有 3% 门槛。
本次 W8A16 使用默认逐行 INT8 量化，没有加载校准后的 `packed_weights`。
其首步误差仍为 **5.69%**，**未通过原有 3% 门槛**；新旧运行时逐位一致，不代表量化质量通过。
两种模式本次均生成以下 32-token 片段，但单条文本相同不能代替 logits 质量检查：

> Once upon a time, there lived a beautiful and gentle bird named Whiskers. Whiskers was a tiny bird with feathers that shone like stars in the

另外试了预填充只对最后一个位置做 CPU norm / lm_head 的候选。
21 轮局部交错计时中，原 FP16 输出头中位数 4.51→2.30 ms，W8A16 4.09→1.72 ms；
但 FP16 logits 最大差异 0.0078125、W8A16 最大差异 0.00390625，未逐位一致，未接入生产路径。
离线诊断确认最终归一化逐位一致，差异发生在 CPU 输出头矩乘改变形状之后。
这只影响预填充输出头的候选时间，不是每个 decode 步骤的加速。

## 检查与证据

原始分组对照：[fp16.json](fp16.json)、[w8a16.json](w8a16.json)。
本次输出准备对照：[FP16](preparation/fp16.json)、[W8A16](preparation/w8a16.json)。
后者保留的样本标签 `1` / `4` 分别表示原 / 新输出准备函数，两者实际都为四层分组；
可由每次生成的 `chain_graphs=930`、`chain_groups=248` 核对，另有 30 个 prefill 图。

[host.json](host.json) 保存 Python / 原生边界剖析。
[audit.json](audit.json) 核对原始样本统计、冻结基线与当前源码的准确差异、
跨轮保存输出的逐位对照、检查点指纹、原生 / 数值内核指纹及各进程最终资源状态。
性能记录中的 `status=passed` 表示执行和新旧一致性检查通过，量化质量门槛单独记录。

回归记录保存在 `regressions/`，复用既有绑定、链式接口、默认策略、模型保护、设备和训练检查。
共 **110 项通过**：绑定 17、链式接口 20、默认选择 8、模型保护 21、设备 33、训练 11。
快速训练包含 FP16 / FP32 的 MLP、卷积、TinyResNet，六项均未逐步重设 CPU 参考；
本轮未重跑更大 `wide_mlp` / `small_resnet` 的扩展训练集。
所有进程退出后上下文正常，活动 / 缓存 / 隔离缓冲区与缓存会话均归零。
所有新输出均压缩保存为 NPZ；未复制大模型编译 blob 或重建 native 工具链。

## 复现

在项目根目录、专用 NPU 环境串行运行。脚本会覆盖对应报告，复测前应保留本目录的冻结证据；
冻结审计对应本轮源码与数据。`before/` 保存优化前运行时和最初生成脚本。

```powershell
# 当前版本的完整生成分组剖析。
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_generation_profile.py -ScriptArguments '--mode','fp16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_generation_profile.py -ScriptArguments '--mode','w8a16'
# 原 / 新输出准备同进程对照，均为四层分组。
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_preparation_probe.py -ScriptArguments '--mode','fp16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_preparation_probe.py -ScriptArguments '--mode','w8a16'
foreach ($taskCase in @('bindings','chain','defaults','guards','device','training')) {
    .\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_generation_regressions.py -ScriptArguments '--case',$taskCase
}
```

只核验本轮保存证据可直接运行 `smollm_generation_audit.py`。
后续优先细分原生执行内部的图绑定 / 提交 / 等待与 DPU / SHAVE 任务耗时；
另需扩展校准和连续 logits 评估，解决 W8A16 的已有质量缺口。
