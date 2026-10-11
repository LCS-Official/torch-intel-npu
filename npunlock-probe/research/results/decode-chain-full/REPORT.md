# 完整 30 层验证与默认分组范围扩展

2026-10-09，Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动 `32.0.100.5540`、编译器 8.3、Torch `2.14.1+cpu`。

## 结果与实现

SmolLM2-135M 全部 30 个 decoder layers 的 FP16 / W8A16、batch 1 / 4 / 8 六组测试通过。
同进程配对中，四层分组使 decoder 延迟中位数减少 **18.2%～22.2%**，每组 21 / 21 轮更快。
据此将 `FrozenSmolStack.decode` 的默认四层分组从 B1 FP16 扩展到这六种配置。

生产模型源码仅改变默认条件：多于一层、batch 为 1 / 4 / 8，且每层均支持整层融合 RMSNorm。
构造器只允许 FP16 / W8A16 两种模式。其他 batch、单层或未融合配置继续逐层执行。
显式 `group_size=1` 保留原路径，`2` / `4` 仍可显式选择；不支持融合的配置显式要求分组会报错。

```python
# 兼容配置自动每四层合并提交。
hidden, next_cache = stack.decode(token, cache)
# 可随时用原路径对照。
hidden, next_cache = stack.decode(token, cache, group_size=1)
```

本轮属于运行时调度优化，复用既有正式链式接口。有依赖的层仍按顺序计算，预热后
30 层的提交 / 等待由 30 次降为 8 次；不是多层同时计算。
原生 DLL / PYD、C / C++ 源码、RMSNorm 和 KV Select 的源码 / ELF 指纹均未变化。
prefill、普通 forward 与通用 ATen 训练策略保持原实现。
研究脚本同时更新了默认策略断言和主机剖析的预期提交计数。

## 性能

六组均为完整 30 层、前缀 32、固定 KV 容量 64。驻留输入与权重，正式 DLL / PYD，无诊断计数器。
每种策略先预热三次；21 轮 ABBA / BAAB 交替，每轮每种策略两个样本，共 84 个样本。
下表为调用加输出释放的样本中位数；胜出轮数按每轮两次调用的均值比较。

| 模式 | batch | 逐层，ms | 四层分组，ms | 耗时减少 | 更快轮数 |
|---|---:|---:|---:|---:|---:|
| FP16 | 1 | 16.488 | 12.869 | 21.9% | 21 / 21 |
| FP16 | 4 | 17.118 | 13.669 | 20.2% | 21 / 21 |
| FP16 | 8 | 18.987 | 15.455 | 18.6% | 21 / 21 |
| W8A16 | 1 | 15.691 | 12.206 | 22.2% | 21 / 21 |
| W8A16 | 4 | 16.915 | 13.446 | 20.5% | 21 / 21 |
| W8A16 | 8 | 18.548 | 15.164 | 18.2% | 21 / 21 |

时间包括输出分配、计划构建、绑定、执行、等待、返回和输出释放。
计时区间内新增驱动分配、图会话创建、显式主机上传 / 下载 / 设备复制 / 打包均为零；
复制计数不覆盖驱动内部 DMA，不能据此认定完全没有数据搬运。
计时不包含首次编译、权重准备、CPU embedding、最终 norm、lm_head 和采样。
这是驻留 decoder 延迟，**不是完整生成 tokens/s、TOPS 或 TFLOPS**。
桌面背景负载和频率未固定；不将不同运行时段的绝对耗时直接作为加速比。

## 正确性与回退

- 每组首次 decode 直接走冷分组，再与逐层路径比较；输出、30 层 K / V / position 共 91 个数组逐位一致。
- 显式分组 2 / 4 以及修改前的默认路径均与逐层结果一致。B1 FP16 另与历史旧二进制的输出及 seed 共 181 个数组逐位对照。
- 每组独立运行逐层与分组路径各 32 步，从位置 32 推进到 64，每步的 91 个数组逐位一致；原始 seed 的 90 个数组保持不变。
- 每组另做真实 63-token prefill → decode 到 64 → 拒绝第 65 个 token；拒绝前后执行与复制等计数均未增加。
- 修改默认策略后，使用四层和随机独立 batch 输入复查六种默认分组配置，另测 B2 默认逐层及未融合默认逐层，共 8 项通过。
- 模型保护回归共 21 项通过，覆盖非法分组、缓存类型 / 顺序 / 快照 / 位置、满缓存、CPU / FP32 输入、训练 / autograd 拒绝、inference_mode，以及单层默认逐层且与显式原路径逐位一致。
- 六组完整模型、默认策略验证及保护回归进程退出后，上下文均正常，活动 / 缓存 / 隔离缓冲区及缓存会话均归零。

连续步骤使用固定驻留 token，属于缓存与调度一致性测试，不是语言模型回答质量或完整生成评测。
W8A16 沿用原有量化权重与计算方式；上述逐位对照不等于量化模型与 FP16 模型等价。
未扩展模型、训练、序列长度上限或精度支持范围。

## 证据与复现

每项独立进程串行运行，避免累计图会话影响实验。原始记录：
[FP16 B1](fp16-b1/models.json)、[FP16 B4](fp16-b4/models.json)、[FP16 B8](fp16-b8/models.json)、
[W8A16 B1](w8a16-b1/models.json)、[W8A16 B4](w8a16-b4/models.json)、[W8A16 B8](w8a16-b8/models.json)。
[defaults.json](defaults.json) 保存实际默认计数与回退结果；[model-guards.json](model-guards.json) 保存 21 项保护检查。

[audit.json](audit.json) 复核六组报告与源码 / 二进制指纹、原始样本统计、资源结束状态和
**637 对保存数组**：六组最后一步各 91 对，加七组默认选择各 13 对。
`before/` 保存此次实验开始时的实现、二进制和测试脚本；审计确认模型源码仅有默认条件变化。
`models-outputs.npz` 与 `default-outputs.npz` 使用压缩保存，未额外复制编译后的大模型 blob。

在项目根目录运行，六项 benchmark 必须串行，且应先于默认策略修改：

```powershell
foreach ($taskMode in @('fp16','w8a16')) {
    foreach ($taskBatch in @(1,4,8)) {
        .\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_full.py -ScriptArguments '--case','benchmark','--mode',$taskMode,'--batch',([string]$taskBatch)
    }
}
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_full.py -ScriptArguments '--case','defaults'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_full_guards.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_full.py -ScriptArguments '--case','audit'
```

源码基线冻结审计：恢复 `before/torch_npu_smollm.py` 和 `before/npunlock-probe/research/decode_chain_api.py`
后执行 benchmark，再应用当前默认条件和断言后执行 defaults / guards / audit。
仅测试当前版本可直接运行 defaults / guards；不要用更新后的源码覆盖本目录的冻结 benchmark 证据。
