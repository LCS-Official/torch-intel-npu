# 单行 RMSNorm 消融与精确 SIMD 半精度舍入

2026-10-09，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动 32.0.100.5540，编译器 8.3。

本轮已将精确 SIMD 慢分支接入默认 RMSNorm 源码 / ELF。
FP32 Neumaier 归约顺序、整数平方根、FP16 中间舍入后再乘 gamma 的语义不变；
普通输入的编码分支逻辑不变，subnormal / overflow / NaN 等范围改用 SIMD 整数处理。
编译器将新 encode4 内联，输出循环中每四元素的两次辅助函数调用消失。
修改自动作用于已有 SmolLM RMSNorm 路径，无需模型用户设置新参数，未新增依赖。

## 单行消融定位

使用当前生产代码生成 9 个消融内核，加原始完整内核，共 10 个版本。
保留同一 carrier 的布局 / 调度 / 缓冲区，只替换代码窗口。
所有 MemRef 合同均为完整一行，`elements=576`；没有跨调用读取相邻行。
普通图每种输入交错 31 轮、每内核 62 个样本，预分配输出，暖路径驱动分配为零。
另建 PERF_COUNT 图，每种输入 / 内核重建 query 后采样 3 次，共 90 份独立任务记录。
60 项普通 / profiling 输出参考检查通过，两种图输出逐位一致。

以下为单图 SHAVE 任务区间并集的中位数，单位 µs：

| 消融内容 | 随机输入 | 真实 embedding | subnormal 压力输入 |
|---|---:|---:|---:|
| 完整 RMSNorm | 71.61 | 60.83 | 100.94 |
| 填零 | 6.77 | 5.94 | 4.90 |
| 原始 FP16 位复制 | 7.86 | 6.82 | 5.52 |
| FP16 解码 / 编码往返 | 27.19 | 24.89 | 52.55 |
| 解码 / 平方 / 编码 | 29.79 | 30.68 | 44.95 |
| 解码 / gamma 乘法 / 编码 | 34.90 | 35.94 | 51.30 |
| 两次舍入的输出处理，inverse=1 | 54.69 | 52.81 | 98.44 |
| 原 FP32 归约 / variance 输出 | 30.94 | 26.77 | 21.82 |
| 原 FP32 归约 / rsqrt 输出 | 31.35 | 26.67 | 21.46 |

输出处理消融使用动态构造的 inverse=1；其输出幅值和慢分支频率与实际归一化不同。
各版本会改变编译器调度、内联和活跃寄存器，不能将这些时间相加或相减，声称得到精确阶段占比。
这些结果用于定位转换 / 舍入 / 调用开销，随后以完整模型配对计时验证优化。
普通图墙钟与 profiling 设备时间分别记录，不能混为一组分解。
原始数据：[ablation.json](ablation.json)，包含额外 sqrt 消融、全部普通图样本与任务记录。

## 精度与边界验证

新 SIMD 慢分支覆盖 FP32→FP16 的 subnormal、舍入中点、溢出、±0、Inf 和 NaN。
移位仅使用 14..24；非激活 lane 先钳制移位量，避免越界移位。

直接在 NPU 执行新旧编码内核，共检查 **319,494 个 FP32 值 / 内核**：
所有有限正负半精度值、相邻半精度值的中点及两侧 FP32 邻点、
65520 附近溢出边界、随机位模式，以及特殊值。
有限值与 signed zero 对 CPU 参考逐位一致，NaN 对参考按 NaN-equivalent 比较。
新旧 NPU 转换之间则全部逐位一致，包括测试中 258 个 NaN 的符号与 payload。
原始输入、参考与新旧输出保存在 [encoding-outputs.npz](half-slow/encoding-outputs.npz)。

另外 **158 项 RMSNorm 数值检查**通过：随机值、全部有限 half 编码、
全部 65,536 个 half 编码往返、舍入压力、特殊值与真实 embedding，覆盖 1..512 行。
往返使用本轮新编码实现单独构建的 ELF，并未复用旧往返内核。
原始检查：[encodings.json](half-slow/encodings.json)、[numeric-half-slow.json](half-slow/numeric-half-slow.json)。

## 完整 30 层 decoder 配对对照

同一个模型 / 编译 carrier 只换 RMSNorm 代码窗口，KV Select 使用原生产内核。
每组 31 轮，轮内正反交错执行，每版本 62 个样本；不启用设备任务 profiling。
计时只含驻留 NPU decoder，不含 CPU embedding / 最终 norm / lm_head 和边界传输。
decode 均使用原四层默认分组。

| batch / prefill 长度 | prefill 原→新，ms | 更快轮数 | decode 原→新，ms | 更快轮数 |
|---|---:|---:|---:|---:|
| 1 / 24 | 17.786→16.659 | 29 / 31 | 12.557→12.181 | 27 / 31 |
| 4 / 24 | 43.386→37.791 | 31 / 31 | 13.466→13.151 | 21 / 31 |
| 8 / 63 | 174.564→157.386 | 31 / 31 | 15.287→15.051 | 21 / 31 |

三组各 182 个输出 / KV / position 数组逐位一致，90 个原 seed 数组不变。
batch 8 / 63 覆盖多行 SIMD 路径及第 64 个 token；暖计时未新增驱动分配或显式主机复制。
prefill 延迟下降约 6.3%～12.9%，decode 约 1.5%～3.0%，较大 batch 的 decode 收益较小。
原始数据：[B1](half-slow/full30.json)、[B4](half-slow/model-b4-s24.json)、[B8 长前缀](half-slow/model-b8-s63.json)。

## 真实 32-token 生成

30 层全部在 NPU，CPU 保留 embedding / 最终 norm / lm_head / argmax；B1、24-token 提示，
固定生成 32 token、不提前停止 EOS。预热后 7 轮 ABBA 对照，每版本 14 次完整生成。
计时包含 CPU 输出头、输入上传和输出下载。
保存样本中的标签 `1` / `4` 分别表示旧 / 新内核，两者实际均使用四层分组；
每次生成有 960 次图执行，其中 decode 的 chain_graphs=930、chain_groups=248。

| 精度 | 整体 tokens/s，原→新 | 相对吞吐增加 | 更快轮数 | 首 token，原→新 ms |
|---|---:|---:|---:|---:|
| FP16 | 60.99→64.43 | 约 5.6% | 7 / 7 | 23.92→24.21 |
| W8A16 | 66.85→70.69 | 约 5.7% | 7 / 7 | 24.47→22.26 |

全部 32 步 logits / token 逐位一致；FP16 首 token 的本轮中位数略增，不能说所有阶段都提速。
CPU 阶段也有波动；整次吞吐是实际配对观测，不是纯 RMSNorm 算力或峰值 TOPS。
未锁定桌面后台负载 / 频率，不能与以前不同运行时段的 tokens/s 计算累积增益。
原始数据：[FP16](half-slow/generation/fp16.json)、[W8A16](half-slow/generation/w8a16.json)。

首次生成验证因研究 wrapper 持有模型强引用而未通过资源释放检查，
已改成弱引用并完整重跑；上表只使用资源检查通过的重跑结果。
失败记录保留为 [reference-failure.log](half-slow/generation/reference-failure.log)。
这个问题位于研究 wrapper，未修改生产 runtime 的生命周期逻辑。

## 正式接入与审计

更新的生产文件只有 RMSNorm C、ELF、汇编和构建清单。
标准构建得到的 ELF / 汇编与已测候选指纹完全相同；文本 6080→7120 字节，仍在 21216 字节代码窗口内。
FP32 归约、sqrt32、模型图、native runtime、Torch 扩展和通用精度转换路径均未改动。
默认内核 ELF SHA256：`6c75489f4be7ba9e452c670a6b1fed994b748a1891970ef012a82b5bf6a93377`。
源文件 / 构建清单与前后备份见 [promotion.json](half-slow/promotion.json)。

正式内核复用已有归一化、数值、边界、保护、因果性、混合执行与堆叠模型检查，
8 个回归任务共 **74 项通过**。所有成功验证进程的上下文正常，活动 / 缓存 / 隔离缓冲区及会话归零。
[audit.json](audit.json) 离线核验原始任务、转换输出、模型 / 生成样本、生产指纹与全部回归记录。
未复制整套模型编译 blob，模型输出压缩保存；没有新增 Python 包或 OpenVINO runtime 依赖。

本轮未重跑通用训练全集；本项改动用于固定宽度 576 的 SmolLM RMSNorm。
W8A16 已有首步 logits 相对 CPU FP16 的 L2 误差仍为 **5.69%**，超过 3% 门槛；
FP16 同项为 0.48%。新旧逐位一致和性能通过不能替代量化质量通过。
下一项可继续优化 KV 更新 / 掩码小任务，或细查单行归约流水线；需继续保留完整模型收益与精度门槛。

## 复现

本目录是冻结证据。正式内核已更新，直接重跑构建会改变基线；
重新对照前请保留本目录，并使用 `half-slow/before/` 中的原版本建立独立实验。
仅离线核验当前保存证据：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_single_audit.py
```

研究入口依次为 `rmsnorm_single_row.py --case build / ablate`、
`rmsnorm_half_slow.py --case build / encodings / numeric / model`、
`rmsnorm_half_validate.py --case generation --mode fp16 / w8a16`，
以及 `--case model --batch 4`、`--case model --batch 8 --prefix 63`。
首次接入使用 `rmsnorm_half_promote.py`，其保护条件阻止重复覆盖生产备份；全部硬件任务串行运行。
