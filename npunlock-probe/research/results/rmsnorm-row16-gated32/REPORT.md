# SHAVE RMSNorm：大分片跨 16 行 SIMD 归约

2026-10-08，本机 Lunar Lake NPU 8086:643e，驱动 1005540，编译器 8.3，PyTorch 2.14.1+cpu。

**已接入默认 C/ELF。** SmolLM2-135M 的 30 层 FP16、B8/S32 预填充从 95.88415 ms 降到 88.60715 ms，约快 7.6%；31 轮交错 A/B 中 30 轮胜出。B1/S32、B4/S16 及解码基本持平。该优化由内核自动选择，用户无需为模型手动启用。

## 实现与精度

每个 SHAVE 分片至少 32 行时，16 个 FP32 SIMD 通道分别处理 16 个独立行；每行仍按原有的 576 列顺序累加。`Float16` 表示 16 个 FP32 元素，不是 FP16 算术类型。小分片和尾行继续使用原归约代码，审计验证原循环体在统一换行后完全一致。

平方及累加和非负，因此用 IEEE 位模式的无符号整数比较替代向量浮点比较；保留两个 NaN 检查，使 unordered 比较仍为 false。FP32 Neumaier 补偿、开方、gamma 前后的两次 FP16 舍入保持原语义。编译参数仍为 `-O2 -ffp-contract=off`，没有 fast-math 或精度降级。

门限必须按**分片**计算。本机当前 256 行独立 norm 被分成 8 次 SHAVE 调用，每次 32 行；512 行每次 64 行。模型每层有两次 norm，因此补丁契约中出现更多调用。曾按完整张量设置 256 行门限，导致被测载体没有启用新路径，该失败对照保留在 `../rmsnorm-row16-gated/`。

修改文件是 `npunlock/tests/fixtures/npu4000/rmsnorm.c` 及其标准构建输出 `rmsnorm/rmsnorm-f16.elf`、`.s`、`manifest.json`。原构建器重新生成的 ELF、汇编与实测候选 SHA-256 完全一致；前后版本在 `before/`、`after/`，绑定记录在 `promotion.json`。

- C SHA-256：`73160d5729b7d214b98429c2ba02672ec6bcbdd62a8a2e09fe0aa0d58bde2fa6`
- ELF SHA-256：`7ba8cb06fa4ded1b1654c926f9b05a585ee69ebbb0f2ed5723b9d6a7e212e5b8`
- text：6,080 字节，低于 21,216 字节的载体窗口。

## 同载体独立 norm

同进程 ABBA/BAAB，21 轮，每个版本 42 个暖样本，预分配输出。热计时中显式复制、新建会话和驱动分配均为零。

| 完整张量行数 | 原内核 ms | 最终内核 ms | 耗时下降 | 胜出轮数 / 21 |
|---:|---:|---:|---:|---:|
| 1 | 0.23960 | 0.24140 | -0.8% | 11 |
| 4 | 0.46455 | 0.48690 | -4.8% | 8 |
| 32 | 0.53465 | 0.53660 | -0.4% | 11 |
| 64 | 0.64000 | 0.63925 | 0.1% | 9 |
| 128 | 0.86730 | 0.86840 | -0.1% | 12 |
| 256 | 1.41040 | 1.28580 | 8.8% | 21 |
| 512 | 2.64815 | 2.41615 | 8.8% | 20 |

小尺寸没有稳定收益；4 行的单次进程结果偏慢，胜出轮数不足以支持加速结论，因此额外核查真实 B1/B4 模型。原始样本在 `numeric-gated32.json`。

## 30 层模型

每组均为同模型、同输入、同种子 KV cache 的交错 A/B，31 轮，每版本 62 个暖样本。候选只替换 RMSNorm 代码窗口；decode 重新组合并验证原有 KV Select 补丁，窗口外字节保持一致。原 allocator 128 块 / 16 MiB 设置保留。

| 形状 | 阶段 | 原内核 ms | 最终内核 ms | 胜出轮数 / 31 |
|---|---|---:|---:|---:|
| B1/S32 | prefill | 21.02415 | 20.99540 | 17 |
| B1/S32 | decode | 16.68465 | 16.50410 | 25 |
| B4/S16 | prefill | 32.98900 | 32.89365 | 20 |
| B4/S16 | decode | 17.40110 | 17.39065 | 15 |
| B8/S32 | prefill | 95.88415 | 88.60715 | 30 |
| B8/S32 | decode | 19.48330 | 19.50585 | 14 |

另一个独立进程中的无门限 16 行候选，B8/S32 prefill 为 95.20435→88.29050 ms，31/31 轮胜出。最终门限版本复现了较大批次的收益。不同进程的绝对时间有波动，收益判断依赖各自进程内的成对对照。

**上述计时只包含驻留 NPU 的 30 层 decoder。** 排除首次准备、CPU embedding、最终 norm 和 lm_head，不能换算成完整 LLM 生成速度。没有把 profiling 开关混入普通 A/B。B1、B4、decode 的小差异不作为稳定加速承诺。

三个最终模型测试各验证 182 个输出/KV 数组逐位一致，以及 90 个输入种子 KV 数组保持不变。热路径没有新增显式主机复制、图编译或驱动分配。

## 验证

- 最终候选 158 项数值检查通过，包括不同长度、255/256/257 与 511/512 边界、尾行、真实 embedding、随机舍入压力、零、极小值、大数、Inf/NaN。
- 所有 63,488 种有限 FP16 编码在 **512 行**输入中覆盖新 SIMD 归约路径。早期 111 行输入被分片后可能走原循环，最终测试已扩大；65,536 种编码的 codec 往返也通过。
- 有限值及符号零与旧内核逐位相同，NaN 按 NaN 等价比较，不保证 payload 相同。
- 正式内核 8 组现有回归、74 项检查、1,159 次 NPU 调用通过：单层 norm/numeric/boundary/guards/causality/hybrid，多层 numeric/boundary，包含 FP16/W8A16 和最大支持尺寸。
- 最终离线审计通过：累计 1,046 项数值记录、423 次图证据校验、1,360 次数组对照，以及构建源码、ELF、汇编、模型图补丁、前后快照和回归指纹。计数包含候选及重复实验，不表示唯一测试数。
- 完整验证进程结束后 live/cache/quarantine 字节归零，未出现 context failure。

审计在 `audit.json`，模型原始样本在 `model-b1.json`、`model-b4.json`、`model-b8.json`，正式回归在 `regressions/`。

四行整数比较、先选择大小数、仅平方路径去掉符号恢复等候选均保留原始源码和计时，未采用。汇编仍有栈读取及分散输入读取，尚有优化空间；本次没有测硬件流水线利用率或通用 FP32 GEMM/INT8 TOPS，也没有验证其他 NPU 型号。

## 复现

在项目根目录执行，NPU 命令须串行；同名输出会覆盖。基线 ELF 已冻结，模型 A/B 不会把当前默认内核误当成旧版本。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_row4_bits.py -ScriptArguments '--case','numeric','--lanes','16','--min-rows','32','--label','gated32'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_row4_bits.py -ScriptArguments '--case','model','--lanes','16','--min-rows','32','--batch','8','--prefix','32','--label','model-b8'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_row4_bits.py -ScriptArguments '--case','regression','--lanes','16','--min-rows','32'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_row4_bits.py -ScriptArguments '--case','audit'
```
