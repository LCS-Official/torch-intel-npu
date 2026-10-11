# SHAVE RMSNorm 输出转换优化：已接入默认内核

2026-10-07，Core Ultra 7 258V / Lunar Lake NPU40xx，Windows，驱动
32.0.100.5540 / 编译器 8.3，沿用 `.conda/npunlock`。

本轮将 FP32→FP16 四路输出转换的范围判断改为 SIMD 无符号范围判断。
已测候选 `mask` 接入默认 C、ELF、汇编和构建 manifest；现有 SmolLM
模型调用在新进程中自动使用。固定宽度仍为 576，最多 512 行。

256 / 512 行独立 RMSNorm 耗时下降约 18% / 15%；四层模块在两个独立进程中
测得 prefill 耗时下降约 4%～12%。decode 收益较小，部分结果接近波动范围。
本轮未新增 OpenVINO runtime 依赖。

## 实现与边界

改动仅在 `rmsnorm.c` 的 `encode4` 中。对清除符号位后的 FP32 位模式，
原来的下界、上界分别比较改成：

```c
UInt4 valid = (UInt4)((magnitude - 0x38800000u) <= 0x0effe000u)
            | (UInt4)(magnitude == 0);
```

`0x0effe000 = 0x477fe000 - 0x38800000`。无符号减法在低于下界时回绕，
因此同时表达原有上下界条件；零的条件单独保留。
四路均满足条件时沿用原整数舍入公式，否则沿用原来的四路 `round_half`。
FP16 次正规数、溢出、NaN、Inf 和带符号零仍经过原有对应路径。

FP32 补偿归约顺序、平方、平方根、除法、gamma 乘法及两次 FP16 舍入均未改变，
MemRef 检查也未改变。继续使用 `-O2 -ffp-contract=off`。
这次没有优化 FP32 GEMM，也没有增加新的精度类型。
原 FP16 转换实现的 MIT 许可和版权声明完整保留在 `FP16_LICENSE`。

另测 `lanes` 候选：异常输入只回退对应 lane。它增加了慢路径分支，
一组 B4 FP16 decode 还变慢约 0.95%，没有证明优于更简单的 `mask`，因此未采用。
上一轮 O3 候选同样没有采用。

## 性能

采用相同编译载体，仅替换 SHAVE 指令窗口；输入、输出绑定、其他图字节保持相同。
预热后按 ABBA / BAAB 顺序交错计时，每轮每个版本两次。
计时是包含提交、执行、同步和接口返回的主机耗时，输出清理在计时外。
不是裸 ALU 时间或芯片峰值吞吐。

独立 RMSNorm，21 轮、每个版本 42 个样本，单位 ms：

| 行数 × 576 | 旧版本中位数 | 新版本中位数 | 耗时下降 | 新版本胜出轮数 |
|---|---:|---:|---:|---:|
| 1 | 0.26055 | 0.23970 | 8.00% | 13 / 21 |
| 4 | 0.57830 | 0.52515 | 9.19% | 19 / 21 |
| 32 | 0.60825 | 0.54690 | 10.09% | 20 / 21 |
| 256 | 2.01930 | 1.66165 | 17.71% | 21 / 21 |
| 512 | 3.16355 | 2.69030 | 14.96% | 21 / 21 |

四层 SmolLM2-135M 模块 prefill，W8 使用已冻结的激活校准权重。
每组 31 轮、每个版本 62 个样本，旧→新中位数，单位 ms：

| 配置 | 进程 0 | 耗时下降 | 独立确认进程 | 耗时下降 |
|---|---:|---:|---:|---:|
| B1 S32 FP16 | 3.24795 → 2.95015 | 9.17% | 3.53425 → 3.17430 | 10.18% |
| B1 S32 校准 W8A16 | 3.46900 → 3.20690 | 7.56% | 3.14810 → 3.01645 | 4.18% |
| B4 S16 FP16 | 4.89660 → 4.48280 | 8.45% | 5.87700 → 5.17820 | 11.89% |
| B4 S16 校准 W8A16 | 4.89350 → 4.37335 | 10.63% | 5.30055 → 4.70780 | 11.18% |

这八组 prefill 中，新版本分别胜出 28～31 / 31 轮。
同八组 decode 中位耗时下降约 0.23%～5.59%，部分组仅胜出 16～20 / 31 轮，
不能据此承诺稳定的 decode 加速。所有原始样本、逐轮配对及重算结果在
`mask/`、`confirmation/mask/` 和 `audit.json`。

桌面视频、温度和频率未受控，因此只比较同进程交错样本；两个进程的绝对耗时
不宜直接比较。四层 NPU 模块收益也不能直接换算为完整 30 层 LLM 的 tokens/s，
更不能换算为 40 TOPS 的利用率。

## 验证

- 两个候选各通过 104 项数值检查，合计 208 项。包括全部 63,488 种有限
  FP16 编码的转换往返、特殊值、次正规数、零、极值、舍入压力、真实 embedding、
  尾部行数和两种 gamma；所有检查的有限值位差均为零。
- 三组模型实验逐位核对 456 项输出、KV、位置和旧 seed 数组。
  载体代码窗口以外的字节一致，实际嵌入的 ELF 与构建指纹一致。
- 默认内核接入后，8 个原有回归任务共 74 项检查通过，包括边界、非法输入拒绝、
  因果性、单层 / 多层及混合生成。
- 当前内核重新生成独立的校准配置，16 条独立提示和三组各 32 个相同输入 token
  的 decode 全部通过原质量门槛：logits 相对 L2 < 3%、下一 token KL < 0.02。
  校准 W8 最大 L2 分别为 1.9786%、2.9215%，112 个位置的 argmax 均与 CPU 相同。
- 接入前后的权重、校准输入、holdout 和 decode 原始数组共 811 项逐位一致。
  离线重算 320 组质量指标；没有放宽容差。
- 审计覆盖 16,735 次 NPU 调用；各进程结束后 live / cache / quarantine
  缓冲归零，上下文没有执行失败。

旧 `w8-calibration` 记录保持原样，其研究加载器严格绑定当时的内核源码指纹。
新配置与输出单独位于 `calibrated-current/`，没有绕过该验证或修改历史证据。
公共 `packed_weights` API 和模型权重语义保持原样。

## 构建与复现

默认 ELF 已与实测 `mask/codec.elf` 逐字节一致，代码段从 4592 缩至 4528 字节；
SHAVE 载体窗口仍为 21,216 字节。

| 文件 | SHA-256 |
|---|---|
| 旧 RMSNorm C | `986b945072713de3d6ae56a6d35b934e3535d8658ca6a6d7fa414818cbb25e29` |
| 旧 RMSNorm ELF | `93d3c4895adfcaefebdf5e095472e22dad645e14c86dba46cbb65958651a65dd` |
| 新 RMSNorm C | `d8058ab62589510ba5567036d8f79d621c9ea1bba13db196736867a7cb6c35ec` |
| 新 RMSNorm ELF | `a661d8175e599a6ab0e48e8139a75b480216ca7fc9298d041cfe22aba58681b4` |
| 新 RMSNorm 汇编 | `1e1bf9d8993036f71a49cbf35edf4c70b85c159b1f346abb369baca26811bea7` |

原 C / ELF / 汇编 / manifest / 许可保存在 `before/`，当前源码快照位于 `sources/`，
当前构建产物位于 `after/`。runtime、Torch 扩展、模型实现和量化器指纹未变。

在 `E:/Pytorch_xfire_NGPU` 的 PowerShell 中运行已验证的当前内核检查：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_encode_checks.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_encode_checks.py -ScriptArguments '--case','regression'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_encode_checks.py -ScriptArguments '--case','holdout'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_encode_checks.py -ScriptArguments '--case','decode'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_encode_audit.py
```

上述检查会覆盖本轮对应输出，复测前应保留本目录。审计本身只读取证据并更新
`audit.json`。计时用的 `rmsnorm_encode.py` 和 `confirm` 分支保留当时的研究配置；
直接重跑其模型分支会被旧校准源码指纹保护拒绝。重新构建候选还会把当前新内核
当作 baseline，因此应先在独立副本中恢复 `before/`，再复现历史完整计时。

后续优先继续检查 SHAVE 输出转换、访存与调度，以及 decode 的 KV Select；
当前结果支持继续沿热点优化，尚不足以说明 DPU 矩乘或整个模型已接近硬件上限。
