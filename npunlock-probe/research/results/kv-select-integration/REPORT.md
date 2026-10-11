# KV Select 默认接入与回归

2026-10-07，Intel Lunar Lake `8086:643e`，驱动编译器 8.3。

已将上一轮验证的 SHAVE FP16 KV 行复制 / 遮罩 Select 接入 `FrozenSmolLayer` 和
`FrozenSmolStack` 的默认 fused decode 路径。在新 Python 进程中，现有 `prefill` / `decode`
调用自动使用该内核；输入仍是 FP16 `intel_npu` hidden states，调用参数保持一致。

正式 fixture 为 `npunlock/tests/fixtures/npu4000/kv_select.c` 及 `kv_select/` 中的 ELF、汇编和 manifest。
重新构建的 ELF 与上一轮候选逐字节相同，text 为 1,984 字节。
运行时加载正式 fixture，并校验源码及 ELF 指纹；不从研究目录加载 candidate 模块。

**实现**

`npunlock.npu4000.patch_kv_select` 复用现有代码窗口和重定位验证，在共享检查中增加四个 MemRef
及 KV / mask 广播布局约束。它检查设备/编译器、Select 代码指纹、完整行、CMX 范围、FP16 dtype、
16-byte 对齐、非别名输出和实际 C 内核要求的 stride。检查使用显式异常，在 Python `-O` 下仍然有效。

`compile_fused_rmsnorm(..., select_kernel_elf=...)` 顺序应用 RMSNorm 和 Select 补丁，分别保留两份记录。
不传这个可选参数的调用继续使用原有流程。模型的全局 graph cache digest 纳入 Select ELF，
且模型证据新增 `kv_select` 字段，使新内核与旧 session 明确区分。
Prefill 沿用原有缓存键及编译路径。现有 DPU、DMA、barrier 和 task 元数据均保留。

设备 profiling 审计原先假定每个图只有一份 RMSNorm 补丁；本轮增加对两份补丁的衔接验证，
检查中间/最终 SHA、实际代码字节、各自 manifest、设备版本及代码区间不重叠。
该检查同时覆盖单补丁和双补丁的 152 张保存图，并拒绝破坏补丁链或代码字节的记录。

适用范围是已验证的 SmolLM2-135M 配置、B1～B8、最多 64 个缓存位置。
KV / mask 选择只做原始位复制，不引入浮点舍入或精度转换。
沿用现有 Intel NPU 驱动编译器和运行库，无新增 Python runtime 依赖。

**默认路径回归**

| 检查 | 覆盖 | 结果 |
| --- | --- | --- |
| Batch 与 KV 边界 | B1～B8，FP16 / 校准 W8A16，prefix=1、7、32、63；prefill、decode、推进及旧 seed 重用 | 176 项检查通过 |
| 全部 decoder layers | 30 层，FP16 / 原始 W8A16，32-token prefill + 32 步连续 decode 至缓存长度 64 | 每模式 33 项检查通过 |
| 既有模型回归 | norm、numeric、boundary、guards、causality、hybrid、stack numeric / boundary | 74 项通过 |
| Select 布局保护 | 16 张历史候选图逐字节复建，以及精度、别名、对齐、stride、版本、指纹等拒绝检查 | 普通及 `-O` 各 35 项通过 |
| 既有 npu4000 离线测试 | FP16 / FP32 carrier、GEMM、传输、归约及非法参数 | 16 项通过 |

新旧默认路径的 hidden states、每层 K/V 和 position 共 6,710 项数组逐位一致，全部有限。
全 30 层测试的最终 logits 另有 66 项逐位比较通过。
76 个新图可从各自旧图只替换 Select 代码字节精确重建；图 IR 也逐字节一致。
所有 warm 检查都保持每层一次 graph 提交，无新增编译或显式主机载荷复制。

全 30 层测试中，embedding、final norm 和 LM head 使用显式 CPU 参考，30 个 decoder layers 全部在 NPU 执行。
连续输入采用相同的 CPU greedy continuation，以便比较相同位置和输入；本轮没有采集完整模型吞吐。

| 全 30 层 NPU 对 CPU FP16 参考 | 比较位置 | 最大 logits 相对 L2 | next-token 一致 |
| --- | ---: | ---: | ---: |
| FP16 | 33 | 0.00563739（约 0.56%） | 33/33 |
| W8A16，未做全层激活校准 | 33 | 0.10343292（约 10.34%） | 33/33 |

W8 的数值偏差在接入前后完全相同。本轮全层测试验证执行和接入一致性；
该 W8 结果尚不满足现有 3% logits 相对 L2 质量门槛。
下一阶段应把激活校准扩展到全部 30 层，并在独立提示上验证。
本次已校准的权重 map 仍覆盖前四层，不应将它当作全 30 层校准 map。

**校准及独立质量回归**

在 `calibrated-current/` 重新生成了绑定当前源码的四层 GPTQ 校准 profile。
保留原有严格源码指纹校验，并在新 manifest 中额外绑定 `npu4000.py`、Select C / ELF / manifest。
旧实验目录及其权重、输出和 JSON 保持原样。

16 条独立提示和三条各 32 步的 continuation 均通过原有校准 W8 质量门槛：
logits 相对 L2 < 0.03，next-token KL < 0.02。局部执行门槛继续为相对 L2 < 0.003、最大缩放误差 < 0.01。
这些测试的前四层使用 NPU，剩余 decoder layers 使用显式 CPU 参考，和历史测试的方法一致。

| 校准 W8 质量 | 位置数 | 最大 logits 相对 L2 | 最大 KL | next-token 一致 |
| --- | ---: | ---: | ---: | ---: |
| 独立提示 | 16 | 0.01978616 | 0.00041478 | 16/16 |
| 连续 decode | 96 | 0.02921538 | 0.00151911 | 96/96 |

与上一阶段相比，校准权重、输入、holdout 和 decode 输出共 811 项数组逐位相同。
校准权重文件 SHA256 仍为 `68381e22371c1c0c84b936731569b2c486b99ceebd43aad2c6033ecaae9300df`。
因此，本轮接入没有改变既有模型精度或放宽质量门槛。

**审计与性能范围**

最终离线审计通过：8,043 次成功 NPU 调用，152 张保存的 decode 图，
6,710 项默认路径数组、66 项全层 logits、811 项历史质量数组核对，320 项质量指标复算。
全部运行最终 context 健康，live / cached / quarantined buffer bytes 均为零。
31 个当前源码/运行库快照保存在 `sources/`，接入前源码位于 `before/`，详情见 `audit.json` 和 `snapshots.json`。

上一轮同一 ELF 的四层交替计时测得约 3%～8% decode 中位耗时下降，设备端 Select 累计时间下降约 46%，
见 `../kv-group/REPORT.md`。本轮主要验证默认接入与数值回归；没有把四层的百分比推算为 30 层模型收益。

| 指纹 | SHA256 |
| --- | --- |
| Select C | `a268f1dc64e3ddbfef7eb91316538f674d94d024fcf537aa74d55a85004a544a` |
| Select ELF | `8988946b9a6a2ac014f99143f30cbf37c2d0e76eaa8e6457887e0f8a7a8b4807` |
| torch_npu_smollm.py | `850578dc79b3b424689c4e85c7d328457b81a1e45dcc2c0ba63257b215b4a50f` |
| npu4000.py | `bacdcc0d857827c10de0009cdb60bb4cceef976ce3d8ce8133e57364b70649cd` |

**复现**

仓库根目录、现有 `.conda/npunlock` 环境。硬件测试串行；这些命令会写入本阶段的固定结果目录，
保留该目录后再重跑可以避免覆盖历史记录。上一轮候选脚本的配套源码位于其自己的 `sources/` 快照。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\build_kv_select.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_guards.py
.\run_torch_npu.ps1 -Script .\npunlock\tests\python\test_npu4000.py

.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_checks.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_integration.py -ScriptArguments '--case','batch'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_integration.py -ScriptArguments '--case','full','--mode','fp16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_integration.py -ScriptArguments '--case','full','--mode','w8a16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_checks.py -ScriptArguments '--case','holdout'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_checks.py -ScriptArguments '--case','decode'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_checks.py -ScriptArguments '--case','regression'

.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_integration_audit.py
```

以各 JSON 的 `status=passed` 和最终审计结果为成功依据。实验权重 map 的加载应指向本阶段
`calibrated-current/`；旧 profile 会按其原有源码绑定规则拒绝与当前模型混用。
