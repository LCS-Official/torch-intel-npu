# RMSNorm 输出转换与舍入向量化

2026-10-05，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动
32.0.100.5540，编译器 8.3。结果：`passed_and_promoted`。

默认 RMSNorm 已采用四元素输出循环。真实 embedding 输入下，256/512 行的
norm 同进程复测约快 1.59/1.64 倍；B4/S64 的 W8A16 单层从 9.620 ms
降到 6.994 ms，延迟减少约 27.3%。没有新增依赖。

## 改动及精度语义

唯一算术实现改动在 `npunlock/tests/fixtures/npu4000/rmsnorm.c`，同时重建
默认 ELF、汇编和 manifest。宽度仍为 576、eps=1e-5，FP16 IO。

- 输出正常范围每组四个元素，整数向量位运算解码 FP16 与执行 RNE 舍入，
  FP32 乘法使用 VAU。没有使用错误的 `convertvector(Half4,Float4)`。
- 一组包含零、subnormal、特殊编码，或输出超出正常范围时，保留原标量转换处理。
- FP32 平方、Neumaier 补偿归约、除法、sqrt/倒数的代码和顺序完全保留。
- 归一化后先舍入到 FP16、重新解码，再乘 gamma 并舍入到 FP16，顺序保留。
- ABI guards 和多 tile 完整行调度保留；未编辑 Python 后端、DLL、DPU 图、DMA、barrier。

正常解码使用有限 normal half 的精确位映射，舍入公式来自已有 `half_bits`。
汇编可见 `CMU.CP.128.u16.u32`、VAU 整数操作及 `VAU.MUL.f32`，没有将
half 位编码按整数转换成 FP32 的那条错误指令。特殊路径没有扩大 NaN payload
等保证，测试目标仍是有限模型输入与现有精度契约。

默认 C SHA256：
`0987b3ec89abc49d05570522ffd0e9af0ca216aa230687b8156ea8717ea04f6e`。
默认 ELF SHA256：
`146e55974d78e4fb4cf3c553176e25daa339b2c521159954c32c34c834de48b3`。
`.text` 从 3,952 增至 5,312 字节，仍小于已验证的 21,216 字节窗口。
正式 build 与通过隔离检查的 `bits4.elf` 完全相同。

## 隔离检查

`build_rmsnorm_output.py` 生成原标量对照、仅输出算术四通道展开的 `unroll4`、
批量位转换的 `bits4`，以及仅用于精度检查的 `codec4`/`affine4`。
编译、汇编、链接继续使用既有隔离 worker，未修补汇编或修改工具链。
最终重新构建一次，全部诊断 C/汇编/ELF hash 与受测 manifest 一致。

每个行数由同一原始 Power carrier 替换代码窗口生成各版本。窗口置零后
五份 blob 的 hash 一致，输入输出、CMX、DMA 与任务布局相同。原始证据在 JSON
及 `carrier-*.blob`。两个隔离进程分别完成 110 和 121 项检查：

- 全部 63,488 种有限 FP16 编码，包括两个零及全部 subnormal，往返位模式一致；
  复测另打乱编码顺序，覆盖混合向量通道。
- 全部有限编码与 0.5、1.5、2^-14 及真实 gamma 相乘，与 CPU FP32→FP16 参考逐位一致。
  另覆盖舍入 ties、underflow、overflow 和 signed zero。
- 11 个行数 1、3、7、8、9、17、32、64、256、259、512，两个真实 gamma，
  随机值、混合 half exponent、零、subnormal、±65504 和真实 embedding。
  两种完整 norm 原型均与原标量 NPU 输出逐位一致；CPU 参考使用原有容差。
- 另在六组 FP16/W8A16、B1/S1、B1/S32、B4/S64 整层比较中，旧新输出逐位一致。

`unroll4` 的代码体积增至 8,368 字节，大输入收益较小，未接入默认路径；
采用收益更明显、体积更小的 `bits4`。

## 同进程 norm 性能

以下为 `confirm.json` 的真实 embedding 输入；每个 norm 图的中位数，单位 ms。
输入与 gamma 已在设备，图/会话已准备。各版本/输出方式预热 5 次、轮换计时
41 轮；复用模式在计时前分配输出，新输出模式计入分配/释放。

| 行数 | 原标量复用 | unroll4 复用 | bits4 复用 | 标量/bits4 | 原标量新输出 | bits4 新输出 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.407 | 0.411 | 0.363 | 1.12× | 0.494 | 0.491 |
| 32 | 0.750 | 0.693 | 0.572 | 1.31× | 0.857 | 0.703 |
| 256 | 3.775 | 3.359 | 2.374 | 1.59× | 3.945 | 2.597 |
| 512 | 7.257 | 6.401 | 4.438 | 1.64× | 7.456 | 4.738 |

第一进程 31 轮的 256/512 行复用输出加速为 1.57/1.61×，方向与复测一致。
两组输入分布（单位尺度随机值、真实 embedding）均有大输入收益。
原始样本及 P10/P90 全部保留；小尺寸波动较明显，不承诺每次单行调用都加速。
特殊值密集分布仅做精度检查，没有证明相同速度收益。

## 同进程整层收益

`confirm.json` 同进程只替换 norm kernel，attention/FFN 路径相同，比较旧/新
完整层。各模型预热 3 次，轮换 21 轮，每次仍为 4 张图。

| 输入 | 模式 | 原标量 norm 的层 ms | 新 norm 的层 ms | 延迟减少 |
|---|---|---:|---:|---:|
| B1/S1 | FP16 | 2.504 | 2.382 | 4.9% |
| B1/S1 | W8A16 | 2.607 | 2.325 | 10.8% |
| B1/S32 | FP16 | 3.925 | 3.395 | 13.5% |
| B1/S32 | W8A16 | 3.832 | 3.123 | 18.5% |
| B4/S64 | FP16 | 9.928 | 7.218 | 27.3% |
| B4/S64 | W8A16 | 9.620 | 6.994 | 27.3% |

正式接入后，原 demo 的 performance case 另在同进程比较新 NPU 与官方 CPU 层，
CPU 8 线程、RoPE/mask 已准备，暖运行排除准备和显式传输：

| 输入 | 新 NPU FP16 ms | 新 NPU W8A16 ms | CPU FP16 ms | CPU FP32 ms |
|---|---:|---:|---:|---:|
| B1/S1 | 2.883 | 2.149 | 1.025 | 0.896 |
| B1/S32 | 3.257 | 2.555 | 2.520 | 1.662 |
| B4/S64 | 7.366 | 6.949 | 6.315 | 5.698 |

此处 CPU FP32 对 B4/S64 的 NPU W8A16 仍快约 1.22 倍。两张整层表来自不同
进程，不拼接其绝对延迟来计算优化收益。桌面负载/频率未锁定，没有功耗或完整
网络 token/s 测量。统计无显式上传、下载、packing 和会话创建发生在计时区域，
不代表驱动内部 DMA 为零。

## 接入后的回归及记录

复用原 demo 和原门槛，结果写在本目录 `regression/`，保留旧 `smollm/` 数据。
七组检查全部通过，共 60 项记录、6,231 次 NPU 提交：
norm 22 项、numeric 8 项、boundary 2 项、guards 16 项、causality 2 项、
hybrid 4 项、performance 前置检查 6 项。Norm 另含五项 carrier 拒绝检查。

22 项 norm 与旧逐算子 NPU 路径逐位一致；相对 CPU 的原有 17 个 half 编码
差异仍保留，未放宽容差。真实层覆盖层 0、10、20、29 和 B8/S64。
混合生成的两条提示 × 两种模式，8 个生成 token 均与 CPU 参考相同。
整层 L2/scaled-error、分段和量化质量门槛保持原值，未声称消除所有 CPU/NPU 差异。

本阶段两个隔离进程加七组回归合计 11,614 次 NPU 提交，结束时各进程的
live buffer、quarantine、cached session 为零，context 未失败。
`before/`、`confirm-sources/` 与 `after/` 保存各阶段源码；`summary.json`
汇总结果和 hash，原始 JSON/日志保留全部证据。生产 Python/C++/DLL 未修改。

当前仍是显式冻结 SmolLM2 单层，FP16 IO、hidden576、B<=8、S<=64；
混合生成仍为 1 个 NPU 层加 29 个 CPU 层，未增加 KV cache 或训练支持。

## 复现

在工作区根目录串行执行。隔离 builder 使用本目录 `before/` 保存的原标量 C/ELF，
防止默认内核已更新后，对照也变成新版本。Probe 不修改默认内核。

```powershell
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_rmsnorm_output.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/rmsnorm_output_probe.py -ScriptArguments '--label','confirm','--rounds','41'
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_rmsnorm.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/rmsnorm_output_regression.py
```
