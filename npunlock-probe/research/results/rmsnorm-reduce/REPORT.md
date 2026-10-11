# SHAVE RMSNorm：公开方案调研、SIMD 半精度转换优化与实测

2026-10-06，Core Ultra 7 258V / Lunar Lake NPU4000，8086:643e，
驱动 32.0.100.5540，图编译器 8.3，Torch 2.14.1+cpu。

已把有稳定收益的 SIMD FP16 编解码接入默认 RMSNorm 内核。
256 / 512 行独立 RMSNorm 调用耗时下降约 12% / 13%；
四层 SmolLM2-135M 的 prefill 在两个进程中下降约 5%～9%。
decode 变化较小，首轮有一组略慢，不能据此承诺稳定加速。
这是内核源码中的默认实现，用户无需修改模型调用或启用额外技巧。

## 先调研公开实现

| 公开来源 | 可借鉴内容 | 本轮处理 |
|---|---|---|
| [Intel NPU compiler 的 SHAVE 构建配置](https://github.com/openvinotoolkit/npu_compiler/blob/develop/sw_runtime_kernels/kernels/CMakeLists.txt) | SHAVE 内核构建、预编译固件分发、编译选项 | 核对工具链路线；没有启用其近似数学/重排浮点加法选项 |
| [llama.cpp SYCL norm.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/norm.cpp) | 并行部分和，再进行 reduction 与归一化 | 移植算法思路，试验有序向量补偿和四路独立补偿和 |
| [PyTorch CPU moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h) | 分层向量归约及数值稳定性处理 | 作为归约组织参考；没有照搬均值/方差算法 |
| [Maratyszcza/FP16](https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h) | IEEE FP16→FP32 的位运算与正常浮点值运算实现 | 改成 SHAVE 四路 SIMD 转换；这是最终采用的来源 |

所检查的 Intel 仓库完整树快照 `upstream-tree.json` 对应提交
`0b38f7d42113ff329ac2bdd33583d123de4ccf2f`，含 `rms_norm.4000xx.elf`
等预编译内核及编译器算子代码，但该树没有对应的 SHAVE RMSNorm C 源码目录。
这一结论只针对已检查的仓库快照，不能据此断言 Intel 所有相关源码均未公开。
GPU 的 SYCL 工作组接口不能直接用于当前 SHAVE ABI，需要适配算法。

FP16 来源固定于提交 `782eea126dc5c755827be751a099eb01826175cf`。
原始头文件、完整 MIT 许可证、下载 URL 和 SHA256 保存在 `references/`。
生产 `rmsnorm.c` 保留版权说明，完整许可证放在
`E:/Pytorch_xfire_NGPU/npunlock/tests/fixtures/npu4000/rmsnorm/FP16_LICENSE`。
未新增 FP16 Python 包或 OpenVINO runtime 依赖。

## 两个归约候选先试过，但没有采用

第一种把补偿误差计算向量化，并保持逐元素的加法顺序；
第二种使用四路 Neumaier 部分和，再合并结果。
153 个数值检查、828 次 NPU 调用通过，测试输出没有位差异，
但性能没有改善，因此保留为实验记录，没有接入默认内核。

| 行数，宽度 576 | 原内核 ms | 有序向量补偿 ms | 四路部分和 ms |
|---|---:|---:|---:|
| 32 | 0.52665 | 0.58780 | 0.54075 |
| 256 | 1.94600 | 1.98015 | 1.99255 |
| 512 | 3.66170 | 3.75190 | 3.79390 |

完整代码、构建记录和测量在 `reduction-trial/`；
更早的浮点向量比较版本只做了编译检查，位于 `initial-float-comparison/`，
没有硬件性能结论。它们暴露了本工具链的比较/选择开销，
说明 GPU 上常见的并行归约形式不能直接保证 SHAVE 上更快。

## 最终实现具体改了什么

1. FP16 解码改用四路整数移位、位选择和正常 FP32 值的运算。
   半精度次正规数转换过程中不依赖 FP32 次正规值，避免硬件 FTZ 丢失数值。
2. FP32→FP16 的现有 SIMD 快速路径允许零值参与，保留正负零的符号。
   次正规数、溢出和特殊值仍使用原有标量编码路径。
3. 中间 FP16 舍入结果直接在寄存器中解码，省去取地址后的临时存储/重载。

宽度仍为 576，epsilon 仍为 1e-5。FP32 Neumaier 累加的源码顺序、
除法、平方根及倒数保持原值；仍以 `-O2 -ffp-contract=off` 编译。
归一化先舍入到 FP16，再解码并乘 gamma，最后舍入到 FP16，
保留原模型的两次半精度舍入语义。MemRef 边界检查保持一致。

正式构建与受测候选的 C、汇编、ELF 指纹一致：

| 项目 | 原值 | 新值 |
|---|---|---|
| SHAVE text 字节 | 5136 | 4592 |
| 载体代码窗口字节 | 21216 | 21216 |
| ELF SHA256 | `b8bf4b2a205587c29cdd25422c83b1033634f2f53161a8d713ed1ba603bf4799` | `93d3c4895adfcaefebdf5e095472e22dad645e14c86dba46cbb65958651a65dd` |

`before/` 和 `after/` 保存完整快照。`production-before.json` 中的七项
Torch 适配器、原生扩展、session runtime 和图载体实现指纹均保持一致。

## 性能对照

同一形状共用编译器生成的载体，仅替换 SHAVE 代码窗口。
四层模型的旧/新图在窗口之外逐字节一致。
所有测量使用普通执行 API，没有启用 task profiling。
预热后以 ABBA / BAAB 顺序交替，每轮每种实现测两次。
每次计时均包含同步执行；表格用各实现原始样本的中位数。
百分比表示耗时下降 `(old-new)/old`，不是峰值 TOPS 或 TFLOPS。

独立 RMSNorm 采用预分配输出，21 轮，每种实现 42 个样本：

| 行数 | 旧 ms | 新 ms | 耗时下降 | 新版更快的配对轮数 |
|---|---:|---:|---:|---:|
| 1 | 0.26505 | 0.24965 | 5.81% | 17/21 |
| 4 | 0.60500 | 0.54445 | 10.01% | 19/21 |
| 32 | 0.65885 | 0.61570 | 6.55% | 21/21 |
| 256 | 2.09545 | 1.84295 | 12.05% | 21/21 |
| 512 | 3.67085 | 3.20960 | 12.57% | 21/21 |

相同的四层 SmolLM2、真实 embedding、相同 FP16 / W8A16 权重和 KV 状态：

| 模式 | 操作 | 首进程旧→新 ms，31轮 | 耗时下降 | 复测旧→新 ms，21轮 | 耗时下降 |
|---|---|---:|---:|---:|---:|
| B1 S32 FP16 | prefill | 3.47985→3.22615 | 7.29% | 3.53850→3.35210 | 5.27% |
| B1 S32 W8A16 | prefill | 3.36730→3.18510 | 5.41% | 3.73375→3.45195 | 7.55% |
| B4 S16 FP16 | prefill | 5.23100→4.76045 | 9.00% | 5.29950→4.84015 | 8.67% |
| B4 S16 W8A16 | prefill | 5.19520→4.76485 | 8.28% | 5.23340→4.88220 | 6.71% |
| B1 S32 FP16 | decode | 2.32240→2.34060 | -0.78% | 2.32315→2.27960 | 1.87% |
| B1 S32 W8A16 | decode | 2.24505→2.21875 | 1.17% | 2.25285→2.16865 | 3.74% |
| B4 S16 FP16 | decode | 2.54620→2.49065 | 2.18% | 2.54860→2.47675 | 2.82% |
| B4 S16 W8A16 | decode | 2.49820→2.44960 | 1.95% | 2.56065→2.46065 | 3.91% |

prefill 首进程 29～31/31 配对轮更快，复测 19～21/21 轮更快。
decode 的配对一致性较弱，完整样本和轮数见 `audit.json`。
这次测量未限制桌面后台负载，采样中存在频率/热状态变化；
交替顺序和独立进程复测有助于辨别收益，不能消除所有环境影响。

计时期间没有新增显式主机上传、下载、设备复制、打包或图编译。
复制计数不覆盖驱动内部 DMA。独立 RMSNorm 没有分配新输出；
模型测试按正式 API 分配返回值，B4 decode 仍有驱动分配：
首进程 FP16 / W8A16 分别为 248 / 124 次，复测为 168 / 84 次。
因此不能把模型收益解释成完全无分配，也不能把较小的 decode 变化全归于计算。

## 精度与接入回归

- 数值候选共 104 项检查、554 次 NPU 调用：随机行数与尾块、真实 gamma、
  所有 63,488 个有限 FP16 编码、全 65,536 编码转换、正负零、次正规数、
  最大有限值、混合指数、Inf/NaN、随机舍入压力和真实 embedding。
  新/旧 RMSNorm 输出位差异为零；NaN 只要求 NaN 等价，不保证载荷位。
  与 CPU 的既有 `rtol=.002, atol=.001` 检查保持不变。
- 四层模型每进程检查 104 个输出/KV/位置数组逐位相同；
  重复 decode 保留原始 seed。两个进程分别完成 4288 和 3008 次 NPU 调用。
  已保存的 NPZ 指纹与两个进程的记录均一致。
- 接入默认内核后，8 个原有回归任务通过，共 74 项检查、1159 次 NPU 调用。
  覆盖单层、2/4 层、FP16/W8A16、B8 S64 上限、拒绝非法输入、因果性及混合生成。
  `norm` 还验证了中间半精度舍入不能省略。
- 修复测试中仍假定每层 4 次提交的历史断言；当前默认整层融合应为每层 1 次。
  没有提高任何数值容差。初次失败、日志及原测试源码保存在
  `failures/stale-call-count/`。
- 各进程关闭后 live/cache/quarantine 缓冲归零，context 未失败。
  离线核对还检查了源文件、ELF、许可证、模型 NPZ、载体窗口外字节、
  原始计时样本及正式回归源码指纹，结果在 `audit.json`。

## 复现

在 `E:/Pytorch_xfire_NGPU` 的 PowerShell 中运行；沿用 `.conda/npunlock`：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_reduce_audit.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_reduce_regression.py
```

`measurement-builder.py` / `measurement-probe.py` 是原测量源码快照。
当前 `rmsnorm_reduce_probe.py` 只追加了旧 ELF 的显式选择和恢复，
使默认内核升级后仍能对照真正的旧版本；离线核对检查了这一差异。
隔离验证新复现路径，避免覆盖原始控制数据：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\results\rmsnorm-reduce\repro-smoke.py
```

接入后的隔离复现检查已通过：4 组配置、104 个输出/KV/位置数组、
704 次 NPU 调用；仍能检查真正旧/新 ELF 的窗口外一致性。
该检查只跑 3 轮，保存在 `post-promotion/`，不用于报告性能百分比。

重新进行完整实验会覆盖对应实验输出，应先保留本目录。命令为：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\build_rmsnorm_reduce.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_reduce_probe.py -ScriptArguments '--case','numeric','--rounds','21','--label','codec'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_reduce_probe.py -ScriptArguments '--case','model','--rounds','31','--label','codec'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_reduce_probe.py -ScriptArguments '--case','model','--rounds','21','--label','confirm'
```

本轮只改善当前固定宽度的 RMSNorm SHAVE 路径。
四层模块的延迟变化不能直接换算为完整 30 层 LLM 的 tokens/s，
也不表示新增精度、训练模型支持或通用 FP32 GEMM 加速。
后续值得继续检查转换后的指令调度/访存及 decode 的 KV Select/分配开销。
