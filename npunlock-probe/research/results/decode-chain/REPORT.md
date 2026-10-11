# B1 decode：连续 2 / 4 层合并提交实验

2026-10-08，Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动 `32.0.100.5540`，
编译器 8.3，Torch `2.14.1+cpu`。本轮完成独立实验原型，尚未接入正式 runtime 的默认路径。

## 正式性能结果

SmolLM2-135M-Instruct，全部 30 层，FP16，batch 1，前缀 32 token，固定缓存容量 64。
每个候选分别与原 API 在同进程中交错比较：3 次预热、21 轮 ABBA / BAAB，
每种策略 42 个样本。下表采用**不带计数器 DLL**的中位数，包含计划构建、
输出分配、参数绑定、NPU 执行、等待、Python 返回和本次输出释放。

| 策略 | 同组原 API，ms | 候选，ms | 耗时减少 | 更快的配对轮数 | 每步提交 / 等待次数 |
|---|---:|---:|---:|---:|---:|
| 计划执行，仍每层提交 | 18.229 | 16.456 | 9.7% | 21 / 21 | 30 |
| 连续 2 层合并提交 | 18.661 | 15.147 | 18.8% | 21 / 21 | 15 |
| 连续 4 层合并提交 | 18.718 | 14.584 | 22.1% | 21 / 21 | 8 |

各行有各自交错的原 API 对照，不能把不同时间段的候选当作直接配对比较。
每轮先对该策略的两个样本求平均，再计算相对原 API 的耗时减少；21 轮的中位数
分别为 9.0%、18.0%、24.2%。百分比统计口径不同，以上表两组总体中位数之比作为主要结果。

计时包含 30 层驻留 decoder。CPU embedding、最终 RMSNorm、lm_head、采样、
初次编译及权重准备不在计时内；这不是完整生成速度，也不是芯片 TOPS。
计时前及计时后做正确性检查，计时区间内不下载输出。
桌面后台负载未固定，结果适用于本机此次实验。

## 本轮确认的开销

独立计数 DLL 记录每次原生 API 的主机耗时。四层候选对应的同组分段中位数：

| 分段 | 原 API，ms | 四层合并，ms |
|---|---:|---:|
| 设置图参数 | 0.314 | 0.242 |
| 追加图命令 | 0.119 | 0.063 |
| 提交队列 | 1.582 | 0.931 |
| 等待 fence | 11.657 | 9.996 |
| 重置 command list | 0.160 | 0.063 |
| 重置 fence | 0.028 | 0.009 |
| API 时间减去全部已计数分段 | 5.091 | 3.009 |

最后一行是**逐样本相减后取中位数**，包括 Python、C++ 校验、锁、依赖屏障等未单独计时的部分，
不能称为纯 Python 开销。各行分别取中位数，不能直接相加还原总中位数。
等待中包含设备计算、图内 DMA、调度和同步；本轮没有把它们进一步拆开。

group1 保持 30 次提交，已经获得一定收益；减少 Python 到原生的逐层往返有价值。
group2 / group4 进一步减少提交和同步次数，也获得了收益。
因此 B1 decode 同时有调用组织及逐层同步的优化空间，不能把当前慢速全部归因于 SHAVE FP32 算术。
本轮计算图和内核没有改变，不能据此声称矩阵计算吞吐提高了 22%。

全部计时组均执行 2,520 次图计算。预热后每个计时区间内新增驱动分配、
会话创建、显式上传 / 下载 / 设备复制 / packing 均为零。
这里的复制计数不覆盖驱动内部 DMA。连续缓存验证另有输出下载和分配，未算入性能样本。

## 实现与依赖顺序

复用既有会话、图参数绑定和 c10 共享分配，Python 收集完整执行计划及输出 Tensor，
再以一个同步 C++ 入口按 1 / 2 / 4 层分组执行。
每组使用首个会话已创建的 command list / fence；每个图仍绑定当前地址，完成后重置。
同组要求图会话不同、已经预热且属于相同上下文；拒绝同图输入输出别名和组内重复输出。
Python 计划与 C++ Tensor 容器保留所有输入、中间值和输出引用，直到同步调用完成。

组内图之间追加 `zeCommandListAppendBarrier`，每组提交一次并等待最终 fence。
这里是**依赖链合并提交**，有依赖的层仍按序完成，并非让四层同时计算。
屏障同时提供执行和内存顺序，依据 [Level Zero 官方 API](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_list.html)。

未增加 OpenVINO runtime 依赖；沿用既有驱动编译 blob，包括最新 Power RMSNorm 和 Select KV 补丁。
正式 DLL、PYD、模型适配器和内核源码 / ELF 的冻结指纹保持一致。
实验二进制及复制的 backend 源码均在本结果目录中。

## 正确性与失败处理

- 真实 NPU：FP16 / FP32 四段 add1 依赖链，group1 / 2 / 4 均精确正确；冷会话、别名、重复会话拒绝后仍可正常调用，共 8 项记录。
- 模型：原路径及三种候选的 hidden、30 层 K / V / position 共 91 个数组逐位一致。
- 连续 decode：原路径与 group4 独立推进 8 步，位置 32 → 40，共 728 对数组逐位一致；原 seed 未被覆盖。
- 主机计数确认每步追加 30 个图，提交 / 等待 / 重置分别为 30、30、15、8 次。
- 模拟驱动：成功、追加失败、屏障失败、队列提交失败、等待超时、device lost、两种重置失败、close 失败及参数绑定失败，共 10 种链路情况；另运行已有会话失败保护套件。
- 提交 / 等待结果不确定时上下文隔离，禁止销毁 / 释放可能在用的设备资源；确认未提交或已完成的失败可安全清理。模拟测试不故意破坏真实 NPU。
- 三个真实设备进程结束后均关闭成功，存活缓冲区、闲置缓存、会话缓存及隔离字节归零。

离线审计共 **1,637 次数组比较、60 个跨进程图哈希、27 项冻结文件指纹**通过。
审计包括已保存的 NPZ、构建源码 / DLL / PYD 指纹、原始计时样本和提交计数。
其中包含 baseline 自身及跨策略 / 跨进程比较，不能把 1,637 当作独立模型输入数量。

## 复现与证据

在工作区 PowerShell 中运行，所有硬件实验串行执行：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain.py -ScriptArguments '--case','build'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain.py -ScriptArguments '--case','checks','--profile'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain.py -ScriptArguments '--case','model','--profile','--label','host'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain.py -ScriptArguments '--case','model','--label','plain'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_audit.py -ScriptArguments '--case','failures'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_chain_audit.py -ScriptArguments '--case','audit'
```

原始样本：[plain.json](plain.json)、[host.json](host.json)。
正确性：[checks.json](checks.json)、[failures.json](failures.json)、[audit.json](audit.json)。
构建：[build.json](build.json)、[extension/build.json](extension/build.json)。
完整数组保存在压缩 NPZ 中；编译图仅存哈希，没有额外复制 60 个大 blob。

host 首轮使用 `before` 内的初版 runner；plain 的精确 runner 另存 `plain-runner.py`。
plain 增加连续 8 步检查，计时算法及原生实现相同；当前 runner 复现 host 时也会执行新增检查。

## 下一步

将经过校验的同步批量入口整理为正式 API，在冻结 decoder 适配器中显式使用，
移除研究原型对 `_run_many` 的临时替换；先保留普通逐图路径作为对照。
接入前补齐 B4 / B8、W8A16、prefill、缓存边界及生命周期回归，再选择启用范围。
目前只认证 B1 FP16 的这一冻结模型；未测试通用 Torch 捕获、训练、异步 stream 或 GPU / NPU 联合调度。
