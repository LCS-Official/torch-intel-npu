# Tensor 分配缓存：测量、实现与模型对照

2026-10-06，Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动
32.0.100.5540，驱动编译器 8.3，Torch 2.14.1+cpu。

正式 C++ Tensor 分配器现默认缓存最后一个 Storage 引用释放后的共享内存块。
相同字节大小的下一次分配可以复用它；缓存同时限制为 **16 MiB、64 块**。
普通 Torch 调用自动受益，模型适配器和计算内核未修改，无 OpenVINO runtime 新依赖。

## 测量决定优化方向

复制正式 session C 源生成隔离 DLL，通过 Windows QPC 计时以下主机 API：
`zeMemAllocShared`、`zeMemFree`、参数绑定、append/close、submit/wait、list/fence reset。
正常 DLL/PYD/source 构建指纹检查保留，通过单独绑定清单选择这个 DLL。
未开启设备 task profiling；`wait` 同时包含内核、图内 DMA 和设备调度等耗时。

四层 SmolLM2 的暖 decode 中，分配和释放 API 合计约占调用及返回结果清理总时间的
**25%～30%**；绑定、记录、关闭和重置合计约 0.067～0.176 ms，分配/释放约
1.269～2.558 ms。独立阶段中位数相加是近似占比，不是单次调用的严格时间分解。
`shared_free` 包括调用内中间输出释放和调用外返回值删除。

| Batch / 模式 | 旧版 decode API ms | 旧版结果清理 ms | 驱动分配+释放 ms | 绑定/记录/重置 ms |
|---|---:|---:|---:|---:|
| 1 / FP16 | 4.4817 | 0.5893 | 1.2824 | 0.0734 |
| 1 / W8A16 | 4.1227 | 0.6144 | 1.2690 | 0.0671 |
| 4 / FP16 | 7.2253 | 0.9584 | 2.3392 | 0.1683 |
| 4 / W8A16 | 7.5736 | 1.1163 | 2.5577 | 0.1755 |

加入缓存后的相同隔离 DLL 计时中，248 次计时操作（每次 4 张图）的驱动分配和释放
计数全部为零；返回结果清理中位数降至约 **0.005～0.008 ms**。
等待 NPU 图完成约占调用耗时的 **56%～88%**。
这表明暖路径的驱动分配成本已被移出循环；不能据主机 wait 推断图内计算或 DMA 的各自占比。
两个 profile 批次的 104 项输出/KV/位置数组按原始字节完全一致。

原始分段样本：`profile-before/host-profile.json` 和 `host-profile.json`。
原始输出/计数：两个目录内的 `profile-model.json`、`profile-model-outputs.npz`。
每个 profile 进程 1248 次成功图调用，两次共 2496 次。

## 生命周期和 API

缓存只接收最后一个 Storage/view 引用释放后、已完成同步执行的内存块。
新 Tensor 仍有独立 Tensor/Storage 所有权；仍存活的输出、旧 KV 或 view 的内存不会被回收。
只按完整分配字节大小匹配，不切块、合并、异步提交或共享活跃输出。
缓存索引最多线性搜索 64 项，容量在配置时预留，Storage deleter 不再分配索引内存。
低层 `NativeLibraries.create_buffer` 的行为未修改。

可显式释放闲置块：

```python
torch.intel_npu.empty_cache()
# 或 torch_intel_npu.empty_cache()
```

该调用保持活跃 Tensor 和图有效，shutdown 自动排空缓存，shutdown 后最后一个活跃
Storage 释放时直接归还资源。`torch.empty` 内容不保证清零；缓存块可能包含旧值，
需要零值时使用现有 `zeros` / `zero_` 行为。

统计口径：`buffer_allocations` 仍计逻辑分配；新增 `driver_allocations`、
`allocation_reuses`、`cached_buffer_blocks`、`cached_buffer_bytes`。
`live_buffer_bytes` 只计活跃 Tensor 载荷；总共享载荷保留量可看 live + cached。
这些统计均不包括编译图和驱动内部内存。

驱动执行失败后阻止分配和复用；未知完成状态继续采用原生上下文隔离，闲置缓存也包含
在 `quarantined_buffer_bytes` 中。模拟 fence 失败检查创建了 8192 字节闲置缓存，
最终隔离量为 8960 字节，且驱动 free/reset/submit 计数为零。
这项检查由隔离 DLL 返回模拟错误，未触发真实硬件故障。

## 四层模型对照

旧版使用保存的实际 PYD、C++ 源和匹配清单；新版使用本轮重建 PYD。
四个隔离进程按 **旧、新、新、旧** 执行。session DLL、模型适配器、checkpoint、编译器相同，
Tensor 入口/输出和 KV 快照语义相同。真实 tokenizer/embedding 输入，连续四层，
batch 1 / prefix 32 与 batch 4 / prefix 16，FP16 和 W8A16 模式。
每组每个操作预热 5 次、计时 31 次；表中取两个进程各自中位数的算术平均。
计时包含 fresh 输出 Tensor 的逻辑分配，返回值删除在计时外。

| Batch / 模式 | 操作 | 旧版 ms | 新版 ms | 延迟降低 |
|---|---|---:|---:|---:|
| 1 / FP16 | prefill | 7.0566 | 4.3029 | 39.02% |
| 1 / FP16 | decode | 5.4222 | 3.0581 | 43.60% |
| 1 / W8A16 | prefill | 5.5491 | 4.0804 | 26.47% |
| 1 / W8A16 | decode | 4.0593 | 3.1345 | 22.78% |
| 4 / FP16 | prefill | 7.5704 | 6.6105 | 12.68% |
| 4 / FP16 | decode | 4.0045 | 3.0993 | 22.60% |
| 4 / W8A16 | prefill | 9.7357 | 6.8256 | 29.89% |
| 4 / W8A16 | decode | 6.1852 | 3.0076 | 51.37% |

桌面负载和频率未锁定，旧版 batch 4 / W8A16 两次 decode 中位数分别为 4.3488、
8.0215 ms，波动很大。不能将 51% 当作稳定加速保证；表中保留所有测得结果。
此处只测四层 prefill 与从 seed KV 开始的一次 decode，没有测完整 LLM 生成速度。

每组两种操作共 72 次暖调用，1008 次逻辑载荷分配：旧版驱动分配 1008 次，新版
包含预热在内仅 **0～2 次**，复用 **1006～1008 次**。
四个计时阶段的新驱动分配分别为 2、0、2、2，两次新版进程一致。
5 次预热后的隔离 profile 计时循环驱动分配为零；逻辑 Tensor 工厂仍会执行。

四个模型进程共 4992 次成功 NPU 图调用。每个进程保存 104 项输出/KV/位置数组，
其余三个进程与首个进程的全部字节一致；此前返回的输出和 seed KV 也未被后续调用覆盖。
暖循环上传、下载和会话创建计数没有增加。shutdown 后活跃载荷、闲置缓存归零。
原始结果见 `allocator-control.json` 和 `allocator-*-before/after.json/log/outputs.npz`。

## 验证和构建

分配器专项检查覆盖相同大小复用、FP16/FP32 特殊位模式及活跃 view、手动 empty_cache、
16 MiB 和 64 块边界、4 线程 80 次 Storage 搬运，以及持有 view 时 shutdown。
专项结果见 `allocator-guards.json`，未知完成状态检查见 `allocator-failure.json`。
设备 Quick 和训练 Quick 回归结果保存在 `regressions/`；构建和原始证据核对见 `audit.json`。

设备回归 33 项、4160 次 NPU 调用通过：4431 次逻辑分配中 1714 次实际驱动分配，
2717 次缓存复用，关闭后缓存及活跃载荷均归零。训练回归 11 项、2732 次图调用通过，
2807 次逻辑分配对应 1042 次驱动分配、1765 次缓存复用；梯度、SGD 和 FP16/FP32
MLP、卷积、轻量 ResNet 均通过，关闭后缓存及活跃载荷归零。

旧 PYD SHA256：`de5b037c50a1dcd253a37e9027b7a780513b0a6483644aef19aa7c856cd73411`。

新 PYD SHA256：`25d48cf772022ca7908934483c123e38cb8923ddd86223c00c8d994f1a39ef60`。

Session DLL 仍为 `d959fc5b1fb9a3171c192d1f607b11431af634203805d46cdcd6bba243933a51`。
原生 session C、公开 ABI 和内核未变。生产 diff 为 `production.patch`；构建快照为
`before/`、`after/`，模型计时脚本快照为 `measurement-script.py`。
`torch_npu_warm_profile.py` 后续新增的 audit 入口不属于计时源码。
之前 session-reuse 的 audit 是当时构建的历史证据；本轮 audit 核对当前新 PYD。

在 `E:\Pytorch_xfire_NGPU` 中运行本轮工具：

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_warm_profile.py -ScriptArguments '--case','guards'
.\run_torch_npu.ps1 -Script .\torch_npu_warm_profile.py -ScriptArguments '--case','control'
.\run_torch_npu.ps1 -Script .\torch_npu_warm_profile.py -ScriptArguments '--case','profile'
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Training -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_warm_profile.py -ScriptArguments '--case','audit'
```

上述命令写入固定结果目录。重复采样前应保存旧批次；重新编译 PYD 后需更新快照及采样
记录，不能直接沿用本轮指纹审计。隔离 profile DLL 构建使用工具的 `--case build`；
模拟错误 DLL 构建/运行使用 `npunlock-probe/research/allocator_failure_probe.py --case build/run`。

后续方向：优先拆分融合图内 SHAVE、DPU 和 DMA 时间，定位当前 NPU wait 的主要来源。
