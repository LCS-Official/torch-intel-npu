# S45：SmolLM 图缓存身份与部分回收问题

2026-10-27；Windows x64，Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，驱动32.0.100.5540，编译器8.3，Python3.12 / Torch2.14.1+cpu。

本轮修正校正因子的磁盘缓存身份，并为可复现的FP16部分图淘汰故障加入有界回避措施。默认仍为1536项、512MiB图blob；没有用提高默认预算或永久保留旧图掩盖问题。驱动或原生会话内部的最终根因仍未确定。

## 已接入的改动

`torch_npu_graph_cache.prefill_cache_graph` 原来的元数据捷径遗漏了S44的 `_corrections`。开启磁盘缓存后，普通W8与不同校正因子可能共用同一个prefill身份。现在哈希包含校正模块源码、投影名称、矩阵形状、rank及两个不可变FP16因子的实际字节，且与字典插入顺序无关。完整图序列化路径本来包含因子；S44正式实验关闭磁盘缓存，其质量数值不受此遗漏影响。

`torch_intel_npu._make_session_room` 对SmolLM的三个图种类采用同族失效重建：预算承压时，整体移除可重建的Smol图族，随后按需冷重建，避免释放部分旧图后继续使用同族剩余图。其他算子继续原LRU策略。此措施增加承压时的重编译成本，不是性能优化，也不证明任何图的部分释放都安全。

`FrozenSmolStack.decode` 构建及执行分组计划期间禁止拆散该图族；含固定图的族也拒绝部分淘汰。预算容纳不了单个Smol图时不再走临时导入后释放的捷径。这些情况提前抛出明确错误，不执行可能失效的计划。

新增标准设备入口，允许在重试前显式调整预算，不立即驱逐现有图：

```python
import torch
import torch_intel_npu

# 默认512MiB通常无需修改；只在预算保护提示后按需求提高。
torch.intel_npu.set_graph_cache_budget(768 * 1024**2)
```

参数必须是1字节至1GiB的整数；拒绝bool、低于当前缓存字节数的缩减，以及计划构建期间修改。它约束图缓存，不是NPU Tensor内存上限，也不表示已认证14GiB整网训练。

## 故障定位与反证

最小完整模型流程：30层FP16，单token prefill，两次分组decode，两次64-token前向，再重复同一个KV decode并逐位核对。图缓存关闭磁盘持久化，固定输入种子451。原始512MiB策略在第一次64-token前向中淘汰19图；第二次前向第14层稳定报 `zeFenceHostSynchronize: 0x70000001`，完成254次提交。该值为 [Level Zero的DEVICE_LOST](https://oneapi-src.github.io/level-zero-spec/level-zero/1.3.7/core/api.html)，不能直接解读为显存不足。

| 试验 | 结果 | 解释与处理 |
|---|---|---|
| 原始30层 / 512MiB | 254次提交后失败，19次淘汰 | 正式复现，进程失败隔离生效 |
| 同预算，额外强引用被淘汰图 | 330次通过，保留19图 | 仅诊断，真实保留量超过名义预算，不采用 |
| 原始单层 / 16MiB | 11次通过，2次淘汰 | 不是每次释放必然失败 |
| 图镜像改为独占内存页 | 同图失败 | 已撤回 |
| 图释放前队列同步 | 首次330次通过，重复仍失败 | 不能算修复，已撤回 |
| 保留子图的list/fence，仅释放图 | 330次通过 | 有临时资源保留，只作线索，不采用 |
| 上下文共用list/fence | 同图失败 | 已撤回 |
| fence后、reset前另加队列同步 | 同图失败 | 已撤回 |
| 原始原生库＋同族整体失效诊断 | 两轮660次通过，215次淘汰 | 无额外强引用，所有检查点在预算内 |
| 正式Python接入 / 默认512MiB | 两轮660次通过，216次淘汰 | 原始原生库，无永久旧图保留，正常关闭 |
| 正式256MiB保护＋同进程512MiB重试 | 420次通过，73次淘汰 | 先安全拒绝，恢复后前向及decode通过 |

最后一项在256MiB下构建decode计划时安全报错，尚未发生图淘汰，context未失败；计划保护计数复位后通过公开setter改为512MiB，同一进程完成全部后续流程。它验证可恢复的容量保护，不是宣称256MiB能容纳这套全部工作集。

逐次二进制和阶段事件用于对照，临时原生修改均已撤回。不能从某个只通过一次的原生试验推断最终根因。本轮查询了 [Intel图扩展头](https://github.com/intel/level-zero-npu-extensions/blob/ddbf8d65e2e22150e23efd383839aa80f84d4dac/ze_graph_ext.h)及 [Linux驱动的图实现](https://github.com/intel/linux-npu-driver/blob/main/umd/level_zero_driver/source/ext/graph.cpp)；后者不是本机Windows驱动源码，未复制其代码。

## 缓存与模型验收

真实checkpoint第0层、校准W8、两token prefill，对照普通W8、rank32校正、同rank半强度因子、rank16。每种首次编译后移除内存图，再创建新实例强制读取磁盘缓存。

- 四个身份互不相同、四个输出两两不同；第二次加载分别与各自首次输出逐位一致。
- 磁盘统计：4次miss / compile / write，4次hit / IR跳过，0次rejected。
- 8个局部结果均满足既定相对L2 <0.003、scaled max <0.01；共24次NPU提交，退出后所有受跟踪资源清零。
- 低位KV针对性回归、三模型整网前向及其质量限制见[低位报告](../w8-guarded-quality/REPORT.md)。

正式两项生命周期验收、缓存验收、98位置KV诊断和三模型前向合计10,374次NPU提交，均正常关闭；另同步旧四层容量验收脚本，FP16/W8各56次，正式合计10,486次。开发性失败及资源保留试验不计入正式通过。另一次前向脚本因局部变量仍引用末层而未通过退出资源计数，设备未失败；修正测试脚本并完整重跑后归零，质量数值逐项相同，其3150次提交不计入正式通过。

240项离线检查通过（根目录203、Python API32、源码发布工具5）；原始原生failure/chain夹具重新编译运行通过。审计复算4222个质量位置并检查原门槛；66枚ELF、66份汇编、27份C来源、3项原生组件和8项session构建来源/头文件指纹一致。

## 复现与边界

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_cache_lifetime.py `
  -ScriptArguments @('--layers','30','--budget','512','--label','family-production','--cycles','2')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_cache_lifetime.py `
  -ScriptArguments @('--layers','30','--budget','256','--label','safe-budget-retry','--retry-after-guard')
```

每次使用新label；重现旧故障需旧版Python缓存策略，不要将正式版的 `normal` 误当成原始LRU。`hold` 和 `flush` 仅为研究对照，不是用户性能选项。磁盘因子验收先按低位报告重新prepare，再运行 `w8_correction_cache.py`。

本轮没有新增依赖、环境、模型下载、图表或内核算法。原session C源码及DLL、Level Zero头文件、C++源码均恢复原值；DLL仍为 `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90`。本地PYD在诊断期间用未修改的C++源码重建，当前指纹 `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd`，无新增运行接口或功能；S45最终改动本身只需同步Python文件并重启进程，不要求用户重建原生库。

未认证其他驱动、任意常量图的释放生命周期、任意长上下文或任意模型；没有宣称彻底修复驱动。source-release未修改，未推送GitHub。

## 本轮源码指纹

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `433389e1627e8b2b24803a8f50d3e1142565f498102d14ad43fc67ac3f94540e` |
| `torch_npu_smollm.py` | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| `torch_npu_graph_cache.py` | `7eecc32f98679bcc7774c4da733024c01a121b713fa98b534dae15502af9aa26` |
| `test_torch_npu_graph_capacity.py` | `4bbecd81b2d2002144547ed80f70b8f5050b88ff30df7c7616d35349648f823d` |
| `test_torch_npu_smollm_correction.py` | `0b5d93b01da8ef6f1c11c9f1cfdc37b6616dce1779022f3e466305ffada53d7c` |
| `test_torch_npu_correction_guard.py` | `9aff891c8068e0a0e29e00abb58ac45babd46db507d7a787209de3b1183ee2b2` |
| `npunlock-probe/research/smollm_cache_lifetime.py` | `f4f053208ca797afcdf82afb98ff2468d55f3bc69c7def84c6edaf146d9817f4` |
| `npunlock-probe/research/w8_correction_cache.py` | `a6b2a1c4e0781cd5c6f3168e0d2cee99750f8bb5fb73c1e595aaba8b153bed60` |
| `npunlock-probe/research/w8_guarded_quality.py` | `2cd0cf6a8e280f86876f47251f2adf3dfd084ce4f774787134436cca8d79db24` |
| `npunlock-probe/research/w8_guarded_audit.py` | `af57384b3610cfe2f3f724ab21b5309b87a3ae65f653483fc14a163a93220ff4` |
| `.github/workflows/offline.yml` | `67ee41e759bd270cd38fb6578e6d7eda6638845b819563f78f66aacb84d981c4` |
| `torch_npu_frozen_capacity_demo.py` | `1413aab725a72ee84d2c39a9a49fca1b775e233f782e192a7866e99c84156739` |

旧四层plan验收不再要求Smol超预算图临时执行。FP16和W8分别验证字节预算保护、两项entry limit下的计划保护、同进程恢复至16项后两轮输出/KV逐位一致及旧snapshot不变；每项56次提交，11次图淘汰，正常关闭。公开setter只改字节预算，研究脚本人为缩小的entry limit由脚本恢复。

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','plan','--mode','fp16')
.\run_torch_npu.ps1 -Script .\torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','plan','--mode','w8a16')
```

## 清理与交付

两个S45实验目录及补充回归共享同一次交付统计：两阶段删除54份原始JSON/JSONL/NPZ、临时图缓存、重复诊断源码/库和本轮obj/exp，共156,547,698字节（约149.3MiB）。每个阶段清理前后均核对2533份保留文件SHA，内容一致；随后补充本清理记录。最后阶段指纹集合摘要为 `7e3dabfe5abb3bb8a975fd89413a2f341721d192db432d18b2c6a0e69629a5f8`。

两个S45结果目录现各只保留REPORT.md；实现、校准/审计脚本、测试、既有模型及运行组件保留。未清理全局缓存或创建新环境，全部测试进程已退出，source-release仍干净，未推送GitHub。原始产物不随报告保留，完整复现需按文中顺序重新生成。
