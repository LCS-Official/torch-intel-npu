# W8与混合精度：prefill缓存提前命中

2026-10-16，Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu。

已将现有FP16提前命中扩展到校准W8和显式混合精度；命中后跳过整层prefill IR重建。使用相同30份编译blob的新进程配对，首token耗时结果如下：

| 配置 | 完整IR首token s | 提前命中首token s | 中位耗时降低 | 脚本入口：完整IR s | 脚本入口：提前命中 s |
|---|---:|---:|---:|---:|---:|
| 校准W8 | 1.525 | 0.723 | 52.60% | 6.215 | 5.410 |
| 混合50% | 1.701 | 0.824 | 51.55% | 6.641 | 5.780 |
| 逐层37.5% | 1.660 | 0.797 | 51.97% | 6.581 | 5.732 |

FP16原有快路径作为回归对照：首token中位0.928秒，脚本入口中位4.997秒；没有用FP16旧完整IR路径混入本轮低位收益。

## 计时口径与固定条件

- 本地SmolLM2-135M-Instruct、完整30层decoder及KV在NPU；CPU embedding/final norm/lm_head/argmax。batch1、固定24-token提示、CPU8线程、interop1线程；prefill的完整24位置CPU输出头保持原实现。
- 每个配置复用前一轮固定逐行GPTQ校准权重，SHA256 `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62`；混合50%保护全部Up/Down，逐层37.5%复用先前45项策略，没有重新选择或校准超参数。
- 先建立30份提前命中缓存，再把同一blob/patch reports关联到完整IR键。逐层核对I/O契约与blob SHA256；每次正式请求均30次命中、0次编译。完整IR由透明graph包装器关闭提前命中，其他执行代码一致。
- 每种低位策略3对全新进程，顺序full→fast、fast→full、full→fast；FP16另3个新进程快路径对照。每进程首次请求后预热2次、测7次热prefill，hash/数值验收在计时区间外。
- 首token从模型准备完成后的embedding到argmax；脚本入口另外包括库导入、NPU初始化、模型和校准权重加载。未把IR准备挪到计时区间外，不含PowerShell/Python解释器创建前的时间。
- 没有清空Windows文件系统页缓存、锁定频率或桌面负载；三对样本不能代表所有模型/机器。收益只针对已有磁盘缓存的新进程，不用于无缓存首次编译或热decode吞吐。

## 全部正式样本

| 配置/进程 | 首token ms | 脚本入口到首token ms | 热prefill中位 ms | 命中/编译/跳过IR |
|---|---:|---:|---:|---|
| fp16-fast-1 | 928.052 | 4997.481 | 26.462 | 30/0/30 |
| fp16-fast-2 | 930.024 | 5007.925 | 26.894 | 30/0/30 |
| fp16-fast-3 | 920.961 | 4958.424 | 26.968 | 30/0/30 |
| w8-full-1 | 1524.837 | 6215.368 | 27.570 | 30/0/0 |
| w8-fast-1 | 729.530 | 5396.033 | 29.675 | 30/0/30 |
| w8-fast-2 | 722.736 | 5412.002 | 27.496 | 30/0/30 |
| w8-full-2 | 1521.643 | 6199.940 | 27.959 | 30/0/0 |
| w8-full-3 | 1550.994 | 6229.429 | 27.121 | 30/0/0 |
| w8-fast-3 | 720.745 | 5410.443 | 26.979 | 30/0/30 |
| mixed-full-1 | 1755.535 | 6977.224 | 29.143 | 30/0/0 |
| mixed-fast-1 | 835.362 | 5840.003 | 26.564 | 30/0/30 |
| mixed-fast-2 | 824.172 | 5778.318 | 26.118 | 30/0/30 |
| mixed-full-2 | 1688.175 | 6635.648 | 28.003 | 30/0/0 |
| mixed-full-3 | 1701.131 | 6640.771 | 27.210 | 30/0/0 |
| mixed-fast-3 | 813.069 | 5779.889 | 26.704 | 30/0/30 |
| selected-full-1 | 1659.900 | 6593.609 | 26.001 | 30/0/0 |
| selected-fast-1 | 794.354 | 5731.817 | 26.513 | 30/0/30 |
| selected-fast-2 | 797.214 | 5716.465 | 27.795 | 30/0/30 |
| selected-full-2 | 1638.349 | 6541.874 | 29.036 | 30/0/0 |
| selected-full-3 | 1668.918 | 6581.180 | 26.413 | 30/0/0 |
| selected-fast-3 | 801.495 | 5837.599 | 27.718 | 30/0/30 |

## 实现与缓存身份

- 生产改动主要在torch_npu_graph_cache.py。FrozenSmolLayer只增加对原weight_graph函数的身份记录，模型计算、量化器、C++、SHAVE、DLL/PYD、默认精度与公开API保持原样。
- 沿用FP16的绑定元数据、延迟完整建图工厂与所有读取校验。W8身份额外包含mode、排序后的保护投影、torch_npu_lowbit源码，以及实际使用的每份PackedWeight的名称、shape、format、group_size、codes和scales。
- 保留原FP16投影字节、RoPE、内核、shape/IO契约及源码依赖。保护投影使用原FP16，未被图使用的对应packed副本不影响身份；两组gamma仍为运行时输入，不进入常量指纹。
- 替换graph、serializer、compiler或W8 weight_graph函数的研究路径绕过提前命中；子类也沿用完整路径。decode与诊断分段的路径没有扩展。
- 键仍绑定设备、驱动、编译器、graph extension和缓存实现。因源码指纹变化，旧缓存需要重新建立一次；不会自动删除旧目录，旧条目仍计入容量。缓存保持显式opt-in、每目录512MiB上限、预留256MiB。

## 数值、失效与资源验证

- 10项模型/保护策略单元测试、19项既有cache检查、19项prefill检查、5项source-export检查通过，共53项。新增检查覆盖W8/混合绑定契约、codes/scales与保护策略失效、保护项原FP16字节、策略顺序无关、运行时gamma、研究hook和低位建图源码变化。
- FP16身份算法与修改前实现对照一致；适配器删除新增的weight_graph身份记录这一行后，SHA256恢复为修改前值。源码整体指纹变化仍会使已有缓存失效，不能理解为旧缓存可直接复用。
- 全部正式新进程与填充进程的最终hidden、30层K/V/position共91个Tensor、首token logits及argmax逐位一致；每进程7次热样本也逐位一致。
- 所有30层图检查原生DPU证据：FP16没有整数权重矩阵任务；W8及混合的每层都包含amode=0/wmode=2的W8A16矩阵任务。
- 故意损坏一个W8缓存IO契约，结果为29命中、1拒绝、1编译、1写回、29跳过IR；输出仍逐位一致，首token1466.623ms。原ZIP、SHA、profile及native检查保留。

| 单层边界策略 | 阶段 | 检查项 | 缓存命中 | 编译 | 跳过IR | NPU调用 |
|---|---|---:|---:|---:|---:|---:|
| FP16对照 | fill | 8 | 0 | 10 | 0 | 40 |
| FP16对照 | load | 8 | 10 | 0 | 4 | 40 |
| 校准W8 | fill | 8 | 0 | 10 | 0 | 40 |
| 校准W8 | load | 8 | 10 | 0 | 4 | 40 |
| 混合50% | fill | 8 | 0 | 10 | 0 | 40 |
| 混合50% | load | 8 | 10 | 0 | 4 | 40 |

边界形状为(1,1,576)、(1,24,576)、(4,1,576)、(8,64,576)，共48项检查。覆盖prefill、允许长度下的后继decode、旧KV不可变，以及两组gamma改变后共享图而输出仍与完整路径一致。边界测试使用第0层；完整30层只验证batch1/24-token启动，不能推广为全部30层在所有边界上的认证。
本轮共 **9,600次真实NPU调用**。所有进程关闭后live/cache/quarantine/session归零，context_failed=False。低位原质量门槛与默认配置不变；输出与旧路径一致不等于低位误差已消除。

## 剩余启动开销

下表为3个新进程cold请求的hook中位ms；hook含嵌套，不能相加。full_ir包括完整建图和序列化，identity为轻量绑定与指纹；读缓存、ELF证据检查、session和gamma准备仍保留。

| 配置/路径 | full_ir | identity | cache_read | inspect_blob | session | gamma |
|---|---:|---:|---:|---:|---:|---:|
| 校准W8/full | 833.043 | 0.371 | 89.454 | 197.101 | 63.006 | 96.974 |
| 校准W8/fast | 0.000 | 105.005 | 95.392 | 201.578 | 62.128 | 107.663 |
| 混合50%/full | 756.998 | 0.471 | 127.777 | 319.654 | 85.163 | 101.199 |
| 混合50%/fast | 0.000 | 90.864 | 130.560 | 243.339 | 84.070 | 97.245 |
| 逐层37.5%/full | 752.923 | 0.382 | 116.447 | 306.784 | 80.028 | 105.374 |
| 逐层37.5%/fast | 0.000 | 96.578 | 120.741 | 233.082 | 77.322 | 97.714 |

提前命中新进程在首请求之前的准备（3次中位ms）：

| 配置 | CPU模型/tokenizer/适配器/校准权重准备 | 其余入口准备、导入与后端初始化 |
|---|---:|---:|
| FP16对照 | 2193.914 | 1883.987 |
| 校准W8 | 2853.115 | 1836.152 |
| 混合50% | 3100.939 | 1865.881 |
| 逐层37.5% | 3117.726 | 1819.737 |

若继续优化程序入口到首token，下一步优先细分并减少CPU模型和权重的重复准备；现有测量中该部分约2.85–3.12秒，导入及后端初始化另外约1.82–1.87秒。模型准备后仍有约0.20–0.24秒ELF/DPU证据检查，可另行研究重复解析。当前数据没有证明这些开销都能安全消除。若优化稳态decode，则需另测图提交与设备任务，不能把本轮启动收益套用到token/s。

## 复现与交付

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:NPUNLOCK_GRAPH_CACHE=$null
.\run_torch_npu.ps1 -Script .\test_torch_npu_smollm_policy.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\graph_cache_checks.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\prefill_cache_checks.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_prefill_cache.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_prefill_cache.py -ScriptArguments '--case','study'
foreach ($label in @('fp16','w8','mixed')) {
  foreach ($phase in @('fill','load')) {
    .\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_prefill_cache_hardware.py -ScriptArguments '--label',$label,'--phase',$phase
  }
}
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_prefill_cache_report.py
```

按用户约定，验收后清理校准权重、NPC、计时/验收JSON等产物，仅保留生产与研究源码、修改前缓存源码及Markdown报告。研究A/B的系统临时缓存已逐策略删除；未新增依赖、模型、图片或持久图缓存，未推送GitHub。

## 指纹

| 文件 | SHA256 |
|---|---|
| `torch_npu_graph_cache.py` | `2cb5adabe7f4e380f3fda108843f2e6f031f135257a93bf87ed279090c81591c` |
| `torch_npu_smollm.py` | `a6a97f8a2d433eaf85671c98678215aee6f3af93443d9085d4edb486400d676c` |
| `torch_npu_lowbit.py` | `e9f5b12b09275f73933ef1ab325346776c75a905d48486e3aad5256d08a46f21` |
| `torch_intel_npu.py` | `bd147eb2adbf49d092084ae4f1de767b395424016e9f8862eb850ab571b55cce` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `3325f761ea55485271e26ecc3667b820f428d31f665e4c31fb916e8c30aefc8b` |
| `npunlock-probe/research/w8_prefill_cache.py` | `b2712d9e044ffd838a2b7d7273d6270357fd441a861d554ddee25ffdc530b335` |
| `test_torch_npu_smollm_policy.py` | `602f7288959eb13cd123a48fa04efd94b4f14acabb0d39c03499f6e07f45ba72` |
| `npunlock-probe/research/w8_prefill_cache_hardware.py` | `33b5aa536ae03dfb099aeba0db0a771823077cb12ee69fcda679375d32808612` |
| `npunlock-probe/research/w8_prefill_cache_report.py` | `ff2142459bec29015b64b0afc3f34fbfa70efc109762153531179f9853183cbc` |
| `npunlock-probe/research/w8_efficiency_startup.py` | `005195e4ca62cee082cc8341a4ee399178fc8e238c849d3a0a922868e2a449b1` |
| `npunlock-probe/research/w8_efficiency_report.py` | `9e0ccae160a877c1e3d9bdf68c416a36d69d798df4f3d0c260a25957f98cab9d` |
| `graph-cache-before.py` | `62e2932b35597b703e56c760297eb49e474bf41fbe13a7c8f93072e82d397ce6` |

清理前临时权重与配对清单指纹：

| 材料 | SHA256 |
|---|---|
| `weights.npz` | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| `prepare.json` | `09d07ed1f72f32b3dcb52d7017cfa2c515602a7fee80a8bd2c3188cd6b1fcddb` |
| `selection.json` | `649f68afe9b8397c78f94ddea456fdccc753f2927597a1d3c5811bb9e40e9872` |
| `study.json` | `ff2de72469ced60cd465a43ad05ce8a0aee95d069ace2a5d9726383fb48a3c17` |

## 清理记录

验收后清理本轮68份临时文件，共273,302,816字节（260.64MiB），包括硬件检查NPC、校准权重、计时JSON和两份离线检查JSON；保留源码与本报告。
系统临时目录中的配对缓存共210份、1,138,717,874字节（1085.97MiB），已逐策略删除。删除后重做审计或复现需重新prepare并运行对应硬件步骤。
