# KV 布局与 SHAVE Select 优化实验

2026-10-07，Intel Lunar Lake `8086:643e`，驱动编译器 8.3。

本轮跑通了一个专用 SHAVE FP16 Select 内核，用于 SmolLM 的 KV 行更新及注意力遮罩选择。
两个独立进程中，四层 FP16 / 校准 W8A16 的 decode 中位耗时均下降；101 轮确认测得约 3%～8%。
设备任务分析中，四层 Select 时间区间的累计并集从 0.571505 ms 降到 0.308902 ms，下降约 45.95%。
它仍是隔离实验候选，尚未接入默认模型。当前生产源码、RMSNorm 内核、设备扩展及 session DLL 均保持原样，
现有校准权重的源码指纹校验也保持原样。

实验先检查了两种图布局，再尝试直接缩短 Select 的 SHAVE 实现：

| 候选 | 实现 | 结果 |
| --- | --- | --- |
| grouped GQA | 每个 KV head 对应的三个 query heads 变为三个矩阵行，省去 K/V repeat Gather | 数值逐位一致；B1 逐轮比较没有收益，未采用 |
| joint Select | K/V 合并后做一次 Select，再分开 | 数值逐位一致；B4 FP16 中位耗时回退约 8%，未采用 |
| generic copy | 自定义四维广播 Select | 数值逐位一致；四组中三组中位耗时回退，未采用 |
| specialized copy | 只处理已验证的 64 元素 KV 行复制及标量值遮罩广播 | 两次独立测量的四组中位耗时均下降，保留候选 |

这些布局由本地模型数学关系和实际编译图推导，没有移植外部实现代码。

**模型计时**

下表每次操作均为四个连续 decoder layers 的单步 decode，不包含 embedding、最终 norm、LM head、采样或完整模型。
B1 使用 32-token prefill，B4 使用 16-token prefill；两者最大缓存长度均为 64。
W8A16 使用现有经过校准的逐输出通道 INT8 冻结权重，激活和输出为 FP16。

每轮交替排列 baseline / candidate；每种方案连续测两次，保留原始样本和逐轮均值。
百分比表示 `1 - candidate 中位数 / baseline 中位数`，为耗时下降比例。
“胜出轮数”表示同一轮内 candidate 均值更小；它与两组样本中位数的比例不是同一个统计量。
运行期间没有控制温度、频率或其他桌面负载，所以不能跨进程直接比较绝对时间。

| specialized copy 测量 | 模式 | baseline ms | candidate ms | 耗时下降 | 胜出轮数 |
| --- | --- | ---: | ---: | ---: | ---: |
| special-0，31 轮 | B1 FP16 | 4.93170 | 4.65630 | 5.58% | 16/31 |
| special-0，31 轮 | B1 W8A16 | 4.19990 | 4.11890 | 1.93% | 21/31 |
| special-0，31 轮 | B4 FP16 | 4.10935 | 3.98025 | 3.14% | 17/31 |
| special-0，31 轮 | B4 W8A16 | 3.76215 | 3.50695 | 6.78% | 24/31 |
| special-1，101 轮 | B1 FP16 | 2.69825 | 2.60725 | 3.37% | 57/101 |
| special-1，101 轮 | B1 W8A16 | 2.41195 | 2.27940 | 5.50% | 61/101 |
| special-1，101 轮 | B4 FP16 | 3.08330 | 2.99170 | 2.97% | 60/101 |
| special-1，101 轮 | B4 W8A16 | 3.97390 | 3.64375 | 8.31% | 69/101 |

101 轮测量的逐轮耗时下降比例中位数分别为 4.14%、5.09%、4.78%、8.21%。
全部预热计时区间内 driver allocation 增量为零。候选重用同一 baseline 编译 blob，
仅替换对应 Select 的代码字节；两组模型 IR、权重、DPU 工作和其他 SHAVE 内核相同。

**设备任务分析**

采用已有的隔离 profiling DLL，每张图重新创建 profiling query 后采集三次。
普通执行和 profiling 执行输出逐位一致。下面各项为四层时间累计的三次中位数，单位 ms。
不同类型可能重叠，因此不应把分类时间直接相加；DMA 时间当前驱动不可用，不能据此估计 IO 带宽。

| B4 校准 W8A16 decode | baseline | specialized copy |
| --- | ---: | ---: |
| Select 区间并集累计 | 0.571505 | 0.308902 |
| 全部 SHAVE 区间并集累计 | 1.848204 | 1.541008 |
| RMSNorm 区间并集累计 | 0.704372 | 0.674787 |
| DPU 区间并集累计 | 0.134822 | 0.128024 |
| 设备可观测时间跨度累计 | 2.340269 | 2.013126 |
| Python API 耗时 | 4.3609 | 3.8596 |

RMSNorm 和 DPU 未修改，其测得差异说明仍有运行波动；profiling API 时间也不替代上面的交替计时。
每层仍为 106 个 SHAVE 任务、76 个 DPU 任务，其中 Select 为 20 个任务。
本候选没有增加任务数量。Prefill 没有命中的 Select，blob 完全相同，其设备时间跨度为
5.908953 / 5.907912 ms；本内核不优化 prefill。

**内核与约束**

源码是 `npunlock-probe/research/kv_select.c`，构建和补丁检查位于 `kv_select_kernel.py`。
专用 text 为 1,984 字节；首版通用实现是 3,328 字节。
原编译器共享的 Select 窗口为 23,088 字节，补丁保留其剩余字节，编译图大小没有缩小。

KV 路径每行读取一次条件，选择当前行或旧缓存行，然后用 16-byte 向量原样复制 64 个 FP16 元素。
遮罩路径用整数向量选择两个 FP16 原始标量。两个路径都不做浮点计算，也不引入 FP16↔FP32 转换。
这个版本只加速数据选择与复制，没有改变矩乘精度或低位权重计算。

主机检查原 Select 代码指纹、range / invocation 重定位、四个 MemRef、FP16 dtype、CMX 范围、
连续布局、广播关系、16-byte 对齐及输出不与输入重叠。只允许实际验证的 KV / mask 形状：
物理最快维 64，KV 行数 1..64、head 数 1..24、N=1，以及 64 元素遮罩。
实际 Select 的条件 MemRef 在该编译器上是 FP16；这不改变模型 Equal / Greater 的逻辑布尔语义。
候选加载器另外验证 `8086:643e` 和编译器 8.3。普通执行的补丁从已准备好的 baseline session 取得同一 blob。

这些是研究脚本中的断言式检查，尚未整理成公开、可用于任意图的生产 API。
其他编译器版本、布局、宽度或硬件需要重新验证；不能把这个补丁视为通用 ATen Select 支持。

| 指纹 | SHA256 |
| --- | --- |
| 原编译器 Select 窗口 | `2bb6d89b2a4d94c392058f5b15e77418d68891e6e060e56feaf19a6040774af7` |
| 专用 C 源码 | `a268f1dc64e3ddbfef7eb91316538f674d94d024fcf537aa74d55a85004a544a` |
| 专用 ELF | `8988946b9a6a2ac014f99143f30cbf37c2d0e76eaa8e6457887e0f8a7a8b4807` |

完整构建参数和汇编指纹见 `select-kernel/build.json`。首版实现及对应构建脚本保存在
`select-kernel/initial-generic/`，其失败的性能样本没有覆盖。

**数值与记录审计**

五个模型测量进程各完成 172 项检查，包含 prefill、单步 decode、连续推进至第 64 个位置、
旧 seed 重用和缓存独立性检查。比较输出和每层 K/V/position，共 11,180 项候选数组逐位一致，
另有 8,944 项跨进程 baseline 数组、520 项 seed 重用数组和 102 项 profiling 数组逐位一致。

独立 Select 图覆盖 1、3、12、16、24 个 heads 和位置 0、1、7、8、15、31、32、63。
两轮共 80 个位置检查、320 项与 CPU 原始位复制参考的数组比较通过。
原缓存包含全部 65,536 种 FP16 位模式；所有位模式在候选输出中都被验证保留。
第二轮还在每个新写入行放入正负零、正负次正规数边界、最小正规数、最大有限数、
正负无穷大及不同 NaN 载荷，全部逐位一致。遮罩的 `-inf / +0` 同样逐位一致。
这验证了复制路径的位保真性，不代表该 NPU 的算术单元对这些值均具有同样的语义。

离线审计重新计算所有计时统计、校验源码及输出指纹、逐字节重建候选 blob，并重新解析原始 task 二进制。
累计 22,720 次成功 NPU 调用、196 张保存的图、7,056 条任务记录；所有运行最终 context 健康，
live / cached / quarantined buffer bytes 均为零。详细计数、逐轮统计和代码窗口证据见 `audit.json`。
28 个当前源码/运行库快照位于 `sources/`，清单见 `snapshots.json`；首轮图布局脚本位于 `measurement-group.py`，
首轮数值脚本位于 `select-kernel/numeric/measurement.py`。

本轮没有更改模型数学运算，现有检查中模型输出全部逐位一致。
默认接入前仍需覆盖完整模型回归、实际支持的其他 batch，并把严格布局检查收口到生产补丁接口。
3%～8% 是当前四层测试的观测值，不保证其他模型、batch 或桌面负载下的收益。

**复现**

在仓库根目录运行，所有硬件测试串行。以下新 label 避免覆盖历史样本。
内部沿用旧工具的 `gather` 名称作为候选 key；报告的 `implementation=select_shave_copy`
才是这里实际执行的实现，图中没有采用 Gather 更新候选。

```powershell
# 使用仓库已有工具链构建专用 Select
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_kernel.py

# 四层连续 decode 数值检查及 101 轮交替计时
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_probe.py -ScriptArguments '--label','recheck','--rounds','101'

# 独立 FP16 原始位复制检查（输出目录 numeric-special）
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_numeric.py

# 设备任务分析，baseline 完成后再运行 candidate
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_profile.py -ScriptArguments '--variant','baseline'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_select_profile.py -ScriptArguments '--variant','gather'

# 无硬件执行的历史样本审计
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_group_audit.py
```

数值和 profiling 命令使用固定结果目录；保留原目录后再重跑即可避免覆盖记录。
以生成 JSON 的 `status=passed` 和审计结果为成功依据。
时间线已导出为 `select-kernel/device-*/b4-s16-w8a16-decode-trace.json`，可用 Perfetto 查看。
