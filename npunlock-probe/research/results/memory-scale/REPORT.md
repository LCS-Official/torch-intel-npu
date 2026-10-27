# S30：大Tensor、图复用与实用模型容量

日期：2026-10-23。本地开发版；Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。

用户最初给出训练占用约14GiB的实用上界，随后要求按本机NPU算力收敛，不必追求大型模型。本轮扩大软件容量并减少图缓存增长，用适中模型训练验收；约14GiB仍是可规划目标，尚未实测认证。这里计算参数、梯度、优化器状态及临时缓冲的合计，不要求单个Tensor达到14GiB。底层仍为Level Zero共享系统内存；后端跟踪的allocation bytes、整个进程内存及任务管理器NPU专属统计不是同一个口径。

## 实现范围

| 图片中的限制 | 本轮处理 | 当前边界 |
|---|---|---|
| 单buffer 256MiB | 扩到512MiB，并同步三层校验/构建 | 单Tensor仍对应一个buffer，各dtype还有逻辑元素上限 |
| 最多1536张图 | 减少偏移和同元素数形状重复编图 | 1536张硬上限保留，不是无限动态图缓存 |
| 分块/分页与运行中淘汰缺失 | 用检查过的子范围绑定和流式写回降低实际压力 | 不提供跨buffer虚拟Tensor或运行中图淘汰 |

- 单allocation从256MiB扩为512MiB，Python、C++ allocator和native session参数保护同步；rank0–64不变。FP16/FP32/BF16/INT8/INT16/INT32/BOOL最多134,217,728元素，INT64最多67,108,864，UINT8字节载体最多536,870,912。FP16/BF16的逻辑单Tensor上限为256MiB，BOOL/INT8为128MiB。
- 新增显式`graphinfer_session_infer_ranges`，绑定仍引用原始拥有者buffer，附加offset/size；检查同context、原始buffer身份、图参数字节数、4096字节起始对齐、减法式越界检查，以及输出与其他argument的部分/完全重叠。原infer和chain保持完整buffer契约，未扩大为任意视图入口。
- C++只在独立`run_ranges`入口接受连续、非空、页对齐的子范围；Tensor/Storage引用持有到同步调用返回，不建立另一个拥有者、不转移或重复释放allocation。Torch入口继续禁止输出与输入/另一输出共用storage，保留比C接口更保守的别名保护。
- 对齐切片按分块长度复用同一DMA图，偏移作为绑定值传入；不对齐的切片保留原完整buffer Slice路径。普通分块结果逐块写回最终allocation，不保存所有中间tile再完整拼接。大块逐元素算术直接绑定原输入和最终输出的范围，减少搬运及临时allocation；小块按元素数而非逻辑shape建图，保留标量、rank64和精确out返回对象。
- 保留65,536元素SHAVE计算块、1,536缓存图、同步fence、失败隔离和16MiB空闲buffer缓存。没有新增SHAVE C/ELF、OpenVINO runtime、pip依赖、模型下载、磁盘图缓存或checkpoint。

更新须依次重建session DLL和Torch PYD。已有62枚resident ELF保持不变；native失败/链回归以及新增范围测试和实际NPU验收见下文。

## 验证口径

入口：`torch_npu_memory_scale_demo.py`。所有NPU进程串行运行。每个case捕获输出/异常、提交数、活跃与峰值allocation、缓存图、关闭状态及源码指纹；关闭后live/cache/scalar/session/quarantine应归零，context_failed=False。

- `ranges`：FP32、FP16、INT64、BOOL、UINT8，每种dtype三个不同页对齐输入/输出范围，逐字节对照CPU；验证别名、未对齐范围和把部分buffer送入旧入口均在提交前拒绝。
- `offsets`：同一个Tensor的2,050个不同页对齐位置，读取值为精确整数，逐项验证对应全局偏移；新的范围图仅一张。另验证非对齐偏移仍走原Slice。旧设计的静态key包含start，会为这些偏移创建不同图。
- `shapes`：FP16/FP32各1,820种不同逻辑shape、每个4,096元素，公开`torch.neg(...,out=...)`逐次完整对照CPU；每种dtype只新增一张图。只消除同元素数逐元素图的shape差异，不把不同大小或其他算子强行合并。
- `storage`：九种dtype分别申请新逻辑边界、GPU读取最后一页，并检查上限+1在执行前拒绝。这是allocation/末页验证，不冒充每种dtype在整个最大容量上的计算认证。
- `large`：真实67,108,881元素FP32 Tensor，268,435,524字节，超过旧256MiB保护；执行缩放、平移、平方、归约及反向，完整loss和完整输入梯度对照独立CPU结果；计算期间不下载payload。
- `paired`：3层、宽32、3,168参数的普通残差Linear/Tanh网络；CPU/NPU独立初始化副本、三步AdamW，逐步比较loss、所有参数梯度和更新参数。
- `training`：相同类型的宽4096残差网络，按层数扩大；使用meta/to_empty创建NPU参数，仅保留一层CPU初始化缓冲，避免完整CPU模型副本。两步FP32 AdamW、foreach/fused=False，所有层参与前向/反向，所有参数保留完整梯度与两份moment状态。

大规模训练不建立完整CPU模型。它对每个权重的首/中/尾对齐块、每个bias的首块采样真实NPU梯度与旧状态，用CPU原生AdamW计算局部更新，比较更新后的参数与两份moment；这证明的是基于真实梯度的采样更新正确性，不能替代同规模全模型CPU反向认证。小型paired则覆盖完整独立反向。

FP32逐元素门槛为rtol=4e-5、atol=2e-6；纯DMA复制逐字节比较。检查loss有限且下降，第二步缓存图不持续增长。大模型每个参数的gradient/moment/parameter实际字节数须精确等于模型状态计划，不用闲置padding buffer凑内存。

训练前要求可用物理内存和commit headroom覆盖计划峰值及额外3GiB；执行中逐层、反向hook和优化器前后检查，另每250ms采样可用物理/commit最低值。该采样不是瞬时硬件峰值监控，也不读取功耗。

55层、宽4096时参数为922,972,160，FP32参数+梯度+AdamW两份moment合计14,767,554,560字节（13.753GiB），再加计算/更新工作区，目标为约14GiB训练峰值。是否实际达到以及完成的步骤以最终实测表为准。

## 边界与复现

512MiB是单allocation软件边界，14GiB是用户给出的实用上界，不是硬件总显存或后端统一硬上限。保留各算子专用workspace、旧FP32元数据16,777,216保护、硬标签分类packet、Embedding/卷积/排序/动态索引等限制；当前实测也不能推导任意14GiB Transformer/ResNet/LLM都能直接运行。原长期Half训练及低位整网质量边界不变。

```powershell
.\build_npu_session.ps1
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','large')
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','training','--width','4096','--layers','2')
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','training','--layers','8')
```

训练峰值随模型、batch、序列长度、优化器和精度变化；这里为batch1、合成输入的残差稠密网络，不是公开预训练模型或质量基准。训练脚本默认目标2GiB；55层约14GiB仅作计划计算，不作为默认验收。全部算术仍走已有SHAVE路径，没有DPU训练或吞吐优势声明。实测汇总、指纹及清理记录见下方。

## 最终代码的专项实测

以下case均使用返回语义修复后的最终后端。秒数包括CPU参考、回读、关闭等，仅作本机记录，不是隔离的内核速度基准。

| 专项 | NPU提交 | 结束前缓存图 | 峰值allocation GiB | 总耗时 秒 |
|---|---:|---:|---:|---:|
| ranges：通过 | 45 | 11 | 0.003662 | 1.573 |
| offsets：通过 | 2,085 | 5 | 0.007843 | 1.309 |
| shapes：通过 | 7,280 | 2 | 0.000061 | 3.250 |
| storage：通过 | 9 | 1 | 0.500004 | 1.145 |
| large：通过 | 8,226 | 15 | 2.000001 | 43.182 |
| paired：通过 | 775 | 52 | 0.000071 | 7.445 |

`offsets`的5张总缓存包括验证/拼接图；2050个offset读取本身仅新增1图。`shapes`的2张缓存分别属于FP16和FP32的neg图，每种shape同时检查out和新结果两种入口；新结果必须没有内部_base。paired另外验证FP16/FP32 neg→原地ReLU的完整输出与梯度。

最大allocation末页检查实际覆盖512MiB FP32/INT64/UINT8，以及其他dtype各自的逻辑上界；完整算术/梯度对照覆盖256MiB+68字节FP32，不把末页测试写成整个512MiB的全量算术认证。

## 适中规模真实训练验收

8层、宽4096、134,250,496参数的残差Linear/Tanh网络完成两步FP32 AdamW。参数、梯度、两份moment共2,148,007,936字节，即2.000488GiB；后端allocation峰值2,416,459,836字节（2.250504GiB）。

- loss：0.006518414710 → 0.005454545841，有限且下降。
- 第一/第二步缓存图均为289张，模型各层复用既有图，未随训练步数增加。
- 所有参数有完整梯度和两份moment；采样实际梯度后的CPU AdamW更新共检查65,536个元素。没有闲置padding buffer，也没有同规模完整CPU模型副本。
- NPU提交117,828次，总耗时450.180秒，包括初始化、参考采样、比较和关闭。
- 250ms采样最低物理余量16.261GiB、commit余量18.159GiB，均高于3GiB保护。
- shutdown后live buffer、空闲buffer、scalar、session和quarantine归零，context健康。

开发阶段同一8层模型在范围DMA已接通、逐元素直接范围绑定尚未接通时记录285,764次提交、450.705秒。最终提交数减少约58.8%，总时间变化较小；这是开发记录，不是隔离对照实验，不据此宣称总体加速比例或确定硬件瓶颈。

按用户最新的算力/实用范围要求，本轮在这一级训练后收敛。55层约14GiB只验证计划公式，未执行全流程；不认证任意大模型、长上下文或14GiB训练。容量与复用改善不表示SHAVE训练吞吐已接近DPU峰值。

## 收尾回归与返回语义修复

容量/复用代码的首轮23个旧回归进程全部通过，共43,553次NPU提交：D Quick33组、T Quick11组，原SDPA契约，FP16/FP32的rank/pointwise、hard/soft/MSE/native/training损失、GQA training/contracts，以及storage和AMP MLP。各进程关闭后均无活跃buffer/cache/scalar/session/quarantine，context健康。

收尾检查发现，统一小点算graph key时不需要reshape结果。原返回reshape视图会让新结果带内部_base，影响后续原地ReLU；新增离线断言先复现失败，再去掉输入/输出reshape，仅统一缓存key。现在原始逻辑shape只作为Torch结果元数据，native仍按完整buffer字节数绑定。硬件shape用例增加每种shape的非out新结果身份检查，paired增加FP16/FP32的neg→原地ReLU输出和梯度对照。随后补充最终代码回归及8层训练，结果见后文。

## 离线与原生完整性

最终代码128项离线检查通过（根目录91、Python API32、发布工具5）。新增5项CPU检查覆盖规划、Windows内存探针、固定大小流式写回、大块范围偏移，以及不改变返回/标量/rank64/out语义的逐元素key复用；返回视图问题先被断言复现，再修复。原生mock覆盖范围边界/别名拒绝、旧入口保护、重绑定、故障隔离及chain的拒绝/失败恢复。62枚ELF/汇编和25份SHAVE源文件与manifest一致，DLL/PYD的source/header/输出构建指纹一致。

后端Python SHA-256：`543ef3ea0400e80b2d21de67887f8d9a66fc587dfa4cf52e2dc8c28442a90d98`。resident manifest：`329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc`；DLL：`bffad21e197ece9349c8f763c4146a454ed2889746645c188850b2493878f89d`；PYD：`cc724867bca40929bebf6bc19a1ba78a9a28baae0b0f978e28eed6443d77866f`。

## 返回语义修复后的最终回归

| 分组 | 真机进程数 | NPU提交 | 结果 |
|---|---:|---:|---|
| 专项范围/偏移/shape/storage/大Tensor/配对训练 | 6 | 18,420 | 全部通过 |
| 8层4096宽真实训练 | 1 | 117,828 | 全部通过 |
| D Quick / T Quick | 2 | 6,941 | 全部通过 |
| FP32/FP16 rank与pointwise | 4 | 15,555 | 全部通过 |
| FP32/FP16组合loss三步训练 | 2 | 1,134 | 全部通过 |
| FP32/FP16 GQA训练及契约 | 4 | 2,664 | 全部通过 |
| AMP MLP | 1 | 1,372 | 全部通过 |

最终20个真机进程全部通过，共163,914次NPU提交。已有23进程的前轮回归另列于上文，未冒充最终源码重复认证。所有最终进程关闭后live buffer、缓存buffer、scalar、session和quarantine归零，context健康。

本轮交付为本地开发版：session DLL/PYD已重建、源码与双语说明/CI清单已更新，source-release与远端GitHub未修改。未增加依赖、模型、图表或磁盘checkpoint/图缓存；清理原始结果后仅保留源码、必要运行时和报告。

## 产物保留与清理

2026-10-23 S30验收后清理：删除本轮89份原始JSON/日志、可重建backend.obj/导出表、旧DLL/PYD临时回退副本及相关bytecode，共8935169 字节（8.521 MiB）；清理前后989份保留源码、报告、内核与必要运行时SHA-256一致，随后追加报告及本台账。保留512MiB/rank64容量、受检范围绑定与图复用/流式写回源码，重建DLL/PYD和构建指纹、验收/5项新增CPU测试、CI及双语说明，原62枚ELF/汇编不变。无新增依赖、模型下载、图表、磁盘checkpoint或图缓存；source-release和GitHub未修改。详见S30 memory-scale/REPORT.md。
