# S34：Embedding工作区、索引反向与排序容量

日期：2026-10-24。本地开发版，未推送GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

## 实现与容量口径

本轮解决的是算子内部的工作区/元数据保护，保留S32的1GiB单allocation与S33的1536项/512MiB缓存预算。

| 入口 | 旧限制 | 当前软件界限与执行方式 |
|---|---|---|
| Embedding与内部index_select | 源/结果<=1,048,576元素；完整索引包须满足`6+2*count+max(rows,count)*columns<=65536`，工作量`rows*columns*count<=1048576`，仅拆特征列，行数/索引数量稍大即拒绝 | 小形状保留旧路径；大形状复用byte gather和selection backward，必要时按特征列拆分。完整源和结果分别<=268,435,456元素，行数和索引数各<=16,777,216；每个列块的源/结果<=16,777,216 |
| Embedding梯度 | 只能在旧完整索引包及列块中生成 | 设备INT64生成行列偏移，有序累加现有选择内核；完整结果按列合并，仍是稠密梯度 |
| sort/argsort | 输入/输出<=524,288元素 | 输入/输出<=16,777,216元素，沿用既有稳定merge/宽行bitonic和计算包 |
| topk | 输入<=1,048,576、输出<=524,288；整数/大k排序分流还受524,288限制 | 输入/输出<=16,777,216元素；整数/BOOL、大k、超小k工作量或宽轴界限走已有排序，其余路径保持 |

表中是保护边界，不是极限形状真机认证。FP16逻辑Tensor受268,435,456元素/512MiB限制，FP32为同元素数/1GiB。SHAVE carrier仍<=65,536 words，selection内核的浮点元数据仍<=16,777,216；没有把这些C检查盲目放宽。Embedding按列拆分使完整逻辑Tensor可以大于单个元数据工作区。词表行数和展开后的索引数仍各需放得进一个列块。

一般公开index_select/gather/scatter_add仍走S25的通用地址路径，其反向保留16,777,216元数据上限；本轮扩大的是Embedding专用接口及其内部index_select，不能混为统一扩容。Embedding仍仅二维FP16/FP32权重、连续同设备INT64索引rank0–3，sparse=False、scale_grad_by_freq=False、max_norm=None；空索引/零维度等原专用限制未补齐。

## 正确性与内存语义

- 原完整索引包及特征列小分块保留；大路径调用已有raw-byte gather，不把INT64索引变为数值FP32，保留高字/负数越界检查及验证失败后的恢复。
- 反向先在设备生成完整行列偏移；每个目标块按输入索引原顺序累加，不拆分后改变重复项的相对顺序。Embedding Half每次加法后舍入，内部index_select采用FP32 opmath后转回原dtype。
- padding梯度以where选择零，避免乘零引入未选NaN。原autograd保存真实索引及version guard，不缓存索引payload到CPU。
- 特征列分块与布局整理使用已有设备DMA，最后按原布局合并；没有CPU算术回退或新的主机梯度镜像执行路径。
- 大路径需要扫描源分块，校验会同步读取4字节状态。该方式优先扩大兼容容量；没有证明比旧小路径、CPU或GPU更快。完整列主序临时数据及最终合并仍占额外内存，没有跨buffer分页。

验收中的`exact=True`/`bit_exact`字段实际采用`torch.testing.assert_close(rtol=0,atol=0,equal_nan=True)`：表示零容差数值相等，**不单独认证NaN payload及正负零的编码一致性**。训练门槛保持FP32 rtol=4e-5/atol=2e-6，FP16 rtol=.004/atol=.001，不为本轮放宽。

## 真机验收

11组专项全部通过，生产后端、排序、通用索引与autograd四份源码的指纹一致，共61,490次NPU调用。另3个进程做旧Embedding/拒绝接口回归，合计本轮最终验收62,109次调用。全部进程shutdown后live/cache/scalar/session/deferred/blob/factory/quarantine归零，context健康。

| 专项 | 精度 | NPU调用 | 最终缓存图 | 峰值 MiB | 总耗时 秒 |
|---|---|---:|---:|---:|---:|
| embedding | FP32 | 3,065 | 287 | 12.766 | 37.062 |
| embedding | FP16 | 2,857 | 299 | 13.516 | 39.610 |
| indexing | FP32 | 3,920 | 199 | 17.005 | 26.007 |
| indexing | FP16 | 3,526 | 207 | 12.755 | 26.788 |
| columns | FP32 | 28,840 | 1536 | 256.007 | 289.501 |
| columns | FP16 | 2,513 | 171 | 8.501 | 21.898 |
| sorting/topk | FP32 | 3,318 | 179 | 34.306 | 28.515 |
| sorting/topk | FP16 | 3,695 | 199 | 30.274 | 30.816 |
| INT64/BOOL sorting | 两种dtype | 1,524 | 92 | 24.000 | 13.327 |
| embedding model training | FP32 | 4,102 | 235 | 17.003 | 31.122 |
| embedding model training | FP16 | 4,130 | 257 | 24.127 | 33.982 |

具体数值范围：

- Embedding：65,537×17词表、17个索引，以及4×1词表、32,769个索引；超过旧行数/源元素/完整索引数量保护。完整CPU前向/梯度零容差相等；重复索引、padding NaN隔离，负数、词表边界、2^32及signed64最大值拒绝后能正常继续。
- 索引：17×65,537数据、非首轴及重复索引，公开index_select和专用内部前后向均与CPU零容差相等。这些形状仍小于一般索引反向16,777,216界限，不推导该通用界限已解除。
- FP32列分块：131,073×128词表，共16,777,344元素，实际超过完整源16,777,216；两个列块，完整前向及全部梯度与CPU零容差相等。643次LRU淘汰，最终1536张图/60,192,720字节blob，图回收中正常完成。约4.83分钟及256MiB分配峰值仅为容量/正确性实测，不是高效lookup的证明。
- FP16列分块：65,537×17、1,114,129元素，临时把Python元数据预算降到131,074以强制9个大路径列块，完整输出/梯度零容差相等；不是FP16超过16,777,216元素的实际认证。
- 排序：4097×129，共528,513元素，包含完整梯度、稳定索引和双out；同形状完整topk输出超过旧524,288输出界限。另8193×129、1,056,897输入元素的小k路径，完整梯度对照通过。FP16/FP32均零容差数值相等。
- INT64/BOOL：65,537×8，共524,296元素，稳定并列索引与CPU一致；INT64包含超2^53的整数。
- 两步训练：1,114,474参数的Embedding(65537,17,padding_idx=0)→Flatten→Linear网络。FP32损失0.18707587→0.18391749，FP16为0.18708582→0.18390936；两步独立CPU/NPU SGD比较完整loss、所有参数梯度和更新，无按NPU权重重置CPU的轨迹折中。

旧接口回归：原Embedding demo两种精度、24个padding/rank组合及冻结模块共385次调用；topk/pooling contracts共10次，11项提交前拒绝及非法设备索引；视觉/排序contracts共224次，12项提交前拒绝及索引/参数version guard通过。原topk contracts把早已支持的k=0误作拒绝，已改为真实非法的k=-1；超旧容量拒绝样例同步改到新上限，未为适应旧测试而收窄生产实现。

## 离线与来源

最终113项根目录离线检查、32项Python API检查和5项发布工具检查通过，共150项。新增4项覆盖源/目标存储界限、重复索引/NaN padding/两种累加舍入、特征列重排，以及topk输入/输出边界；sort拒绝检查更新为新元数据界限。最大容量的CPU meta测试只验证软件形状保护，未申请对应物理NPU存储。列布局单测曾因CPU替身未模拟materialize的contiguous契约失败，修正测试替身后通过，没有据此修改生产数值实现。

FP32 columns真机记录在该单测替身修正前已经启动，记录的单测文件SHA因此属于前一版；本表生产四文件及demo与最终版本一致。没有把单测替身修正写成再次重跑该5分钟真机案例。

实现仅为本项目Python编排，复用`resident_index.c`、`resident_dynamic_index.c`、`resident_selection.c`和`resident_sort.c`及其既有ELF；CPU PyTorch提供独立数值与训练参考。无新pip依赖、模型下载、SHAVE内核、OpenVINO runtime、宿主DLL/PYD构建或低位质量策略变更。发布工具根目录源码/报告选择规则自动覆盖新demo和报告，公开source-release不变。

## 复现

已有S32原生组件时串行运行：

```powershell
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','embedding','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','embedding','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','columns','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','columns','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','indexing','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','indexing','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','sorting','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','sorting','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','integers')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','training','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','training','--precision','float16')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_operator_capacity test_torch_npu_compat_extended
```

FP16 columns用例临时降低Python元数据预算以强制多个大路径列块，执行后恢复；不能把这个缩小预算用例写成FP16最大形状认证。实际跨越16,777,216完整源元素的列分块为FP32用例。所有NPU进程串行，时间包含编译、CPU参考、完整回读和关闭，不是隔离吞吐基准。

## 尚待改进

- 一般索引反向、卷积窗口/折叠、池化、take_along_dim及次序统计仍有独立工作区、行长、rank或packet界限。
- Embedding大输出、最大词表/特征组合、全部原生out/backward接口和极值/空输入未全面认证；没有稀疏梯度、频率归一或max_norm支持。
- 大路径的扫描、INT64地址与中间布局仍需效率/峰值内存优化；宽行bitonic依旧O(n log²n)。
- 仍未认证14GiB完整模型训练、任意Hugging Face模型、长期Half/AMP收敛或GPU/NPU协同。

## 最终源码与原生组件

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `9a5416c35b50aa80c78083f701b38ee2df0100b8eec483bd26cbc40c6e80d46a` |
| torch_npu_sort.py | `092103de79bacd4f441b4c743023cb91af33762b76b019e5b23afc8e98defceb` |
| torch_npu_index_capacity_demo.py | `536c057fcc10db2df793ee4eb705b58c536fd0bffd136383f3fb068ec8394f1c` |
| test_torch_npu_operator_capacity.py | `18bd7bbf83d3244a0fa55331887d6076cdae0f8774c3d3144f8dbacd8e8e5111` |
| test_torch_npu_compat_extended.py | `c5d6a77d771841a9a3c0e85891154d1ddbe85a0c607a0099fb42681ab1fc3dce` |
| torch_npu_vision_sort_demo.py | `1f64f1c31ed5d748a781b042ba139117a4a517ac84c1153fecf56838d8cdd75e` |
| torch_npu_pool_topk_demo.py | `55c1b117bcebda92ca69c7924866862d8e6f3500e05d156bb96348c248a72fff` |

未修改的通用索引`torch_npu_compat_extended.py`为`8241b2925456ccb23128223e42779f0f8d0f8658727298c6f84830a9fd1db266`，autograd为`a0831dc63ef447f8e25dbc232a9d0d6df029d19e12c5410292ad3747c62a5c6b`。

原session DLL仍为`1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90`，PYD为`4fed86d100f9ca48ecda3e8d810ea2526400485a690465ba96e1cf92d7150e4a`，resident manifest为`329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc`。源码、当前构建组件、可复现检查和本报告保留；原始JSON验收归纳后按用户要求清理。

## 清理记录

全部NPU进程退出后，删除本轮11份专项及3份旧接口回归JSON，共68,277字节；清理前后497份保留源码、报告、内核和构建组件SHA-256一致。无模型/环境下载、图表或持久图缓存产物；未修改公开source-release，也未推送GitHub。
