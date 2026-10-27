# S35：普通Conv2d工作区与流式组装

日期：2026-10-24。本地开发版，未推送GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

## 改进范围

单allocation1GiB、rank/dtype及1536项/512MiB图缓存界限没有扩大。此次降低普通Conv2d的内部临时需求，使已支持的模型更少提前消耗内存。

| 部分 | 原行为 | 当前行为 |
|---|---|---|
| 路径选择 | 完整窗口或fold上界超过16,777,216元素才流式处理，较小形状仍创建完整索引/反向映射并保存im2col | 超过262,144元素即使用已有流式路径；这是选择预算，不是Tensor上限 |
| 输出与输入梯度组装 | 所有结果块放入列表，再拼接完整结果；中间块与最终结果同时占用内存 | 一个最终FP32目标，边生成边写回；不足1024个FP32元素的尾部合并到下一块 |
| 地址与对齐 | 沿用S30范围绑定 | 目标写入偏移始终4096字节对齐；末尾长度不必满一页，仍核对完整覆盖/边界 |
| 行块规划 | 主要按输入窗口宽度确定行数 | 同时考虑窗口和矩乘结果宽度，通常两者均<=65536元素；单个宽行例外保留 |
| 权重梯度 | 二叉树归约部分梯度 | 不改归约树和现有FP32计算/FP16转换规则 |
| 小窗口、多输出通道 | 原路径 | 继续保留原路径，输出大小不单独触发切换 |

选择流式后，卷积前向保存原输入/权重并重算小窗口，不保存完整im2col；输入梯度同样按窗口生成后逐块组装。CPU只生成几何索引，Tensor计算不回退CPU。小形状仍使用原lru_cache(maxsize=32)索引表；普通Conv2d自动路径中，单个正向/反向数组的上界各为262,144个INT64项，理论最多约4MiB/缓存几何（不含Python对象和瞬时构造开销）。手工调用私有`_conv_indices`、其他视觉路径不自动获得该约束。

不只看输出大小：开发时曾尝试让大输出单独触发流式，但1×1、多通道案例的主峰值来自完整输出/梯度与布局副本，额外分块只增加开销。最终不启用这条选择规则。流式组装本身也会增加范围复制/提交次数，本轮不宣称所有卷积加速。

## 真机结果

14份最终基线/验收记录全部通过，共4,998次NPU调用；所有进程关闭后live/cache/scalar/session/deferred/blob/factory/quarantine归零，context健康。以下峰值是后端Tensor分配计数，不包含CPU参考、索引数组、驱动图对象或整个进程RSS。

| 案例 | 精度 | 基线峰值 MiB | 当前峰值 MiB | 基线→当前NPU调用 | 基线→当前缓存图 | 基线→当前总秒数 |
|---|---|---:|---:|---:|---:|---:|
| 窗口工作区 | FP32 | 6.999035 | 1.843460 | 550→718 | 289→123 | 40.573→17.121 |
| 窗口工作区 | FP16 | 7.354527 | 1.630043 | 601→735 | 304→132 | 42.966→18.575 |
| 小窗口、多通道输出 | FP32 | 16.850796 | 16.850796 | 290→290 | 84→84 | 12.542→12.046 |
| 小窗口、多通道输出 | FP16 | 16.593353 | 16.593353 | 539→539 | 100→100 | 14.808→14.304 |
| 不对齐块组装 | FP32原始位 | 2.523720 | 2.012142 | 1→26 | 1→10 | .288→1.478 |

窗口工作区案例为输入1×4×96×97、4个输出通道、3×3/padding1。完整窗口有335,232元素；基线缓存一份完整几何，当前为0份。输入/权重/偏置全部梯度及完整输出通过独立CPU对照，分配峰值分别降低约73.7%和77.8%。基线两个INT64几何数组按已知shape计算约5.12MiB，当前不创建它们；这不是CPU总峰值的实测值。耗时含编译、CPU几何生成、独立参考、完整回读和关闭；没有交替多轮热基准或隔离NPU吞吐测量，不能由这些秒数推导TFLOPS。

小窗口输出案例为输入1×1×1×4097、257个1×1输出通道，输出1,052,929元素。最终保持原路径，NPU调用和分配峰值均与基线相同。没有宣称该案例有额外内存收益。

组装对照输入分块为`17,1007,1,65535,33,2049,3,65536,65536,65536,65536`，合计330,789个FP32值；基线先保留完整块列表再concat，当前逐块组装。完整回读按INT32编码逐位一致；生成器检查证明没有长期持有以前的块。该纯组装控制的峰值降低约20.3%，调用数上升；它不能替代完整模型的性能结论。

其他当前版本检查：

| 案例 | 精度 | NPU调用 | 缓存图 | 峰值 MiB | 总秒数 |
|---|---|---:|---:|---:|---:|
| 特殊几何 | FP32 | 112 | 54 | .185669 | 7.211 |
| 特殊几何 | FP16 | 163 | 73 | .174294 | 9.887 |
| 两步卷积训练 | FP32 | 156 | 63 | .234615 | 8.686 |
| 两步卷积训练 | FP16 | 278 | 82 | .233234 | 11.406 |

特殊几何包含batch2、非方形kernel、非对称步幅、膨胀以及groups=2。为覆盖小形状的流式分支，临时把选择预算降到1，结束后恢复；因此不把这些形状写成默认路径自然触发的证据。前向、输入/权重/偏置完整梯度全部与CPU对照。

两步训练为Conv2d(2,3,3,padding1)→Tanh→Conv2d(3,2,1)，同样强制流式，独立CPU/NPU SGD且逐步比较所有参数梯度与更新。FP32损失0.12802210→0.12772729，FP16为0.12801452→0.12770167。训练前后向区间没有显式payload下载或跨设备复制，验证回读另计；没有把CPU权重每步重置为NPU权重。

数值门槛沿用既有stream卷积检查：FP32 rtol=7e-5/atol=6e-6，FP16 rtol=.004/atol=.0005，equal_nan=True。浮点运算不是逐位轨迹认证；分块、部分归约与CPU/native Half可能有舍入次序差异。

## 离线、来源与复现

118项根目录离线检查、32项Python API和5项发布工具检查通过，共155项。新增5项覆盖页对齐与尾块、逐位组装、旧块所有权释放、计数/dtype/容量拒绝、路径选择及输入/输出行块保护；已加入离线CI列表。无需新增依赖、SHAVE C/ELF、OpenVINO runtime或DLL/PYD构建，沿用S32组件及62枚resident ELF。

`npunlock-probe/research/conv_workspace_before.py`保留S34四个卷积函数，基线标志在同一后端命名空间安装它们；没有重新初始化NPU上下文。四份卷积基线在S34主模块上记录，组装基线在最终共用运行时上使用原concat列表方式；最终九份after记录使用同一生产源码版本。基线后benchmark只扩展了独立assembly控制，四份卷积基线的数据/数值检查不变。S34完整主模块原SHA为`9a5416c35b50aa80c78083f701b38ee2df0100b8eec483bd26cbc40c6e80d46a`。

串行运行，`--baseline`只用于workspace/output/assembly对照；geometry/training测试当前流式路径：

```powershell
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','workspace','--precision','float32','--baseline')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','workspace','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','workspace','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','output','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','assembly','--baseline')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','assembly')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','geometry','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','training','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','training','--precision','float16')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_conv_capacity
```

测试收尾显式清理CPU几何cache并关闭后端；常规小形状cache仍供运行期间复用，不宣称生产shutdown已改变所有CPU缓存的生命周期。源码发布选择规则自动包含新脚本、基线和报告；没有修改公开source-release或推送GitHub。

## 剩余边界

- 仍有完整输出、NCHW/NHWC复制、FP32中间量、权重梯度归约树和分组结果拼接；预算不是总内存硬上限。
- 单个很宽的行仍需容纳，未新增K/列方向的卷积窗口分页。没有跨buffer分页、单allocation扩容或最大形状认证。
- 小窗口的完整索引表保留，其他视觉算子的工作区未统一修改。Conv3d、转置/可变形卷积、池化及全部原生out/backward变体不属于本轮新增认证。
- 没有长期FP16/AMP收敛、14GiB完整训练、全模型加速、功耗或硬件峰值算力认证。

## 最终指纹

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `db4e01e455973ee7048afc40d2b1d30bb00e8bf5324ef8a944c7f6798ade3df6` |
| torch_npu_conv_capacity_demo.py | `a8eaa8834293e7177c2fe6aaec089f570fdad24ab8b6be904c7988fda2229d9e` |
| test_torch_npu_conv_capacity.py | `ce18f339efc2103555888b35df9766c06b26fc71f9b57c88b816f35ef4f48867` |
| npunlock-probe/research/conv_workspace_before.py | `03aef0c656b44e16342907dfeff6b243e95564d0e94b20b037871db557bbb65e` |

session DLL、PYD及resident manifest分别仍为`1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90`、`4fed86d100f9ca48ecda3e8d810ea2526400485a690465ba96e1cf92d7150e4a`、`329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc`。无新模型、环境、图表或项目持久图缓存。

## 清理记录

全部进程退出后删除14份最终原始JSON，共75,356字节；清理前后501份源码、报告、内核和构建组件SHA-256一致。保留本轮实现、基线、复现脚本、离线检查和Markdown报告，公开source-release未改，未推送GitHub。
