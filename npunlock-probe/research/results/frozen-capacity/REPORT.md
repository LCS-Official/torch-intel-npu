# S33：冻结图重建、缓存字节预算与模型容量

日期：2026-10-24。本地开发版，未同步GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

## 实现与实际作用

S32已允许普通算子图回收，但低位Linear、Transformer、SmolLM和矩形GEMM图仍固定保留。多个模型/形状或带权重的图可能耗尽槽位，也会持续占用图blob及驱动资源。本轮为这些生产入口接通调用期间的重建工厂，使它们与普通图共用LRU；同时按图blob总字节数限制缓存。

| 项目 | 当前行为 |
|---|---|
| 图重建 | QuantizedLinear的INT8/INT4/NF4、FrozenTransformer、SmolLM whole/split/plain/KV/RMSNorm，以及4/8任务矩形GEMM都可重新生成图 |
| 缓存预算 | 最多1536项，且serialized graph blob合计最多512MiB；先淘汰最近最少使用的可重建项 |
| 单个图大于预算 | 可重建图临时导入、不留在缓存；执行调用或prepared plan持有到结束，最后使用者释放后关闭 |
| 模型生命周期 | 调用者在栈上持有factory，运行时只存WeakValueDictionary；不由缓存长期持有模型或CPU权重 |
| 活跃图生命周期 | 沿用S32的NativeSession强引用与retired弱集合；prepared chain可同时持有多个已移出缓存或临时导入的图 |
| 未提供factory的手工/研究图 | 保持pinned，无法回收或超过预算时仍明确拒绝；不猜测未知图的重建方法 |

新统计为`cached_graph_blob_bytes`、`graph_cache_limit_bytes`、`graph_factory_calls`、`live_graph_factories`和`transient_graph_imports`。factory调用次数包含首次加载及失败尝试，不等于底层编译器实际编译次数；已配置的可选磁盘缓存仍可能命中。1536项/512MiB只约束缓存，不约束root bootstrap、所有活跃计划、native图对象或整机物理内存。

弱factory在模型每次调用中登记，需由调用者持有到提交返回；模型对象自身保留原权重/形状信息。重复加载沿用原有device/compiler profile、ELF、DPU mode和量化约定检查，没有绕过验证。SmolLM的已有IR/disk cache快捷路径保留。首次编译证据由真正执行的编译入口更新，避免为收集证据提前导入一份大图、执行时再重复导入。

单allocation仍为1GiB、rank0–64，FP16/BF16逻辑Tensor仍最多512MiB；不扩展各算子独立工作区或C/SHAVE任务。没有跨buffer分页Tensor。本轮只有Python宿主和测试变更，S32 DLL/PYD、62枚resident ELF及manifest不变，无新增依赖、模型下载或OpenVINO runtime。

## 冻结模型与字节压力验收

入口为`torch_npu_frozen_capacity_demo.py`。最终生产源码版本一致的9组通过，共277次NPU调用、25次超预算临时导入。各进程退出后live/cache/scalar/session/deferred/blob/factory/quarantine归零，context健康。

| 用例 | 压力与结果 | 数值控制 | NPU调用 |
|---|---|---|---:|
| INT8/INT4/NF4 Linear | 2项缓存，3种batch×3种格式×3轮；25次淘汰、27次factory调用，全部无pinned | CPU显式量化参考，后两轮重建输出与首轮逐位相等；删除模型后weakref失效 | 27 |
| blob字节预算 | 64项槽位但仅37,960字节预算；3个同形状不同权重图轮换，只保留1图 | 与充分缓存基线逐位相等；预算降至37,959字节后仍成功执行，临时图结束后deferred与缓存blob归零，恢复预算后可继续缓存 | 14 |
| FrozenTransformer | FP16/W8A16、长度2/3，2项缓存强制回收后重复运行 | CPU staged参考及重建前后逐位对照 | 20 |
| SmolLM FP16 whole | 单层prefill+2次decode，2项缓存、普通图扰动后重建 | CPU staged prefill；输出/KV/position与充分缓存基线逐位相等，旧KV快照保持 | 19 |
| SmolLM W8A16 whole | 同上 | 同上；不推导任意量化整网质量 | 19 |
| SmolLM FP16 split | attention/FFN分别融合，2项缓存；19次淘汰 | 同上，覆盖fused segment重建 | 28 |
| SmolLM FP16 plain | 独立RMSNorm、KV图及FFN，2项缓存；29次淘汰 | 同上，覆盖继承的Transformer执行入口和普通KV入口 | 46 |
| 四层W8A16 prepared chain | 字节预算低于完整decode图大小；24次临时导入，链执行前最多4张deferred图同时保留 | 全部输出/KV与充分缓存基线逐位一致，旧快照不变；四层用例不建立CPU整网参考 | 52 |
| 矩形SHAVE GEMM | 4/8任务两条路径，2项缓存及普通图扰动；50次淘汰 | 独立CPU FP32矩乘及重建前后逐位相等 | 52 |

数值门槛沿用原有入口：低位Linear rtol=.003/atol=.0003；FrozenTransformer rtol=.004/atol=.002且相对L2<.002；单层SmolLM staged prefill相对L2<.003且token-RMS归一最大误差<.01；FP32 GEMM rtol=4e-5/atol=2e-6。逐位对照用于验证图回收没有改变执行结果，不把缓存基线当作独立模型质量参考。KV在测试中显式下载核对，生产prefill/decode调用自身无payload下载。

四层plan的完整图大于测试缓存预算时，factory临时导入的session仍由plan持有，直到`_run_chain`完成；native句柄没有因缓存移除而提前失效。预算37,959字节的测试同样表明缓存预算不是单个可执行图的大小上限。缓存blob预算与Tensor单buffer1GiB保护是两种独立限制。

## 真实训练容量与回归

沿用`torch_npu_memory_scale_demo.py`，把原8层测试扩到16层4096宽残差Linear/Tanh网络，batch=1，FP32 AdamW（foreach/fused关闭）完成两步。模型在meta上构造后直接分配NPU参数；初始化只保留单层CPU临时权重，不建立同规模CPU整网。

| 指标 | 实测 |
|---|---:|
| 参数 | 268,500,992，约2.69亿 |
| 参数+完整梯度+两份AdamW状态 | 4,296,015,872字节，4.0009765625GiB |
| 后端Tensor分配峰值 | 4,564,467,772字节，4.250991877GiB |
| 两步前向损失 | 0.0062046605 → 0.0047463635 |
| 每步末缓存图数 | 289 → 289 |
| 缓存blob | 10,170,576字节，约9.70MiB |
| NPU调用 | 238,220 |
| 总耗时 | 2,110.44秒，约35.17分钟，含构造/初始化、校验与释放 |
| 250ms采样最低可用物理/commit内存 | 约7.41/8.03GiB |

每层反向hook完成，所有参数均有完整梯度；两步共抽样131,072个参数位置，每个位置的真实梯度、旧参数和旧AdamW状态交给独立CPU AdamW更新，再与NPU参数/两份新状态比较（rtol=4e-5/atol=2e-6）。采样覆盖各权重首/中/尾及bias；**这不是同规模整网CPU前向/全梯度对照，也不是长期收敛或吞吐优势认证**。前向+反向区间无显式payload下载/跨设备复制，损失与抽样校验下载另计。状态字节数与计划完全一致，退出后资源归零、context健康。35分钟两步说明目前该SHAVE训练路径仍受大量有界提交与更新工作限制；没有从本轮时长推导硬件峰值利用率或具体瓶颈比例。

另用原S32入口进行普通图回归：8项缓存下，FP32/FP16各128种元素数与回访共258次调用，分别121/129次淘汰；prepared plan/手工pinned图/全pinned拒绝与旧图恢复检查共18次调用通过。单独启用临时磁盘图缓存的单层SmolLM用例再次通过19次调用：2次写入、2次命中、1次IR重建跳过，0次拒绝，验证新的factory路径保留原缓存快捷入口。上述13个NPU进程合计238,792次调用，全部报告通过；退出后live/cache/scalar/session/deferred/blob/factory/quarantine均归零。

## 离线与构建检查

109项根目录离线检查、32项Python API检查、5项发布工具检查通过，共146项；另19项已有prefill磁盘缓存契约检查通过。新增4项覆盖byte budget、factory回收/plan生命周期、模型弱所有权、编译/profile失败重试；原native导入失败、disabled/failed context、pinned和shutdown保护继续通过。源码发布选择规则已自动包含新的根目录demo及本报告；没有修改公开source-release。

原session DLL SHA-256为`1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90`，原PYD为`4fed86d100f9ca48ecda3e8d810ea2526400485a690465ba96e1cf92d7150e4a`；resident manifest仍为`329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc`。

## 剩余边界

- 缓存字节统计仅为serialized blob，不是驱动/系统物理占用。NativeSession backing、副本、IR临时数据、当前模型CPU权重及prepared plan持有的图都可能增加内存。
- 重新生成/编译会增加下一次使用的延迟；小缓存压力验证不代表推荐的性能设置，临时超预算图每次使用都需要重新导入。
- 没有factory的手工图仍pinned。私有`_sessions`字典的直接插入/修改不提供完整缓存管理契约。
- 单allocation1GiB、dtype元素数、rank64及Embedding/索引/卷积/排序等专用界限保留；没有跨buffer分页或任意大模型保证。
- 保留既有长期Half、近并列排序、任意Hugging Face生成、完整低位质量和GPU/NPU协同的未认证范围；14GiB完整模型训练仍未认证。

## 复现

已有S32组件时无需重建DLL/PYD或安装新依赖，串行运行：

```powershell
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','lowbit')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','bytes')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','transformer')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','smollm','--mode','fp16')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','smollm','--mode','w8a16')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','smollm','--execution','split')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','smollm','--execution','plain')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','plan','--mode','w8a16')
.\run_torch_npu.ps1 -Script torch_npu_frozen_capacity_demo.py -ScriptArguments @('--case','gemm')
.\run_torch_npu.ps1 -Script torch_npu_graph_capacity_demo.py -ScriptArguments @('--case','retained','--limit','8')
.\run_torch_npu.ps1 -Script torch_npu_graph_capacity_demo.py -ScriptArguments @('--case','churn','--limit','8','--count','128')
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','training','--target-gib','4','--width','4096','--layers','16')
.\run_torch_npu.ps1 -Script npunlock-probe/research/prefill_cache_checks.py
```

SmolLM复用本机`models/SmolLM2-135M-Instruct`；未下载新模型。磁盘缓存回归临时设置`NPUNLOCK_GRAPH_CACHE`为工作区内独立目录，单层FP16 whole用例在回收重建时命中；默认仍关闭。原始JSON和临时缓存在结果归纳后清理，保留源码、构建组件和Markdown报告，不生成图表或模型副本。

验收后删除14份原始JSON及2份临时磁盘图缓存，共15,351,313字节（约14.64MiB），并移除空缓存目录。清理前后495份保留源码、内核/构建组件和报告的SHA-256一致；未推送GitHub，公开source-release状态干净。

## 最终生产源码指纹

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `e2e72012d5897c09e9fab5c064fd3c8ef585dabf2925aeb007c2e62be7d5b97d` |
| torch_npu_lowbit.py | `1c579ad4f2de9528d3fbd407c30df52bcfd29ac11b63cc202ffacd9c9137b01d` |
| torch_npu_rect_gemm.py | `78adc9fc5d06ca804d618ec0c10982cb1511c3c231179134b01b1c5d6b26fc9c` |
| torch_npu_transformer.py | `63e79f4e603ebcd63f36ab1ab86ce967833739dff7e4aa081ee3d73c340e09d2` |
| torch_npu_smollm.py | `fccb3d437c515ebac0aa33ba6cb4c5fecfa86f091d46cd0712aed347a735b95b` |
| torch_npu_frozen_capacity_demo.py | `9254d5234b893aaae7edd394d62ef0423bf82193754ed2976548cf7cdd16d1b8` |
| test_torch_npu_graph_capacity.py | `ff9be7a12790fd01cbebf2dc59a5533f624f941d0575974719a3f927cd88849b` |
| torch_npu_memory_scale_demo.py | `2f0adfe6b69becf3210501c4b4f85851b45a583911b2646c351ae4ea7c5a2c0a` |
