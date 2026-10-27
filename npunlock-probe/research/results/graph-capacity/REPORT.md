# S32：大 Tensor 存储与普通算子图回收

日期：2026-10-24。本地开发版，未同步 GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。复用原62枚resident SHAVE ELF，不新增依赖、模型或OpenVINO runtime。

## 本轮解决的限制

| 原限制 | 当前实现 | 范围 |
|---|---|---|
| 单allocation最多512MiB | 扩为1GiB，C session、C++ allocator和Python检查同步更新并重建 | 逻辑FP16/BF16最多512MiB；普通FP32/INT32/INT64和U8字节载体最多1GiB，详见dtype表 |
| 普通图累计到1536项后直接报错 | 有界LRU，普通图可回收，后来使用时重建 | 缓存保持1536项；无需无限保留曾使用的所有元素数/形状 |
| 淘汰可能关闭仍被使用的图 | 缓存仅移除自身引用，执行调用和prepared chain持有session；最后一个Python owner释放才关闭 | shutdown关闭缓存及仍由外部plan持有的已淘汰图；失败context继续使用原隔离策略 |
| 自定义冻结图不能由普通算子入口重建 | 明确保留pinned，低位/SmolLM/Transformer/矩形GEMM共用缓存登记入口 | 新自定义图可回收普通图腾出位置；全部槽位为pinned时仍拒绝 |

`stats()`新增`graph_cache_evictions`、`pinned_graph_sessions`、`deferred_graph_sessions`。1536是缓存保留项上限，外部plan持有的已淘汰图另外计数；不承诺所有活跃图/驱动内存均受此数值约束。缓存按真实使用更新LRU；不会修改Tensor存储所有者，也不会为回收执行CPU数值回退。

普通softmax、SDPA及S31稠密工作区随共享存储检查扩容；不盲目扩大浮点索引元数据、完整行宽度或专用packet。旧SHAVE计算块仍最多65,536元素。单次分配保护不等于整网总内存上限，也不等于所有算子都已认证最大形状。

## 实际存储界限

| dtype | 最大逻辑元素数 | 对应单Tensor字节数 |
|---|---:|---:|
| FP32 / INT32 | 268,435,456 | 1GiB |
| FP16 / BF16 / INT16 | 268,435,456 | 512MiB |
| BOOL / INT8 | 268,435,456 | 256MiB |
| INT64 | 134,217,728 | 1GiB |
| UINT8字节载体 | 1,073,741,824 | 1GiB |

rank仍为0–64。九种dtype均实际申请对应边界、用NPU DMA读回末页，并检查逻辑上限+1在任何NPU提交前拒绝。该检查不表示对这些dtype的所有计算均支持。

## 真机验收

| 项目 | 实际结果 | CPU/边界控制 |
|---|---|---|
| 默认1536项缓存压力 | FP32/FP16各1600种不同元素数，3202次调用，1666次淘汰；缓存保持1536，重访旧形状成功 | 每次完整结果与独立CPU逐元素精确相等 |
| 执行计划与冻结图 | 8项缓存下，已从缓存移除的图仍可完成prepared chain；最后owner释放后deferred归零 | 全pinned时新图在提交前拒绝，原有图继续执行正确 |
| 动态batch训练 | 32项缓存，17→23→7 Tanh网络；FP32/FP16各6步SGD+momentum；合计784次淘汰 | CPU/NPU模型独立推进，逐步比较loss、所有梯度及更新参数；前后向无payload下载 |
| 超过旧512MiB的完整梯度 | 134,217,745个FP32元素，即512MiB+68字节；16,434次NPU调用，15张图，后端分配峰值约4.000001GiB | 完整CPU输出/输入梯度对照；没有只比较采样点 |
| 14GiB同时驻留 | 14个各1GiB的FP32 buffer同时保留，共15,032,385,536字节；70次NPU调用 | 每个buffer由NPU写入并读回首尾页，读取中间零页；不是14GiB整网训练 |
| dtype边界 | 9种dtype全部通过，9次NPU调用，最大单allocation为1GiB | 末页NPU读回、上限+1拒绝及退出资源检查 |
| 低位缓存保护 | INT8 QuantizedLinear的输入拒绝、禁用缓存拒绝与恢复通过 | 无效请求0次NPU提交；恢复后1次DPU图调用 |
| 安装入口 | FP32尾部算术、FP16矩乘、Linear/Conv2d前后向及SGD通过，73次NPU调用 | 独立CPU参考；无OpenVINO导入 |
| Device Quick回归 | 33项全部通过，4171次NPU调用 | 包含更新后的广播/矩乘容量拒绝、线程与shutdown存储生命周期；资源全部归零 |
| 冻结SmolLM回归 | 使用本地SmolLM2-135M-Instruct，单层FP16 prefill/decode及旧KV长度保护通过；连同安装检查77次NPU调用 | 不新增下载/依赖，无OpenVINO导入；不代表重新认证完整30层模型质量 |

FP32沿用rtol=4e-5/atol=2e-6，Half沿用rtol=.004/atol=.001；指定纯DMA/neg采用精确比较。没有为扩大容量放宽门槛。动态batch用例的输入每步不同，不要求跨batch的loss单调下降；它验证独立训练轨迹与图回收正确性。

14GiB测试使用native共享分配初始化真实内存，250ms监控的最少物理可用内存为4,565,958,656字节（约4.25GiB），最少commit余量7,380,611,072字节（约6.87GiB），高于预设3GiB余量。后端峰值15,032,393,728字节，仅比14GiB存储多8KiB；首尾页测试不是14GiB全量计算或梯度验证。

六个容量/回收专项退出后live buffer、allocation cache、scalar cache、session、deferred session与quarantine均归零，context健康。105项根目录离线检查、32项Python API检查、5项发布工具检查通过，共142项；native会话重绑定/失败隔离与chain模拟驱动套件也通过。

## 开发过程与实际限制

- 旧运行中销毁/重新导入曾导致设备丢失的记录保留。本轮先做96次小图回收，再做默认缓存压力、保留plan及带回收的训练；只据此确认当前设备/驱动和已列工作负载，未证明任意驱动/模型都不会出现device-lost。
- 第一次训练末尾与下一项压力测试短暂重叠。最终训练已在独立进程串行重跑，本文训练结果取后一次；本轮不作速度排名或能效结论。
- 默认缓存压力期间，验证脚本只追加了retained用例的全pinned拒绝/恢复检查；churn执行体与生产源码未改变。retained和训练随后按最终脚本独立重跑。
- 扩容后，旧广播/矩乘/attention/卷积负向fixture中的尺寸可能已合法，已改为根据当前上限生成超限尺寸；不把执行了合法新形状后产生的测试失败视为设备应拒绝的证据。
- Device Quick运行期间触发原脚本120秒faulthandler定时堆栈打印，进程随后完成33项检查并以0退出；这是诊断输出，不记作执行超时或删除失败记录。
- 保留单allocation1GiB、rank64、dtype逻辑元素及各算子专用限制；没有跨buffer分页Tensor或虚拟连续大storage。
- 自定义冻结图仍pinned，缺少通用reload recipe；全部1536项被这类图占用时仍可能提前触限。很多同时持有的prepared plan也会保留额外图，不能称为无上限动态图。
- LRU会带来重建/编译开销；训练压力故意用32项小缓存，不代表推荐的性能配置。14GiB完整模型训练、最大形状attention、长期Half收敛、异步stream及GPU/NPU协同仍未认证。

## 复现

```powershell
.\build_npu_session.ps1
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script torch_npu_graph_capacity_demo.py -ScriptArguments @('--case','churn')
.\run_torch_npu.ps1 -Script torch_npu_graph_capacity_demo.py -ScriptArguments @('--case','retained','--limit','8')
.\run_torch_npu.ps1 -Script torch_npu_graph_capacity_demo.py -ScriptArguments @('--case','training','--limit','32')
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','large')
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','storage')
.\run_torch_npu.ps1 -Script torch_npu_memory_scale_demo.py -ScriptArguments @('--case','resident','--target-gib','14')
```

14GiB驻留测试会先检查物理/commit余量不足时拒绝，可用`--target-gib 2`降低预算。所有NPU测试应串行运行。参考实现仅使用本机PyTorch CPU、Python标准库对象所有权和现有npunlock/Level Zero接口；未复制外部LRU、内核或编译器代码。

## 构建指纹

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `8fedef49501cc6843672a67d50ab1b499ea80e762d338a986963db8f72353745` |
| npunlock/python/npunlock/npu4000.py | `80daa21a9c0e69db88d3f8ce8d6c519433b9d26f9538ae529d247fddc26c2533` |
| session_win32.c | `cfdd2f3241ae28a234f4161ed71b6e15e8cf0d1c20b716f27c9c61074228a70e` |
| npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| backend.cpp | `34497308b7ea4b010afa5794147ee74d538af031e48fb662dada2685113a376b` |
| _torch_intel_npu.pyd | `4fed86d100f9ca48ecda3e8d810ea2526400485a690465ba96e1cf92d7150e4a` |
| torch_npu_graph_capacity_demo.py | `6aa6e90ed78aa308ba31be4da959915405197b9333d39795c4a1dc56bc1c9778` |
| torch_npu_memory_scale_demo.py | `2f0adfe6b69becf3210501c4b4f85851b45a583911b2646c351ae4ea7c5a2c0a` |
| 原resident manifest，未修改 | `329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc` |

源码、测试、DLL/PYD、构建记录与Markdown报告保留；已删除9份原始JSON/日志，共49,051字节，清理操作前后386份源码、内核及构建/报告文件的SHA-256一致。不增加模型、图表或磁盘图缓存。公开source-release保持原状，没有推送GitHub。
