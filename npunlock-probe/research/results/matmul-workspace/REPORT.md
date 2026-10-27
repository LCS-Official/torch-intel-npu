# S27：矩乘工作区、高维与边界兼容性

日期：2026-10-23。本地开发版；Windows x64、Core Ultra 7 258V / Lunar Lake `8086:643e`、驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

本轮继续处理“大Tensor／大模型”支持：移除长K矩乘完整partial堆叠，扩展高维与有界只读视图，补齐空维度，并修复密集梯度复制误走动态scatter的问题。它没有扩大S26的256MiB单allocation或1,536图缓存，也没有把普通矩乘改成DPU路径。

最终21份真机验收记录、66组检查全部通过，共28,056次NPU图提交；111项离线检查通过。以下是实现和用例范围，不是完整CUDA或任意大模型认证。

## 实现

| 项目 | 变更 |
|---|---|
| 长K矩乘 | K>65,536时保留原K分块与顺序，用生成器逐块提供FP32结果；滚动Neumaier状态为每个输出两个FP32值，状态空间O(MN)，不再保留/拼接全部K分块结果 |
| SHAVE | 新增`resident_stream_sum.c`及12,316字节`stream-sum-f32.elf`；复用`arithmetic.c`的subnormal安全helper与`resident_reduce.c`的补偿顺序；全仓库resident manifest共62枚ELF |
| carrier | 复用ReferenceSW完整行carrier，逻辑形状`(1,2*count)`；每任务最多32,768个输出的sum/correction，必须是一组完整输入/输出，不能用普通逐元素Power载体的分任务布局 |
| rank | `matmul`/`@`从rank1–4扩到rank1–64；向量提升、批次广播、dtype/内维、输出及广播操作数容量检查保留 |
| 只读输入 | `mm`/`mv`/`dot`/`matmul`允许有界非负stride偏移、步长、展开/重叠读取；先DMA整理，native绑定仍是完整连续allocation；不允许输出与输入别名 |
| 空维度 | 空M/N/批次返回正确空结果且不提交图；零K的向量/矩阵/广播matmul及公开bmm输出零，保留out对象与一阶梯度；避免bmm除零和ND reshape中零元素推导`-1`的歧义 |
| 梯度复制 | S25通用可写视图路径对完整密集目的也使用动态scatter。现在这类目的用现有physical-view及DMA排列复制，避免索引上传与验证标志读回；真正部分目的仍保留动态验证 |

长K的滚动状态和临时DMA缓冲仍可能较大；本轮没有给所有输出尺寸提供总内存保证。普通FP16矩乘仍使用FP32 SHAVE计算，再转换为FP16；优化过的128/256方阵派发不变。

## 数值与真实训练

新入口为根目录`torch_npu_matmul_workspace_demo.py`。FP32逐元素rtol=4e-5/atol=2e-6，FP16为.004/.001，不放宽S26阈值。梯度另增加整体相对L2门槛：FP32<=1e-4、FP16<=.01。

- `stream`：257×129个输出、三块partial，覆盖两个状态tile。与原NPU完整stack后的Neumaier归约比对原始INT32位模式；除NaN payload外逐位一致，NaN位置一致。包含`1e8+1−1e8=1`、最小subnormal `2^-149`、溢出、Inf相消、NaN及负零。检查两个完整行carrier的实际memref和单invocation布局，并验证out身份及空流拒绝。
- `stream-large`：1025块257×257的合成device partial，其完整逻辑stack为270,800,900字节（258.26MiB），超过256MiB单allocation。341组三元抵消加两个零，所有输出精确为341；记录到的活跃Tensor buffer峰值为1,581,088字节（1.508MiB）。这是NPU实际执行的合成partial流，**不是同规模完整GEMM**，峰值也不含驱动/编译器内部分配，不能当作任务管理器的全部NPU内存。
- `wide`：真实`(3,65539)@(65539,5)`，两种精度各比较前向、两个输入梯度及out，覆盖真实长K派发。
- `batch`：rank5广播、rank64批次矩乘和rank64输入乘向量；两侧梯度和out均与CPU独立对照。
- `views`：两侧偏移/步长切片mm及完整base梯度、展开重叠读取、部分向量dot；空M/N/批次无提交，零K的2D/批量/广播/向量matmul前后向及out，公开bmm零K前后向、输出别名提前拒绝。零K bmm的out身份与数据另由离线逻辑测试检查。
- `train`：196,644参数网络，`Linear(65539,3)`→tanh→rank5 matmul→mean，输入`(1,2,1,2,65539)`。CPU/NPU各自独立两步MSE+SGD（lr=.1，foreach/fused=False），比较loss、输入/全部参数梯度与全部更新参数；不在两步之间重置CPU参考。

| 新组合网络 | 两步NPU loss | 最大梯度相对L2 | 两步缓存图数 |
|---|---|---|---|
| FP32 | 0.0188740641 → 0.0149012972 | 2.27028e-6 | 117 → 117 |
| FP16 | 0.0189514160 → 0.0165863037 | 0.00224065（约0.224%） | 157 → 157 |

FP16/FP32随机初始化序列不同，不用两列loss比较精度优劣。这里测试的是新宽K和高维组合，参数量小于S26的1704万参数网络；不是增加模型参数规模的训练基准，也不是实际数据集、长时间收敛或LLM质量认证。

## 开发诊断与修复

1. SHAVE编译器最初为FP32 helper留下四个绝对函数调用，构建脚本的安全检查在执行前拒绝。对复用helper强制inline后完成构建；保留`-ffp-contract=off`、无jump table及无绝对call/rodata检查。
2. 初次使用普通Power carrier，编译器把两个状态word拆成两个一word任务，C端偶数长度保护提前返回，抵消探针读出零。检查真实memref后改用已有完整行ReferenceSW carrier；没有修改算法容差或跳过C保护。修复后依次读出`[1e8,0]`、`[1e8,1]`、`[0,1]`，最终结果1。
3. 正式stream脚本曾只把actual转CPU、expected仍留NPU，比较器报设备不匹配。修正为双方detach/cpu；不修改生产计算。
4. 旧S2 matmul回归发现四次额外主机复制，定位到S25通用copy对完整转置叶梯度进行动态scatter，而非矩乘数值差异。修复共用copy路径后，原严格复制计数检查通过；未放宽原检查。另回归S25切片写入和梯度。
5. 扩展空张量检查时补上公开bmm的零尺寸除零和高维零K matmul的`view(-1,...,0)`歧义。空输出不提交；非空零结果用既有设备fill。旧容量负向fixture改用当前容量上限，原“部分向量必须拒绝”改成真正内维不匹配；合法部分dot另做正向/梯度验收。

## 保留边界

- 每Tensor及广播操作数仍受S26 dtype/元素/单allocation限制；没有跨allocation分页存储。图缓存仍最多1,536，运行中不淘汰；动态偏移/形状可能先耗尽图资源。
- 扩大rank或GEMM工作区不自动扩大SDPA完整attention矩阵、Embedding/卷积/排序/统计或其他FP32元数据内核的边界。selection backward的16,777,216几何保护保留。
- 普通矩乘仍以展开乘积/归约与同步图提交为主；长K滚动解决临时存储增长，没有声称提高峰值TOPS。没有测试会产生超过256MiB真实partial stack的完整大GEMM，其运算量远大于本轮专项。
- 一阶autograd以已验收用例为准；没有新增高阶梯度、FP64/复数、异步stream/event、GPU+NPU协同或完整CUDA兼容性。低位量化/整网质量未修改。
- 不新增依赖、不下载模型、不生成图表，不重建宿主C++/PYD/session DLL；`source-release`和GitHub未修改。代码和配套ELF属于本地开发版。

## 复现

在已构建的S26环境运行；升级自更早版本应先按INSTALL重建session DLL与设备PYD。

```powershell
.\run_torch_npu.ps1 -Script torch_npu_matmul_workspace_demo.py -ScriptArguments @('--case','stream','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_matmul_workspace_demo.py -ScriptArguments @('--case','stream-large','--precision','float32')
foreach ($precision in @('float32','float16')) {
    foreach ($case in @('wide','batch','views','train')) {
        .\run_torch_npu.ps1 -Script torch_npu_matmul_workspace_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
```

新增CPU测试`test_torch_npu_matmul_workspace.py`包含六项：生成器不保留全部partial（weakref最多两个输入存活）、独立FP64对照、补偿/非有限类别、无效流拒绝、rank/容量几何、bmm及NDmatmul零维度/out。它检查主机分块逻辑；C内核数值另由真实NPU专项验证。离线CI清单已纳入新模块。

## 最终验收汇总

表中数字来自每个配置最后一次通过的JSON，不把开发探针或重复运行累计进去。零K ND路径补齐后专门重跑两种精度的views；非空矩乘、滚动内核与训练沿用本轮已通过记录。

| 配置 | 记录数/检查组 | NPU调用 | 活跃Tensor buffer峰值（字节） |
|---|---|---:|---:|
| stream FP32 | 1/1 | 65 | 1,452,576 |
| stream-large FP32 | 1/1 | 10,265 | 1,581,088 |
| wide FP32 | 1/1 | 392 | 6,554,124 |
| wide FP16 | 1/1 | 494 | 7,602,748 |
| batch FP32 | 1/1 | 73 | 14,304 |
| batch FP16 | 1/1 | 118 | 13,968 |
| views FP32 | 1/1 | 87 | 1,835,932 |
| views FP16 | 1/1 | 149 | 917,966 |
| train FP32 | 1/1 | 716 | 6,029,844 |
| train FP16 | 1/1 | 1,022 | 6,160,984 |
| S2 matmul两种精度 | 1/2 | 1,603 | 3,888 |
| S2 layouts两种精度 | 1/2 | 79 | 3,206 |
| S2 contracts | 1/1 | 34 | 17,380,812 |
| S2 training两种精度 | 1/2 | 4,252 | 4,468 |
| S25 write两种精度 | 2/2 | 882 | 2,061（两记录最大值） |
| S25 gradients两种精度 | 2/2 | 880 | 1,449（两记录最大值） |
| S4 vector contracts | 1/1 | 4 | 160 |
| Device Quick | 1/33 | 4,171 | 1,051,726 |
| Training Quick | 1/11 | 2,770 | 9,511,052 |
| 合计 | 21/66 | 28,056 | 各进程独立计量，不累加峰值 |

每份记录退出时`closed=True`、`context_failed=False`，live buffers、buffer cache、scalar cache、session cache和quarantined bytes均为0。部分编译器/驱动常驻内存不属于这些计数。

离线检查按`.github/workflows/offline.yml`运行：根目录74项、npunlock Python API32项、source-release工具5项，共111项。62枚resident ELF/汇编及全部manifest源码哈希一致；S26的backend.cpp/PYD与session DLL及其C/H构建来源哈希一致，未重建。发布快照Git状态仍干净。

## 关键指纹（SHA-256）

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `0c82b945cdcc04399322d7af0b094debdb8bf1b5331d9626b264254529e47f9b` |
| torch_npu_compat_extended.py | `8241b2925456ccb23128223e42779f0f8d0f8658727298c6f84830a9fd1db266` |
| torch_npu_matmul_workspace_demo.py | `ac31254a042118dfe67a945c9e91e71ec02a6b1bce94c78b272a853a15f7da1f` |
| test_torch_npu_matmul_workspace.py | `e1aa604cef725800ad440215ddb5691b6ee6f6feace83d2f29a8601cc966db2c` |
| resident_stream_sum.c | `6fe062ff276f8d457a6758217c2e2252eaf02ac2ed396f1d9204afeddd870da1` |
| stream-sum-f32.elf | `00d5e82d80a4f4eac9f0df377ccd0d664814a651e5886f287decf8bf093f68ac` |
| stream-sum-f32.s | `5e59ebb9fc3a53c815d398fe74a9966f66d49692c19eefbade48dca97fdb178f` |
| resident/manifest.json | `329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc` |
| build_stream_sum_kernel.py | `6093eec206da52049acd7f6183eb793b5f118a71a1e0ddbf92abfc63877ac0ee` |

## 验收后清理

按用户保留策略，仅删除本轮55份原始JSON/日志、临时诊断脚本及Movi worker响应，共281,837字节（0.269MiB）。所有删除路径先验证为工作区内的普通文件；未递归删除目录。清理前后964份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充本段和WORKSPACE_STORAGE台账。保留新SHAVE C/ELF/汇编/manifest、构建/派发/复制修复、正式验收与离线测试、双语说明及本报告；无依赖安装、模型下载、图表或磁盘checkpoint/图缓存。
