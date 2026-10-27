# S25：容量、索引写入与常用算子/dtype兼容性

日期：2026-10-22至2026-10-23。平台：Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动`32.0.100.5540`、图编译器8.3、Torch2.14.1+cpu。当前为本地开发版。

本轮完成三类主要阻塞的集中修复：扩大普通Torch存储并复用有界计算分块；接通动态索引和非重叠NPU目的写入；扩展整数/BF16及常用数学。41个最终验收进程全部通过，101项离线检查通过，共53,492次NPU图提交。它不是任意CUDA程序、任意dtype/算子或完整模型的全面兼容声明。

## 实现

| 方向 | 当前结果 | 执行方式 |
|---|---|---|
| 普通存储 | 单allocation从4MiB扩到64MiB，逻辑rank从5扩到8；FP16/FP32等16,777,216元素、INT64为8,388,608、UINT8字节载体为67,108,864 | C++ allocator/factory同步更新；Torch存储与冻结DPU适配器的旧容量分开；SHAVE carrier仍最多65,536 words |
| 大Tensor | 超旧百万元素的点算/均值/clone与反向；2359296元素卷积权重可存储/复制；实际64MiB FP32缓冲区上传及首尾DMA读回 | 数据仍在Level Zero shared allocation，现有FP32/FP16计算继续分块；没有把整机共享内存容量当作单Tensor无上限 |
| 动态索引 | nonzero、单参数where、masked_select、BOOL及INT32/INT64高级索引、负高级索引、广播及分离轴、scalar BOOL | SHAVE计算计数、规范化索引和原始字节读取；仅动态计数/校验标志同步到CPU |
| 写入 | 非重叠NPU切片/转置目的的broadcast copy/fill/masked_fill及no_grad算术；index_put覆盖/累加、scatter覆盖、scatter_add | 源/目标分块扫描，重复覆盖采用确定顺序；目的重叠拒绝。INT64值始终使用原始字，不经FP32保存 |
| 一阶梯度 | 高级索引重复累加、原索引version guard、index_put/scatter、原生inplace索引写入、切片copy/fill/masked_fill | index_select/gather/scatter_add的Half使用FP32 accumulator；index_put累加保留逐次Half舍入；支持范围见测试 |
| dtype | 新增INT8/UINT8/INT16/INT32/BF16存储与设备转换；整数加减乘、位运算、整除/余数、div floor/trunc、整数幂、sum及all/any | U64原始字算术和转换在SHAVE；超过2^53的INT64做精确对照。BF16部分算子采用FP32中间计算并舍入 |
| 数学与拼接 | sin/cos/tan、pow的Tensor/Scalar入口、expm1/log1p、floor/ceil/trunc/round、相应一阶导及已列out；新增数学函数除pow外的无梯度inplace；混合dtype cat/stack及梯度 | 链接自有SHAVE ELF，复用项目IEEE helper；三角函数采用192位常量范围约减。未使用CPU Tensor算术回退 |

原生图绑定仍须完整、连续、offset-zero；公共布局入口先DMA整理，或在设备内生成更新后的完整backing storage。存储扩大没有解除排序/统计、极值、softmax/attention、卷积窗口、Embedding及冻结LLM的独立工作区/模型限制。

## 真机验收

新增19个专项进程，9,923次图提交；另22个回归进程，43,569次图提交。各专项的最终计数表附后。

- 9种存储dtype、整数运算与转换、INT64高位/符号回绕、空/标量/部分view、混合拼接、动态和重复索引、覆盖/累加写入、两线程整数运算、version guard及非法输入拒绝。
- 公开gather/index_select/scatter_add另测130行，取消旧公开入口128行限制；S20旧精确初始化累加对照及两种精度梯度仍通过。Embedding和take_along_dim专用入口保留原边界。
- sin/cos另做8192个随机FP32位模式全量级对照，有限结果的最大绝对差均为`1.1920928955078125e-7`；NaN按类别比较。这是样本结果，不是全FP32正确舍入证明。
- S24整数排序/统计、池化、3D、grid及两类三步训练共14个进程通过；旧设备Quick33组、旧训练Quick11组通过。另回归布局/mask契约、SGD更新及迁移探针；原生checkpoint仍作为预期拒绝检查。
- 所有41个记录shutdown后live/cache/scalar/session/quarantine均归零，`context_failed=False`。NPU提交按本后端图调用计数，不是SHAVE指令数或硬件利用率。

## 较大模型

模型为`Linear(2048,1024) → sin → Linear(1024,4)`，共2,102,276参数。两种精度分别建立独立CPU和NPU模型，保持各自训练轨迹；每步比较loss、所有梯度及SGD参数，没有逐步把CPU权重重置成NPU权重。

| 精度 | 步数 | NPU loss | NPU图提交 | 后端峰值live bytes |
|---|---:|---|---:|---:|
| FP32 | 2 | 0.01031090971082449 → 0.009166275151073933 | 1142 | 41984052 |
| FP16 | 2 | 0.01031494140625 → 0.00939178466796875 | 2188 | 50352166 |

数值判据：FP32 rtol=`4e-5`、atol=`2e-6`；FP16 rtol=`.004`、atol=`.001`；整数/指定字节复制精确比较。峰值仅为后端跟踪的live buffer，不包含编译器、驱动内部或全部进程内存。两步合成训练不代替默认ResNet/任意Transformer或长期训练质量认证；未测速度排名。

## 开发中发现与修复

1. 将旧全局容量直接扩大，会让冻结低位适配器和部分旧carrier接受未经验证的范围。改为独立`MAX_STORAGE_ELEMENTS`；冻结DPU旧容量保留，sort/topk/行极值及旧索引packet分别检查自己的边界，避免静默产生错误结果。
2. 首版三角代码超过Power carrier代码窗口。将约减后的正常范围多项式使用普通FP32运算，保留极小数/非有限值及整数范围约减分支；最终所有新增构建均拒绝绝对函数调用、跳转表和rodata。失败发生在构图校验阶段，没有因此提交不安全内核。
3. CPU对FP32标量`pow(x,±0.5)`的快捷计算，与Half/逐Tensor exponent在负零和负无穷上的行为不同，分别接入并测试。Scalar-base的Half前向会舍入base，而导数使用原始Scalar的log；将静态Python Scalar的log折叠成常量并上传，Tensor导数仍在NPU计算。另测base为0、1e-300、接近1及负数，避免前向舍入导致导数错误变为0或无穷。
4. 本机CPU直接对百万长度生成Half linspace会产生917553个NaN；容量测试改为先生成FP32再转Half，原始字节复制按byte比较。没有把CPU测试数据生成问题当作NPU运算故障。
5. 旧“超容量”卷积fixture在扩大存储后不再超限，旧索引“每行两个标志下载”的断言也不再对应新packet。更新为实际超限几何和计数/校验/静态地址元数据的精确传输计数；同时保留CPU数值与梯度对照。
6. 新模块共享原后端RLock，避免session/scalar缓存被并行修改；两线程INT64操作经过真机检查。全量离线检查也确认原低位校准范围没有被无意扩大。
7. BOOL底层字节不一定只有0/1：`UINT8.view(bool)`还可带2、128、255等值。BOOL转换和sum现在复用已有、按非零判真的mask内核，而非将原始字节当成数值；新增真机精确对照。尝试在新通用C读取器中加入归一化时，Movi编译器生成了冲突的load指令，汇编阶段即拒绝；最终复用既有内核，所有源码/ELF指纹重新核验通过。

开发期失败与重复诊断没有计入41个最终通过记录。原S23近似并列排序放大训练差异、原生CPU Half与FP32-opmath卷积等已有数值边界仍保留，本轮不宣称全部修复。

## 剩余边界

- 单allocation64MiB、rank<=8及1536缓存图仍为软件边界，不能承诺在满上限上完成所有算子；运行中不淘汰图，宽Tensor和动态形状可能先碰到图缓存。任意大模型仍需更广泛分块/模型验证。
- sort仍限524288元素、次序统计单行32760、quantile仅FP32；topk和行极值保留独立旧carrier边界。新增存储dtype不自动获得所有排序/统计/神经网络算子的实现。
- 非负stride重叠读取已接通，重叠目的写入及有孔洞的empty_strided工厂仍拒绝；任意可微算术inplace、所有out/backward重载及高阶梯度未完成。
- 当前通用gather/scatter需要扫描源/目标块，可能明显慢于专用索引内核；动态nonzero计数和校验有同步。兼容优先，本轮没有做性能优化或承诺零搬运。
- BF16不是完整矩乘/卷积/AMP后端；BF16累加写入、FP64/复数和全部特殊值/溢出组合未认证。INT8原始字算术不等于DPU量化矩乘或完整低位模型质量达标。
- 更多数学/激活、loss配置、采样、稀疏优化器、异步stream/event、普通设备通用compile/直接checkpoint、GPU+NPU协同及任意CUDA扩展仍缺；详见根目录兼容性矩阵。旧低位整网质量没有改变。

## 复现

旧本地版本先重建PYD；全新源码环境使用根目录安装步骤。SHAVE C有改动时用已有MoviTools重建compat和selection内核，默认随源码提供配套ELF及manifest，无新pip依赖。

```powershell
.\build_torch_npu_device.ps1
foreach ($precision in @('float32','float16')) {
  foreach ($case in @('capacity','math','pow','index','write','gradients','expanded','model')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_compat_extended_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
  }
}
# dtype/全FP32位模式/拒绝契约各做一次
foreach ($case in @('types','trig','contracts')) {
  .\run_torch_npu.ps1 -Script .\torch_npu_compat_extended_demo.py -ScriptArguments @('--case',$case,'--precision','float32')
}
```

离线检查为64项根目录测试、32项npunlock Python API、5项源码导出测试，共101项。CI离线列表包含新增的packet/索引几何和独立排序边界保护。61枚resident内核的源码/ELF/汇编，以及C++/PYD/session DLL构建指纹均已校验；新增16枚ELF，不重建session DLL。

来源与CPU参考见根目录`REFERENCES.md`。本轮无模型下载、依赖安装、图表或磁盘图缓存；source-release与GitHub未修改。按用户约定，仅保留源码/内核/必要运行组件及报告，最终清理统计附后。

## 最终验收计数与指纹

下表每格对应一个独立真机进程的图提交数；“—”表示该专项只需一个FP32宿主进程，其中types覆盖全部9种存储dtype。

| 专项 | FP32进程 | FP16进程 |
|---|---:|---:|
| capacity | 100 | 147 |
| types | 583 | — |
| math | 112 | 213 |
| trig | 22 | — |
| pow | 132 | 235 |
| index | 713 | 722 |
| write | 441 | 446 |
| gradients | 419 | 469 |
| expanded | 915 | 914 |
| contracts | 10 | — |
| model | 1142 | 2188 |

回归计数：
| 回归 | 进程数 | NPU图提交 |
|---|---:|---:|
| 设备Quick33组 | 1 | 4171 |
| 训练Quick11组 | 1 | 2770 |
| layout-migration | 1 | 5 |
| bool-mask | 1 | 5 |
| torch-migration | 2 | 908 |
| index-pool | 2 | 6758 |
| vision-sort-extended | 14 | 28952 |

以下SHA-256标识最终本地源码与运行件，配套manifest另记录全部61个内核的源码、ELF与汇编指纹。

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `983180fad1d94239985b961df36ef669de6e4584c1686ca26e9191d02c1e0251` |
| `torch_npu_compat_extended.py` | `0922c77c951b0d0afb8d4c509db71ce66e355708fba1016cc9d39e112b1f89ad` |
| `torch_npu_compat_extended_demo.py` | `aea075ff50a354058e046425fb5c027d9150f9fc1dec547ec6b66dfe78ef50f6` |
| `npunlock/python/npunlock/npu4000.py` | `bbc0f5b38df5c806c9da341dc859eb9644657834b89ba79831c877b3d5c632ce` |
| `npunlock-probe/torch-device/backend.cpp` | `e2b01b73304ac741ac73f580f2f6024a9ea61c4015625f834f7a19a276334a59` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `84bc85770e39c8f1bf13af910a4694ff6c3a17bbb51add11f4c3d7a35992b3f0` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `b1254c2600c3bbdf74fcf3db69e495d43b41070a4518868054f72ab24ef27f31` |

## 清理记录

2026-10-23 S25容量/索引/dtype兼容性验收后清理：删除本轮227份原始JSON/日志、Movi响应及可重建backend.obj/导出表，共9443183 字节（9.006 MiB）；清理前后954份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与台账。保留64MiB/rank8 C++与重建PYD、精确整数/部分BF16及动态索引/数学源码、一阶梯度、构建/验收/离线检查、61枚resident ELF/汇编/manifest和双语文档。41个最终真机进程全部通过，共53492次图提交；101项离线检查通过。无依赖安装、模型下载、session DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未修改。

