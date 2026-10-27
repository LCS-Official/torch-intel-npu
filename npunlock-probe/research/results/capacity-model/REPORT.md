# S26：扩大大Tensor与模型支持

日期：2026-10-23。本地开发版；Windows x64、Core Ultra 7 258V / Lunar Lake `8086:643e`、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。

本轮针对“大Tensor／大模型”扩展支持范围。17个最终真机验收进程全部通过，共57,484次图提交；105项离线检查及两个原生故障/链测试可执行程序通过。它仍有软件容量、图缓存和算子专用限制，不表示任意大模型或完整CUDA兼容。

## 已实现

| 项目 | S25 | S26 |
|---|---|---|
| 单allocation上限 | 64MiB | 256MiB；C++ allocator和native session参数检查同步扩大 |
| 逻辑存储rank | 0–8 | 0–64；点算与普通Linear/softmax入口同步使用新rank检查 |
| FP32/INT32元素 | 16,777,216（64MiB） | 67,108,864（256MiB） |
| FP16/BF16/INT16元素 | 16,777,216（32MiB） | 67,108,864（128MiB） |
| INT64元素 | 8,388,608（64MiB） | 33,554,432（256MiB） |
| BOOL/INT8元素 | 16,777,216（16MiB） | 67,108,864（64MiB） |
| UINT8字节载体 | 67,108,864（64MiB） | 268,435,456（256MiB） |
| 普通softmax/log_softmax | 总元素<=1,048,576，rank<=5 | 总元素检查<=67,108,864，rank<=64；每行仍<=65,536，完整行分块不变 |
| scalar pow(x,1)/pow(x,2) | 通用4096元素packet | 复用已有DMA clone与分块乘法；平方反向的pow(1)也走新路径 |

FP32 scalar exponent±0.5、-1原有特殊值快捷路径保持不变。

复制、转换、广播、普通FP16/FP32点算和sum/mean使用扩大的存储范围；Linear/矩乘输出检查同步更新。SHAVE任务仍最多65,536元素，没有重建或新增SHAVE C/ELF。本轮扩大native shared buffer及图参数允许的字节数，不改变buffer结构、同步、绑定和失败隔离机制；图绑定仍须完整连续allocation。

## 真机专项及判据

- 九种dtype各检查逻辑边界allocation、超限拒绝，以及末端原始字节DMA读回；FP32另实际上传完整256MiB，并比对首尾带已知值和负零的设备读回。不是仅申请地址或只测CPU读取。
- 64维张量的clone、转置整理、点算/归约反向、Linear及输入/权重梯度、softmax，与CPU独立对照。
- 16,777,233元素张量的点算、平方、FP32 sum及一阶梯度、精确clone；FP32输入超过旧64MiB。使用FP32 loss，避免大Half sum溢出或mean产生全部下溢的梯度。
- rank9、4097×256共1,048,832元素的softmax/log_softmax及梯度，超过旧百万元素上限；另验证65,537列仍在提交前拒绝。没有把这个用例称作最大上限softmax认证。
- 普通`nn.Linear(4096,4160)`共17,043,520参数；FP32单权重65MiB，超过旧单allocation64MiB；FP16权重32.5MiB，超过旧逻辑16,777,216元素。两步MSE/SGD训练分别使用独立CPU/NPU模型，比较loss、全部梯度及全部更新参数，不在步骤间重置参考模型。

浮点比较沿用S25：FP32 rtol=4e-5/atol=2e-6；FP16 rtol=.004/atol=.001。指定复制按原始字节精确比较。测试不是长期训练或实际数据集质量认证，不评估CPU/GPU/NPU速度排名。

## 开发诊断

1. 运行期间销毁再导入图曾导致设备丢失，因此不通过运行中LRU淘汰或无限增大图缓存来绕过限制。当前仍最多1,536缓存图，进程shutdown后统一释放。大模型两步均记录缓存数量，检查固定形状没有持续增长。
2. 旧通用pow每4096元素一个packet，会为较大张量产生过多偏移图。平方与恒等幂改为复用已验证的分块乘法/复制，不新增内核代码；既减少图种类，也让大张量平方及其反向能在当前缓存预算内完成。恒等幂仍返回独立存储，out及特殊值另有检查。
3. 初次大Linear的SGD lr=.03时，FP16两步loss显示相同，虽然CPU/NPU梯度和更新对照通过。该学习检查判为未通过；最终合成训练用例统一采用lr=3，两种精度都保留原数值容差和下降断言重新验收，不放宽门槛。
4. 旧selection backward使用FP32 header表示全局计数/地址，C内核限制16,777,216。扩大普通存储后，这条路径必须在Python侧明确拒绝超限几何，防止地址舍入或内核无输出。新增离线检查在任何NPU执行前捕获该情况；冻结DPU/其他旧工作区也保留独立原边界。
5. 旧BOOL→INT64负向fixture仍用S25的16,777,216个BOOL元素，S26下该尺寸不再因输出容量被拒；实际执行后触发旧用例的“无提交”断言失败。该进程共4,602次图提交，退出context健康、资源和quarantine归零。最终改为INT64逻辑上限+1个BOOL元素，重新验证容量错误在提交前拒绝。大尺寸通用转换仍受图缓存预算影响，未认证全部存储上限的转换。

## 仍有的容量边界

- 单allocation256MiB、dtype元素数与rank64仍是保护边界。没有实现大于单allocation上限的分页Tensor/跨allocation虚拟存储，也没有承诺跑满设备共享内存。
- 图缓存1536仍可能在动态形状、很多不同偏移或通用小packet算子中先耗尽。没有通用安全图淘汰机制；整个模型的活跃存储、临时缓冲和图资源仍受系统内存限制。
- 普通softmax扩展不自动扩大SDPA的完整attention矩阵或冻结KV容量。卷积/池化、Embedding、排序/统计、部分索引反向和其他FP32元数据内核各自保留原界限。
- 很宽K的矩乘可能因完整partial结果堆叠等临时工作区先超限；本轮只认证已列形状。高rank存储不表示全部算子均接受64维。
- 本轮没有优化计算吞吐、扩展DPU训练、补齐更多dtype或低位整网质量，没有对任意ResNet/Transformer、长时训练或全部上限形状给出保证。

## 复现与来源

```powershell
.\build_npu_session.ps1
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_capacity_demo.py -ScriptArguments @('--case','storage','--precision','float32')
foreach ($precision in @('float32','float16')) {
  foreach ($case in @('rank','pointwise','softmax','model')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_capacity_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
  }
}
```

CPU测试、native失败/链回归、既有模型/索引验收和最终指纹在下文记录。native DLL重建后必须重建PYD，使session指纹一致。没有新依赖、模型下载、图表或磁盘图缓存；source-release/GitHub未修改。参考与授权范围见根目录REFERENCES.md、THIRD_PARTY_NOTICES.md；复用本项目既有Level Zero及byte-DMA路径。

## 最终验收明细

新增9个进程共43,525次图提交：

| 专项 | FP32提交 | FP16提交 |
|---|---:|---:|
| storage | 11 | — |
| rank | 35 | 70 |
| pointwise | 5191 | 13236 |
| softmax | 174 | 592 |
| model | 8406 | 15810 |

storage进程覆盖全部9种dtype；其余4个专项分别检查两种精度。上限处申请/读写与常见计算形状分开验收。

| 大Linear精度 | 两步loss | 参数数 | 单权重bytes | 两步结束缓存图 | 峰值live bytes |
|---|---|---:|---:|---:|---:|
| float32 | 0.01302294246852398 → 0.011508163996040821 | 17043520 | 68157440 | 551 | 409010960 |
| float16 | 0.0130157470703125 → 0.011505126953125 | 17043520 | 34078720 | 824 | 477135246 |

缓存图在两步结束时与第一步相同（FP32为551、FP16为824）；live bytes仅为本后端跟踪值，不包含驱动/编译器/CPU参考及全部进程内存。

另8个回归进程共13,959次提交：

| 回归 | 进程数 | NPU提交 |
|---|---:|---:|
| 设备Quick33组 | 1 | 4171 |
| 训练Quick11组 | 1 | 2770 |
| 布局/布尔契约 | 2 | 10 |
| S25幂/特殊值/梯度 | 2 | 358 |
| S20索引/FP32初始化累加与梯度 | 2 | 6650 |

以上17个最终记录的shutdown后live/cache/scalar/session/quarantine均归零，context_failed=False。开发期lr=.03探针、过时fixture及重复诊断不计入该最终计数。

离线检查为根目录68项、npunlock Python API32项、源码导出5项，共105项。新增4项保护较大DMA边界、调用者可控的完整行计划、scalar pow(1)/pow(2)的值和独立存储，以及旧selection元数据上限。native两个测试程序覆盖安全失败、隔离、rebind和链的成功/拒绝/失败恢复；测试使用mock，不计入真机进程数。

已核验重建session DLL的编译源码/头文件及DLL/PYD指纹；既有61个resident内核的源码/ELF/汇编与manifest一致，无新增SHAVE构建。

| 最终本地文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `3ae7db4055119639c0e5067a5180a7d5591a98d61980376dd4151589fe54a0dd` |
| `torch_npu_compat_extended.py` | `41c49ec601291decd68e330f66eea7475b9dc13f537cd34ab1fbb925ef1b438b` |
| `torch_npu_capacity_demo.py` | `002a101957d6ce67841f7382da2145e7b39836972e02e5cb302ba98ce13a62b2` |
| `torch_npu_geometry.py` | `383ebde84929ef4b9fe2e5580ef555d92cf457c9496bf6f8bbf6b86384c17c3f` |
| `test_torch_npu_capacity.py` | `22e1671b4fbb07eb4486af060a37165a8a451832bf699600c4917922b9d64f6c` |
| `npunlock/python/npunlock/npu4000.py` | `7295f5cfb00c1d685f0adf307edaa8c4411b867f3468eb5f60ae0cc8831384ee` |
| `npunlock/src/graphinfer/session_win32.c` | `9cd08d6def3d9e5320453f80e585af3627f5eb0e665fe801f3eac96979f2538f` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `4b9378aa5d028c17e8db545e566f2eabe3d9a13adf19ae0971a4a0a01b50eee3` |
| `npunlock-probe/torch-device/backend.cpp` | `02f09c05054382364169de1cf91f47851ce29566b5e6ba31c9f890d20a584f35` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `e2852223d84081333485066cbbaad5152c5ad70afcb185216ce388bc39e6e666` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `b1254c2600c3bbdf74fcf3db69e495d43b41070a4518868054f72ab24ef27f31` |

## 清理记录

2026-10-23 S26大Tensor/模型支持验收后清理：删除本轮44份原始JSON/日志、可重建backend.obj/导出表和已改动模块的bytecode，共8158505 字节（7.781 MiB）；清理前后957份保留源码、报告、内核与必要运行时文件SHA-256一致，随后补充本段与台账。保留256MiB/rank64存储与DMA源码、重建session DLL/PYD及构建指纹、较大点算/softmax/1704万参数训练验收、4项新增离线保护、双语说明及原61个resident ELF/汇编/manifest。17个最终真机进程通过，共57484次提交；105项离线检查及原生故障/链回归通过。未新增依赖、模型下载、SHAVE构建、图表或磁盘图缓存；source-release和GitHub未修改。

