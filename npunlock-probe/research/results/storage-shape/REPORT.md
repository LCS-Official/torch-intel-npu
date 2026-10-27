# S38：按字节分配、密集MaxPool及最大逻辑形状

本地开发版，2026-10-25，未同步GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU 8086:643e、驱动32.0.100.5540、编译器8.3；Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

## 已实现的改变

存储不再为大部分dtype统一规定268,435,456个元素，而是由`MAX_ALLOCATION_BYTES / dtype.itemsize`计算。C++工厂、Python验证和byte-DMA的slice/broadcast/permute/concat容量同步；单allocation仍明确保留1GiB，rank仍<=64。PYD已重建并校验源码/二进制指纹，session DLL及其1GiB原生边界不变，没有跨buffer分页。stats新增逐dtype元素容量；旧max_tensor_elements字段仍表示FP32基数。

| dtype | 旧单Tensor | 当前单Tensor元素数 | 当前字节上限 |
|---|---:|---:|---:|
| FP16 / BF16 / INT16 | 512MiB | 536,870,912 | 1GiB |
| BOOL / INT8 | 256MiB | 1,073,741,824 | 1GiB |
| FP32 / INT32 | 1GiB | 268,435,456 | 1GiB |
| INT64 | 1GiB | 134,217,728 | 1GiB |
| UINT8 | 1GiB | 1,073,741,824 | 1GiB |

密集普通MaxPool前向改为SHAVE内生成窗口/索引，不再逐窗口生成并搬运整张采样矩阵。新内核pool-dense-f32为4,428字节，比较逻辑强制内联，汇编构建检查禁止内部函数调用；输出坐标以计数器推进，避免逐像素整除。任务限制为32,768个采样点及32,768-word carrier；源范围、输出步长和残余数据按页面整理，普通大块输出数按512对齐以减少不同图形状。单窗口源跨度太大或参数不满足该路径时继续使用原gather路径。

反向利用几何只扫描可能到达当前输入块的窗口，保留原输出顺序、FP16每次重复累加后的舍入。它只用于autograd保存且通过version检查的前向索引；手动传入任意合法索引的ATen backward入口仍使用通用扫描，避免假定索引必须在窗口内。每个目标块61,440元素，梯度源包最多21,504条；复用原始UINT32 selection内核，并把其元数据保护扩至字节预算可能产生的范围。

流式组装从仅FP32扩展为按目标dtype写回，配对数值与INT64索引均不保留完整分块列表。普通池化、宽选择反向和极值按局部块转换精度，避免FP16单Tensor已合法但完整FP32副本会超过1GiB。显式整张转换仍检查目标dtype字节上限，不承诺1GiB Half可转成单个2GiB FP32 Tensor。

原62枚ELF保留，S37的selection-backward-u32仅扩大整数metadata保护，另新增上述pool内核，总计64枚。没有新增pip运行依赖、OpenVINO runtime、模型或session DLL构建。

## 开发阶段记录与验收口径

最初的65,536-word大任务版本虽然小案例数值正确，但出现413秒、275秒、211秒等forward墙钟异常值。只读原生栈采样显示主线程在Intel驱动同步等待；不能仅凭该栈确定所有停等的根因。随后缩小carrier/工作量、内联比较并改为坐标计数器。旧的大形状开发进程在源码迭代期间最终被任务/源范围保护拒绝，耗时1,358.627秒、11,520次提交，正常关闭且context_failed=False、live/session/quarantine为0；该结果不计入最终通过或加速比。

曾准备排队Python中断的诊断脚本，但自动审批拒绝进程注入，用户明确选择等待。没有执行注入或强制终止，旧进程自行退出；最终只保留文字记录。临时只读采样工具及宿主参考构建件在验收后清理。

热运行单独记录，同进程第2/3次的中位数包含同步提交和设备分配，排除CPU参考及下载。旧方案没有完成大形状的正式吞吐测量，因此不提供虚构的大形状实测加速比。小形状旧方案由保留的pool_dense_before.py四个S37函数复现，新增known_windows参数只被忽略。

离线套件174项（根目录137、Python API32、发布工具5）通过；随后页步长调整另由规划与宿主C检查复核。宿主参考只扩大入口和MemRef指针宽度，核对真实pool C算法，共468个几何×dtype组合、663个包，通过CPU原生MaxPool对照；另核对索引536,870,908。它不是SHAVE运行或吞吐证据。

数值/完整梯度默认rtol=0、atol=0、equal_nan=True，索引完整INT64比较；负零原始位和非法包另测。最大kernel1/stride形状用CPU分块解析参考检查每个输出、索引和输入梯度，不推导任意大核/步长组合均已认证。各真机记录保留当时源指纹；后续极值块的页对齐修改由独立回归验证，最大池化算法及内核不因此改变。

## 最终真机汇总

10个主专项、10个既有回归进程全部通过，共1,165,819次NPU提交；另2个正式旧方案对照进程通过，计入对照后22个正式进程、1,168,033次提交。开发阶段5份大包/冷启动记录不计入此数。每个正式进程均正常关闭，context_failed=False，live/cached buffer、scalar、session、图blob、factory和quarantine归零。

### 同进程冷热吞吐

257×257的3×3/stride1/padding1，每次完整CPU前向、索引、输入梯度检查。热运行取第2/3次的中位数，样本数只有2，不把小幅前向差异当作稳定加速承诺。

| 精度 | 方案 | 冷前向/反向秒 | 热前向/反向秒 | 每步NPU提交 |
|---|---|---:|---:|---:|
| float32 | S37对照 | 27.795 / 5.017 | 0.2343 / 0.0532 | 355 |
| float32 | S38 | 3.664 / 3.300 | 0.2181 / 0.0273 | 251 |
| float16 | S37对照 | 27.699 / 4.586 | 0.2625 / 0.0748 | 383 |
| float16 | S38 | 5.572 / 4.605 | 0.2401 / 0.0394 | 335 |

热反向在该小形状上约1.95×（FP32）/1.90×（FP16）；完整一步中位数约1.17×/1.21×。数据为有限本机样本，包含大量同步提交开销。

| 4097×4097密集平面 | FP32 | FP16 |
|---|---:|---:|
| 元素数 | 16,785,409 | 16,785,409 |
| 热前向秒 | 46.122 | 52.199 |
| 热反向秒 | 8.123 | 11.000 |
| 每步提交 | 63,115 | 82,326 |
| 峰值Tensor分配GiB | 0.377 | 0.252 |

两种精度均三次完整零容差CPU对照。几何规划的选择反向包数由旧算法203,841包降至1,093包；逻辑梯度扫描量由4,380,991,749降至19,022,917。这是代码规划统计，不是旧大形状实测速度比；提交、打包、状态检查和设备复制仍占明显成本。

### 最大存储及配对形状

| 用例 | 完整或边界检查 | 结果 |
|---|---|---|
| 九dtype存储 | 每种精确1GiB，首/中/末页NPU DMA写入与读回；多1元素在提交前拒绝 | 通过 |
| Half逐元素运算 | 536,870,912元素/1GiB输入，完整neg输出逐元素核对；算子阶段22.752秒 | 通过 |
| FP32最大配对池化 | 输入268,435,456元素/1GiB，kernel1/stride2；输出134,217,728，INT64索引1GiB；完整输出/索引/输入梯度核对 | 通过 |
| FP16最大配对池化 | 输入536,870,912元素/1GiB，kernel1/stride4；输出134,217,728，INT64索引1GiB；完整输出/索引/输入梯度核对 | 通过 |

| 最大配对池化 | FP32 | FP16 |
|---|---:|---:|
| 前向秒 | 74.983 | 142.242 |
| 反向秒 | 61.082 | 110.324 |
| NPU提交含完整读回检查 | 210,246 | 472,633 |
| 峰值Tensor分配GiB | 4.008 | 3.508 |

上述峰值来自后端分配计数，不包括所有驱动图对象、编译器或CPU参考，不能视为整机RAM峰值。最大形状为指定kernel1/stride的容量认证，不表示同样大小任意3×3/大核网络已完成认证。测试保留至少3GiB物理/commit余量。

### 既有回归

| 入口 | FP32 / FP16提交 | 覆盖 |
|---|---:|---|
| pool_topk / edges | 188 / 275 | 负零梯度原始位、NaN/并列、Half重叠、索引version |
| vision_sort_extended / pooling | 347 / 491 | 每精度21组3D/自适应/平均池化、Unpool |
| conv_capacity / workspace | 718 / 735 | 完整卷积输出和输入/权重/bias梯度 |
| pool_capacity / training | 294 / 466 | 205通道池化模型两步SGD，全部参数梯度和更新 |
| selection_capacity / extrema | 14,128 / 16,326 | 16,777,219长轴极值、完整梯度及NaN首索引 |

最后一项验证了末次极值源块页对齐修改。部分早期主专项的torch_intel_npu.py指纹仅在该未执行的极值分块语句上不同，已按字节重构旧语句并核对原SHA；池化模块、内核及存储实现指纹保持一致。最终174项离线检查在全部修改后重新运行通过。

## 复现

```powershell
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script torch_npu_pool_dense_demo.py -ScriptArguments @('--case','small','--precision','float32','--baseline')
.\run_torch_npu.ps1 -Script torch_npu_pool_dense_demo.py -ScriptArguments @('--case','small','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_pool_dense_demo.py -ScriptArguments @('--case','large','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_storage_shape_demo.py -ScriptArguments @('--case','storage','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_storage_shape_demo.py -ScriptArguments @('--case','pointwise','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_storage_shape_demo.py -ScriptArguments @('--case','maximum','--precision','float16')
```

另一精度将precision替换即可；全部NPU进程串行。宿主C检查可通过verify_pool_dense_host.py的--emit-source生成临时C，使用tools/msvc_environment.ps1中的既有编译器以/LD /O2 /TC构建pool_host.dll，再执行该核对脚本；只改变宿主指针宽度，不用于性能对照。原始JSON归纳后清理，保留复现源码和报告。


## 边界

- 1GiB单buffer和物理RAM预算仍存在；跨buffer分页、任意14GiB整网训练未在本轮认证。
- BF16/整数有完整字节存储不等于完整NN/AMP/整数算子支持，其他算子仍各有专用工作区或dtype限制。
- 未认证所有最大shape、全部布局、任意kernel/dilation/ceil组合；原生手动索引的通用反向可能仍慢。
- 提交仍同步；热吞吐、冷编译、驱动等待和数据搬运分开记录，不声称达到NPU硬件TOPS峰值。

## 最终源与组件指纹

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `f853e056cbfaaa9e0862ac33b7c6b8392f91d1dcc16db2146d62bc326cb8dcb0` |
| torch_npu_geometry.py | `b097a4bd0d066041f528c6cf7985b19d5b8bb2f7e767f9484ea6dd3e2de51540` |
| torch_npu_autograd.py | `09ec6aea68b7c410378243b8324d2bc1b6803f4691e48241b67a4d4c3d40dff2` |
| torch_npu_pool_dense.py | `93fde28c04501822566233b49d45cf0343985c1e2e5ad2d207736f612358b4ce` |
| torch_npu_pool_dense_demo.py | `fb5a740e6690c9d6692390769c726f8911ae42574b1d8ea6e6e9891f80faf00f` |
| torch_npu_storage_shape_demo.py | `d8121c9f67bb227df44d618d04a47135acb1cddc0393b72b95228b941ba5e2ce` |
| test_torch_npu_pool_dense.py | `e770901623563e8a600d55780b53ef3feae47ce2ba5f448d81d69c242977f3cc` |
| test_torch_npu_conv_capacity.py | `ed5a5cc8ff70aa5da65b1c0e0a926ad693cc4c6b0c1ea0b984d55cfc8550b9cc` |
| test_torch_npu_pool_capacity.py | `11be31571092f69646ff8ef66593d558990e15bd84f59b33f25cb6b187d4114a` |
| test_torch_npu_selection_capacity.py | `8760512f7afc2dea8823bbabc58062aaa4004e02d109de1373f64d7ecb16e30d` |
| torch_npu_selection_capacity_demo.py | `3ad3ffc116a56937cf3f50c4ab834c6a4074eeb69fc0f447cb71dad164e71bb2` |
| npunlock/python/npunlock/npu4000.py | `a88338f78deb4b091e6048b40671e142ee8a9a3aeb61679dc330f04fe286a741` |
| npunlock-probe/torch-device/backend.cpp | `9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `433757698bbc704a44da3ca9c3c3355d3fac49ab19b82d06f0d09790aa4c6ea0` |
| npunlock-probe/torch-device/build.json | `c41e55c5c3044c45084e74c9ed25b094832aa86637b14fcaa2326b6827a3171f` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock/tests/fixtures/npu4000/resident_selection.c | `c05ab1156907478bda701fb01cf219bf52d0f478c450e203f54f1fdf7cc0960a` |
| npunlock/tests/fixtures/npu4000/resident_pool_dense.c | `8e3ae5e180c4261fe1188243a9d71ece5cb6ee701a304ca49c2ddb0c5ba3b535` |
| npunlock/tests/fixtures/npu4000/resident/pool-dense-f32.elf | `8d0836327bc408151300ea2397f78ef9bcb288540d46531c3720dd5138618da4` |
| npunlock/tests/fixtures/npu4000/resident/pool-dense-f32.s | `a26ad8dd01249866de6b3fd77ef69513160fcd9f415357da5214ee99b9827f0c` |
| npunlock/tests/fixtures/npu4000/resident/selection-backward-u32.elf | `6ea053fcb19be594870acab628c2e8dd77d169ed9cefea354c95f279a8dcc01a` |
| npunlock/tests/fixtures/npu4000/resident/selection-backward-u32.s | `4c0c7c796c7c6a00d1603bc5099ce5ab25e79d1452d6e1f754ec1c6a1fe24905` |
| npunlock/tests/fixtures/npu4000/resident/manifest.json | `d7b35afdcd6270c435f1b644af2c3fec5d3e0e9a16cb44c8860a3316356eb6bb` |
| npunlock-probe/research/pool_dense_before.py | `04e4ad294e3e46d98a787eeafe16a365d65d55de210ac2c0b9510b267bdf7d2a` |
| npunlock-probe/research/build_pool_dense_kernel.py | `5a7792f3a213b229654d748713ddb2f95cb249289eabd101fa9630173b7a6b89` |
| npunlock-probe/research/verify_pool_dense_host.py | `2748a4b896368a0f5cdb07783e5bda8fa9ffdf86533afcd364a773f67fe25a77` |
| .github/workflows/offline.yml | `d54026b2ed110dbae170c902bdf4f9f25920f2b5457c57afa0a6485eb5444f85` |

## 清理及交付记录

全部测试/诊断进程退出后，删除74个本轮原始结果、构建响应/日志、可重建backend.obj/导出表和临时采样/宿主参考工具，共14,765,213字节（14.081MiB）。其中46份JSON包含22份正式真机记录、5份开发记录、1份宿主C核对和18份构建阶段记录；失败的开发结果已在上文归纳，不计入最终通过。没有删除历史Markdown报告。

逐项核验工作区内绝对路径，拒绝联接/符号链接和保留文件；清理前后522份选定源码、报告、内核及必要构建组件SHA-256相同，随后追加本段及空间台账。临时进程注入脚本在未执行状态下移除，所有诊断二进制已清理。保留实现、64组ELF/汇编、26份来源指纹、更新的PYD/构建记录、对照/复现/检查源码与报告；无模型、磁盘checkpoint、图表或项目持久图缓存。公开source-release未改，未推送GitHub。
