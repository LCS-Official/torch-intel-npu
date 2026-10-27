# S31：损失函数分块与算子专用容量

日期：2026-10-24。本地开发版，未同步GitHub。Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。沿用S30的512MiB单allocation、rank64及1536图缓存；没有增加依赖、OpenVINO runtime、SHAVE内核或宿主构建。

## 改进范围

| 项目 | 原限制 | 当前实现与实际验证 |
|---|---|---|
| 硬标签CE/NLL | 整批 `rows*(classes+4)<=65536` | 按完整类别行分块，每个packet仍<=65536；总输入按当前存储界限检查。FP16/FP32加权/平滑分类、原生反向out及分割训练有独立CPU对照 |
| NLL训练保存状态 | 保存整批稠密类别mask | 多packet批次只返回空mask标记，autograd保存原标签/权重/total等；反向重建有界mask，并保留版本保护 |
| 高样本数NLL | 每个切片偏移可能增加图 | 多packet且行块足够大时，以1024行为单位对齐，FP32输入及INT64标签复用范围图；mean梯度统一缩放一次 |
| 普通Conv2d输出 | 16,777,216元素 | 检查扩大到134,217,728元素；真机验证16,778,240元素输出及完整输入/权重/偏置梯度。窗口与fold的浮点元数据保护不变 |
| 原生Transformer工作区 | MHA score/projection、encoder FFN沿用16,777,216元素 | 改用134,217,728元素的普通存储检查；FFN实际覆盖16,793,600元素。MHA大score只有元数据正反边界测试，不冒充全量大attention认证 |
| 原生Half掩码softmax | 默认最多1,048,576元素 | 调用当前存储界限的完整行分块，保留CPU Half舍入路径；真机覆盖1,049,600元素及全屏蔽行 |
| 多输出纯DMA拼接 | 旧16,777,216元素检查 | 改用DMA存储界限；两个16,777,233元素输出逐字节对照CPU |

136项离线检查通过（根目录99、Python API32、发布工具5），包括本轮新增的3项损失与5项算子容量检查。CPU meta用例只验证形状/拒绝逻辑，不代表实际申请了上限内存。各专项数值、提交、缓存和指纹汇总见下方。

## 损失函数实现与数值口径

`nll_row_tiles`保持完整类别行，每行四个额外word携带INT64标签/ignore_index及校验结果。非空类别行最多65,532类；空batch不需要packet。单packet继续使用原C/ELF，没有盲目增加MemRef维度或SHAVE计算块。

前向逐块生成mask及所选loss/weight，FP32行结果拼接后按整批计算total和分子；保留S29的Half乘积、分子和mean除法舍入边界。所有packet非法标签标志在设备归约，整次非空前向仅下载一份4字节标志。原始标签、logits和梯度没有CPU计算回退。

多packet反向从已校验且保存版本的标签重建mask，不长期保存整批one-hot。手动native backward重新验证标签，也只下载一份标志；autograd利用保存的标签版本，不重复下载。输出梯度仍会分块拼接，未实现跨buffer分页或完全流式反向组装。

FP32门槛为rtol=4e-5、atol=2e-6，Half为rtol=.004、atol=.001，equal_nan=True；不放宽已有门槛。百万行mean梯度很小，额外检查全梯度相对L2<=1e-4，防止绝对容差掩盖缺失梯度。纯DMA逐字节比较。

小分割网络为Conv2d(2,17,1)，输入2×2×33×65，每步4290个标签；独立CPU/NPU两步SGD，比较loss、所有参数梯度与更新参数，Half模型显式采用FP32损失。CPU oneDNN不支持本机该Half卷积反向，参考前后向关闭MKLDNN，使用原生CPU实现；该开发期失败来自CPU参考端。

收尾修复连续空间标签的不必要clone：现在直接复用contiguous视图，减少副本并保留原标签版本。新增CPU断言先复现修改原标签未被检测的问题，再修复并验证FP16/FP32真机版本检查。非连续空间标签仍需要复制，其原始来源版本追踪尚未补齐；不将这类布局的版本语义写成完整兼容。

## 专用工作区的边界

普通Conv2d的输出、原生MHA/encoder的稠密投影和FFN，以及纯DMA拼接，后续已经委托有界矩乘/逐元素/DMA，所以可沿用存储界限。保留类别行宽、softmax行宽、索引packet、Embedding、卷积窗口/fold、排序/selection及其他浮点索引元数据的独立边界；没有统一把这些常量全部扩大。

512MiB和134,217,728是软件检查上限，不是每个算子在极限形状上的认证。普通Half逻辑Tensor最多256MiB，计算中间量往往为FP32；大batch、许多宽类别行、不同形状或不对齐偏移仍可触及1536张图、物理内存或专用工作区。原生encoder仍仅支持已列的推理入口。没有认证14GiB训练、任意大模型、长期Half收敛或GPU/NPU协同。

本轮扩容主要针对硬标签CE/NLL；软标签CE和MSE继续使用S29/S30的广播与存储范围。BCE/KL/CTC等完整损失家族未在本轮新增。

## 复现入口

```powershell
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','tiled','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_loss_extended_demo.py -ScriptArguments @('--case','tall','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_operator_capacity_demo.py -ScriptArguments @('--case','conv','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_operator_capacity_demo.py -ScriptArguments @('--case','encoder','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_operator_capacity_demo.py -ScriptArguments @('--case','masked','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_operator_capacity_demo.py -ScriptArguments @('--case','dma','--precision','float32')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_losses test_torch_npu_operator_capacity
```

所有NPU进程串行。每个进程记录原始结果、复制计数、缓存/峰值、关闭状态与源码SHA；汇总后按用户要求清理原始JSON/日志，保留报告和可复现代码。时间包含编译、CPU参考、完整回读和关闭，不是隔离内核吞吐基准。实现复用本仓库原内核，CPU PyTorch只作参考；没有复制外部内核代码。

## 真机验收汇总

24份专项/回归记录全部通过。按子功能推进的记录对应不同源码阶段，表中列整体后端SHA前缀；算子容量测试后仅迭代NLL对齐/缩放及连续标签复用，没有修改算子本体或C/ELF。最终连续标签修复另外通过两种精度tiled案例和完整99项根目录离线检查，不将所有旧回归写成同一源码版本全量重跑。

| 案例 | 精度 | NPU提交 | 缓存图 | 峰值 MiB | 总耗时 秒 | 后端SHA前缀 |
|---|---|---:|---:|---:|---:|---|
| contracts-float16 | float16 | 118 | 40 | 0.001 | 5.103 | 1d759e9aa93f |
| contracts-float32 | float32 | 78 | 30 | 0.001 | 4.057 | 1d759e9aa93f |
| hard-float16 | float16 | 5,754 | 144 | 0.001 | 19.785 | 1d759e9aa93f |
| hard-float32 | float32 | 3,292 | 116 | 0.001 | 15.619 | 1d759e9aa93f |
| mse-float16 | float16 | 866 | 42 | 0.000 | 5.959 | 1d759e9aa93f |
| mse-float32 | float32 | 490 | 28 | 0.000 | 4.125 | 1d759e9aa93f |
| native-float16 | float16 | 1,066 | 76 | 0.002 | 9.826 | 1d759e9aa93f |
| native-float32 | float32 | 763 | 58 | 0.002 | 8.030 | 1d759e9aa93f |
| soft-float16 | float16 | 2,585 | 81 | 0.000 | 11.334 | 1d759e9aa93f |
| soft-float32 | float32 | 1,198 | 44 | 0.000 | 6.203 | 1d759e9aa93f |
| tall-float32 | float32 | 2,026 | 43 | 56.275 | 8.790 | 1d759e9aa93f |
| tiled-float16 | float16 | 7,087 | 327 | 3.273 | 49.352 | ba421e174aa7 |
| tiled-float32 | float32 | 5,779 | 246 | 3.513 | 37.421 | 1cbed14fe7aa |
| training-float16 | float16 | 672 | 101 | 0.002 | 12.997 | 1d759e9aa93f |
| training-float32 | float32 | 468 | 82 | 0.002 | 10.914 | 1d759e9aa93f |
| conv-float16 | float16 | 8,262 | 632 | 256.623 | 101.199 | 1d759e9aa93f |
| conv-float32 | float32 | 4,653 | 621 | 256.902 | 98.598 | 45a95a38ee22 |
| dma-float32 | float32 | 1 | 1 | 192.000 | 0.419 | 45a95a38ee22 |
| encoder-float32 | float32 | 6,097 | 671 | 192.656 | 124.257 | 45a95a38ee22 |
| masked-float16 | float16 | 565 | 40 | 19.050 | 8.040 | 45a95a38ee22 |
| torch-npu-device | 双精度/组合 | 4,171 | 295 | 1.003 | 45.086 | 未记录整体SHA |
| torch-npu-training | 双精度/组合 | 2,770 | 323 | 9.563 | 44.618 | 未记录整体SHA |
| container-float16 | 双精度/组合 | 7,139 | 0 | 0.007 | 0.000 | 1d759e9aa93f |
| container-float32 | 双精度/组合 | 3,838 | 0 | 0.008 | 0.000 | 1d759e9aa93f |

合计69,738次NPU提交；各进程shutdown后live buffer、空闲buffer、scalar、session和quarantine归零，context_failed=False。复制计数覆盖显式Tensor复制，不包含编译器/驱动内部DMA。

百万行NLL：1,048,593行，3,145,779输入元素，114个packet；43张缓存图，峰值56.275MiB；全梯度相对L2误差为0.0。前向+反向只下载一次校验标志，之后完整下载结果作独立CPU比较。

两种精度的tiled案例覆盖三种reduction、加权/平滑CE、空间布局、全忽略标签、未选中NaN/Inf权重、65532类上边界、末块非法INT64标签、native backward out、标签/权重版本与两步独立分割训练。普通Conv2d在FP32/FP16均完整比较16,778,240元素输出和三类梯度；encoder大FFN为FP32推理输出认证，未做同规模Half认证。

## 保留产物与指纹

新增优化为项目内Python编排与保护检查；无新增SHAVE C/ELF或宿主DLL/PYD构建。核对25份SHAVE源及62对ELF/汇编与原manifest一致，现有DLL/PYD的源/header/构建输出指纹有效。source-release Git工作树保持干净，未推送GitHub。

- torch_intel_npu.py：`ba421e174aa76c762b5ef7f405413f3c51d5e3da92e7463c7984fd8169512e28`
- torch_npu_autograd.py：`a0831dc63ef447f8e25dbc232a9d0d6df029d19e12c5410292ad3747c62a5c6b`
- torch_npu_geometry.py：`6b99bb12e83fa9573ec8710ab5cf9ff2f3a5fc29e87785c8a342c10349619ab1`
- torch_npu_loss_extended_demo.py：`6abdc88ec31a39e95508d19ff108893712abd4d2b32a63ac94764d20ba809be1`
- torch_npu_operator_capacity_demo.py：`83bc03b76fbbd1954a0179565e7139afdd0417ed05119765ba27668e64161fd1`
- test_torch_npu_losses.py：`02ba2c2c0ac571661ede73c7544686155befbb4f3e854feb26bfa2589be776ab`
- test_torch_npu_operator_capacity.py：`028ed37004158bc1c6af70a939a22c2403f7dce0bcbf3a2ff973a176b65e07eb`

resident manifest仍为`329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc`。

## 清理记录

2026-10-24 S31验收后清理：删除本轮55份原始JSON/日志，共250573 字节（0.239 MiB）；清理前后992份保留源码、报告、内核及必要运行时SHA-256一致，随后追加报告及本台账。保留硬标签损失分块/反向重建/标签复用、稠密工作区容量检查、验收及8项新增离线测试、CI清单和双语说明。24份真机记录通过，共69738次提交；136项离线检查通过。原62枚ELF、DLL/PYD不变，无新增依赖、模型、图表、磁盘checkpoint或图缓存；source-release及GitHub未修改。边界与数值口径见operator-capacity/REPORT.md。
