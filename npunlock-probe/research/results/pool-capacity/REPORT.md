# S36：普通MaxPool、规则平面反向与极值容量

本地开发版，未同步GitHub。平台为Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。本轮验收跨会话日期更新继续执行，表中时间为各进程计时，不用跨日时长推导性能。

## 生产变更与边界

| 部分 | 旧行为 | 当前行为 |
|---|---|---|
| 普通MaxPool1d/2d空间输出 | 每平面最多524,288 | 每平面最多16,777,216，仍按原包计算 |
| 普通MaxPool总采样量 | 超过8,388,608直接拒绝 | 不以总采样量拒绝；总输出仍受数值/INT64存储，窗口面积<=32,765 |
| 前向包 | 每个通道平面单独分包 | 包可跨完整/部分平面，输出索引仍相对于对应空间平面 |
| 池化及规则行选择反向 | 整个源/梯度总量须进入16,777,216元数据范围 | 按完整平面分组，组内沿用原C内核；单平面的输入/选择数仍<=16,777,216 |
| 浮点行极值 | 输入最多1,048,576元素，C offset也限制在该范围 | 输入保护16,777,216；超旧offset的分块用局部索引，再通过已有设备INT64加法恢复全局位置 |
| 无用输出梯度 | 不可微索引默认仍实体化零梯度 | MaxPool、TopK、sort/max/min索引输出及扩展MaximumPool关闭该实体化；backward已忽略相应参数 |
| 几何与读取 | 合法性检查枚举整个轴；窗口源可能触发大Tensor不对齐Slice编译 | 仅枚举左padding边界，内部起点已合法；窗口源范围向页边界扩展并继续限制<=65,536元素 |

本轮不增加存储/allocator上限、SHAVE carrier或C元数据上限，也不重编任何ELF。普通池化总配对输出最多134,217,728元素（当前INT64存储限制）；输入仍按原dtype存储。FP16逻辑Tensor最多512MiB、FP32最多1GiB。每个空间平面的反向仍需独立放进元数据界限；很大的单平面推理不等于可训练，多平面扩容不等于任意扁平高级索引/scatter扩容。

分组保持每个平面内的原有顺序，不打散重复项：MaxPool的Half逐次累加舍入，sort/topk的不累加赋值模式保留负零。只需一组时直接调用原入口；其他组逐个产生完整局部结果，借S35的FP32流式目标组装，再转回原dtype。原索引version guard仍在数值反向前检查；关闭无用索引梯度不会改变公开返回值。

普通MaxPool增加正通道的空batch前后向，拒绝零通道。3D/自适应MaximumPool只复用规则平面反向与读取逻辑，前向静态采样量、窗口和存储界限不在本轮放宽。AveragePool、插值、Unpool及一般扁平索引不是本轮全面扩容对象。

## 正确性与限制口径

专项的算子输出、索引与完整梯度使用CPU原生PyTorch参考，`rtol=0,atol=0,equal_nan=True`；这是零容差数值相等，不单列NaN payload/正负零编码认证。负零另由既有原始位梯度回归与分组CPU原始位测试核对。训练保持FP32 rtol=4e-5/atol=2e-6，FP16 rtol=.004/atol=.0005。

初期FP16空间输出用例在`ir2blob.compile`阶段出现一次旧大源不对齐Slice编译timeout；该进程正常关闭，context_failed=False、quarantine/live/session为0。改用页对齐范围后重跑，不能把编译超时记录当作数值错误或硬件设备丢失，也不宣称彻底消除了驱动所有编译问题。

窗口对齐在连续且storage_offset=0的源上启用；源区间包含原窗口、范围仍<=65,536元素，所有相对索引同步调整，-1 padding保留。其他布局仍由原路径处理。已有C/ELF验证、设备profile和同步生命周期不变。跨平面前向仍保留若干结果列表，整体布局、索引缓冲和CPU参考不包含在单包预算中；不是总内存硬上限。

## 真机专项结果

最终生产源码上12个独立进程全部通过，FP32/FP16各6个；每个都完成正常shutdown，context_failed=False，live buffer、缓存buffer/scalar、session、图blob、工厂及quarantine均归零。

| 用例 | 实测几何与验证范围 | FP32 / FP16 NPU提交 | FP32 / FP16峰值设备buffer MiB | FP32 / FP16进程秒 |
|---|---|---:|---:|---:|
| contracts | 空batch池化前后向、非法高位INT64索引拒绝后恢复、saved index版本错误不提交数值图、topk k=0反向 | 38 / 48 | 0.002 / 0.002 | 2.839 / 3.115 |
| spatial | `(1,1,1,524289)`，kernel/stride=1；完整输出及索引，推理测试 | 372 / 435 | 16.000 / 15.000 | 15.359 / 16.366 |
| extrema | 长轴1,048,593，超旧offset位置的并列值/NaN，max/min/argmax/argmin；另测`(524289,3)`完整值/索引/梯度 | 1,303 / 1,960 | 26.000 / 24.000 | 32.654 / 36.991 |
| sampled | `(1,4,513,513)`，kernel3/stride1/padding1，总采样9,474,084；完整值/索引/输入梯度 | 6,857 / 7,244 | 40.230 / 38.222 | 250.271 / 252.996 |
| planes | `(8193,1,32,64)`，kernel2/stride2；源16,779,264元素，输出4,194,816元素；完整值/索引/输入梯度 | 10,623 / 13,742 | 208.049 / 224.051 | 30.182 / 47.708 |
| training | 205通道bias + MaxPool + FP32均值 + Linear；与独立CPU模型对照两步SGD，全参数梯度及更新 | 320 / 480 | 2.468 / 2.336 | 15.667 / 19.783 |

训练loss：FP32 `0.0176839251 → 0.0170685854`；FP16 `0.0176782925 → 0.0170133468`。这是两步集成检查，不推导长期收敛。`spatial`未测该形状反向；`planes`实测跨越旧整源16,777,216元数据限制，但每个平面仍为2,048元素。软件最大空间/总输出形状没有全部实测。

峰值来自本后端`peak_buffer_bytes`，不是任务管理器共享内存、进程RAM或整机峰值。时间包含图编译、CPU参考、完整回传和关闭，不是热内核延迟。重叠窗口的`sampled`仍需约四分钟及数千次同步提交；两种精度缓存都达到1,536项，发生332/350次安全图淘汰。这说明容量和缓存生命周期检查通过，不代表吞吐已优化。

## 既有路径回归

以下各用FP32/FP16独立串行运行，共8个进程全部通过；正常关闭后的资源计数同专项归零。与12个专项合计20个正式进程、50,295次NPU提交。

| 入口/用例 | 覆盖 | FP32 / FP16 NPU提交 |
|---|---|---:|
| pool_topk / edges | 原始位负零TopK梯度、末NaN池化、并列值、重叠Half累加、saved index版本检查 | 188 / 275 |
| vision_sort_extended / pooling | 每个精度21组3D/自适应池化、平均池化及Unpool检查 | 347 / 491 |
| index_capacity / sorting | 528,513元素完整sort梯度及out；1,056,897源元素TopK完整输入梯度 | 1,801 / 2,318 |
| conv_capacity / workspace | `(1,4,96,97)`的4输出通道3×3卷积完整输出、输入/权重/bias梯度 | 718 / 735 |

卷积回归峰值设备buffer为FP32 1.843MiB、FP16 1.630MiB，保留S35工作区收益；没有据此宣称所有卷积形状峰值相同。扩展池化的旧前向限额没有被这次回归解除。

## 离线检查与复现

最终126项根目录、32项Python API和5项发布工具检查通过，共163项。新增8项覆盖大空间无整轴数组、与完整轴枚举一致的边界验证、跨平面位置、组计划与存储、负零/重复项、长索引恢复、页对齐读取原始位及padding。已加入离线CI列表。

本轮代码均为项目Python编排，使用既有`resident_selection.c`、`resident_row_extrema.c`、原始字DMA和整数算术，不新增依赖、模型、OpenVINO runtime、SHAVE内核或DLL/PYD构建。

```powershell
.\run_torch_npu.ps1 -Script torch_npu_pool_capacity_demo.py -ScriptArguments @('--case','contracts','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_pool_capacity_demo.py -ScriptArguments @('--case','spatial','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_pool_capacity_demo.py -ScriptArguments @('--case','extrema','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_pool_capacity_demo.py -ScriptArguments @('--case','sampled','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_pool_capacity_demo.py -ScriptArguments @('--case','planes','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_pool_capacity_demo.py -ScriptArguments @('--case','training','--precision','float16')
.\run_torch_npu.ps1 -Script torch_npu_pool_topk_demo.py -ScriptArguments @('--case','edges','--precision','float32','--label','s36')
.\run_torch_npu.ps1 -Script torch_npu_vision_sort_extended_demo.py -ScriptArguments @('--case','pooling','--precision','float32','--label','s36')
.\run_torch_npu.ps1 -Script torch_npu_index_capacity_demo.py -ScriptArguments @('--case','sorting','--precision','float32')
.\run_torch_npu.ps1 -Script torch_npu_conv_capacity_demo.py -ScriptArguments @('--case','workspace','--precision','float32')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_pool_capacity
```

所有NPU进程串行，正式结果记录源SHA、复制/分配与图缓存统计，归纳后按用户要求清理原始JSON，保留源码和报告。单次时间含编译、CPU参考、完整下载和关闭，未做交替热吞吐或功耗认证。

## 最终源码与组件指纹

12个专项进程均记录并核对下列五个生产模块的同一SHA-256；既有回归中带源码指纹的记录也逐项核对。没有修改C/C++或重建原生组件，原62枚resident ELF继续使用。

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `90da94329e319f4ca44a3d1a69b01c429a0752ee5f1ad8778e862eda7c3d4cd4` |
| torch_npu_geometry.py | `ad9ebdb85ee78cf3771c43de98bd8de210c161ffeff9f82b684d8a76cba398b7` |
| torch_npu_autograd.py | `f86896cda34d997dac94fca94a815f8a97cdb96e4497608deefa02ccdee9cb04` |
| torch_npu_sort.py | `a63d0c6f5a91e035890f90ce181d22508b7ae4d0d3be73b9395b80df5503a789` |
| torch_npu_vision_extended.py | `b45a81e66eba6a27de34de146c8df6fc66fb24d4d7ccd2731153837b9880c7af` |
| torch_npu_pool_capacity_demo.py | `e3058eb78ad04111479241c3adcc4d1f05ca266446181552be9220d0b82027ae` |
| test_torch_npu_pool_capacity.py | `879b9166a1d7354d68b7664473dd56cabd9571ab7e55d32160cdbb48b8e0c666` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `4fed86d100f9ca48ecda3e8d810ea2526400485a690465ba96e1cf92d7150e4a` |
| npunlock/tests/fixtures/npu4000/resident/manifest.json | `329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc` |
| npunlock-probe/torch-device/backend.cpp | `34497308b7ea4b010afa5794147ee74d538af031e48fb662dada2685113a376b` |
| npunlock/src/graphinfer/session_win32.c | `cfdd2f3241ae28a234f4161ed71b6e15e8cf0d1c20b716f27c9c61074228a70e` |

## 尚待改进

- 一般高级索引/扁平scatter反向、AveragePool、插值、3D/自适应前向、单平面大于16,777,216的反向仍有独立边界。
- 许多不对齐切片、每包状态同步及大量不同图的编译/提交仍可能较慢；没有通用异步或吞吐优势承诺。
- 原有稀疏/高阶梯度、长期Half/AMP训练、任意Hugging Face迁移、14GiB完整训练和GPU/NPU协同未在本轮认证。

## 清理记录

全部NPU/Python进程退出后，按明确的20条路径清理专项及回归原始JSON共96,164字节。删除前逐项核对通过状态、工作区内绝对路径、非联接/符号链接及非保留文件；清理前后504份选定源码、报告、内核及必要构建组件SHA-256一致，随后补充报告与空间台账。保留全部实现、验收/离线测试、CI及Markdown报告；没有生成模型、磁盘checkpoint或图表。公开source-release未改，未推送GitHub。
