# S24：视觉与排序兼容性扩展验收

日期：2026-10-22。本地开发版，尚未同步至公开 v0.1.0。

S23只补了常用子集。本轮补上原清单的主要缺口：3D、抗锯齿、网格采样、池化/Unpool、padding/图像重排，以及整数/BOOL排序、空结果和次序统计。
下述明确范围已经实现并在真实NPU验收；不能把它扩大成全部视觉API、任意配置或通用CUDA兼容。

## 环境、实现与数值口径

- Windows x64 / Core Ultra 7 258V / Intel NPU4000 `8086:643e`；驱动 `32.0.100.5540`、图编译器8.3。
- Python3.12.15、Torch2.14.1+cpu、NumPy2.5.3；沿用 `.conda/npunlock`，没有安装新依赖或下载模型。
- 真实 `intel_npu:0` Tensor：C++ allocator与复制、Level Zero共享存储、自编SHAVE C及设备DMA/FP32算术。
- 五维是Torch逻辑shape；执行通过展平或DMA整理使用原有载体，没有改固件MemRef ABI。
- CPU仅生成公开shape/stride/滤波参数等元数据并运行独立参考；模型grid、indices、量化外的计算数据留在NPU。动态读取/统计每包会回读4字节校验标志；这不是零同步路径。
- 独立CPU轨迹、seed824、CPU单线程并关闭MKLDNN。新专项预设FP32 `rtol=2e-4, atol=2e-5`，FP16 `rtol=.008, atol=.002`；整数、排序值/索引和数据重排采用精确比较，NaN按equal_nan检查。
- 本机CPU不提供Half抗锯齿插值、AvgPool3d和grid_sample。对应Half检查显式使用相同已舍入Half输入/参数，CPU升至FP32计算再转回Half；不称为原生CPU Half对照。

## 已验收范围

| 家族 | 实现与覆盖 |
|---|---|
| 存储、空Tensor | rank0–5；FP16/FP32/BOOL最多1,048,576元素，INT64最多524,288。零元素不分配驱动buffer、不绑定空图；复制/clone、五维排列与广播/归约、空sort/topk及空切片的梯度回传 |
| sort/argsort/topk | FP16/FP32/INT64/BOOL；稳定排序、升降序、合法轴、空结果、k=0及scalar k=0的Torch语义、配对out/argsort out。INT64直接比较两个原始32位字，检查signed64极值、超过2^53、低字进位与9,001元素宽行，不经过FP32数值转换 |
| 极值与次序统计 | max/min整数值/索引；kthvalue、median/nanmedian、mode、全局median梯度分摊；FP32 quantile/nanquantile的五种插值、scalar/vector/empty q、NaN/Inf及极小分位点；pair out与quantile out、输入/索引版本保护 |
| 3D卷积 | Conv3d/ConvTranspose3d、分组、不同stride/padding/dilation/output_padding、bias、输入/权重/bias一阶梯度；窗口DMA与FP32 GEMM，输出恢复原dtype |
| 插值与归一化 | nearest/exact/trilinear3d；2D bilinear/bicubic抗锯齿；BatchNorm3d、InstanceNorm3d、GroupNorm、与原地ReLU组合；train/eval和inference_mode |
| 池化 | 有/无batch的MaxPool3d、AvgPool3d，ceil、max dilation、include_pad；AdaptiveMaxPool2d/3d、AdaptiveAvgPool3d，2D自适应均值扩展超过64格（81格验收），空间INT64索引、重叠梯度 |
| Unpool | MaxUnpool1d/2d/3d的覆盖式设备索引写入及动态读取反向；检查非重叠池化/反池化组合 |
| padding/重排/展开 | constant/reflect/replicate/circular、正负extent；pixel_shuffle/unshuffle、channel_shuffle；Unfold/Fold及重叠累加的一阶梯度 |
| 动态网格采样 | affine_grid2D/3D；grid_sample2D nearest/bilinear/bicubic、3D nearest/trilinear，zeros/border/reflection和两种align_corners；输入、grid及theta一阶梯度 |
| 五维分类 | 任意合法轴softmax/log_softmax，以及(N,C,D,H,W)硬标签CrossEntropy与label smoothing；标签仍在NPU，继承原packet界限 |
| 小模型训练 | 两种精度各3步独立CPU对照：分组Conv3d→GroupNorm→ReLU→AvgPool3d→ConvTranspose3d→trilinear→均值池化→Linear，逐步检查loss、参数梯度和带动量SGD更新；另有3步可学习affine_grid→grid_sample训练 |

新增入口：`torch_npu_vision_extended.py`、`torch_npu_grid.py`、`torch_npu_order_stats.py`；扩展原sort、vision、geometry、autograd和设备派发。
旧安装须重编 `npunlock-probe/torch-device/backend.cpp` 对应PYD，session DLL未重建；正常运行使用配套ELF，不需要OpenVINO Python/runtime。

## 新专项结果

22个独立进程最终均通过，共30,930次记录NPU图提交。下表是最后保留的验收记录；`checks`是检查组或训练步数，不是所有逐元素/参数断言的数量。

| case | FP32流程检查组 | FP16流程检查组 |
|---|---:|---:|
| foundation | 21 | 21 |
| integer | 24 | 24 |
| statistics | 85 | 31 |
| images | 39 | 39 |
| pooling | 21 | 21 |
| volume | 13 | 13 |
| grid | 34 | 34 |
| training | 3步 | 3步 |
| warp | 3步 | 3步 |
| contracts | 15 | 13 |
| outputs | 12 | 11 |

`integer`两列都实际测试INT64/BOOL，并不是新增INT4/INT8计算精度。Half流程不测试Torch本就不提供的Half quantile。
22个进程shutdown后live/cache/scalar/session/quarantine均为零、context_failed=False。它们包含普通no_grad、autograd及inference_mode路径。

旧设备Quick的33组检查与旧训练Quick的11组检查也已通过。原先拒绝空Tensor/rank5/k=0/整数排序/超过64格自适应均值的测试，改成相应正向检查和真实非法/仍未支持边界；没有通过继续拒绝新功能来让测试变绿。

## 修复与开发期发现

1. 首版排序行统计在真实设备发生一次 `zeFenceHostSynchronize 0x70000001`。汇编出现绝对跳转表及独立函数调用；禁用跳转表并全部内联后又触发安全代码窗口上限，故将插值算术复用到已有NPU算子，压小SHAVE控制内核。最终统计/grid构建强制拒绝Function call/LJTI/rodata，后续通过进程均无context failure；不把初次失败计为通过。
2. 极小分位点暴露FTZ：q=1e-39本身位模式713624与普通NPU乘法均可保留，但初版统计输出的权重为0，使 `[0,1e38]` 的参考约0.100000016变成0。最终在设备内用IEEE helper计算rank、按正数位序判断上端点，并用整数位模式传递权重（微小分支保留volatile整数读），恢复权重713624和正确结果。保留 `quantile_subnormal_probe.py` 与正式回归，不用CPU补算。
3. circular padding与负extent混用须先裁剪再循环；原先对原图直接取模错误，已修复并加入伴随参考和真机梯度检查。
4. constant padding使用Inf/NaN时，`mask*constant`会污染未填充位置；改为BOOL条件选择。非插值统计梯度也避免创建不必要的 `Inf*0` 分支；全局极值/median用条件选择保留未选位置的零梯度。
5. Unfold的DMA前向精确相等，但重叠反向最初要求逐位相等失败，FP32最大绝对差约2.38e-7。重叠加法采用不同归约顺序，因此按本轮原先声明的算术容差验收其梯度，前向仍要求精确；未写成逐位一致。
6. CPU Half缺少上述三类算子时，原参考调用直接报NotImplementedError；改为明确的FP32-opmath参考，保留此区别。早期验收脚本的检查计数覆盖问题已修，表中为最终计数。

## 仍然保留的边界

- 单allocation最多4MiB；逻辑rank最多5；sort输入/索引最多524,288元素；全局最多1,536个缓存图。宽行bitonic是O(n log²n)，兼容通过不意味着快。
- 新次序统计每行最多32,760元素；quantile仅FP32、一次最多128个同设备分位点，q不求导。更长统计行、其他dtype、全部NaN/Inf/并列值组合仍不承诺；mode/topk并列索引及重复Unpool indices不保证跨平台相同路由。
- 视觉新路径主要验收非空batch/空间；单窗口容量、最多8,388,608采样格及既有GEMM/分类packet限制保留。空Tensor基础不表示所有视觉算子接受空输入/输出。
- grid坐标及展开的像素坐标须有限且绝对值<=1,048,576；NaN坐标实现为-1，Inf拒绝。正式对照采用有限grid；不承诺所有非有限数据组合。原生out/backward变体并未逐个补齐，正式范围是上述公共模型入口和已列出的out接口。
- S23少数显式scale的旧CPU插值前后向不一致配置仍在训练前拒绝；使用size或recompute_scale_factor=True。旧Half/AMP近似并列值经排序放大整网轨迹差异的记录保留，本轮不宣称修复了任意长期模型收敛。
- torchvision自定义NMS/ROIAlign/deformable convolution、searchsorted/bucketize/unique不在本轮认证范围。默认大型视觉网络还会受4MiB/元素上限和其余算子约束，不能保证仅改设备名就运行。
- 不提供二阶梯度、异步stream/event、CUDA运行时/PTX/Triton扩展或GPU+NPU协同。本轮未改低位量化整网质量，也未做性能优化或速度排名。

## 复现

```powershell
.\build_torch_npu_device.ps1
foreach ($precision in @('float32','float16')) {
  foreach ($case in @('foundation','integer','statistics','images','pooling','volume','grid','training','warp','contracts','outputs')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_vision_sort_extended_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
  }
}
```

只修改SHAVE源码时，使用已有MoviTools运行sort/selection/order_stats/grid各构建脚本；完整resident构建入口已包含它们。默认运行只读取配套ELF及其manifest。
离线几何/控制测试58项、Python API32项、源码导出5项，总计95项通过；CI离线列表已包含新增几何测试。
来源与独立参考见根目录 `REFERENCES.md`，普通软件契约见 `COMPATIBILITY.md`。

本轮原始实验JSON/日志和可重建构建中间物在汇总后按用户要求清理；源码、内核、必要运行组件与本报告保留。清理统计与最终源码指纹附后。

## 最终旧回归与指纹

旧S23的sort、interpolation、normalization、transpose、contracts、extrema两种精度共12个进程全部通过，9,635次NPU提交；D/T两个旧进程共44组检查、6,936次提交。合计36个最终记录、47,501次提交；全部shutdown资源归零且context健康。开发失败及重复诊断不包含在这个通过总数中。

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `fd3484e87ea2cc05d09b195fb49a8dc3858caf221794fbbe0116fa8409854495` |
| `torch_npu_autograd.py` | `be3928e07dc3f45f00d3f80f280d960aa4e58041fc7fd3fb7da498a55568e416` |
| `torch_npu_sort.py` | `8db90d96a639a65cfa22eb041f650529be789a3816a3566e7758171aa882d26e` |
| `torch_npu_order_stats.py` | `04290372402aaa86349997cd631a4a281ac318897785402d7d856c3f12bf66bd` |
| `torch_npu_vision.py` | `f580c6d98ed4513f18ff1b2af68041e0e088d81c21c202c06ec68e2425f89015` |
| `torch_npu_vision_extended.py` | `4b0ac45981f185c170aea399ba546c8519f043e450b7f607c472eee64993a553` |
| `torch_npu_grid.py` | `fc5b71d94b21404e8cc26f1572e0280907117e778f1490baaab694026815c26a` |
| `torch_npu_geometry.py` | `9e3af3a903d9a405aaebc8d4a50210633c9bea11cf4a149b4adf0ad12d63f48c` |
| `torch_npu_vision_sort_extended_demo.py` | `d67dfa1c4e8f8082b34665ff574df416dd3b697c5d66b6627a4ad906b86618f5` |
| `test_torch_npu_vision_extended_geometry.py` | `17a271f32302bd953f5dd9793c3e27313261b8d68dea3260dc40fbfa212f7729` |
| `npunlock-probe/torch-device/backend.cpp` | `1632051878a2c218e807c289641a40be7f93cd504553cdc0f827901efe9e1c75` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `ef175e1d1e8f16c1a3dc159b8424eb36ad82d6c0fdbdd6a4fee8c72c22c29b95` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |
| `npunlock/tests/fixtures/npu4000/resident_sort.c` | `6a223644b53c91b64a1c42bd691fbaecb9aa52544f8cb75413f7614c5d2e9578` |
| `npunlock/tests/fixtures/npu4000/resident_selection.c` | `4344079c18b895caffb5cd77b747f7cf3e9fdce26d07f388982fb88a0aab9fcd` |
| `npunlock/tests/fixtures/npu4000/resident_order_stats.c` | `86f267c587eb4afcacc6c74e655c27a27ba0114f20b4be79b35efc598f92a2aa` |
| `npunlock/tests/fixtures/npu4000/resident_grid.c` | `a60919eb7742b0b27c2918001a52e6778696e812d3548a0e42142585e61c7979` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `02a39979c5cef0f62043f3c1d214b95d82ffc9c833c47c2e8efc6022c27407c2` |

全部45个resident内核的源码、ELF和汇编manifest已验证。新增ELF大小：order-stats-f32=8540 bytes, grid-stencil-f32=7132 bytes, indexed-read-f32=4380 bytes。sort-f32和selection-backward-f32也已更新；其余旧内核沿用原实现。

## 清理与发布状态

按用户约定删除103份原始JSON/日志/Movi响应和可重建backend.obj/导出表，共8,093,635字节（7.719MiB）。清理前后834份保留源码/报告/内核及必要运行时的SHA-256一致，随后追加本段和存储台账。无新增图表、模型、依赖、磁盘图缓存或session DLL重建。source-release与GitHub未修改，未创建发布。
