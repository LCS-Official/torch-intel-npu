# 视觉与浮点排序兼容性（S23）

2026-10-22，Windows x64 / Core Ultra 7 258V / NPU4000 `8086:643e`，驱动32.0.100.5540，
图编译器8.3，Python3.12.15、Torch2.14.1+cpu、NumPy2.5.3；沿用`.conda/npunlock`。
本轮以兼容性和数值语义为目标，没有新增依赖、模型下载、OpenVINO调用或宿主C++/PYD/DLL重建。
新增代码直接接入普通ATen及AutogradPrivateUse1，不修改torch全局函数，设备名仍为`intel_npu`。

## 新增范围

| 项目 | 已实现 | 保留界限 |
|---|---|---|
| 插值/上采样 | NCL nearest/nearest-exact/linear；NCHW nearest/nearest-exact/bilinear/bicubic；size/scale_factor、align_corners、普通模块及一阶反向；非vec原生out/backward入口 | 非空、rank<=4、输出<=1,048,576；无antialias、3D/trilinear、grid_sample/affine_grid；area沿用S20有限自适应均值池化；下文的旧CPU尺度边界在训练前拒绝 |
| GroupNorm | rank2–4、合法groups、可选affine、FP16输入配FP32 affine；native forward/backward | 正有限FP32 eps，mixed bias-only拒绝；无native out、二阶导 |
| InstanceNorm1d/2d | affine/无affine入口、train/eval、可选running stats；模块无batch输入；FP16/FP32一阶梯度 | 输入统计要求空间元素>1；running buffers成对、无梯度且不别名；不是InstanceNorm3d |
| ConvTranspose1d/2d | 合法stride/padding/dilation/groups/output_padding、可选bias；输入/权重/偏置梯度；包含output_padding>=stride但<dilation情形 | 同dtype NCL/NCHW及权重；输出<=1,048,576；每组核面积×最大输入/输出通道<=65,536；无3D/DPU专用实现 |
| sort/argsort | FP16/FP32、rank0–4、任一合法轴、ascending/descending、stable、只读partial输入、双out/argsort out；值梯度及INT64索引 | 输入/输出<=524,288；stable=False也返回稳定的合法结果；无整数/BOOL排序、空Tensor；图缓存仍上限1,536；不保证所有NaN payload逐位相同 |
| max/min | dim值与索引、keepdim及双out；全局归约及out；两Tensor极值重载；一阶梯度 | dim首个tie/NaN接收梯度；全局平分相等极值或多个NaN，不能混用amax/amin的NaN导数；输出沿用INT64容量 |
| 更大topk | 超出S22的k<=128/比较预算时复用排序；largest、sorted、out及反向 | 输入须<=524,288、输出<=524,288、1<=k<=轴长；旧小k路径仍可处理全局FP输入上限；k=0仍拒绝，tie不承诺CPU相同索引 |

以上是有界算子及组合支持，不代表任意视觉模型或CUDA生态兼容。全局空Tensor、rank>4及更大存储仍未开放。
原地ReLU仍要求非叶、非view、完整密集目的存储；S23在Conv1d/ConvTranspose1d及InstanceNorm的公开reshape出口
使用真实可微clone，让常见后接原地激活获得独立存储，未隐藏调用者view别名或绕过version保护。

## 设备实现

**排序。** 自写SHAVE底部向上merge sort，每包最多10,922个值，输出载体同时提供scratch。
更宽行用分块bitonic比较交换，partner地址仅由shape/XOR生成；原始INT64高低字通过FP32字节载体搬运，
不作数值浮点索引转换。NaN按ascending末端/descending首端排列，相同值按原始位置稳定处理，±0视为相等。
局部O(n log n)，长行O(n log²n)；这是兼容实现，不声称比CPU或原小k选择更快。
sort/max/min反向复用S22覆盖式索引回填，保留索引version保护及上游负零语义。
全局max/min另行处理多个NaN的共享梯度；amax/amin原有语义保持。

旧训练Quick的负向清单仍要求合法ConvTranspose被拒绝，首次回归因此报`Accepted transposed`；
这条旧断言已改为检查非法output_padding（stride=dilation=1而output_padding=1），保留提交前拒绝覆盖。
合法转置卷积的独立正向/导数检查由本轮专项承担，没有删除错误检查来隐藏算子故障。

**视觉。** CPU仅生成插值坐标、系数、转置卷积窗口地址等静态几何元数据。
模型/激活/梯度的DMA读取、FP32加权归约和矩乘均在NPU；转置采样窗口构造反向，不下载payload计算梯度。
插值按轴分离，先收缩后扩张以约束中间存储，反向采用相反次序；FP16仅在输出/梯度边界转换。
ConvTranspose按输出位置读取符合stride整除条件的输入，复用既有FP32 SHAVE矩乘；反向分别收集输出梯度，
计算输入梯度及树式归约权重梯度。groups串行处理，不声明并行、DPU或异步加速。
GroupNorm复用每组FP32统计/归约；InstanceNorm在设备上重排为(1,N*C,1,S) BatchNorm，
将running mean及无偏running variance跨实例平均回原buffer。

新增`resident_sort.c`及配套ELF/汇编/manifest，ELF **8,636字节**。
正常运行不需要MoviTools；修改C时使用`build_sort_kernel.py`。
完整resident构建入口也补上此前遗漏的index/row-extrema/selection及新sort步骤，避免完整重建丢失manifest条目。
本轮只实际重编新sort内核；没有再重编全部既有内核或宿主运行时。

## 修复与真实数值边界

**Half归一化权重梯度。** 开发轮InstanceNorm(2,3,4,5)的一个通道，NPU梯度-0.072265625、CPU -0.0703125，
差0.001953125，原容差未通过。逐实例比较显示一个贡献为3.19140625 vs 3.193359375，而保存invstd逐项相同。
CPU先计算sum(g*(x-mean))、再乘invstd；原实现先把invstd乘进每个元素，再归约。
调整共享BatchNorm的权重梯度次序后，原失败种子/形状通过，未放宽容差。
为保留该顺序，训练Function额外保存一个FP32 centered buffer至backward；普通无梯度入口仍返回原三个内部结果。

**恒等插值及别名。** 最初FP32恒等路径可能复用输入存储。现仅在静态采样确为恒等时真实clone，
检查其后relu_不改变原输入或其version；普通变化尺寸和显式scale不会因“输出尺寸恰好相等”错误走恒等路径。
Conv/GroupNorm/InstanceNorm的原地组合另有前后向检查，叶/view拒绝及保存参数/索引版本检查保留。

**本机旧CPU显式scale的不一致。** 实测Torch2.14.1：nearest1d输入[0,1,2,3]、scale_factor=1.24，
输出[0,0,1,2]，但对梯度[1,2,3,4]返回原样[1,2,3,4]，不是该前向映射的转置[3,3,4,0]。
nearest1d部分恰好2倍尺寸及bicubic2d显式scale而总尺寸不变，也可触发旧backward快捷分支。
本项目没有静默采用已知不一致的导数：这些配置在可微前向/原生反向的设备提交前拒绝；无梯度前向按CPU映射执行。
使用`size=`或`recompute_scale_factor=True`可绕过该配置；普通nearest-exact、linear/bilinear及其他已验收配置不因此受限。
四项离线几何检查包括独立CPU前后向及这类不一致的显式区分。

**含排序的整网敏感性。** 独立sort用同一输入逐位核对值、索引及梯度；这不保证不同设备的整网浮点激活总有相同排序路由。
合成网络为Conv2d→GroupNorm→原地ReLU→MaxPool→ConvTranspose2d→bilinear→InstanceNorm→可选sort→Linear→FP32 CE。
每项模型有同初始化、同输入/标签的独立CPU控制。FP32含排序三步通过；FP16/AMP严格轨迹及固定路由诊断见下方正式记录。
固定路由只在验收脚本中使用设备产生的索引来重算CPU参考，帮助排除连续算子导数错误；生产后端始终在NPU排序，
不采用CPU路由，不把诊断参考替代严格独立整网通过，不增加容差来掩盖分叉。

**AMP运行统计的严格探针。** `vision-amp --precision float32`在首步的一个FP32 running mean上，
与CPU相差1.45789235830307e-5，超过全局FP32检查的rtol=8e-5、atol=1e-5；该探针保留为未通过。
AMP的前级激活是Half，FP32保存running buffer不消除输入量化/运算差异。
正式AMP使用`--precision float16`选择既定半精度模型比较阈值，实际主参数/优化器仍为FP32；
不能把它记为上述FP32统计阈值通过。混合模型参数/梯度始终使用预先声明的rtol=0.012、atol=0.002。

## 正式验收结果

<!-- RESULTS:START -->
**22个主专项进程：20个通过，2个排序整网数值边界；另有1个AMP运行统计严格FP32探针未通过。**
主专项记录29,207次NPU图提交；含严格统计探针共23进程、29,865次。开发失败/诊断不计入这些数量。

| 专项 | FP16调用 | FP32调用 | 结果 |
|---|---:|---:|---|
| 稳定sort/argsort、双out、梯度 | 1,538 | 1,166 | 通过 |
| max/min值/索引及全局梯度 | 1,142 | 848 | 通过 |
| 21,847元素长行、跨包bitonic | 5,791 | 5,785 | 通过 |
| 20类基础插值＋尺度/单值/布局/out/backward | 540 | 354 | 通过 |
| 21类规范化模块＋无affine/混合affine | 1,440 | 794 | 通过 |
| 7类转置卷积及全部一阶导 | 596 | 426 | 通过 |
| 12项提交前拒绝＋别名/version/原地组合 | 370 | 223 | 通过 |
| 66,049排序元素/67,600视觉元素分块 | 635 | 547 | 通过 |
| 无排序视觉网络三步独立SGD | 1,741 | 1,055 | 通过 |
| 含排序视觉网络独立SGD | 429 | 1,103 | FP32三步通过；FP16第0步严格差异，固定路由参考通过 |
| FP32主参数/FP16 autocast，无排序三步 | 2,054 | — | 混合精度阈值通过 |
| FP32主参数/FP16 autocast，含排序 | 630 | — | 第0步严格差异，固定路由参考通过 |
| AMP running mean严格FP32探针 | — | 658 | 未通过，见上文原误差 |

**91项离线检查通过**：根目录54项、npunlock Python API 32项、源码导出5项。
算子浮点容差为FP32 rtol=8e-5/atol=1e-5、FP16 rtol=0.006/atol=0.001；排序/索引/值梯度的指定检查使用零容差。
模型FP32参数/梯度容差rtol=0.0002/atol=2e-5；FP16/AMP为rtol=0.012/atol=0.002。
AMP的running buffers另按显式precision所选的模型比较阈值核对；严格FP32探针的失败单列，不合并为通过。

| 独立训练 | 三步NPU loss | 每步最大参数梯度绝对误差 |
|---|---|---|
| FP32无排序 | 1.36602974 → 0.96472323 → 0.65354037 | 3.5762787e-07 / 2.9802322e-07 / 3.5762787e-07 |
| FP16无排序 | 1.46117163 → 1.04294062 → 0.73739028 | 0.0014648438 / 0.0010375977 / 0.0012207031 |
| FP32主参数/FP16 AMP无排序 | 1.36619222 → 0.96439797 → 0.65343636 | 0.001953125 / 0.0014648438 / 0.00085449219 |
| FP32含排序 | 1.57893896 → 1.13355124 → 0.83870441 | 2.3841858e-07 / 2.3841858e-07 / 1.1920929e-07 |

FP16含排序与AMP含排序的严格独立训练均在第0步记录边界，**不计为三步训练通过**：
| 任务 | 不同排序位置 | 前级激活最大差 | 原严格卷积梯度最大差 | 固定路由后最大梯度差 |
|---|---:|---:|---:|---:|
| FP16含排序 | 6 | 0.001953125 | 0.08197021484375 | 0.0009765625 |
| AMP含排序 | 4 | 0.00390625 | 0.199798583984375 | 0.001953125 |

各dtype内部使用相同CPU/NPU输入；不同dtype的随机输入不作为跨精度质量或性能比较。
普通无排序/含排序训练每步仅回读2/3个四字节分类/索引标志，AMP增加一个scaler BOOL标志；显式跨设备payload copy计数不增加。
普通热步12次上传均为池化地址/插值系数等几何表；首步及AMP scale变化另有标量常量上传。

<!-- REGRESSION:START -->
**12个既有回归进程全部通过，共10,006次NPU提交。**
| 回归 | 结果 | NPU调用 |
|---|---|---:|
| Device Quick | 33组通过 | 4,162 |
| Training Quick | 11组通过；更新过时转置卷积负向断言 | 2,774 |
| s23-regress-contracts-float16 | 通过 | 12 |
| s23-regress-contracts-float32 | 通过 | 11 |
| s23-regress-relu-float16 | 通过 | 219 |
| s23-regress-relu-float32 | 通过 | 126 |
| s23-regress-training-float16 | 通过 | 1,516 |
| s23-regress-training-float32 | 通过 | 964 |
| division | 通过 | 6 |
| dense | 通过 | 151 |
| unary | 通过 | 35 |
| reduction-cast | 通过 | 30 |

主专项加既有回归为34个进程、39,213次提交；连同单列的AMP严格统计探针，共35进程、39,871次记录提交。
旧S22四步CNN/clip/checkpoint恢复在两种dtype下通过；设备、复制、原地/version、训练拒绝、除法/密集布局/单目/归约转换回归通过。
23个新增记录对应的生产Python、C及manifest指纹与当前源码一致；新增排序ELF/汇编及全部42枚resident内核/source指纹审计通过。
未重跑已冻结低位/完整LLM质量、性能或长时间功耗测试；旧严格数值边界未因此消失。
<!-- REGRESSION:END -->
<!-- RESULTS:END -->

每个独立进程检查释放后的非标量live storage没有增长；shutdown后live/cache/scalar/session/quarantine归零且context_failed=False。
训练区间显式copy计数区分几何/标量常量上传、索引/分类校验标志以及scaler标志；不包括驱动内部DMA。
数值参考下载在计数区间之外；有限模型训练不认证长期收敛、真实数据集质量、完整ResNet/UNet或性能优势。

## 复现、来源和剩余工作

同一环境串行运行`torch_npu_vision_sort_demo.py`，参数`--case`及命令见根目录[INSTALL.md](../../../../INSTALL.md)。
`sort/extrema/wide/interpolation/normalization/transpose/contracts/tiled`为算子与容量专项；
`vision-training/vision-amp`为无排序训练；`training/amp`保留完整排序整网与严格/固定路由的区别。
CPU离线入口为`test_torch_npu_vision_geometry.py`，已加入offline CI列表；没有声称本轮GitHub远程CI运行过。
参考本机ATen schema、`ATen/native/UpSample.h`及CPU API；来源关系见[REFERENCES.md](../../../../REFERENCES.md)。

剩余兼容性按优先级：

1. 更完整视觉：antialias、AdaptiveMaxPool/Unpool、padding与grid_sample/affine_grid、默认大视觉网络；
   3D卷积/池化/插值首先需要rank5存储和布局扩展，不能只增加Python算子注册。
2. 排序/数值：INT64/BOOL输入、k=0及空Tensor、kthvalue/median/mode/quantile，更大容量/缓存压力；
   含近似平局离散路由的整网轨迹、全部非有限值/NaN payload组合尚未认证。
3. 其余通用设备：更广索引/随机/数学/损失配置、二阶导、异步stream/event、torch.compile及第三方CUDA/Triton扩展，
   以及GPU+NPU协同仍未实现。原低位整网质量、S17/S18/S21数值限制仍按原报告，本轮未改量化。

<!-- HASHES:START -->
<details>
<summary>本轮源码/必要产物SHA-256</summary>

```text
torch_intel_npu.py
c100ad781405321bfd207099baf2e55bf3ddb76ceb45db5cd271c63daebcaeda
torch_npu_autograd.py
8ed0cb2fa33c3984720982f344f6f3a3da39d6b6dc24fb9d00b4f2e3513b5fc8
torch_npu_geometry.py
33c350c3c4bc381a687388db3c57b64e235011909bf07be74d79744176ba2196
torch_npu_sort.py
c259e0b16df41c793ed6cc82807c897a4c92f2e4f41c55116b1ff810404ec39f
torch_npu_vision.py
c334614b4b2548f26cca05dc456e0be89922f228b4c45c3a7ad52dec460f23c1
torch_npu_vision_sort_demo.py
89752c342115b088e80762d0273681b17ebf89b869352410749d57373563a097
torch_npu_training_demo.py
5c8462209251fe663d40604c14b5ee5be0ba2706984f70072a6897ec574b7d54
test_torch_npu_vision_geometry.py
760303e81e754e816025c010c88d0290c4c111acfec677794f0619e306b9a7e5
npunlock-probe/research/build_sort_kernel.py
1d6e220a1d593593651ef78defe0d4f0c69a1db5e8d882f0d7359e68cae67874
npunlock-probe/research/build_resident_kernels.py
b01bd6d4a761675bb49dcb9c4a1479d25987e1516e9c1d72c6b71e245373961b
npunlock/tests/fixtures/npu4000/resident_sort.c
b5eb845571fa6e36991f5a6159260a6decd335d9549db4e4399b1571abc5e057
npunlock/tests/fixtures/npu4000/resident/sort-f32.elf
ca6fa3c66c6cde3a8bec2f5a49d3e6d6d49e9952a1f1a1640ca295ca5d634074
npunlock/tests/fixtures/npu4000/resident/manifest.json
b6b5df57f1ca692b152468f531dbaf8723bf578222118f7eae6647cdf820ab9d
npunlock-probe/torch-device/backend.cpp
4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83
npunlock-probe/torch-device/_torch_intel_npu.pyd
3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd
npunlock-probe/session-runtime/npunlock_session.dll
0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
```

</details>
<!-- HASHES:END -->


## 清理与保留

2026-10-22 S23视觉/浮点排序验收后清理：删除本轮65份原始JSON/日志/Movi响应，共271581字节；清理前后817份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与台账。保留1D/2D插值、GroupNorm/InstanceNorm、ConvTranspose、稳定sort/argsort/max/min及更大k的源码/反向、新排序C和8636字节ELF/汇编/manifest、构建/验收、CPU几何检查与双语文档。22个主专项20通过/2排序轨迹边界、12个旧回归全部通过；另有1个AMP运行统计严格FP32探针未通过，原误差和固定路由诊断分开保留，共35个最终记录、39871次NPU提交，91项离线检查通过。无依赖安装、模型下载、宿主C++/PYD/DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未改。
