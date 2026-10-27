# S39：Attention空维度、Tensor概率Bernoulli与randperm

日期：2026-10-25。本地开发版，尚未同步至GitHub。平台为Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

## 实现与语义

本轮为Python派发/编排及autograd边界更新。复用现有SHAVE Philox整数/浮点RNG、原始字INT64排序、BOOL比较/归约和DMA，没有新增依赖、模型、DLL/PYD构建或SHAVE内核。64组ELF/汇编及26份内核源指纹与现有manifest逐一一致。

### SDPA

- 支持有界空batch、空头、空Q/K/V序列、零QK特征与零V特征，含广播、GQA、mask、causal及rank64用例。
- 空K/V序列返回零值；空输出不提交无意义的score/softmax计算。零QK特征不被当作空attention：仍应用mask、softmax、Dropout和V加权，V/mask梯度可非零。
- 空序列/零V特征不推进Dropout RNG；零QK特征且存在实际分数时，0<p<1仍按分数数目推进。p=0/1和所有backward不抽样。
- 对齐本机math SDPA的梯度连接：纯空batch仍给广播浮点mask零梯度；空序列/零V特征的未使用mask梯度为None。Q/K/V需要的梯度保留正确形状的零Tensor。
- 默认QK特征为0时使用有限的中性内部scale，其零收缩与梯度值通过CPU参考核对。显式非法scale、mask、广播和GQA仍拒绝；原有容量/行宽/因果query保护不变。
- GQA允许空Q头，以及相同空K/V头；后者返回零值。非空K/V头仍要求Q头可整除K/V头数。

开发初版将空mask梯度统一写成零；CPU对照暴露了None语义，以及零V特征不消耗Dropout RNG的区别，已修复后重新验收。该初版不计入下表正式结果。

### Bernoulli

接入torch.bernoulli、ATen Tensor/标量/out，以及bernoulli_的Tensor概率入口。概率可为FP16/FP32/BF16设备Tensor，允许广播及有界只读strided view；输出支持九种存储dtype，函数/out输出不必与概率dtype相同。

设备先检查概率在[0,1]且非NaN/Inf，仅回读一个BOOL校验标志；非法数据在消耗RNG或写入目标前失败。空输出不消耗RNG，其余每个输出使用一个Philox counter；p=0/1的非空Bernoulli也遵循此接口计数。原地目标须完整密集存储，可为完整转置；概率与目标可同一Tensor，全部概率读取完成后写回。out仍须精确shape、支持的dtype且无输入别名，不做resize。

保留原生PyTorch Bernoulli的零导数；验证了函数概率梯度、非叶原地目标的源梯度及广播概率梯度。FP32中间工作区受1GiB约束，不能由九dtype输出推导完整九dtype随机初始化或神经网络支持。

### randperm

复用已有NPU整数RNG产生完整有符号INT64键，再以设备INT64排序返回索引。设备检查相邻键是否重复，仅向主机回读碰撞标志；重复时整批重抽，最多8次，避免接受稳定平局带来的偏差。离线注入碰撞验证了重抽、8次失败及调用者out不被部分写入。

支持默认/独立intel_npu Generator、generator/out重载、状态恢复；CPU Generator拒绝。n=0/1不消耗RNG；其余每次尝试消耗n个counter，碰撞重抽会再消耗n个。沿用24字节RNG状态，不保证CPU/CUDA相同种子产生相同结果。

| 输出dtype | 软件n上限 |
|---|---:|
| FP16 | 2,049 |
| BF16 | 257 |
| INT8 / UINT8 | 128 / 256 |
| INT16 | 32,768 |
| FP32 / INT32 / INT64 | 16,777,216 |

n-1须可精确表示，并同时遵守现有排序总量保护。BOOL/FP64/复数拒绝。实测n=0/1/17/128/129/256/257/2049/8193，覆盖全部八种输出dtype；8193元素跨INT64排序包。上限是软件契约，没有最大规模性能认证；随机键排序并非针对大规模洗牌优化的专用内核。

## 正式真机与回归

下表为15个独立进程，每项passed。提交数包含测试准备、校验和CPU对照所需读回，不是纯算子性能计数。

| 入口 / case | 精度 | NPU提交 |
|---|---|---:|
| rng_extended / bernoulli | FP16 | 961 |
| rng_extended / bernoulli | FP32 | 923 |
| rng_extended / permutation（含八dtype） | 综合 | 9,022 |
| rng_extended / contracts | FP16 | 83 |
| rng_extended / contracts | FP32 | 79 |
| sdpa_extended / boundaries（21例） | FP16 | 223 |
| sdpa_extended / boundaries（21例） | FP32 | 188 |
| sdpa_extended / contracts | FP16 | 31 |
| sdpa_extended / contracts | FP32 | 23 |
| sdpa_extended / training（3步） | FP16 | 1,851 |
| sdpa_extended / training（3步） | FP32 | 759 |
| rng_demo / uniform（含标量Bernoulli） | FP16 | 131 |
| rng_demo / uniform（含标量Bernoulli） | FP32 | 103 |
| discrete_demo / generators | FP16 | 132 |
| discrete_demo / generators | FP32 | 114 |
| **合计** | **15进程** | **14,623** |

42例SDPA边界包括mask、causal、GQA、rank64、部分布局、p=0/.25/1。输出、Q/K/V与浮点mask梯度通过CPU对照；非零Dropout使用独立Philox掩码公式参考。最大有效梯度相对L2误差：FP16为0.000230957173，FP32为6.30192415e-8。空维度检查同时要求CPU/NPU mask梯度的None连接一致、RNG推进正确且backward状态不变。

Bernoulli用独立CPU Philox逐元素核对，覆盖七类函数/原地/out用法、九dtype输出、三种概率dtype、广播、完整转置、部分strided概率、原地自别名、空输出、原生零梯度与逐位状态重放。randperm用独立CPU Philox signed-INT64键与argsort参考核对每个元素，并检查完整集合恰为0..n-1、计数器、默认/独立Generator和out逐位重放。张量有效载荷计算无CPU fallback；概率/碰撞校验允许回读小标志。独立Generator专项保持默认NPU和CPU RNG不变。

每种精度11项元数据拒绝不新增NPU提交/不推进RNG；另4类非法设备概率验证在RNG消耗/目标修改前拒绝。既有attention拒绝及保存K/V的version guard回归通过。

现有259参数、rank5、Q头4/KV头2的GQA网络独立CPU/NPU三步训练通过既有逐项容差和全局梯度L2检查，未放宽阈值。FP32 loss为0.0672759339→0.0595084541→0.0531175919；FP16为0.0672282204→0.0594639480→0.0530970991。两者热图数保持109/162；接近零的FP32 key bias梯度沿用绝对误差单列口径，不以病态相对误差充当失败或通过依据。这些小模型三步结果不认证长期收敛。

所有进程shutdown后closed=True、context_failed=False，live_buffers、cached_buffer_blocks、cached_scalar_buffers、cached_sessions、quarantined_buffer_bytes均为0。

离线检查：根目录140项、npunlock Python API 32项、源码发布工具5项，**177项全部通过**。新增SDPA边界契约、randperm精确dtype上限与碰撞/失败测试；现有CI测试模块自动包含这些新增用例。GitHub托管CI未在本轮执行。

## 复现

在已有S38环境中更新Python源码并重启Python进程即可；无需重建原生组件。各真机命令串行执行，FP16/FP32分别运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = ''
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_extended_demo.py -ScriptArguments @('--case','boundaries','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_rng_extended_demo.py -ScriptArguments @('--case','bernoulli','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_rng_extended_demo.py -ScriptArguments @('--case','contracts','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_rng_extended_demo.py -ScriptArguments @('--case','permutation','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_extended_demo.py -ScriptArguments @('--case','contracts','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_extended_demo.py -ScriptArguments @('--case','training','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_rng_demo.py -ScriptArguments @('--case','uniform','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_discrete_demo.py -ScriptArguments @('--case','generators','--precision','float32')
```

permutation本身覆盖八dtype，只需运行一次。完整离线命令沿用.github/workflows/offline.yml；新增用例位于原有test_torch_npu_sdpa_extended/test_torch_npu_rng模块。

## 剩余范围

- 没有实现FlashAttention/NestedTensor、稀疏或高阶梯度；仍保存完整非空score及GQA重复K/V。MHA/Encoder任意空配置和通用HF generate未全面认证。
- 其他随机初始化dtype、全API/布局/最大容量、长期FP16/AMP模型训练和低位整网质量未在此轮认证。
- 原地Bernoulli仅完整密集目标；out沿用精确shape/无别名限制。randperm不支持BOOL，浮点dtype须保持索引精确；8次碰撞耗尽会明确失败。
- 提交同步；randperm复用通用排序的兼容路径，未做性能或功耗基准，不承诺大规模洗牌加速。
- 通用torch.amp.GradScaler入口、原始NPU Tensor保存/deepcopy、通用torch.compile和GPU+NPU协同仍为后续工作。
- 本轮本机验证没有覆盖其他驱动、其他设备或新机器安装；公开source-release保持不变，未推送GitHub。

## 源与组件指纹（SHA-256）

```text
torch_intel_npu.py
a6f5b93c2f1052d680ed89142967040b72cb9140f8253bdbd7222ff9a6980e27
torch_npu_autograd.py
454dc711240e48c121aaba84320ecf2bd39ccf64f897d21fde189ce709046115
torch_npu_rng.py
04506726435d348b6959ca75917592822677ba04e2fc294f4953b55f69b77750
torch_npu_sdpa_extended_demo.py
26ef466fb33dc3f5a89ccff3adaf6d7dbb9717c1b152237cda5b987aad578afd
torch_npu_sdpa_dropout_demo.py
d6de95d6a8590bd3a82e02c1c7ceea19d994ab7389d32fe858520f2258dc5f3a
torch_npu_rng_extended_demo.py
91febf51206b85c4bc4ac999a0b332256d70bbf7d1abafd2cdcccfba76a99791
test_torch_npu_sdpa_extended.py
280fb217a4fbad8e73f917db2d1a765e50b29de91d02e8d79aa9e4368301d591
test_torch_npu_rng.py
74d46de02abaa0d9f2137a2c8dd945a04a71eff5d31b6502329a7b7b96c62da2
npunlock-probe/torch-device/backend.cpp
9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba
npunlock-probe/torch-device/_torch_intel_npu.pyd
433757698bbc704a44da3ca9c3c3355d3fac49ab19b82d06f0d09790aa4c6ea0
npunlock-probe/session-runtime/npunlock_session.dll
1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90
npunlock/tests/fixtures/npu4000/resident/manifest.json
d7b35afdcd6270c435f1b644af2c3fec5d3e0e9a16cb44c8860a3316356eb6bb
```

## 清理与交付

所有测试进程退出后，按空间策略删除本轮15份正式真机JSON，共113,347字节；保留实现、测试、复现入口与本报告。删除前后828份受保护源码/内核/运行组件/报告SHA-256一致，随后补充本节及空间台账。无新增环境、模型、图表、磁盘checkpoint或构建临时文件。公开source-release未改，未推送GitHub。
