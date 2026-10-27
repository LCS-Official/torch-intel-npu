# S18：AMP卷积的分批舍入归因与残差训练

日期：2026-10-20。Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu。
本地开发版，source-release未改、未推送GitHub。继续按兼容性优先处理S17卷积严格差异，未做性能优化或低位质量调优。

## 结论与实际改动

S17的卷积权重梯度严格差异已定位到本机CPU Half逐batch中间舍入。六种局部几何中，NPU与独立FP32累加参考的前向/输入梯度/权重梯度/偏置梯度均通过原严格阈值；200个权重梯度值全部等于FP64参考最终舍入到FP16的结果。
现有NPU FP32累加方式保持不变。原S17与原生CPU Half的严格失败记录保留，没有将CPU原生轨迹改记为通过。
新增独立CPU精度参考与双参考训练验收：原SiLU网络、三卷积残差网络各四步，损失/全部参数梯度/参数更新通过独立FP32累加参考，checkpoint精确重放；原生CPU Half差异同时单独记录。

生产代码只修复一处scaler边界：加载空checkpoint时，此前访问scale键产生KeyError；现沿用PyTorch的明确RuntimeError，并保持原状态。先用单元测试复现失败，再修改共享load_state_dict入口并回归。
没有改变卷积求导公式、autocast策略、普通非AMP路径、SHAVE ELF、C++扩展或session DLL；更新本轮Python代码无需重编，没有新增依赖。

## 归因证据与参考范围

核对[PyTorch v2.14.1 ConvolutionMM2d.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/ConvolutionMM2d.cpp)：逐batch调用GEMM，beta=1，梯度缓冲dtype为scalar_t；本机Half实验可由分批舍入逐位复现。
该证据结合下表的本机对照说明S17这一差异的来源，不推广到所有CPU实现、CUDA/cuDNN或任意模型。

独立CPU参考在F32/F64计算中使用相同、已舍入的FP16输入/权重/上游梯度；CPU参考从未调用NPU计算结果完成整网训练。
FP32累加控制的卷积前向、输入/权重/偏置导数在CPU计算，每个卷积结果/导数最后回到FP16，再经autocast转换的反向回到FP32主参数。它是独立数值控制，不是标准CPU Half实现的别名。
FP64参考只用于局部数值对照，不进入NPU执行路径。FP32主参数/梯度存储不代表所有卷积反向中间量都保持FP32输出；当前autocast的FP16权重节点仍会发生最终half梯度舍入。

另有无随机数的精确二进制分数CPU控制：三个batch贡献为2048、1、-2048时，逐batch Half结果为0，FP32/F64全量累加结果为1。该检查用于验证参考的舍入边界，不依赖某台CPU的随机数或GEMM向量细节。

## 六种局部真机检查

全部为groups=1、非转置、紧凑卷积路径，FP32输入/主参数经标准autocast进入FP16卷积；同时计算输入、权重和偏置梯度。
没有host模型payload计算回退；算子区间host_download与host_device_copy增量均为0。CPU参考和结果下载在区间之后。

| 几何 | 输入NCHW | 输出通道/核 | stride/padding/dilation | 原生CPU Half与全量单次half舍入不同的权重元素 |
|---|---|---|---|---:|
| batch1 | 1×2×4×5 | 3 / 3×3 | 1,1 / 1,1 / 1,1 | 0/54 |
| batch2 | 2×2×4×5 | 3 / 3×3 | 1,1 / 1,1 / 1,1 | 12/54 |
| batch4 | 4×2×4×5 | 3 / 3×3 | 1,1 / 1,1 / 1,1 | 19/54 |
| stride | 2×2×5×7 | 2 / 2×3 | 2,1 / 1,1 / 1,1 | 5/24 |
| dilation | 2×1×7×6 | 2 / 2×2 | 1,1 / 1,0 / 2,1 | 2/8 |
| pointwise | 2×2×5×5 | 3 / 1×1 | 1,1 / 0,0 / 1,1 | 2/6 |

逐batch Half控制与原生CPU的200个权重梯度全部相同；原生CPU与FP64最终舍入结果有40/200不同，NPU为0/200。
六种几何的NPU全部结果通过rtol=3e-5、atol=3e-6的独立FP32累加对照。FP64未舍入真值仍有正常的half量化误差，权重最大绝对差为约2.20e-4至7.88e-4；不能把“等于最终half舍入”称为FP64精度计算。
有限样本不认证完整数域的正确舍入、极端值或更大streaming卷积路径。

## 原S17网络与新增残差网络

两侧各自从相同初始化独立执行，不逐步用NPU参数覆盖CPU。三个训练对象分别为标准NPU模型、CPU FP32累加控制及CPU原生Half控制，后两者都是CPU参数、数据与优化器；NPU仍使用普通PyTorch模块和原标准SGD。
训练使用FP32主参数/梯度、FP16 autocast、显式CPU控制的FP32 CrossEntropy、lr=.02、momentum=.5、foreach=False/fused=False、init_scale128/growth_interval2。所有网络的梯度检查/反缩放在NPU，每步只允许标签校验与unscale标志各一次下载；step/update无模型payload下载。
严格阈值仍为rtol=3e-5/atol=3e-6；原生CPU Half不符合严格阈值时继续保留错误，并另行检查既有FP16界限rtol=.003/atol=.0005。

| 网络 | 配置 | 四步NPU loss | 独立FP32累加控制 | 原生CPU Half严格梯度 |
|---|---|---|---|---|
| 原SiLU网络 | Conv(1→2,3×3)→SiLU→AvgPool→Linear(2→3)，29参数，batch2/4×4 | 1.288599730 / 1.282931805 / 1.274547219 / 1.264938593 | 损失/全部梯度/更新通过 | 四步仍有stem.weight差异 |
| 三卷积残差 | stem Conv(1→2)，两层Conv(2→2)残差块，ReLU/AvgPool/Linear，105参数，batch4/6×6 | 1.175162077 / 1.172701478 / 1.169088244 / 1.164948940 | 损失/全部梯度/更新通过 | 四步仍有权重/偏置差异 |

原SiLU配置与S17的初始化、输入、标签[0,2]相同，前三步loss完全相同，原权重梯度差异3.814697e-6、7.629395e-6、1.525879e-5也重现；第4步差异3.814697e-6。
残差的原生CPU比较还包含上游梯度与偏置的Half舍入/轨迹差异，最大记录差3.814697e-5；本轮未逐项归因这些残差整网差异，不把全部问题都归为单个局部权重累加。
两个网络在第3步后把CPU权重、优化器及scaler状态存入内存BytesIO，新模型/优化器恢复后重放第4步：loss、参数、动量和scaler状态精确一致。这里的精确参数/动量检查使用rtol=atol=0，CPU独立轨迹比较仍使用上述严格容差。
这是有限小网络四步与断点检查，不代表长期收敛、真实任务质量、完整torchvision ResNet或Transformer AMP已认证。

## 验收、构建与限制

| 独立进程 | 状态 | NPU图提交 |
|---|---|---:|
| 六种局部几何 | passed（独立FP32累加参考） | 288 |
| 原SiLU网络四步/恢复 | passed（独立FP32累加参考） | 1,254 |
| 三卷积残差四步/恢复 | passed（独立FP32累加参考） | 2,333 |
| 既有S17 scaler contracts | passed | 32 |

正式四进程合计3,907次提交，均在shutdown后live/cache/scalar/session/quarantine归零，context_failed=False、closed=True。此处passed不能读作原生CPU Half严格检查全部通过。
探索性ReLU小网络及标签修正前的试跑不计入正式结果。小网络后来按原S17标签[0,2]单独复测，局部/残差的几何、数据和算法未变；所有正式进程使用相同生产Python与原生二进制。
82项离线检查通过：核心/模型45项、npunlock Python API32项、source exporter5项；新增三项CPU参考控制，并在原checkpoint测试内加入空状态报错/状态不变回归。CI沿用已有test_torch_npu_amp入口，无新依赖。

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_amp_conv_demo.py -ScriptArguments @('--steps','4')
.\run_torch_npu.ps1 -Script .\torch_npu_amp_demo.py -ScriptArguments @('--case','contracts','--label','s18')
```

本轮只修改scaler空状态边界、增加参考/验收源码及双语说明。通用torch.amp.GradScaler入口、动态索引、广泛模型/优化器、BF16及低位整网质量仍按兼容性矩阵的有限状态报告。

## 代码指纹（SHA-256）

| 文件 | SHA-256 |
|---|---|
| torch_npu_amp.py | 71234cd54621d6f8f30ba6002af00fff1a2e567698cda70543c3f2d56e2269b8 |
| torch_npu_amp_conv_demo.py（最终小网络标签版本） | 1a7d882cfa83f8859286423a37d264d95c4991abaf2ad16678402a647310de06 |
| 同脚本（局部/残差验收，最后小网络标签修正前） | 2d2c887fd4558e8cf65e31f95f248e91a2e825c354b58f007c43913f35cafc0c |
| test_torch_npu_amp.py | 351f5c20319ae95f84f46b8f0288b788d679cf039e5cf03ee112915c0990f7bb |
| torch_intel_npu.py（沿用S17） | 51d4779778d3a10617fdf9bb936c26ec044a48dd4cbb154d6722a91eafb94ae6 |
| torch_npu_autograd.py（沿用S16/S17） | b31d5a05cf0379e0ba22dfbc6a4eec91a6c770dc228ecd51bc121c5b1a56c52e |
| backend.cpp（沿用S17） | 4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83 |
| _torch_intel_npu.pyd（沿用S17） | 3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd |

四份正式结果的生产Python/原生二进制指纹均与最终文件相符；session DLL仍为0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9。

## 清理记录

验收后仅删除本轮四份原始JSON，共15,282字节；逐项核对实际绝对路径位于指定结果目录，无递归删除。清理前后783份保留源码、报告、内核和必要运行时SHA-256一致，随后有意补充本节及存储台账。
source exporter包含777份源码/报告/内核，覆盖新CPU/硬件参考与报告，全部SHAVE ELF/汇编哈希仍有效；没有创建额外源码快照、下载模型、安装依赖、生成图表、重编宿主/数值内核或保存磁盘checkpoint，source-release未改，未推送GitHub。
