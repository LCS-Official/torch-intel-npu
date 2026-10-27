# S28：SDPA高维、GQA/MQA与容量扩展

日期：2026-10-23。本地开发版；Windows x64、Core Ultra 7 258V / Lunar Lake `8086:643e`、驱动32.0.100.5540、图编译器8.3、Python3.12.14、Torch2.14.1+cpu、NumPy2.5.3。

本轮继续扩展attention兼容性。普通SDPA复用既有FP32 SHAVE矩乘、完整行softmax与byte-DMA；没有新增C/ELF、重建DLL/PYD或安装依赖。配套resident内核仍为S27的62枚。

最终25份真机记录、76组顶层检查全部通过，共28,521次NPU图提交；116项离线检查通过。这里只对报告列出的用例承诺验证，不表示任意Transformer/LLM或通用CUDA兼容性完成。

## 当前契约

| 领域 | S27前 | S28 |
|---|---|---|
| Q/K/V rank | 同rank 3D/4D | 非空rank2–64，可以不同rank并广播批次 |
| GQA/MQA | 公开SDPA拒绝enable_gqa=True | Q头数为K/V头数的正整数倍，K/V头数相同；三者rank>=3 |
| GQA前向 | 无 | 每个KV头按连续组在设备上重复；prefix展平后用rank3 DMA载体，不为rank64输入额外创建rank65 Tensor |
| GQA反向 | 无 | 共享KV头贡献先按组在FP32归约，再按原批次广播形状归约并转回输入dtype；保留原Q/K/V版本检查 |
| score总量 | 1,048,576 | 实现检查上限67,108,864元素，沿用完整行分块；实际超过旧上限的形状见下 |
| 各广播Q/K/V和输出 | 旧16,777,216元素检查，错误文字仍写1048576 | 与普通存储统一为67,108,864元素，错误文字同步更新 |
| 专用保护 | key行宽<=65,536；C内核因果长度编码<=1,048,576 | 两者保留；Python显式检查因果query长度，避免仅扩大总存储后C内核静默提前返回 |

单allocation256MiB、各dtype逻辑元素数和1,536图缓存保持不变。FP16输入仍转换为FP32计算，输出转回FP16。完整attention矩阵与GQA重复KV仍占用设备共享缓冲；容量检查通过不意味着在所有形状下都有足够图资源、总内存或实用吞吐。

mask继续要求设备BOOL或query dtype/FP32浮点，可广播到既定score形状；BOOL True表示保留。显式mask与is_causal不同时使用。GQA接通浮点mask梯度、负scale、因果及Dropout，保持全屏蔽行零输出/梯度。

## 新增真机用例

入口：根目录`torch_npu_sdpa_extended_demo.py`，每个case、每种精度独立进程。保留独立CPU模型/输入，NPU没有CPU payload算术回退。

- `shapes`：rank2、不同rank广播、rank5及rank64因果SDPA。前向和Q/K/V梯度对照CPU Torch math核。
- `gqa`：9个配置，包括4Q/2KV头、6Q/1KV的MQA、不同rank与浮点mask梯度、部分步长view与BOOL mask、rank64、Q/K/V相同头数、Q/K/V各自单侧梯度。部分view比较完整base梯度。
- `dropout`：GQA浮点mask、MQA因果的p=.25，以及p=1。使用独立CPU Philox生成的同一mask参考；核对精确RNG推进量、反向不再抽样、CPU RNG不变和计算期间无payload下载。
- `large`：真实Q=`(1,1,1025,1)`、K/V=`(1,1,1024,1)`，1,049,600个score（旧上限+1024），负scale和因果前向及全部Q/K/V梯度；与CPU Torch math独立比较。这是head特征维度为1的容量边界测试，不是常见大模型吞吐或最大容量认证。
- `training`：259参数的LayerNorm、Q/K/V投影、4Q/2KV头GQA、残差及输出head。输入`(2,1,9,8)`，形成rank5 attention；显式FP32 MSE、SGD lr=.05、foreach/fused=False，独立CPU/NPU各三步，比较全部输入/参数梯度与更新参数，记录固定图缓存。
- `contracts`：非法头数、KV头数不一致、rank2启用GQA、score超存储边界、key行宽、因果query header、CPU mask及mask/causal冲突，验证拒绝前不提交图、不推进RNG；另验证原K版本修改在反向被捕获。

## 参考与数值门槛

逐元素比较沿用S26/S27：FP32 rtol=4e-5/atol=2e-6，FP16 rtol=.004/atol=.001。算子梯度另检查相对L2<=1e-4（FP32）或.01（FP16）。训练中所有参数梯度仍逐个执行同样的逐元素检查，附加相对L2检查用于输入梯度及拼接后的整体参数梯度。

这些门槛按模型精度配置选择。FP16配置中的FP32 loss和浮点mask梯度也使用FP16模型门槛，不宣称它们通过了FP32模型配置的更严格逐元素阈值。

CPU零Dropout参考使用原生Torch math SDPA及enable_gqa。相同Dropout mask的CPU参考复用已验证的项目oracle；CPU离线测试另将head布局与原生repeat_interleave/反向对照，并把GQA输出和梯度与Torch math核独立比较。原生CPU repeat_interleave在rank64上的反向会遇到内部rank65限制；其head布局测试采用4D flatten后原生重复并恢复rank64，避免把CPU参考实现的这个限制误算成NPU失败。

## 诊断记录

1. 首次shapes脚本在保存CPU RNG快照后又生成随机gradient seed，导致“CPU RNG不变”断言失败。把快照移到seed生成/上传之后，生产实现不改，重跑通过。
2. 初版三步训练使用“每个参数都必须通过纯相对L2”附加条件，第一步K投影bias未通过：CPU梯度范数7.00762e-10，最大绝对差1.09685e-9，纯相对L2为1.58682。原逐元素检查已通过。K的恒定bias对softmax是共同平移，理论梯度为零，该参数比率在数值上病态；未把此比率称作模型误差。
3. 最终训练保留每个参数原逐元素容差，记录近零参数的范数/绝对差，附加相对L2统一检查整体参数梯度向量及输入梯度；两种精度保留各自原1e-4/.01阈值。报告中给出这一判据变化及原始失败，不宣称初版“逐参数纯相对L2”已经通过。
4. 旧S2/S13“百万score应拒绝”fixture随当前边界改成8193×8192（超过67,108,864）；原GQA应拒绝变成非法Q/KV头数，原rank2应拒绝变成rank1。新增正向用例覆盖这些原来合法性变化，没有删除容量/维度的提交前保护。

## 仍未覆盖

- 空批次/空序列/零特征SDPA仍在提交前明确拒绝；没有实现FlashAttention、NestedTensor或高阶梯度。单行key和因果query的专用上限保留。
- 没有认证67,108,864 score的全部上限形状，也没有把native MHA/Encoder专用入口的旧工作区保护一并扩大。GQA设备重复KV属于兼容性实现，不宣称内存优化或速度提升。
- 259参数三步训练是新接口的整网组合验收，不是大型LLM训练、实际数据集质量、长时收敛或CPU/GPU/NPU性能对比。
- 没有修改冻结DPU/低位量化质量、GPU+NPU协同、异步执行或通用CUDA二进制支持。source-release与GitHub未修改；全部为本地开发版。

## 复现

```powershell
foreach ($precision in @('float32','float16')) {
    foreach ($case in @('shapes','gqa','dropout','large','training','contracts')) {
        .\run_torch_npu.ps1 -Script torch_npu_sdpa_extended_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
```

新增五项离线检查位于`test_torch_npu_sdpa_extended.py`，已加入offline.yml；不加载NPU后端，用于几何、容量/header保护、head排列/归约以及CPU GQA参考的检查。真实SHAVE和autograd验证由上述硬件用例负责。

## 新专项结果

12个专项配置全部通过，共5,526次NPU图提交。下表每种配置只计最后一次通过记录；开发诊断和重复运行不累加。

| case | FP32调用 / 活跃buffer峰值字节 | FP16调用 / 活跃buffer峰值字节 | 最大附加梯度相对L2（FP32 / FP16） |
|---|---|---|---|
| shapes | 180 / 29,892 | 232 / 29,600 | 2.90410e-7 / 0 |
| gqa | 393 / 16,260 | 497 / 15,806 | 2.15706e-7 / 1.08146e-7 |
| dropout | 179 / 9,652 | 218 / 9,260 | 1.26145e-7 / 0.000514059 |
| large | 575 / 17,609,496 | 588 / 17,603,348 | 4.43274e-7 / 0.0000120826 |
| training | 759 / 34,756 | 1,851 / 31,808 | 1.87131e-7 / 0.00146685 |
| contracts | 23 / 4,526,232 | 31 / 2,264,880 | 无可微数值比较 |

`training`的附加相对误差统计是输入和整体参数梯度，近零单参数仍逐元素验证，不计其病态的纯相对比率。所有其他算子梯度按各输入Tensor比较。buffer峰值是本项目跟踪的活跃Tensor allocation，不包含驱动/编译器内部存储，也不是任务管理器总NPU内存。

| 三步GQA网络 | loss序列 | 缓存图序列 |
|---|---|---|
| FP32 | 0.0672759339 → 0.0595084541 → 0.0531175919 | 117 / 117 / 117 |
| FP16 | 0.0672282204 → 0.0594639480 → 0.0530970991 | 170 / 170 / 170 |

FP32三个步骤K bias参考梯度范数为7.00762e-10、9.42623e-10、1.02330e-9，对应最大绝对差1.09685e-9、3.55612e-10、7.11225e-10。FP16本例没有参考范数<1e-7的参数；两个精度的参考初始化数值不同，不能用loss列直接比较精度优劣。

## 既有路径回归与生命周期

| 入口/配置 | 记录数 / 顶层检查 | NPU调用 |
|---|---|---:|
| S2 sdpa：两种精度 | 1 / 2 | 1,283 |
| S2 sdpa-masks：两种精度 | 1 / 2 | 744 |
| S2 training：两种精度 | 1 / 2 | 4,252 |
| S2 contracts | 1 / 1 | 34 |
| S9 operators FP32 | 1 / 1 | 642 |
| S9 operators FP16 | 1 / 1 | 782 |
| S13 sdpa FP32 | 1 / 4 | 2,032 |
| S13 sdpa FP16 | 1 / 4 | 2,102 |
| S13 contracts | 1 / 1 | 0 |
| S10 training FP32 | 1 / 1 | 3,101 |
| S10 mha FP16 | 1 / 1 | 1,082 |
| Device Quick | 1 / 33 | 4,171 |
| Training Quick | 1 / 11 | 2,770 |
| 回归合计 | 13 / 64 | 22,995 |
| 新专项+回归总计 | 25 / 76 | 28,521 |

76组为各结果文件顶层check数量；部分check内部有多个shape、mask或训练步骤，不把它们再重复计入该总数。每个配置只统计最后通过的结果。

25份最终记录均验证`closed=True`、`context_failed=False`，live buffers、cached buffer blocks、scalar buffers、sessions和quarantined bytes全部归零。S13回归包括因果跨块、BOOL/浮点全屏蔽行、Dropout及65537 query/单key尾块；标准MHA和Encoder训练路径保持通过。

按offline.yml运行的离线检查为根目录79项、npunlock Python API32项、source-release工具5项，共116项。62枚ELF、对应汇编及全部manifest源码哈希通过；S26 C++/PYD、session DLL及C/H构建来源校验一致，未重建。发布快照Git状态干净，未推送GitHub。

## 源码指纹（SHA-256）

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `7388efd44f9bc7d035e1328cbe70b81044c6694633f4a70033e7b59f0dbd1d96` |
| torch_npu_autograd.py | `bc8a022dba6fb1b8701fe3ee230c008d541b4e425b0145b8f187e04c7923bc16` |
| torch_npu_sdpa_extended_demo.py | `547c428aa9de2f3e04c9404b73ac82a95ece2fdeadb2edf6a6f38f5b8a034ca5` |
| test_torch_npu_sdpa_extended.py | `a12eb90588117226baa54b74a3333c8397d24b86305d022acd64af0706a0e4fe` |
| torch_npu_sdpa_demo.py | `7f949995299777a971270a4cbe1b19d19290787787968065253e627dcd9e1813` |
| torch_npu_attention_rows_demo.py | `423315d7df278901f7490d52ae4f1524bc4f2d24f6a6b14d8534e29981846f7f` |
| resident/manifest.json | `329ec9430d4f9fbe2b8b721b5fc5d72aef38f2d6ceb44a33322aa6515cd64cfc` |

## 验收后清理

按用户保留策略，删除本轮52份原始JSON/日志，共177,912字节（0.170MiB）。删除前逐项校验为工作区内普通文件，不递归删除目录；清理前后967份保留源码、报告、内核及必要运行时文件SHA-256一致，随后补充本段和WORKSPACE_STORAGE台账。保留新SDPA/GQA派发/反向、正式硬件与CPU验收脚本、CI列表、更新后的旧边界fixture及双语说明；无依赖安装、模型下载、SHAVE构建、宿主重建、图表或磁盘checkpoint/图缓存。
