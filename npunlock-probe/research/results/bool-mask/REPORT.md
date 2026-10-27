# BOOL、条件选择与布尔 attention mask 验收

日期：2026-10-19。Windows x64 / Core Ultra 7 258V，NPU `8086:643e`，驱动API 1005540、编译器8.3，Torch 2.14.1+cpu；沿用`.conda/npunlock`环境。

本轮继续补普通PyTorch设备后端兼容性：新增真正的BOOL存储、比较、条件选择及布尔SDPA mask。**14组新增硬件验收通过，共2,429次NPU图调用；41组原功能回归通过，共5,778次调用；48项离线检查通过。** 终版验收各进程关闭后资源归零，无context failure。开发初版曾有一次提交失败，原因分析和修复记录保留在下文。

这是本地开发版，未推送GitHub，`source-release`发布快照未修改。没有新增Python运行依赖、下载模型或更换系统驱动。本轮没有做性能、功耗或完整语言模型质量基准。

## 接口与边界

| 功能 | 支持范围 |
|---|---|
| BOOL设备存储 | 每元素1字节，rank 0–4、1–1,048,576元素；含奇数长度、scalar、复制、clone、fill/zero、cat/stack及有界部分/广播view只读整理 |
| BOOL转换 | 与FP16、FP32、INT64双向转换；按非零判真，NaN/Inf为真、±0为假，真转为数值1；目标INT64仍限524,288元素 |
| eq/ne/lt/le/gt/ge | FP16/FP32/INT64/BOOL；广播、设备Tensor和无梯度CPU零维/Python实数常量；输出BOOL、不建立比较梯度。支持浮点promotion，拒绝INT64与浮点混用 |
| BOOL逻辑 | Tensor的logical_not/and/or/xor及bitwise_not/and/or/xor；二元广播。未提供一般整数bitwise和Python标量bitwise重载 |
| 三参数where | BOOL条件，可广播及标量分支；FP16/FP32/INT64/BOOL结果，拒绝INT64/浮点混用；浮点一阶梯度含广播归约、混合dtype和saved mask version保护 |
| masked_fill | BOOL mask，输入可广播；值为Python实数或与输入dtype匹配的零维Tensor。浮点输入与NPU Tensor value的一阶梯度可用；CPU零维值限无梯度常量 |
| masked_fill_ | 完整可写存储、shape不变；不提供一般可微inplace，INT64目的仍须连续；支持no_grad更新及非grad数据，检查version递增 |
| out | 比较/逻辑/where要求精确shape/dtype、完整连续输出且不与输入别名，无可微out。masked_fill的ATen Scalar_out/Tensor_out已测，公开torch.masked_fill没有out参数 |
| 布尔SDPA mask | 沿用rank 3D/4D FP16/FP32 Q/K/V、batch/head广播、分数<=65,536等原限制；BOOL True表示保留，False屏蔽。支持部分mask读取、全屏蔽行零输出/梯度和Q/K/V一阶梯度 |

API语义核对PyTorch官方[where](https://docs.pytorch.org/docs/2.14/generated/torch.where.html)、[masked_fill](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.masked_fill.html)及[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。布尔SDPA mask与masked_fill的True含义不同：前者保留，后者填充值。

没有新增BOOL sum/any/all、单参数where/nonzero、masked_select或动态布尔索引；不支持空Tensor、部分目的写入、通用AMP、RNG/Dropout、GQA、异步stream/event或CUDA专属扩展。BOOL标量item仍须覆盖完整单元素存储。普通算术的输入契约没有整体放开。

## 实现

原生`backend.cpp`仅扩展工厂允许的dtype以加入BOOL，重建PYD及其构建指纹。原生完整allocation、offset-zero、无重叠输出等绑定保护保留；没有把部分指针直接交给NPU。

已有byte-DMA的Broadcast、Transpose、Slice、Gather、Concat扩展到u8传输。Python设备派发、复制/转换及autograd新增对应入口，依然不依赖OpenVINO Python/runtime。INT64与浮点之间的直接数值cast仍未开放。

新增一个SHAVE C内核`resident_mask.c`，使用完整F32载体中的原始字节packet：16字节静态头、BOOL/数据输入、8字节对齐padding。F32仅是载体；数据由DMA打包，CPU只准备shape、操作码、dtype码及字面常量。

每次最多处理8,192个逻辑元素，并检查整个packet<=65,536个F32载体元素；大Tensor分tile并在设备合并。INT64按原始双字比较，浮点按IEEE编码处理NaN、±0、Inf与subnormal；不转成低精度浮点比较。where选择分支的原始字节，不通过`mask*x+(1-mask)*y`模拟，反向也用条件选择，从而避免未选分支的NaN/Inf污染结果。

mask算子及SDPA的测试区间核对无宿主下载。初始化上传、标量常量上传、显式CPU对照下载和已有分类错误标志不在这个结论内；统计也不涵盖驱动DMA。masked_fill的溢出校验只操作CPU字面常量，不下载NPU数据进行计算。

## 新增硬件验收

| 组别 | 主要覆盖 | NPU图调用 |
|---|---|---:|
| storage | scalar至rank4、奇数长度、密集/partial/broadcast view、BOOL双向cast、INT64高字、最小half/FP32 subnormal、BOOL item | 57 |
| compare FP16 / FP32 / INT64 / BOOL | 六种比较、特殊值成对广播、标量、out；half全编码及BOOL逻辑 | 346 / 94 / 94 / 156 |
| where FP16 / FP32 / INT64 / BOOL | 广播及浮点梯度、标量重载、原始字节选择、NaN/Inf梯度、masked_fill input/value梯度、ATen out和inplace version | 95 / 80 / 45 / 49 |
| mixed | FP16输出、FP32零维参数的三组严格梯度归约反例 | 57 |
| sdpa FP16 / FP32 | BOOL True/False标量及部分二维mask、batch/head广播、整行屏蔽与Q/K/V梯度 | 195 / 156 |
| contracts | 20项非法metadata/别名/越界/未支持调用，以及saved mask version | 5 |
| training | 普通FP32带padding/因果mask的attention网络四步独立CPU对照 | 1,000 |
| **合计** | **14组通过** | **2,429** |

FP16比较覆盖全部65,536种编码，六种关系与CPU精确对照。FP32覆盖NaN、Inf、±0、正负subnormal；INT64覆盖有符号64位上下界、`2**53+1`等数值，不丢高字。where前向按选择结果的原始字节对照，梯度包含NaN/Inf种子，未选位置必须为零。

BOOL SDPA中，全屏蔽输出及相应Q梯度执行严格零值检查；整个mask为False时，全部输出及Q/K/V梯度严格为零。普通非零结果及梯度按既有FP16 `rtol=.005/atol=.0006`、FP32 `3e-5/3e-6`对照。比较、where搬运和混合标量反例使用零容差，没有放宽容差掩盖差异。

20项拒绝覆盖错误condition/device/shape/dtype、INT64/浮点混用、错误/别名out、非BOOL逻辑、非零维或不匹配的填充值、half标量溢出、部分BOOL写入、BOOL转INT64容量超限、空/rank5 Tensor、动态where/布尔索引和异步复制。它们均在NPU图提交前拒绝；修改保存的mask后反向也在提交前报version错误。

每组live buffer无净增量；每个最终进程shutdown后live/cache/scalar/quarantine/session均为零，context_failed=False。

## 普通模型训练

95参数的小型attention网络：四个Linear实现Q/K/V与输出投影，双head SDPA、masked_fill、mean、分类head与CrossEntropy。padding由NPU INT64 `tokens!=0`产生，和BOOL因果buffer通过AND组合；masked_fill清除padding输出。模型和BOOL buffer均通过普通`model.to('intel_npu')`迁移。

两个固定合成样本，FP32，SGD `lr=.03, momentum=.5, foreach=False, fused=False`。CPU参考显式选择SDPA math backend，CPU/NPU各自训练四步；逐项比较loss、全部参数梯度和参数更新，阈值`rtol=3e-5, atol=3e-6`。

| 步 | NPU loss |
|---|---:|
| 1 | 1.1517567635 |
| 2 | 1.1427985430 |
| 3 | 1.1297432184 |
| 4 | 1.1151092052 |

四步对照通过。这验证本例的组合训练语义，不表示完整Transformer、真实数据集质量、长期收敛、LLM训练或能效优势。

## 诊断与数值修复

开发初版BOOL存储及cast可运行，但FP32比较发生一次`zeFenceHostSynchronize ... 0x70000001`提交失败。汇编检查发现比较逻辑生成`.rodata4.__jt__`跳转表和独立辅助函数调用，这些绝对地址模式不在新内核的安全代码替换约束内。改为强制内联、`-fno-jump-tables`，构建时拒绝Function call、LJTI或rodata模式。终版ELF为4,828字节。此修复同时改变两项编译约束，没有再次运行故障版本逐项做硬件因果隔离。

另发现CPU `fill_(65520.)`在half中可产生Inf，而`masked_fill(...,65520.)`会拒绝溢出。新校验改为对CPU零维字面常量执行masked_fill的严格转换检查，再提交设备任务；不是读取设备数据回退CPU。

额外混合dtype检查发现：FP16向量和FP32零维参数的where输出为FP16，CPU会先将归约梯度舍入到输出dtype，再转回输入dtype。原实现直接保留FP32归约结果，七项输入时NPU为2.332763671875，CPU为2.33203125。仅修正Where反向的归约/转换顺序，保留其他算子的FP32 opmath策略。

| 归约项数，种子为half 0.3333 | 修复后CPU/NPU的FP32标量梯度 |
|---|---:|
| 7 | 2.33203125 |
| 100 | 33.3125 |
| 127 | 42.3125 |

三组零容差通过。基础13组先完整执行；加入混合反例修复后补测mixed，并重测FP16/FP32 where及storage，表中取最后通过的记录。其他已测路径未再变更。

## 回归及离线验证

| 回归 | 组数 | NPU调用 |
|---|---:|---:|
| device Quick | 33 | 4,162 |
| layout views | 3 | 630 |
| 原浮点SDPA masks，含mask梯度 | 2 | 744 |
| Embedding contracts / edges | 1 / 2 | 33 / 209 |
| **合计** | **41** | **5,778** |

device Quick用时134.11秒；120秒faulthandler堆栈为诊断快照，最终状态通过。上述回归在额外Where混合标量修复前完成，随后只调整该反向的输出dtype归约舍入，原算术/矩乘/浮点SDPA及Embedding路径未改。

离线geometry/lowbit/新packet ABI共11项、Python API 32项、source export 5项，共48项通过。新增3个packet单元检查覆盖奇数BOOL导致的非对齐payload、输出大于输入时的padding、载体容量及操作码拒绝。默认验收按case/精度隔离进程，14任务分派及遇失败停止的mock检查通过；新C/ELF/汇编、脚本、测试和CI workflow均在源码导出选择中。未执行远程GitHub CI。

## 复现与下一步

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_mask_demo.py
.\run_torch_npu.ps1 -Script .\torch_npu_mask_demo.py -ScriptArguments @('--case','mixed')
.\run_torch_npu.ps1 -Script .\torch_npu_mask_demo.py -ScriptArguments @('--case','training')
.\run_torch_npu.ps1 -Quick -Device
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_geometry test_torch_npu_lowbit test_torch_npu_mask_packet
.\.conda\npunlock\python.exe -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
.\.conda\npunlock\python.exe -B tools/test_source_release.py
```

原生扩展通过`build_torch_npu_device.ps1`重建。修改mask C后需已有MoviTools，再运行`npunlock-probe/research/build_mask_kernel.py`；正常使用采用随源码提供并校验manifest的ELF，不需要用户重新安装MoviTools。
已导入旧后端的Python进程需要重启，以加载新增BOOL存储的PYD。

下一步优先实现设备RNG/Dropout的种子、状态恢复、采样和反向契约；BOOL归约和动态索引仍是独立缺口。当前packet路径以正确性/兼容性为目标，未认证吞吐、延迟或跨平台支持。

## 实验产物清理

删除本轮35份原始JSON、日志和Movi worker响应，共146,324字节；清理前后696份保留文件指纹一致，随后补充本记录与存储台账。逐文件核对目标位于实验目录内，并拒绝reparse point，没有递归删除目录。本目录只保留本报告；保留新C源码、最终ELF/汇编、manifest、构建/测试脚本及当前PYD。未生成图表或下载模型。

## 最终源码指纹

```text
torch_intel_npu.py
45367b3b6c88fc5c5b4a08bb2fb963db2bce9212556e162ae9f56cb5537667e0
torch_npu_autograd.py
b950ae2d0c7478d9b16737871d9b9470f8cb615dfefa9a6955e3261c5b302150
torch_npu_mask_demo.py
dad3aea710dad91f0e0277cb4a79d5978a9ce76a2cc82532de55d96b1578051f
test_torch_npu_mask_packet.py
7e2596bd4b014c92b3fa949abb70c555199da56bd6c82194990164d524a686cf
npunlock/python/npunlock/npu4000.py
b7fb8a4b765fe80bcf5c7ae6d86fb19f8c487cd4f056d0036a5f6c9545e19f14
resident_mask.c
20a24b450c7e0ade52939bdd853db2b414d41b16864313e9fef4fe1f1b5b227e
resident/mask-packet.elf
6efd874dd10851cc0b7c950f4f9ce1ae311eb9417bf66abe97897321fb80ba0c
torch-device/backend.cpp
14ec42e5e7e0a669641194a1bd9f1aae6c3e6323067038fd05c35bdddfb7394a
torch-device/_torch_intel_npu.pyd
1b698a835b0608c687d5e56bd23f1674a9c1ef376b7f78a8ca3b4c3630fee3f4
```

文件集的主机DLL/session DLL沿用原版本；当前SHAVE源码及ELF/汇编哈希由resident/manifest.json记录。完整支持边界同步至根目录COMPATIBILITY.md和中英文README。
