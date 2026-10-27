# 设备RNG与普通Dropout验收

日期：2026-10-19。本地开发版，未推送GitHub，`source-release`发布快照未修改。
Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Torch 2.14.1+cpu。
没有安装依赖、下载模型或修改系统驱动。原生allocator/session/PYD沿用BOOL版本，本轮新增SHAVE C及其3,484字节ELF。

## 结论与范围

RNG是普通PyTorch训练迁移的基础：随机初始化、训练态Dropout及随机采样需要它；没有随机操作的eval推理可以不需要。
这次补齐默认设备RNG及普通Dropout子集，不提供CUDA运行时、CUDA扩展兼容或CPU/CUDA相同seed的相同随机序列。

| 接口 | 当前范围 |
|---|---|
| `rand/randn/rand_like/randn_like` | FP16/FP32，rank 0–4、非空、<=1,048,576元素；随机数据由SHAVE计算 |
| `uniform_/normal_` | 默认设备RNG、有限实数参数；uniform上下界在目标dtype范围内且有序，FP32区间宽度有界；normal std非负、参数在FP32范围内；目的存储完整、密集、不可重叠 |
| `bernoulli_(p)` | FP16/FP32目标、Python标量p在[0,1]；不包含Tensor概率分布 |
| 种子与状态 | `torch.manual_seed`及`torch.intel_npu.manual_seed/manual_seed_all/initial_seed/seed/seed_all`；get/set_rng_state和all版本；单设备0，状态是带版本标识的24字节CPU ByteTensor |
| `fork_rng` | 显式`device_type='intel_npu'`，保存并恢复设备状态及PyTorch原有CPU状态 |
| `nn.Dropout/F.dropout` | 非原地、FP16/FP32、一阶梯度；native_dropout返回真正BOOL mask，backward复用保存的mask，不重新抽样 |
| 初始化与训练 | 直接NPU Linear初始化、随机工厂叶Tensor梯度、普通FP32 MLP + ReLU + Dropout + SGD；CPU参考使用相同mask |

普通Dropout的eval/p=0返回输入；native_dropout的eval/p=0提供独立输出和全真mask，p=1提供全假mask。本实现这些特殊情况不推进RNG。
普通有效训练调用生成新mask，前后向在FP32中间精度计算后转回输入dtype。保存的mask受version保护，修改后反向报错；二阶梯度未开放。

未支持或未认证：显式`torch.Generator(device='intel_npu')`及Generator参数、randint/randperm/multinomial、Tensor概率Bernoulli、完整torch.distributions、Dropout2d/3d、原地Dropout、SDPA内部非零dropout、AMP、异步stream/event和动态布尔索引。SDPA仍要求dropout_p=0。
普通随机填充保持原有完整存储保护，部分目的写入/空Tensor仍拒绝。理论容量界限沿用设备存储，不表示本轮已逐一测试全部形状或满容量。

## 实现与可复现性

新增`resident_rng.c`：Philox4x32-10使用32位分肢乘法高字和10轮变换；64位seed作为key，64位counter加元素偏移作为输入，每个元素消耗一个计数器块。
均匀FP16/FP32取首字11/24个高位，形成[0,1)；正态使用首两字的Box–Muller变换，自编range-reduced log、Newton sqrt和cos多项式，无主机libm参与设备计算。
算法参考[Random123 Philox](https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h)，代码为本项目独立实现，未复制第三方头文件或引入依赖。

主机`torch_npu_rng.py`只维护seed/counter及状态序列化；设备提交在现有锁内串行保留计数器区间。状态版本`NPURNG01`，格式为little-endian的8字节magic、uint64 seed、uint64 counter。
新进程注册时读取CPU的initial_seed作为初始seed，不抽取CPU随机值；此后CPU与NPU的随机流独立。`torch.manual_seed`按PyTorch接口同时设置设备，设备manual_seed仅改变NPU状态。
计数器不静默回绕，耗尽时报错；无效参数/无效状态在提交和状态推进前拒绝。运行时提交失败之后不承诺RNG回滚。

动态32字节header上传到NPU，经byte-DMA Concat零填充为完整F32 carrier；seed/counter原始位不作浮点转换。kernel读取header，在SHAVE生成数据，byte Slice/Concat及既有FP16 cast组成结果。
每tile最多8,192元素，header不进入图缓存键，同shape不同seed/offset复用同一图；没有在CPU生成整块随机Tensor再上传。
“无随机数据下载”不等于零I/O：每tile存在32字节控制信息上传，字面参数、编译器和驱动也有自身I/O。
本次计数器/分块测试验证均匀随机数跨调用划分和内部tile边界的逐位一致性；不把它扩大为不同版本/设备之间的可复现保证。

构建沿用MoviTools目标4000xx，`-O2 -ffp-contract=off -fno-jump-tables`。所有辅助函数强制inline，构建断言禁止out-of-line Function call、jump table和rodata，ELF/汇编/源文件哈希写入resident manifest。
开发时发现8-word最小packet会被优化成identity图，输入/输出binding selector歧义在提交前拒绝，NPU调用为0、context健康；修为至少16 words的实际Concat后，以下正式验收全部通过。

PyTorch语义参考[Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)及[random/fork_rng](https://docs.pytorch.org/docs/2.14/random.html)，具体ATen签名核对本机Torch 2.14.1。

## 正式硬件结果

最终相同源码指纹的10个独立进程全部passed，共942次真实NPU调用：

| 组 | NPU调用 | 主要检查 |
|---|---:|---|
| smoke FP32 | 4 | Philox零key/零counter已知向量，8个uniform逐位匹配独立Python整数实现 |
| uniform FP16 / FP32 | 131 / 103 | 8,199元素跨tile逐位匹配、rand_like、非默认上下界、密集转置目的逐位核对、标量Bernoulli；CPU RNG不变；scalar/rank4随机叶梯度、直接NPU Linear初始化 |
| normal FP16 / FP32 | 30 / 24 | 8,199元素对照Python整数Philox + math Box–Muller；randn_like、16,384元素normal_(2,.5)；无计算期数据下载 |
| state | 39 | reseed、uint64最大seed、状态回放/fork_rng、seed/seed_all/all状态恢复、counter跨2^32进位、3+8189+7与8199调用划分一致 |
| dropout FP16 / FP32 | 99 / 72 | 四种p/train组合，前向/反向与相同mask的CPU计算逐位一致；普通nn.Dropout反向、eval及特殊p不消耗RNG |
| contracts | 18 | 9个元数据拒绝不改变state/不提交；in-place叶保护、保存mask的version保护；后面的有效mask创建/梯度准备会提交，不计入拒绝区间 |
| FP32 training | 422 | 四步CPU相同mask梯度与更新对照、参数/SGD/RNG状态恢复后下一步逐位重放 |

normal参考容差：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=atol=.002；近零cos值及多项式的有限误差不等于所有正态数逐位一致。
8,199个标准正态FP32样本mean=0.02323934、std=0.99232501；FP16 mean=0.02324191、std=0.99232876。
16,384个normal_(2,.5)的FP32 mean=2.00314188、std=0.50173116，FP16 mean=2.00313878、std=0.50173223。
这些是有限样本与算法参考检查，没有运行TestU01/完整统计认证，也不认证极端尾部质量或随机采样模型的任务效果。

训练网络是Linear(4,8)→ReLU→Dropout(.25)→Linear(8,2)，3个合成样本、MSE、SGD lr=.03、无动量，四步loss：
`0.1729539037 → 0.1684483141 → 0.1327390820 → 0.1465806812`。
Dropout引入随机性，loss没有逐步单调下降；末步低于首步。
CPU参考使用设备生成的相同mask，而非CPU相同seed的独立随机流，损失/梯度/更新容差rtol=3e-5、atol=3e-6。
第2步后在内存中保存显式CPU参数、SGD state_dict和以CPU ByteTensor表示的NPU RNG状态；恢复后第3步loss、参数及NPU RNG状态逐位一致。未把该检查描述为新进程/磁盘checkpoint认证，原始NPU Tensor序列化仍不支持。
正式训练计数区间内无host payload下载；CPU参考、loss记录与checkpoint快照的显式下载在区间外。

所有10组live_buffer_delta=0；shutdown后live_buffers、cached_buffer_blocks、cached_scalar_buffers、quarantined_buffer_bytes和cached_sessions均为0，context_failed=False、closed=True。

## 离线检查及复现

既有功能回归共35组、5,162次NPU调用，全数passed：设备Quick 33组/4,162次调用/263.745秒；S4 gaps 1组/0次调用；BOOL attention训练1组/1,000次调用。
设备Quick保留43种元数据拒绝且拒绝区间无提交，原“rand不支持”测试替换为仍未支持的整数randint；S4已移除过期的randn/Dropout预期拒绝，仅保留async-copy和native-save两项。
attention网络仍为带INT64 padding比较/BOOL因果mask/SDPA/masked_fill的四步独立CPU训练对照，未使用本次Dropout，检查其原有路径不被影响。
这3个回归进程关闭后相同资源计数也归零，context_failed=False。设备Quick日志中两次120秒faulthandler栈转储为定时诊断，最终在300秒外层截止前完成；未把本轮耗时当作性能基准。

52项离线检查通过：geometry/lowbit/mask/RNG共15项，Python API 32项，source exporter 5项。新增4项覆盖种子/保留、状态往返及无效状态不改变原状态、counter耗尽和packet容量边界。
源码导出selection已检查包含新增RNG Python、C、ELF、汇编、构建及测试/CI文件，没有写入新发布快照。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\run_torch_npu.ps1 -Script .\torch_npu_rng_demo.py
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_geometry test_torch_npu_lowbit test_torch_npu_mask_packet test_torch_npu_rng
.\.conda\npunlock\python.exe -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
.\.conda\npunlock\python.exe -B tools/test_source_release.py
```

本轮只验证此硬件/驱动上的功能与有限数值语义，没有测RNG吞吐、长时间训练、完整Transformer、LLM采样质量或功耗。
下一步优先把已验证的随机mask接入SDPA内部dropout并核对Q/K/V反向，再考虑明确模型所需的离散采样/Generator子集。

## 正式源码SHA-256

```text
torch_intel_npu.py
2e1d013be74663c620e43977a605d2e04988fe448df05d613278919a7b017f21
torch_npu_autograd.py
a2727d655a21e68bc8027357f4e472be88b458b84d82fce5b9f1878ecbd98646
torch_npu_rng.py
44530064b98c910d889a4b0dc91c2426a6c24eb77e70368212070b73f8d1b5e7
torch_npu_rng_demo.py
df29788418378afeb7a3ce386eedb8bcc6e7e31a37b9963252dd2fa6ca3d80bf
npunlock/python/npunlock/npu4000.py
7b18d327e9658a86358147b65504a53506b6c7f78e87528b9824681e9fe8ca4b
resident_rng.c
ab3c8c9437a1df84c0e5d62cb06b428b9f085b0bd4affca6913abd6142e3e928
resident/rng-packet.elf
3d5c8e0019281896d4abfcf516ab592c0a0e3ce26cbc56fa2df8a8f5709d7b52
```

## 临时结果清理

2026-10-19 RNG/Dropout验收后清理：20份本轮原始JSON、日志及Movi worker响应，共80411字节；清理前后704份保留文件SHA-256一致，随后补充文字报告/台账。保留新增SHAVE C/ELF/汇编/manifest、设备RNG与autograd源码及测试；没有下载模型、安装依赖或生成图表。
