# CUDA式模型迁移缺口与向量/标量接口验收

日期：2026-10-19；Windows x64 / Core Ultra 7 258V / Intel NPU 8086:643e，驱动API 1005540、编译器8.3、Torch 2.14.1+cpu。本轮为本地开发版，未推送GitHub。

## 本轮结论

本轮补齐两个常见迁移缺口：向量参与的`matmul/@`及`dot/mv`，以及FP16/FP32的同步`Tensor.item()`。新增9组验收全部通过，共3,400次真实NPU图调用，关闭后live/cache/scalar/quarantine/session归零，无context failure。

普通PyTorch模型迁移仍有明确缺口，尤其是设备索引/Embedding、通用拼接、设备RNG/Dropout、较大Tensor存储、AMP及异步执行。这不是通用CUDA兼容性完成认证，不能给出有依据的“完成百分比”。

## “CUDA兼容”在本项目中的范围

1. **设备使用方式**：已有真实PrivateUse1 Tensor，使用`model.to('intel_npu')`和Tensor运算，部分模型可沿用前向、backward、优化器的写法。
2. **PyTorch算子与训练语义**：需要逐项补ATen派发、布局、dtype、梯度、版本与资源语义；能调用一个API不代表所有参数/模型均受支持。
3. **CUDA专属软件栈**：未提供CUDA运行时、PTX、CUDA扩展、CUDA版Triton/FlashAttention、NCCL或`torch.cuda`兼容层。这些依赖需要移植，不会因实现相似设备接口而自动运行。

本轮改进第2项。没有把NPU并入`torch.xpu`，也没有测试GPU+NPU协同。

## 已补齐的接口

| 接口 | 新的本地范围 | 实现方式与边界 |
|---|---|---|
| `torch.matmul` / `@` | FP16/FP32、rank 1–4；向量×向量为标量，向量×矩阵、矩阵×向量和高维共享向量 | 按PyTorch规则提升向量维度，复用原SHAVE矩乘/bmm，最后移除提升的维度；不新增DPU内核 |
| `torch.dot` / `torch.mv` | dot仅1D×1D；mv仅2D×1D；一阶梯度与精确形状out | 共享matmul路径；反向在FP32中归约批次共享向量的梯度，再转回输入dtype/shape |
| FP16/FP32 `.item()` | 单元素、完整连续存储，允许在grad模式下读取loss | 持锁同步下载2或4字节并返回Python float，不可微，不改变原计算图；保留INT64的精确8字节读取 |
| `float/int/bool(tensor)` | 本轮检查单元素浮点转换 | 复用标量读取；不扩展多元素真值、通用打印、tolist或整数算术契约 |

所有原有dtype/rank/存储、别名与out限制保留。FP16/FP32单Tensor最多1,048,576元素，广播操作数和输出仍须各自满足上限；不支持空Tensor、稀疏Tensor、一般部分存储view、复数或二阶梯度。

实现主要改动在`torch_intel_npu.py`的形状/派发与标量复制、`torch_npu_autograd.py`的向量梯度；原生C++扩展、DLL、SHAVE ELF与编译工具链未修改，也未新增依赖。

[PyTorch matmul语义](https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html)、[dot](https://docs.pytorch.org/docs/2.14/generated/torch.dot.html)、[mv](https://docs.pytorch.org/docs/2.14/generated/torch.mv.html)、[item](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.item.html)为语义参照。版本固定的本机CPU执行用于独立数值对照；当前CPU版本实测允许向量matmul的out参数，本项目只承诺本轮通过的精确shape/dtype、无别名、无梯度out范围。

## 新增硬件检查

入口：根目录`torch_npu_vector_demo.py`，随机种子107，CPU单线程。

| 检查 | 结果与口径 |
|---|---|
| 常规向量前后向 | 每种dtype 48组，共96组；含1D×1D、1D×2D、2D×1D、与3D/4D组合、长度1、完整密集转置、双侧/单侧梯度 |
| 跨tile向量前后向 | 每种dtype 2组，共4组；K=65,537的点积及批次共享向量，覆盖既有65,536元素计算tile分块 |
| out | 共138次调用，覆盖matmul/dot/mv，核对返回同一个out对象、结果形状与CPU数值 |
| dot/mv直接梯度入口 | 在对应常规案例上追加检查，核对共享Matmul autograd入口与CPU一致 |
| 标量item | 每种dtype 27组，共54组；rank 0/1/4的单元素、正负零、正负Inf、NaN、subnormal及最大有限值；每次恰好一个显式下载，无图提交/上传，Tensor版本不变 |
| INT64与Python转换 | ±64位边界、2^53以上整数仍精确；float/int/bool单元素转换通过，读取后原计算图仍可反向 |
| 明确拒绝 | 13项，包括rank-0矩乘、inner/dtype不匹配、CPU混输、dot/mv非法rank、out形状/dtype/别名/可微调用、部分存储标量/向量、多元素item；拒绝区间无NPU图提交或显式主机复制 |
| autograd版本保护 | 保存输入被原地修改后，backward在计算提交前拒绝 |
| 普通模型训练 | FP32 Linear→ReLU→共享向量头，输入(2,3,5)、输出(2,3)，4步SGD(momentum=.5)，独立CPU参数/梯度/损失逐步对照，通过正常loss.item记录日志 |

常规/大向量计算区间显式主机上传、下载、packing相关计数不增加；数值对照的最终下载和用户要求的item读取单独发生。计数不覆盖驱动内部DMA或编译器行为。

CPU比较容差：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005，equal_nan=True。未把通过这些容差当成逐位等同CPU。

模型四步NPU损失：0.0193605479 → 0.0191600230 → 0.0188652482 → 0.0185308438。小合成任务与四步训练不构成长时间收敛、真实数据集精度或纯FP16训练认证。

### 修改前与诊断记录

修改前，FP16/FP32向量matmul均被rank 2–4契约拒绝；带梯度浮点item被旧标量/梯度检查拒绝。初版验收脚本使用了不存在的统计字段名，修正为现有字段后重新采集修改前证据。

第一轮修改后，全部向量/标量数值及模型训练已通过，但版本保护测试使用无参数backward，隐式创建种子梯度本身会提交fill图，导致“版本检查之前零提交”的统计断言误判。改为计数前准备显式种子后，正式整套重跑通过。生产保护检查未放宽；中间失败不计入成功认证。

## 仍未补齐的迁移缺口

下面7个真实API探针均明确拒绝，合计没有NPU图提交；输入索引已放在NPU上，不是因CPU索引误用而得出的结论。

| 探针 | 本轮实际阻塞 |
|---|---|
| `F.embedding`，resident INT64索引 | `aten.index_select.default`未实现 |
| `torch.index_select` | 同上 |
| `torch.cat` | `aten.cat.out`未实现 |
| `torch.randn(device='intel_npu')` | `aten.normal_.default`未实现 |
| `F.dropout(training=True, p=.1)` | `aten.native_dropout.default`未实现 |
| `x.to('cpu', non_blocking=True)` | 现有转换只接受同步执行 |
| 直接`torch.save(npu_tensor)` | 序列化路径请求不受支持的存储dtype；显式CPU checkpoint方案仍可用 |

另外根据源码和现有契约，以下范围仍未实现或未认证，本轮没有另做全面动态测试：

- **通用索引、切片与BOOL/掩码**：内部静态gather不等于公开动态索引。部分view可以创建，但其后计算/复制通常不能绑定部分allocation；stack、scatter、布尔mask与一般高级索引不能从cat/index_select探针推断已覆盖。
- **RNG、初始化与Dropout语义**：manual_seed_all仍没有设备RNG状态。需定义种子、状态恢复、设备采样、掩码和反向，不能用CPU生成随机数作为原生设备RNG实现。
- **公开算子、梯度与混合精度**：内部exp/rsqrt不等于公开算子；sqrt当前无autograd。设备AMP支持dtype列表为空，既有混合训练长期严格CPU轨迹仍有限，不能宣称autocast/GradScaler可用。
- **存储与布局**：FP16/FP32单Tensor最多1,048,576元素，INT64最多524,288；rank<=4。即使补齐索引，普通大LLM权重仍会碰到存储/工作区限制，需要单独扩展与验收。
- **运行时生态**：提交同步、串行；没有CUDA等价异步streams/events、CUDA graphs、分布式后端或本项目自动torch.compile代码生成后端。注册的设备模块只提供有限接口，不能据此推断完整torch.cuda工具、内存统计或第三方CUDA库兼容。
- **性能**：普通ATen FP16矩乘仍经SHAVE FP32计算与转换，未自动接入冻结DPU路径。算子语义补齐不意味着达到GPU性能或能效。

## 推荐的下一轮

优先实现有界`index_select`及其反向，再接普通`nn.Embedding`。验收须包含设备INT64索引、重复索引梯度累加、越界拒绝、稠密权重梯度、padding_idx及独立CPU模型训练；其他参数如sparse/scale_grad_by_freq需分别实现或明确拒绝。

随后补通用cat/stack和必要的切片materialize，再推进设备RNG/Dropout。先用有限的小型Transformer验证完整模型算子链，再扩展Tensor存储、模型规模与性能；这比只增加矩乘形式更能扩大实际可迁移模型范围。

## 复现

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
Remove-Item -LiteralPath Env:NPUNLOCK_GRAPH_CACHE -ErrorAction SilentlyContinue
.\run_torch_npu.ps1 -Script .\torch_npu_vector_demo.py -ScriptArguments @('--case','all','--label','after')
foreach ($case in @('matmul','contracts','training')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments @('--case',$case)
}
.\run_torch_npu.ps1 -Device -Quick
```

仍使用项目已有NPU环境与原生组件；不需要Intel功耗监控驱动、OpenVINO runtime或新的pip包。修改前证据需回到本报告所列旧源码指纹对应实现后重新采集，当前代码不会重现旧拒绝。

## 原有路径回归与指纹审计

| 检查入口 | 通过分组数 | NPU图调用 |
|---|---:|---:|
| matmul | 2 | 1603 |
| contracts | 1 | 34 |
| training | 2 | 4244 |
| device-quick | 33 | 4160 |

以上回归全部通过；设备Quick耗时128.348秒，退出码0。其日志中的120秒“Timeout”来自既有faulthandler周期栈快照，当时处于compile_ir；最终status=passed/stage=complete，不是外层300秒超时或硬件失败。
旧设备/SDPA拒绝测试中“全部向量matmul不支持”分别替换为inner维不匹配与rank-0输入检查，其他原有边界仍保留。
正式新增验收与SDPA回归记录的全部源码哈希与当前文件一致；宿主PYD指纹与修改前一致。没有修改或重编译原生组件、SHAVE内核。

### 正式新增验收源码指纹

- `torch_npu_vector_demo.py` SHA256 `6fe69f612d47597b053666d9f25373f3ba9e7ef41b93622a28a138d2e2188002`
- `torch_intel_npu.py` SHA256 `7ccf5970d4a7b85b5eb5da0784c4993233c950e53a1c29864c8a04748022231f`
- `torch_npu_autograd.py` SHA256 `331e5010f3fd6f0ca56084441ced36c99d10d11d514ab249493c1c3e25ef5775`
- `npunlock-probe/torch-device/_torch_intel_npu.pyd` SHA256 `b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e`

### 修改前实现指纹

- `torch_intel_npu.py` SHA256 `97b87be9ee0d8bc9e8632fb708e2963be94f45acd81ad4a2bc335a93d3f5d1c5`
- `torch_npu_autograd.py` SHA256 `76218869984dedfe75e4b4a4b75a54c43eb8b40814224dc3bdb28c77734a5470`

### 本轮原始记录指纹（审计后清理）

- `npunlock-probe/research/results/vector-migration/after-all.json` 6967字节 SHA256 `43485ca7dadc7b16e8999e87f73c30f2d8b12e543c54825617f65b16c6de0d3f`
- `npunlock-probe/research/results/vector-migration/before-scalar.json` 4182字节 SHA256 `57043984021f2710a49a19f9efc1f8ee435f7f7a9fb2616f8b38541544a5970d`
- `npunlock-probe/research/results/vector-migration/before-vectors.json` 5299字节 SHA256 `0a32167395c620e97c6e2a06032cf5caa5d9ef3d2b913faa010057081a68e304`
- `npunlock-probe/research/results/vector-migration/intermediate-all.json` 6133字节 SHA256 `04eeb91853016da031c3d2b1a04da6d99218c209822b8c92172a3b493df1f68f`
- `npunlock-probe/research/results/sdpa/matmul.json` 10920字节 SHA256 `041fad5b4f0b6231fa38ae2ab339559a521b4173ba4600f9dc85924cd44eb857`
- `npunlock-probe/research/results/sdpa/contracts.json` 4211字节 SHA256 `17a1b111a499023b0a7063491cd3e5d3d59eb875a31c54a89cc7eb73d11bcc18`
- `npunlock-probe/research/results/sdpa/training.json` 5166字节 SHA256 `d9624eb9dd8dccba5a761ed57ac28e645f57faeea2c8e5add1e3073fd8a20b13`
- `npunlock-probe/research/results/torch-npu-device.json` 12278字节 SHA256 `1abad7441eb0df82eff8585254ad1dd645130413faa793bc034ede3cbb67285d`
- `npunlock-probe/research/results/torch-npu-device.log` 2587字节 SHA256 `bb66b028527df91854ba43ec91f5b85f29c06d55482502bc575bb6a7b19fc95b`

## 清理与交付

审计后清理本轮9份原始JSON/日志，共57,743字节；清理前后670份源码、内核、原生组件及报告指纹一致，随后补充本记录与存储台账。本目录仅保留REPORT.md，研究源码与复现入口保留。

## 同日后续进展

后续S5已补齐有界index_select/Embedding及稠密梯度，见[Embedding报告](../embedding/REPORT.md)。上文7项探针保留本阶段的历史状态；当前gaps入口已移除这两个旧拒绝预期，剩余5项仍明确拒绝，最新支持边界以兼容性矩阵与S5报告为准。

