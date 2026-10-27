# 有界设备索引与普通Embedding训练验收

日期：2026-10-19；Windows x64，Core Ultra 7 258V / NPU 8086:643e，驱动API 1005540、编译器8.3，Torch 2.14.1+cpu。本地开发版，未推送GitHub。

## 完成内容

新增两个实际SHAVE C内核及对应PyTorch接口：动态设备索引取数、重复索引的稠密梯度累加。普通`torch.index_select`、`F.embedding`和`nn.Embedding`在下述边界内可前向/一阶反向。索引保留INT64原始双字，不转成浮点数值，也不把索引或权重下载到CPU计算。

正式新增验收8组全部通过，共1,693次真实NPU图调用；每组无live buffer增量，关闭后live/cache/scalar/quarantine/session全零，无context failure。含Embedding的普通FP32分类模型完成四步独立CPU轨迹对照。

这扩展了普通设备Tensor路径，没有修改冻结SmolLM适配器、原生C++/DLL或系统驱动，也不把通用CUDA兼容性宣布为完成。

## API与边界

| 接口 | 支持范围 |
|---|---|
| `index_select` | FP16/FP32，数据rank 1–4，合法正/负dim，完整密集输入可materialize；同设备、连续一维INT64索引；输出连续且独立存储；一阶梯度 |
| `index_select(...,out=...)` | 无梯度、精确shape/dtype、连续完整输出，不允许与数据或索引别名；不自动resize |
| `F.embedding` / `nn.Embedding` | 二维FP16/FP32权重，连续同设备INT64索引rank 0–3，输出rank<=4；稠密权重梯度，padding_idx，冻结权重推理 |
| 未实现参数 | `sparse=True`、`scale_grad_by_freq=True`、`max_norm`明确拒绝；不支持INT32索引、空索引、一般部分存储view、通用scatter/gather、高级索引或二阶梯度 |

令R为被选轴长度（Embedding中为词表行数），D为其余维度乘积，L为索引数量。初版要求：

```
P = 6 + 2*L + max(R,L)*D <= 65,536
R*D*L <= 1,048,576
```

P按FP32载体元素计，包括5个头参数、每个INT64索引的两个原始32位字、数据/输出容量以及一个错误标志。第二项限制串行反向工作量。前向同样采用这套统一边界；未放宽普通Tensor的既有存储上限。

这是小规模索引/词表支持。即使权重能放进原有单Tensor上限，也可能超出本轮更严格的packet/工作量边界；不能据此把任意Hugging Face LLM直接迁移到普通设备后端。

模块应先在CPU初始化，再`.to('intel_npu')`；直接在NPU初始化仍受设备RNG缺失限制。例：

```python
import torch
import torch_intel_npu

embedding = torch.nn.Embedding(16, 6, padding_idx=0).to('intel_npu')
tokens = torch.tensor([[0, 2, 2, 3]], dtype=torch.int64).to('intel_npu')
output = embedding(tokens)
loss = output.mean()
print(loss.item())
loss.backward()
```

## 内核实现

- 源码`npunlock/tests/fixtures/npu4000/resident_index.c`，构建入口`npunlock-probe/research/build_index_kernels.py`；新产物为`resident/index-gather-f32.elf`和`resident/index-backward-f32.elf`及对应汇编/manifest指纹。
- 复用已有`compile_classification_rows`完整单任务载体及严格布局检查，避免把INT64双字、动态索引目标或归约拆到不完整的SHAVE片段。图编译仍使用既有Intel驱动，不需要OpenVINO Python/runtime。
- 前向按原始32位字复制所选FP32数据；FP16先复用设备转换，在FP32载体中取数后转回FP16。反向逐行检查索引并累加梯度，padding行保持零；未使用原子更新或并行scatter。
- 全部索引先在设备上检查高字与行范围，非法索引不会触发越界数据访问。每次前向、反向各下载一个4字节验证标志，必要时向Python抛IndexError；这是显式错误检查，计算数据保持resident。
- 先重排被选轴到最前，在设备上整理完整密集存储，结束后恢复轴顺序。PyTorch autograd保存索引version；反向前被原地修改会明确拒绝。

当前以有界正确性为目标。单完整任务、FP16转换、packet打包/拆分及错误标志同步均有开销；本轮未做性能、功耗或大词表效率测试。

## 发现并修复的FP16精度问题

初版让index_select与Embedding共用“每次加法后舍入到half”。两个重复值时无法区分该方案与FP32累计后最终舍入；新增128次重复检查后，index_select发现0.0078125偏差。

对照固定版本CPU实现，二者的归约策略不同。测试梯度序列`[.3333,-.2,.125,-.1]`重复32轮，FP16输入时：

| 路径 | 选中行的梯度 |
|---|---:|
| CPU index_select，FP32累计后转half | 5.06640625 |
| CPU Embedding，每次half更新后舍入 | 5.07421875 |

修复通过现有内核的模式参数区分两条反向：index_select累计FP32后转回输入dtype；Embedding对half梯度逐次软件舍入，保持subnormal。正式重跑时两条路径均与对应CPU参考精确相等，没有放宽容差。

这只保证本机固定版本与所测案例的数值语义；不承诺任意CUDA/GPU累加顺序或跨版本逐位一致。CPU公开行为与源码参照：[index_select](https://docs.pytorch.org/docs/2.14/generated/torch.index_select.html)、[Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)、[PyTorch Embedding实现](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/Embedding.cpp)。新SHAVE实现沿用本项目已有IEEE辅助函数，无新增第三方代码或运行依赖。

## 实际验收

入口`torch_npu_embedding_demo.py`，种子113，CPU单线程。

| 检查 | 结果 |
|---|---|
| index_select前后向/out | FP16/FP32各10组，含rank 1–4、dim正负、完整密集转置、非连续输出梯度、重复索引、连续独立输出及同一out返回 |
| Embedding前后向 | FP16/FP32各12组，索引rank 0–3，padding_idx=None/0/-1；前向保留padding行数据、反向忽略该行；另有两种dtype的from_pretrained冻结模块 |
| 原始位模式 | FP16/FP32所选值含NaN、Inf、正负零与subnormal，前向CPU/NPU按整数位模式精确一致 |
| 边界与缓存 | P恰好65,536时前后向通过；修改resident索引后复用同一图得到新输出，无新会话编译 |
| 累加与特殊值梯度 | 每种dtype分别检查index_select与Embedding的128次重复、subnormal梯度；padding行NaN不污染权重梯度；CPU/NPU精确一致 |
| 元数据拒绝 | 13项：CPU/错误dtype/非1D/部分存储索引、非法dim、out形状/别名、packet/工作量超限、Embedding索引rank、freq/sparse/max_norm；拒绝区间无NPU图提交 |
| 非法INT64值 | -1、4、2^32、2^53+1、2^63-1、-2^63、0x7fc00000均安全拒绝；包含低字有效但高字非零，以及低字像FP32 NaN的情况，验证没有浮点数值转换 |
| 版本保护 | 前向后修改索引，使用显式种子反向，计算提交前由保存Tensor的version检查拒绝 |
| 普通训练 | 129参数FP32分类器：Embedding→mean→LayerNorm→ReLU→Linear→CrossEntropy，resident INT64输入/标签，SGD(momentum=.5)四步；损失、全部参数与梯度独立CPU逐步对照 |

四步损失：1.0696121454 → 1.0601336956 → 1.0462303162 → 1.0305043459。只有四个固定合成训练样本，不代表真实数据质量、长期收敛或完整语言模型训练。

常规容差FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005；前向、位模式、重复累加、subnormal与padding NaN隔离使用本轮明确记录的精确检查。计算区间的显式复制数等于新缓存标量上传数加错误标志读取数；没有输入索引、权重或输出数据的CPU运算回退。计数不覆盖驱动内部DMA。

另外7项离线INT64/完整行载体检查通过，覆盖完整双字、单任务布局与raw-byte transport拒绝保护。

## 后续工作

优先补公开cat/stack与必要的切片materialize，再用普通小型Transformer验证完整算子链。设备RNG/Dropout、BOOL mask、AMP、异步stream/event、一般checkpoint、较大存储与大词表仍有缺口。现有FrozenSmolIO仍采用CPU两端，不会因本轮小Embedding接口而自动迁移。

## 复现与保留

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
# 仅修改索引C源码后才需要重编译，沿用已配置的MoviTools。
.\.conda\npunlock\python.exe -B npunlock-probe/research/build_index_kernels.py
.\run_torch_npu.ps1 -Script .\torch_npu_embedding_demo.py -ScriptArguments @('--case','all','--label','after')
.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py -ScriptArguments @('--child','boundaries','--label','embedding')
.\run_torch_npu.ps1 -Script .\torch_npu_vector_demo.py -ScriptArguments @('--case','vectors','--label','embedding-regression')
.\run_torch_npu.ps1 -Script .\torch_npu_vector_demo.py -ScriptArguments @('--case','gaps','--label','embedding-regression')
```

交付保留源码、构建脚本、源码对应的两个ELF/汇编/manifest和Markdown报告；审计后清理本轮原始JSON、构建worker响应等实验文件。使用已构建内核不需要额外pip包或编译器运行依赖。

## 最终回归与构建审计

| 入口 | 已支持检查通过 | 明确未支持 | NPU图调用 |
|---|---:|---:|---:|
| embedding/after-all.json | 8 | 0 | 1693 |
| torch-migration/embedding-boundaries.json | 10 | 1 | 220 |
| vector-migration/embedding-regression-vectors.json | 2 | 0 | 2660 |
| vector-migration/embedding-regression-gaps.json | 1 | 0 | 0 |

普通迁移探针现为10项已支持检查通过、1项直接NPU序列化明确拒绝；Embedding探针使用设备INT64索引并核对权重梯度。向量gaps入口剩余5项（cat/RNG/Dropout/异步复制/直接序列化）均明确拒绝。
两个索引内核的编译/汇编/链接六个worker阶段全部成功，manifest的全部来源哈希及两个新ELF/汇编哈希正确。正式验收与最终回归的源码指纹均与本次交付文件一致。

### 新内核

| 名称 | ELF字节 | SHA256 |
|---|---:|---|
| index-gather-f32 | 4956 | `e2d4f0c93d6fc4dde16a2c8849d3806d216c6602649ad9d16ed53fec908f0078` |
| index-backward-f32 | 8364 | `ba890e03251908a4c8d3a5e9504554934f09b82efd6d27e751e2297a4e37edb7` |

### 正式验收源码指纹

- `torch_intel_npu.py` SHA256 `716b85ca573d430c18a8bcab79e762212069a7c86c429672ddd06e21edfd2801`
- `torch_npu_autograd.py` SHA256 `a11b825750bf96c545552df5e8378dcd64583e054ba3c309f28dff7c1bad5327`
- `torch_npu_embedding_demo.py` SHA256 `7e4e0afb46f8c7b0e8de6427b84ae42502b4af5a23a93ab9eb088d5ca5428b51`
- `npunlock/tests/fixtures/npu4000/resident_index.c` SHA256 `e11bc2d4ec1028b3d87280a0dfdc16958a394d273f9ef6c8cfe685c636818635`
- `npunlock/tests/fixtures/npu4000/resident/manifest.json` SHA256 `f386a27dd268b9f532aa6bae8ebdc31399559b4ed5d296f2e4b11d474168dab4`
- `npunlock-probe/torch-device/_torch_intel_npu.pyd` SHA256 `b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e`
- `npunlock-probe/research/build_index_kernels.py` SHA256 `9dabe543280d0efd142cd05b644344cec9a139d33ecb5566b05c4fd240b24d90`
- `npunlock/python/npunlock/npu4000.py` SHA256 `96fd710b05d130b8290ce17d5bf76f1ec0ad3ecdc4176b1b61f18ae4e539e921`
- `npunlock-probe/torch-device/backend.cpp` SHA256 `006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924`
- `npunlock-probe/session-runtime/npunlock_session.dll` SHA256 `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9`

### 原始记录指纹（审计后清理）

- `npunlock-probe/research/results/embedding/after-all.json` 6483字节 SHA256 `b90d06bfe6e8e092775bd4191b8ce5fa13d8f4ecd581d39b66d8498c9f62a666`
- `npunlock-probe/research/results/embedding/first-all.json` 6252字节 SHA256 `3c13401fdbabf80d4276593f7fae111064eede99c7c6c87df6b50faf742556f5`
- `npunlock-probe/research/results/embedding/initial-index.json` 2774字节 SHA256 `063587d9dd9f9a0a095b3d3c20e6832e58b02b8dc25dcbb8939c65750a58ae05`
- `npunlock-probe/research/results/embedding/rounding-before-all.json` 7590字节 SHA256 `28e460ef4ee31f46b3196047cc29eaf1f1ca19b4ef0c9de5e9811c9c9dfa93a3`
- `npunlock-probe/research/results/torch-migration/embedding-boundaries.json` 5520字节 SHA256 `f9d83ad4593510eca95739e35c31d796cb2a9c3e67f99a870d2024170c7280c4`
- `npunlock-probe/research/results/vector-migration/embedding-regression-vectors.json` 2463字节 SHA256 `877acc5d7add90c792c337a32fbf7cb0fd5a1073b1a6e2575a78f86b48ec3823`
- `npunlock-probe/research/results/vector-migration/embedding-regression-gaps.json` 2756字节 SHA256 `921ca0507ce22f5f8ec533543348b61ee6fa6c675aa83273b7e352c88b300922`
- `npunlock-probe/research/results/vector-migration/index-discovery-gaps.json` 3849字节 SHA256 `59fe0aebbd8c4b09035312683be723a4749014f41e5084f9a566e8f08cf030e9`
- `npunlock-probe/research/results/index-backward-f32-assemble-response.bin` 9316字节 SHA256 `086c5d5451bb431bc0ac1aa97bbb2a52bb70632c0a425eaf57a93688ee378c1c`
- `npunlock-probe/research/results/index-backward-f32-assemble.json` 293字节 SHA256 `c923e6805bd41c7e641d47c62e5604755d9847e4d64241e54a28059438833b44`
- `npunlock-probe/research/results/index-backward-f32-compile-response.bin` 76066字节 SHA256 `468cddcb06d03bac1dfc398a5615c2e8f45a4ff81ada973d64b2d8330237d607`
- `npunlock-probe/research/results/index-backward-f32-compile.json` 293字节 SHA256 `da2510b4318b70d36f6359a7e4340e1c35ca9d0a9d060b4ae600d82d97f81595`
- `npunlock-probe/research/results/index-backward-f32-link-response.bin` 8396字节 SHA256 `9af968b1e69bbe00bc3b1630f5b59835fcc04b46ec42b529ab7805bfae1ca1f2`
- `npunlock-probe/research/results/index-backward-f32-link.json` 289字节 SHA256 `8feea0cb5a77800144f2231cbb9a7dda0ed88951e918f2725858a8c2cf8f88c0`
- `npunlock-probe/research/results/index-gather-f32-assemble-response.bin` 5496字节 SHA256 `af4c4ffb344519be5007349515e89b2afcef1cc174ff62b65265339b16f502b7`
- `npunlock-probe/research/results/index-gather-f32-assemble.json` 291字节 SHA256 `df9208d6e331a854f10d4c80427bb0247ef4b889f3326e847b0c939ae690bc5c`
- `npunlock-probe/research/results/index-gather-f32-compile-response.bin` 43106字节 SHA256 `6364c044f34b72afca94b02c4cc9af8a00aee9163a59dfffe13b1437c6ab407d`
- `npunlock-probe/research/results/index-gather-f32-compile.json` 291字节 SHA256 `3ec14a2887e1995b52086fd9ae4aa55c97f5ecf3b8bd1051fd80f3d1300ccd23`
- `npunlock-probe/research/results/index-gather-f32-link-response.bin` 4988字节 SHA256 `24c519831fc782eed061e220c99f63afcb87364758f59eeb943782a39978056f`
- `npunlock-probe/research/results/index-gather-f32-link.json` 287字节 SHA256 `049fff6d51f5cf17ab1feca501ad071977528d11328604ead38780691c3dcb98`

## 清理验收

审计后删除20份原始JSON/worker响应，共186,799字节；清理前后678份源码、原生组件、内核与报告指纹一致，随后补充本记录及存储台账。Embedding与vector-migration结果目录各仅保留REPORT.md；两个新索引ELF、汇编、C源码及manifest均保留。

