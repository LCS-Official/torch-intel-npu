# 标准TransformerEncoder稠密容器与token分类训练

日期：2026-10-19；本地开发版，未推送GitHub，source-release发布快照未改。
平台：Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Torch 2.14.1+cpu。
本轮新增一个7,752字节的SHAVE masked-softmax兼容ELF及C/汇编/构建脚本/manifest记录；复用现有byte-DMA和设备存储，未重建宿主PYD/DLL，没有新增依赖或下载模型。

## 正式真机结果

10组全部通过，共16,312次真实NPU图提交；定位/复核运行不计入正式数。所有正式分组源码指纹一致，默认MHA开关均保持True。

| 正式组 | NPU调用数 | 结果 |
|---|---:|---|
| primitives-float16 | 126 | 通过 |
| primitives-float32 | 126 | 通过 |
| primitives-int64 | 120 | 通过 |
| primitives-bool | 120 | 通过 |
| boolean-bool | 418 | 通过 |
| masked-float16 | 220 | 通过 |
| container-float16 | 7,135 | 通过 |
| container-float32 | 3,834 | 通过 |
| classifier-float32 | 4,207 | 通过 |
| contracts-float32 | 6 | 通过 |

两种精度的容器各完成8种配置，包含一阶参数/输入梯度；三角操作每种dtype各30种shape/diagonal组合（含INT64 diagonal两端）、3项out及浮点各2项梯度；BOOL归约40组合及分块边界；masked组另验原生FP16舍入与非零bias权重输出。

端到端FP32分类网络443个参数，三步loss为0.986053348, 0.834364653, 0.854081511。每步梯度、SGD动量更新和权重与独立同mask CPU参考通过既有容差；这是语义检查，不能据loss走势判断收敛。
第1步后保存CPU权重、动量优化器和NPU RNG状态，在新模型/新优化器中恢复并重放第2步，loss、全部模型状态、动量buffer与RNG逐位一致。恢复后的eval与同权重CPU一致，RNG不推进；Embedding padding行梯度为零，重复索引累加通过。

10个进程均正常shutdown，context_failed=False，live_buffers、cached_buffer_blocks、cached_scalar_buffers、quarantined_buffer_bytes、cached_sessions均归零；每组扣除标量缓存后的live buffer增量为0。调用数不是速度/算力指标。

## 范围与实现

目标是标准`nn.TransformerEncoder`的有界稠密Tensor路径，而非NestedTensor存储。
首次探测：无mask默认训练、`enable_nested_tensor=False`的padding推理已通过；因果mask自动检测被`aten.triu.out`阻塞；默认padding推理在canonical浮点mask的`logical_not`处被现有BOOL限定拒绝。

新增以下设备操作以承接标准容器控制流程，没有修改PyTorch模块源码或全局MHA开关：

| 操作 | 边界与实现 |
|---|---|
| `triu/tril`及精确out | FP16/FP32/INT64/BOOL，rank 2–4、非空，继承各dtype存储上限；部分/转置只读输入先DMA整理；CPU仅生成shape/diagonal索引元数据，设备byte-Gather的-1哨兵生成精确零，保留所选值的原始位 |
| 三角操作一阶梯度 | FP16/FP32，梯度应用相同三角选择；不保存数值输入，不提供可微out或二阶梯度承诺 |
| BOOL `all/any` | rank 0–4、有界非空BOOL，支持None/int/维度列表、负轴、keepdim、精确无别名out；BOOL在NPU转FP32计数，单归约最多1,048,576个0/1，计数可精确表示，再在NPU比较生成BOOL |
| 空维度列表 | BOOL all/any的`dim=[]/()`复制输入并保持shape；sum/mean仍归约全部轴，两者不混用 |

普通sum/mean与BOOL归约复用纯元数据轴/shape检查；原数值实现及限制保留。三角操作沿用完整原生绑定与安全DMA，不放开部分目的写入或输入/out别名。
接口核对[TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)、[triu](https://docs.pytorch.org/docs/2.14/generated/torch.triu.html)、[all](https://docs.pytorch.org/docs/2.14/generated/torch.all.html)及本机Torch 2.14.1源码/schema；空维度列表与极端diagonal另用CPU结果验证。没有复制PyTorch CPU数值内核。

## FP16 padding推理的舍入修复

两层FP16容器的padding eval最初1/24元素超出既有容差：最大绝对误差0.00146484375、相对误差0.00560379；没有放宽阈值。
逐层隔离发现LayerNorm、两段FFN、QKV投影、QK分数及context矩乘在相同输入下逐位一致。普通NPU softmax也与普通CPU softmax逐位一致，但CPU native masked-softmax的5个概率值与普通路径不同，最大差0.00048828125；第一层attention约0.00006103515625的差经归一化放大。

依据[PyTorch 2.14.1 SoftMax.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/SoftMax.cpp)的native masked路径及本机c10 Half算术，CPU会把减最大值、exp值、倒数及最终乘积舍入到FP16；累加这些已舍入exp使用FP32。
独立CPU参考逐步模拟这些舍入，与原生API交叉验证。这里的差异属于融合入口数值语义，不是NPU普通softmax的错误。

新增`resident_native_half_softmax.c`独立实现该舍入序列，复用项目已有IEEE binary16转换和exp近似；不依赖CPU数据计算。native MHA仅在FP16且有mask时采用此内核，并按原生QKV投影/加bias/缩放的阶段组合现有NPU算子；无mask、FP32、普通softmax和训练SDPA保持原路径。
原失败样本修复后，第一层attention、各层及两层容器最终输出均与CPU逐位一致。专门验收还覆盖非零QKV/projection bias、平均/逐头权重、masked NaN、全屏蔽NaN、半精度subnormal和8,193宽行。
本轮新增内核是为接口数值兼容，没有进行性能优化或更改公共softmax语义，也不新增原生masked-softmax训练接口。

## 验收设计

- 三角选择：四种dtype，rank 2/3/4、部分只读输入、多个diagonal、out与独立存储；浮点特殊值按原始字节对照，INT64含超出FP64精确整数范围及正负极限；浮点另验一阶梯度。
- BOOL归约：40种shape/axis/keepdim组合、scalar、空维度列表、转置out，8,193及65,537元素的跨packet/计算分块。
- 完整容器：两层标准EncoderLayer及最终LayerNorm，embedding=4、heads=2、FFN=8、batch=2、序列长3、Dropout=.25；默认训练、因果加padding、显式因果hint、pre-norm GELU、sequence-first、无padding默认eval，以及稠密padding/因果eval/inference_mode。
- 端到端分类：词表16的Embedding、可学习位置参数、两层标准TransformerEncoder、padding感知池化、Linear分类头及CrossEntropy；FP32、三步独立同mask CPU梯度/SGD动量更新；重复token及padding_idx=0梯度；显式CPU checkpoint保存后，在新模型/新优化器中恢复并重放下一步。
- 边界：非法rank/dtype/axis、重复轴、错误shape/out别名在图提交前拒绝；默认NestedTensor padding推理作为明确未支持边界检查。

普通数值比较保持FP32 rtol=3e-5/atol=3e-6、FP16 rtol=.003/atol=.0005；三角数据移动、checkpoint重放要求逐位一致。
CPU训练参考复用上一轮独立Philox同mask控制，仅在CPU作用域替换Dropout/SDPA，并在CPU对照作用域关闭不支持FP16 GELU反向的oneDNN；NPU模块运行在这些CPU作用域之外。

## 控制传输

容器自动检测因果mask时会在NPU生成比较结果、BOOL归约，然后由PyTorch Python控制流读取1字节BOOL。
显式正确的`is_causal=True`可跳过该次检测；hint必须与实际mask一致。
这仍是同步执行，不能称为没有主机交互或已融合的Transformer。

两层非零Dropout训练有8个随机位置，每次上传32字节seed/counter参数头；随机张量仍由NPU生成。
测试单独断言随机参数头上传、标量常量缓存上传与控制下载，未把它们误报为整模型数据往返。
token分类训练每步预期4次标量下载：Embedding前向/反向的4字节索引检查、因果判断1字节、CrossEntropy的4字节标签检查，共13字节；eval无损失/反向时为2次、5字节。
权重/激活/梯度不由这些标量检查下载计算。输出/梯度核对、日志与显式CPU checkpoint下载在计算区间之外。

## 使用与限制

对需要padding推理的稠密模型显式使用：

```python
import torch
import torch_intel_npu

layer = torch.nn.TransformerEncoderLayer(
    4, 2, dim_feedforward=8, dropout=0.25, batch_first=True)
model = torch.nn.TransformerEncoder(
    layer, 2, enable_nested_tensor=False).to("intel_npu")
```

无padding的默认容器及训练分支不必为本后端关闭全局MHA开关。
`enable_nested_tensor=False`是PyTorch提供的容器参数；本轮没有把默认自动NestedTensor路径伪装为已支持，也没有把其predicate强制返回False。
仍不提供NestedTensor、完整encoder-decoder Transformer、通用AMP、空Tensor、浮点/整数all/any、通用逻辑类型扩展或长程FP16训练保证。
继承score<=65,536、4MiB单allocation、有限图缓存及Embedding packet/工作量边界；synthetic小分类模型不是LLM训练、模型质量或性能基准。

## 复现

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_demo.py
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_standard_modules_demo.py -ScriptArguments @('--case','inference')
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_dropout_demo.py -ScriptArguments @('--case','operators')
.\run_torch_npu.ps1 -Script .\torch_npu_mask_demo.py -ScriptArguments @('--case','contracts','--label','transformer-regression')
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_geometry test_torch_npu_lowbit test_torch_npu_mask_packet test_torch_npu_rng test_torch_npu_sdpa_dropout test_torch_npu_standard_modules test_torch_npu_transformer
.\.conda\npunlock\python.exe -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
.\.conda\npunlock\python.exe -B tools/test_source_release.py
```

`--case probe`只用于定位，probe-complete不算正式passed。正式分组、回归与指纹见本报告相应章节。

## 既有路径回归

| 回归 | 检查量 | NPU调用数 | 结果 |
|---|---:|---:|---|
| 标准MHA/Encoder推理 FP16 | 1 | 574 | 通过 |
| 标准MHA/Encoder推理 FP32 | 1 | 333 | 通过 |
| SDPA Dropout FP16 | 1 | 781 | 通过 |
| SDPA Dropout FP32 | 1 | 641 | 通过 |
| 基础设备Quick | 33 | 4,162 | 通过 |
| BOOL/mask拒绝契约 | 1 | 5 | 通过 |

回归共6,496次真实NPU调用，6个进程全部正常关闭，context_failed=False，live/cache/quarantine/session资源计数全零。
覆盖原生推理的无mask/合并mask/全padding NaN、GELU/pre-norm及inference_mode；两种精度各12个SDPA场景（含广播、mask梯度、因果、p=0/1、单独求导）；33项设备检查含原sum/mean轴/特殊值/大矩阵/转置/复制及存储生命周期，确认提取元数据轴检查未改变原数值路径。原20项BOOL/mask拒绝与saved-mask version保护仍有效。
设备日志的120秒faulthandler栈快照是诊断输出；最终33项passed、正常complete且未触发300秒外层超时。

## 离线与构建指纹

64项离线检查通过：基础/低位/mask/RNG/SDPA/标准模块/本轮CPU控制27项、npunlock Python API 32项、source exporter 5项。新增检查包含几何索引/维度规则、全部Dropout位置的CPU RNG计数、CPU checkpoint独立clone及动量状态、CLI精度组合拒绝，以及native FP16 masked-softmax舍入参考。
源码导出选择已确认包含新增C、ELF/汇编、manifest、构建/验收/测试及报告；未写入已有source-release。构建通过项目既有隔离Movi worker，目标4000xx、-O2、-ffp-contract=off；ELF为7,752字节，未下载编译器或安装新运行依赖。

```text
torch_intel_npu.py
6eab0c24884634f2832bdf8664b6257f382ef22bd9b24bd292b70301cbda9f62
torch_npu_autograd.py
7609327d922089b6c4ba20b03a5941405f0cf329a1baca401a907b38ed5ff839
torch_npu_geometry.py
6184fcc8baa2b3b897f46a957ded8851eebc601bdeb12e5a043ece2d0cc85c22
torch_npu_transformer_demo.py
af9632eeaab4d5ea65b17141ca2a9cfc18759aeed81e1262d7d9a1a4f2ff12f1
test_torch_npu_transformer.py
1bdb45d7308e29b949565accf5169046f7ce37748928cc617f3fbea051fdebf6
npunlock/tests/fixtures/npu4000/resident_native_half_softmax.c
a35dd65fcb5e7a318c1954595d835fd8a47a7820096f13791720bded2fbbb044
npunlock-probe/research/build_native_half_softmax_kernel.py
b44f10dccec973af8d673f9a12e446b01df8fd793b2bbc99f3b3b608e1e8a8a2
npunlock/tests/fixtures/npu4000/resident/native-half-softmax.elf
04a8cab7c19b4fd9f6ce3aa47cabce5b5632204e4c53dffe98f8968d49beaad8
npunlock/tests/fixtures/npu4000/resident/manifest.json
9f2e8638e164e7a753c3d191e87f822e8b0abc42d45c2220333b23ded8211889
npunlock-probe/torch-device/_torch_intel_npu.pyd
641087286922730429eceb6e4b9e8ba58792e9a05958933ce17b87e90e6bc6a5
```

## 临时结果清理

2026-10-19 稠密TransformerEncoder及FP16 native masked-softmax验收后清理：26份本轮原始JSON/日志/Movi响应与临时隔离脚本，共177268字节；清理前后720份保留文件SHA-256一致，随后补充报告与台账。保留新增SHAVE C/ELF/汇编/manifest、构建/派发/autograd/几何源码、正式验收与CPU测试、双语文档和报告；未下载模型、安装依赖或生成图表。
