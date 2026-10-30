# S60：INT4/NF4冻结SmolLM2解码器入口与主线覆盖

2026-10-30，Windows x64／Core Ultra 7 258V／NPU 8086:643e，驱动32.0.100.5540、编译器8.3。Torch 2.14.1+cpu、NumPy 2.5.3、Transformers 5.18.0、safetensors沿用现有模型环境；不下载模型、不增加依赖或重建DLL／PYD／SHAVE ELF。

## 结论与范围

新增`FrozenSmolLayer`／`FrozenSmolStack`的`w4a16`、`nf4a16`入口，实际30层前向、8-token prefill和第9个token decode均通过本格式局部数值检查。不是通用低位Tensor／训练，也未认证自由生成、PPL、grouped decode、其他batch／长度或任意HF模型。

这一步把既有四位Linear能力接入真实decoder。压缩常量解压后使用FP16 DPU计算，不宣称原生四位乘加或40TOPS利用率。原权重teacher的整网质量两者均0/257；不能推荐替代FP16，也不改默认。S59已[结束单层末层拟合方向](../w8-native-down-check/REPORT.md)，本轮没有追加校准候选或参数扫描。

## 实现与明确边界

- `w4a16`对应INT4，`nf4a16`对应NF4；自动准备使用group32，也接受全部七个投影、格式／形状匹配的显式`PackedWeight`。FP16保护、rank32修正仍只允许W8，其他模式会明确拒绝。
- 既有图生成器处理packed Const→Convert→scale→MatMul，权重及decoder中间值不在CPU执行回退。CPU加载／量化、embedding、最后norm／lm_head以及独立参考计算是显式边界。
- 原共享检查误把全部非FP16模式当W8，造成“Missing INT8 weight DPU tasks”。现在按模式检查：W8仍要求INT8矩阵证据，四位解压路径要求FP16矩阵证据。任务描述符只证明矩阵任务存在，不把每个IR操作逐一映射为硬件任务。
- NF4单行decode首次被8.3编译器拒绝：`LowFpType or levels attribute should has value`。仅NF4单行投影在同图内补至32行再裁回；公开token、attention和KV形状保持不变。未改驱动、描述符或量化数值。
- 四位模式使用普通内存图缓存，不进入仅针对FP16/W8的canonical磁盘快捷缓存。默认512MiB下NF4的首次全层输出、90项局部检查和重复输出相等均通过，但热decode触发重编译，缓存复用断言失败。显式1GiB预算重跑后无淘汰且热调用通过；生产默认保持512MiB。

## 执行正确性与缓存

局部参考在相同实际输入上，用四位权重实际FP16解压值逐阶段计算；W8保留原精确code×scale参考，FP16参考不变。沿用L2<0.003、scaled max<0.01，未放宽门槛。每个30层配置均有30个forward、30个prefill、30个decode检查；四种模式的单层控制另有12项。

| 模式 | 30层局部通过 | 最大相对L2 | 最大scaled误差 | 实验图预算 | 结束时缓存blob字节 |
|---|---:|---:|---:|---:|---:|
| fp16 | 90/90 | 0.00062985 | 0.00388710 | 1024 MiB | 655.474 MiB |
| w8a16 | 90/90 | 0.00085543 | 0.00576577 | 1024 MiB | 350.445 MiB |
| w4a16 | 90/90 | 0.00140911 | 0.00918667 | 512 MiB | 468.136 MiB |
| nf4a16 | 90/90 | 0.00087726 | 0.00648860 | 1024 MiB | 655.578 MiB |

四种配置的热forward／decode各30次图提交，重复输出逐位一致，无新增显式host拷贝或session创建。旧KV快照长度8及内容不变，新快照长度9；group_size=1。复制计数不含驱动内部DMA／编译器工作；这是正确性及复用检查，不是吞吐、TTFT或功耗基准。

四位30层投影的packed码＋scale均为59,719,680字节（56.953MiB），原FP16矩阵为212,336,640字节（202.5MiB），源表示减少71.875%。这不等于进程／设备内存减少71.875%：CPU模型／原FP16副本仍在，编译器可展开权重；NF4结束时缓存约655.6MiB，接近FP16。表中缓存是结束值，不能当作总内存峰值。

## 独立原模型质量

同一模型、八条固定新提示、64-token CPU greedy teacher轨迹。只评分从prompt末位到首次生成EOS的预测，忽略EOS后续；prompt内EOS不截断。八条共257位置。无新校准或挑选。原权重FP16/eager CPU模型为teacher；CPU embedding、最后norm／head边界相同。

沿用raw logits L2<0.03且KL<0.02，不去均值、不按top-1相等代替质量门槛。

| 模式 | 质量通过 | 平均L2 | 最大L2 | 最大KL | next-token相同 |
|---|---:|---:|---:|---:|---:|
| fp16 | 257/257 | 0.3107% | 1.6170% | 0.00024384 | 257/257 |
| w8a16 | 96/257 | 4.2853% | 18.0785% | 0.01309188 | 247/257 |
| w4a16 | 0/257 | 54.6518% | 207.8990% | 2.33958912 | 188/257 |
| nf4a16 | 0/257 | 26.7244% | 129.7603% | 1.01679409 | 214/257 |

FP16控制257/257通过，排除本语料上的明显评估流程问题。NF4优于简单INT4，但两者均未达到整网要求；不能只看top-1或局部误差宣称质量修复。此处W8为自动逐行RTN，无S44–S59的显式校准因子；语料和EOS评分协议也不同，不能与旧243/290或142位置结果直接作回归比较。

## 验收与失败记录

8个正式进程共2,204次NPU提交、372项局部检查通过并健康关闭。另3个开发失败进程：旧INT8证据检查2次提交、NF4未补行9次、NF4默认缓存复用断言540次，共551次；三个均健康关闭。全阶段11个实际进程共2,755次提交，执行的464项局部检查均通过；3个进程级失败未从记录中省略。关闭后context_failed=False、live_buffers／cached_buffer_blocks／cached_sessions／quarantined_buffer_bytes均为0。

新增10项离线检查覆盖模式、压缩IR、解压参考、输入不变、准备字典／模式拒绝、单行补行边界、DPU证据守卫与EOS评分边界。最终完整工作区349+32+6=387项通过；公开快照336+32+6=374项检查，373通过、1项既有nested-source缺失条件跳过。可选模型依赖已安装；GitHub核心环境可能按依赖显式跳过更多模型测试。

原FP16/W8的forward／prefill／decode共六个图与修改前逐字节一致。230份Git跟踪的NPU4000 fixture文件在工作区／公开副本／父提交间完全相同；未改native构建或内核。本轮早期单层与INT4记录先完成，随后仅增加NF4专用补行及研究记录字段；各临时JSON保存当时实际源码SHA，最终FP16/W8/NF4全栈对照使用当前核心。报告保留原始记录身份，不能把所有旧记录SHA标成最终源码。

## 用法与复现

```python
import torch
import torch_intel_npu
from torch_npu_smollm import FrozenSmolStack

# Experimental frozen inference; original-model quality is not certified.
torch.intel_npu.set_graph_cache_budget(1024 * 1024**2)
stack = FrozenSmolStack.from_checkpoint(model_directory, num_layers=30,
                                      mode="nf4a16")  # or "w4a16"
with torch.no_grad():
    output = stack(hidden_fp16.to("intel_npu"))
```

`hidden_fp16`由调用者显式提供，本轮认证形状(1,64,576)。模型按现有安装说明准备。研究复测先设置`PYTHONPATH`为项目根、`npunlock/python`、`npunlock-probe/research`；先用CPU环境执行`smollm_lowbit_coverage.py prepare`冻结语料，再用`run_torch_npu.ps1`运行同脚本`run --mode ... --layers 1/30`。NF4／FP16／W8的正式30层控制用`--graph-cache-mib 1024`，INT4用默认512。脚本拒绝覆盖已有硬件JSON，重跑需更换本轮结果目录或按存储规则清理旧数据。

下一步以实际模型／前后向接口缺口为主线。低位质量需模型层面的固定方案和独立整网验收，局部SSE改善不再触发无限追加单层扫描。效率工作等执行和质量基线明确后再开展。

公开快照清单还补纳42份此前已跟踪但未入清单的研究文件及S60三份新文件；严格源包校验覆盖1,003份payload文件、约16MiB。研究分支CI此前只运行离线测试，完整清单校验在tag／手动事件运行；本轮另在本地完成严格校验。

## 身份与证据

| 当前文件 | SHA-256 |
|---|---|
| torch_npu_smollm.py | `d82bbfb90241b851edcf0ef10669676b8f112fb911eb5931588030dc594a80f6` |
| torch_npu_transformer.py | `4d5ef74adc3460d1aac234b2b2dd546008ab6db17bf1f6af5168c7155dd06792` |
| torch_npu_lowbit.py | `1c579ad4f2de9528d3fbd407c30df52bcfd29ac11b63cc202ffacd9c9137b01d` |
| test_torch_npu_smollm_lowbit.py | `dc7a5894012fe4ff37b04f0b9898c36e7691f9368c3d96de1c1709b00f9cfcff` |
| npunlock-probe/research/smollm_lowbit_coverage.py | `0b66adab7b2fb2705ef0c6bde8e1be228073f08d3aa374e262631996f5733d1d` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |

| 本轮临时记录（按存储策略清理，仅保留身份） | SHA-256 |
|---|---|
| 临时corpus.json | `1330d6e55b43356677640decf091cd47ad9843b4b15a60dd7f09d045d8d15003` |
| 临时default-graphs.json | `bc94ec2fa0e7a4e725bc95d2026288573388d3d41aff54aeda105e20ffb9ebf0` |
| 临时integrity.json | `22249e57ecad5d79b333de1e3aabd9b7a6ea17c7017cafedc26613dfbce61859` |
| 临时fp16-1.json | `d4e593b79c94422654c06abed9031addd3141dec1dbfac26636ec5c060f7ad30` |
| 临时fp16-30.json | `f3321cfa38683e46102c40ccdc70fcd4fbeb013b1d7a2602e6f77503b7dcabaf` |
| 临时w8a16-1.json | `ce57ab105f9aff90af1eae131aa169584e4cd58317675e8e608feac05d39a2b8` |
| 临时w8a16-30.json | `c2fe06665b00ebf7c6361695a1936b44da11da8b19d0e89491e39778386bd0e4` |
| 临时w4a16-1.json | `c2e76aae7ef46e2baa7ef65f3b857d927252b352b979ccd621b7e1b680bafa01` |
| 临时w4a16-30.json | `4c32c5793421a6aed7a6b722c6d7f272b0dbe4037dd26405fc3ba293ceca83a2` |
| 临时nf4a16-1.json | `799a99e0fe5b9cc43fec484fd6e0c456a6cf5f3bf9eefc99d636f4180baa837f` |
| 临时nf4a16-30.json | `0c1f0c29d676b687d43070630f6d13577b7bececb81c8de5307820bbaccc04ab` |
| 临时nf4a16-1-initial-driver-failure.json | `0f8bdca42444290092c87facdd4ef0ac48c75b7c92c174e192f78d1488b5c513` |
| 临时nf4a16-30-initial-cache-failure.json | `6e544d9282927ffb5137cb6af88f5e0af5cbbe70e300b58d5f67cc8cf0463b4c` |
| 临时w4a16-1-initial-inspection-failure.json | `41dfa7a0588ff679f2410ffb18c7ea021958ae048faccb8411f6bdc9bc7fcd0f` |

## 清理

本阶段删除41份临时JSON／日志，共2,068,465字节（1.973MiB）；清理前后2,016份保留文件SHA一致。随后仅更新本清理说明、存储台账及发布清单。结果目录只留REPORT.md，模型、环境和全局缓存保留。
