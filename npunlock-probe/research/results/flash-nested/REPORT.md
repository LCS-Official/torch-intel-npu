# S42：原生SHAVE分块Attention与strided NestedTensor实验支持

日期：2026-10-26。本地开发版，尚未发布或推送GitHub。平台为Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu、Transformers5.18.0。

## 结果与定位

已实现可选的FlashAttention式分块/重计算路径，以及标准Encoder所用的strided NestedTensor推理子集。两者可通过标准SDPA入口组合使用；原版HF Llama/GPT-2小配置也通过FP32训练与恢复检查。

主要收益是降低Tensor工作区及反向保留状态。本轮128×256测例的后端Tensor峰值降低约86%–88%，但含编译的冷启动前后向约慢3倍，**没有证明速度优势**。这是一版实验实现，默认仍走原数学Attention。

没有下载模型/数据、安装依赖、修改site-packages或重建DLL/PYD。新增一个C源及两组源码对应ELF/汇编，原有64组内核保持；普通运行不依赖OpenVINO runtime。

## FlashAttention式实现

依据[FlashAttention论文v2](https://arxiv.org/abs/2205.14135v2)的分块在线softmax及反向重计算方法，自行实现NPU40XX SHAVE版本，没有移植其CUDA代码或调用flash-attn包。

`resident_flash_attention.c`在一个有界CMX任务内融合QK点积、mask/因果位置、在线最大值/归一化和PV累积。每个query块只保留每行最大值、指数和及未归一化输出，不保存整个L×S分数或概率矩阵。反向读取这些行状态并重新计算块概率，生成Q/K/V梯度；Python层累加广播与GQA共享头梯度，K/V不用先复制成所有Q头。

Python负责切片、FP16→FP32 tile转换、byte-packet组装、调度和输出拼接；计算有效载荷在NPU。现有Power carrier把每个完整packet搬入CMX，宿主提交仍同步串行；不是单个覆盖整个模型的融合图。packet组装、重复搬运及提交开销是后续优化方向。

FP16/FP32输入均采用FP32累计，复用项目原有IEEE运算与exp helper，FP16最终转换使用已有保留subnormal的路径。scale按正负平方根分别应用于Q/K，与原数学实现的防溢出方式一致。头维、tail块、不同Q/K长度、因果全局位置、负scale、BOOL/加性mask、全屏蔽有限输入行和部分非有限值前向已有对照。

### 范围与选择方式

- 同dtype FP16/FP32、rank2–64、有界广播与GQA，QK/V头维各为1–128。
- query block为1–32、key block为1–128；默认16/64。预检最坏前后向packet，最多32,768个FP32 word，包括header和可选mask；超出时在提交前拒绝，须减小block。
- dropout_p目前必须为0；浮点mask作为常量可用，trainable mask明确拒绝。默认math路径仍保留原有Dropout和mask梯度。
- 支持Q/K/V一阶梯度、空batch/序列对应零梯度及保存Tensor版本保护；不提供二阶梯度、BF16或任意head dimension。
- streaming检查不再要求完整分数矩阵能放入单buffer，也不再调用旧65,536列整行softmax检查。原有操作数/输出、rank、因果query header、1GiB分配和图缓存边界保留。离线Meta检查通过4097×65537逻辑分数，不表示已真机运行这一规模或任意长上下文。

```python
import torch
import torch_intel_npu
from torch.nn.attention import sdpa_kernel, SDPBackend

# q/k/v 已在 intel_npu；训练时需要 requires_grad=True。
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
out.sum().backward()

# 也可直接选择块大小；此入口不改变PyTorch backend开关。
out = torch.intel_npu.flash_attention(q, k, v, query_block=16, key_block=64)
```

本后端读取标准sdpa_kernel上下文的选择：math关闭且flash开启时采用新路径。默认开关下math仍开启，因此不会自动切换；只选efficient/cudnn且禁用math/flash时明确拒绝。未提供CUDA/PTX版本或上游flash-attn包兼容性。

## NestedTensor实现与边界

支持的是PyTorch真实`NestedTensorImpl`、`torch.strided`格式，设备为intel_npu，payload使用NPU共享存储；不是把Python list伪装成设备Tensor。形状、stride、offset仍是CPU INT64元数据。

接入内容包括factory已有的NPU构造、CPU↔NPU复制、clone、copy_、batch unbind、values、to_padded_tensor及精确预分配out、CPU布局元数据view、LayerNorm/native stats、native MHA/Encoder和Nested SDPA。copy_拒绝不同对象的同storage别名，训练在进入no_grad或有效载荷计算前明确拒绝。不是通用NestedTensor算子全集。

默认`nn.TransformerEncoder(..., enable_nested_tensor=True)`在eval/no_grad、右侧padding满足左对齐keep mask等上游条件时，可正常进入nested fastpath。对齐检查在NPU，主机只取得检查标志及每样本长度；转换、打包和计算使用设备有效载荷。每个ragged样本复用原有稠密native MHA/Encoder推理，输出重新打包；这条native模块路线仍使用原数学注意力。MHA的平均/逐头weights按最大长度补零，空组成项已有验证。

标准`F.scaled_dot_product_attention`也支持这些Nested输入：无外部mask、dropout=0、非空batch，每样本进入math或新flash路径，支持已验收的GQA/因果配置。为避免上游AutogradNestedTensor复合实现先拆成未支持的mul，本轮安装一个仅对intel_npu改道的SDPA分派；其他设备通过去掉当前AutogradNestedTensor key的redispatch进入其原复合实现。CPU Nested前向及反向已与独立稠密参考核对；CUDA/XPU Nested没有本轮真机验证。

metadata getter和buffer-view构造复用PyTorch已有CPU metadata kernel：前者只返回CPU布局Tensor，后者检查storage边界并包装存储，没有CPU激活算术。这些调用按本机已安装实现核对；布局检查和数值执行分别保留自己的设备保护。

[PyTorch Nested文档](https://docs.pytorch.org/docs/2.14/nested.html)另有推荐的torch.jagged格式；本轮选择标准Encoder当前使用的strided路线，**没有实现jagged/Nested训练、通用torch.compile、所有布局/空批次/算子或AMP契约**。需在eval/no_grad或inference_mode下使用这里的推理接口。

```python
layer = torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True)
model = torch.nn.TransformerEncoder(layer, 2).to("intel_npu").eval()
with torch.no_grad():
    out = model(padded_input, src_key_padding_mask=padding_mask)

nt = torch.nested.nested_tensor(
    [sample_a.to("intel_npu"), sample_b.to("intel_npu")], device="intel_npu")
dense = nt.to_padded_tensor(0)
```

HF共享Parameter仍按[S41迁移步骤](../attention-migration/REPORT.md)在转换期间使用PyTorch swap选项，随后恢复原设置。Nested支持不改变原始设备Tensor序列化、异步stream/event或通用CUDA生态限制。

## 实测内存与时间

每条记录为独立进程，输入Q=(1,2,128,16)、K/V=(1,2,256,16)，随机有限值、dropout=0；包含一次前向及Q/K/V反向，全部输出和梯度与CPU数学路径核对。saved_tensors_hooks按唯一storage去重，包含原始输入及反向保留状态。

| 精度 | math Tensor峰值 | flash Tensor峰值 | 减少 | math反向保存 | flash反向保存 |
|---|---:|---:|---:|---:|---:|
| FP32 | 2,236,424 B | 308,232 B | 86.22% | 393,216 B | 100,352 B |
| FP16 | 2,203,656 B | 275,464 B | 87.50% | 352,256 B | 59,392 B |

这是后端`peak_buffer_bytes`，不包含编译器、图缓存blob、驱动内部对象或CMX总占用，不能当作整个进程物理RAM或任务管理器NPU内存。flash的保存项为Q/K/V及16份query块行状态，没有完整概率矩阵；保留状态随序列长度线性增长。

同一测例的冷启动前后向：FP32 math5.680秒、flash17.204秒；FP16 math6.421秒、flash20.470秒。提交数分别为537/1552和550/1864。时间含首次图编译、packet组装和同步提交，不是稳态内核计时、TFLOPS或通用模型速度结论。当前这一测例更慢，不能把省内存写成已获得FlashAttention级加速。

## 数值、模型与独立参考

正式检查的FP32阈值为rtol=5e-5、atol=5e-6；FP16梯度为rtol=0.01、atol=0.002，推理及Nested结果使用0.005/0.001。整数元数据、CPU往返、clone和指定恢复采用零误差。特殊值前向使用equal_nan；不推广为任意非有限值反向认证。

原C body另通过16组宿主case、288个前后向packet，与独立CPU FP64 SDPA及autograd对照。覆盖负/零/正scale、tail、因果、BOOL/浮点mask和全屏蔽行；宿主仅拓宽入口/指针以运行相同C算法，不用于证明SHAVE调度或性能。

Transformers原版两层随机Llama（5,712参数）和GPT-2（5,504参数，保留embedding/lm_head共享）采用标准SDPA flash选择，FP32两步SGD及新模型CPU checkpoint恢复通过：所有梯度、参数对照通过，恢复后的loss/模型/动量逐位重放。loss分别为3.4469473362→3.2790396214、3.5423476696→3.3871634007，与原math验收一致。未下载权重或修改HF模型源码；不评价语言质量或长期收敛。

## 正式验收

最终21组独立进程检查全部passed，共39,108次NPU提交；重复开发轮、早期probe和独立宿主C检查不计入此数。所有最终进程shutdown为closed=True、context_failed=False，live_buffers、cached_buffer_blocks、cached_scalar_buffers、cached_sessions和quarantined_buffer_bytes均为0。

| case | FP32提交 | FP16提交 | 主要覆盖 |
|---|---:|---:|---|
| masks | 2,967 | 3,569 | 全局因果/tail、BOOL及常量浮点mask、全屏蔽行、GQA/广播梯度、标准flash选择、17×65且head128的边界 |
| contracts | 64 | 78 | dropout、block、可微/错误mask拒绝；拒绝不提交/不推进RNG，空梯度、KV版本检查 |
| special | 234 | 286 | NaN Q、Inf V、Inf mask、全屏蔽且Inf V四种前向 |
| nested | 875 | 1,318 | 真正Nested构造、CPU往返/clone/copy、别名/训练拒绝、LayerNorm stats、默认两层Encoder和可选final norm |
| nested-attention | 399 | 529 | Nested flash SDPA因果/GQA、原生MHA平均/逐头weights、5/2/0长度、CPU Nested前后向保留 |
| memory-math | 537 | 550 | 独立进程的完整前后向内存/状态对照 |
| memory-flash | 1,552 | 1,864 | 同上，线性行状态且无完整概率保存 |

另有HF Llama/GPT-2 FP32训练/恢复各4,140和4,208次提交；Llama FP16 inference为14,574次，含普通batch1及左padding batch2、6个cached decode比较点、use_cache开/关共4次greedy generate，每次3个新token，所有scores及token均匹配CPU，KV保留在NPU。

原路径回归为standard_modules/inference FP16 565次、SDPA extended/gqa FP32 393次、dropout FP16 218次、boundaries FP32 188次，全部通过。全局MHA fastpath保持；CPU Nested对照没有被NPU数值参考替代。

193项离线检查通过：156项根目录检查（含新增7项）、32项npunlock Python API、5项源码发布工具检查。额外宿主C结果为16组/288 packet passed。实际真机范围主要是rank2–4与上述模型；rank64/超旧score边界为实现及Meta保护验证，不是大模型性能认证。

## 复现与更新

已有S38–S41环境同步Python实现、新C/ELF/汇编及manifest并重启Python即可，无需重建DLL/PYD或安装新依赖。拥有现有Movi工具链时，可运行build_flash_attention_kernels.py重建两枚内核；完整build_resident_kernels.py也已接入此步骤。源码发布默认使用已核对的ELF。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=''
foreach ($precision in @('float32','float16')) {
    foreach ($case in @('masks','contracts','special','nested','nested-attention','memory-math','memory-flash')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_flash_nested_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
foreach ($kind in @('llama','gpt2')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','training','--kind',$kind,'--backend','flash')
}
.\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','inference','--kind','llama','--precision','float16','--backend','flash')
```

离线检查入口为test_torch_npu_flash_nested.py，已加入CI，无需NPU或Transformers。宿主C复现用verify_flash_attention_host.py的--emit-source生成临时C，使用现有tools/msvc_environment.ps1选取MSVC，以/LD /O2 /std:clatest /TC构建DLL，再用--dll校验。

## 开发边界与后续

最初把所有IEEE helper强制内联，生成约80KB ELF，被既有代码窗口保护拒绝，未绕过ABI保护；改为链接内局部共享helper后，两枚最终ELF为9,072及10,116字节。没有扩大代码窗口、修改驱动或破坏原有内核。

开发中补正CPU来源Nested复制、部分存储LayerNorm整理、Half native stats dtype，以及AutogradNestedTensor优先级；CPU代理及真实NPU接口对照用于防止只看单例成功。原有数学路径保留，并继续认证其GQA、Dropout、空边界和标准模块。

下一阶段可研究减少byte-packet副本、复用动态header/图、按head并行和批量提交，以及更充分的预训练模型/长上下文验证。jagged、Nested训练、flash Dropout/trainable mask、更广泛AMP和真正稳态性能优化仍有工作，不宣称本轮完成通用FlashAttention/NestedTensor支持。

## 指纹与原组件保护

原64组ELF/汇编及26份C源码的manifest记录与本轮开始快照逐项一致。当前66组ELF/汇编、27份C源码及C++/PYD/session DLL共162项文件指纹均匹配manifest/build.json。未重建旧原生组件，未修改已安装Torch文件。下面为交付源码/组件及相关安装源文件SHA-256。

```text
torch_intel_npu.py
1a31e69264a39c9915324a53db76919e6b8eb3fadae867040f06ab39e751b6a1
torch_npu_autograd.py
5ded3982d6a1f2a06c12da8569725b5bd2ecf324b9abc936272df85715aa9337
torch_npu_flash_attention.py
6f0ce408bac272f2a7814067f3eebad758a2362287da1d57242d8b07a721bd90
torch_npu_nested.py
a5966183d497a2c87eb8959a3e9a4339fb2b1431e9985a0f3b231b9821836d7b
torch_npu_flash_nested_demo.py
ce4b296b90a76db8483474897957ea919822350b4ae9b17445cb0c85e579c0b9
torch_npu_attention_migration_demo.py
85d27b01f70a2f112816c11e2e4e92aee652f6b880aab1f1fe243d7504ebdfd3
test_torch_npu_flash_nested.py
982162cca90fac6342450f425596707b5fe526ba462774ef57c1e7522c2bb43b
npunlock-probe/research/build_flash_attention_kernels.py
56ede9f7a60a413db463d5d03920f480376e406bf43c7702b58c68d77c6c8c0b
npunlock-probe/research/verify_flash_attention_host.py
0f6c528347793c9da14636d8f354eb6291d2dd8c25f37558d1e26fa32a665c14
npunlock-probe/research/build_resident_kernels.py
ea0b9ed0a1394dbf99a83d6f2d6f659fee6715564aac1f84e5d9c2897f356856
.github/workflows/offline.yml
38618915e8361c0f772f9553be2f420d6a39345f744f7b7a63a75efe9cf9e5be
references.bib
ca38243e62bd4fb3d19d6daf68c24bb483514c667e198db3c9e0a61c0d4ba6a9
npunlock/tests/fixtures/npu4000/resident_flash_attention.c
2a402bc60be67ec326a45bc56885498b43fa4f99d89e7f38e96196bb1e3e65a4
npunlock/tests/fixtures/npu4000/resident/manifest.json
ee439860a0a2b19c79d2b23b15b166e08969f3385f54ee5a48b10f98dbeb6fb2
npunlock/tests/fixtures/npu4000/resident/flash-step-f32.elf
8ed2a8296146251b67c6434d08592e6c191765fb20098bfc58efdc2c597784f3
npunlock/tests/fixtures/npu4000/resident/flash-step-f32.s
e4846d055d7d33e1efa9e8c67d7f04f4696df40ac46920a5f7e0b3eef67abc48
npunlock/tests/fixtures/npu4000/resident/flash-backward-f32.elf
1f459e8d655cef7a117918dfd5e5a2e9cab7b81f2b114bfbc88501985df9c276
npunlock/tests/fixtures/npu4000/resident/flash-backward-f32.s
761e40879093a7740c7ef8b406e751dff224f95e28ef021e580b18b4bd52c17e
npunlock-probe/torch-device/backend.cpp
9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba
npunlock-probe/torch-device/_torch_intel_npu.pyd
433757698bbc704a44da3ca9c3c3355d3fac49ab19b82d06f0d09790aa4c6ea0
npunlock-probe/session-runtime/npunlock_session.dll
1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90
.conda/npunlock/Lib/site-packages/torch/include/ATen/NestedTensorImpl.h
556d572508cc33380673d766b3b86b3a929f70d95eeaf916fb77bc6b01bf1c66
.conda/npunlock/Lib/site-packages/torch/nested/__init__.py
7a6e42579e1df9cd9e97f047e201a9e758d1ce44791e84487af49fb29d1d1c75
.conda/npunlock/Lib/site-packages/torch/nn/modules/transformer.py
7c02d7948a34015aa835df9bb4e1886b3ec59e84684c30765d3dc3d321d99098
```

## 清理与交付

本轮删除21份正式进程JSON、5份宿主/早期probe记录、12份Movi阶段JSON/response和5份临时宿主C/DLL/链接文件，共43份、530092 字节；空host目录一并移除。删除前后905份保留源码、内核、报告、原生组件及相关安装源文件SHA-256一致，随后补充本节和空间台账。保留C、两枚新ELF/汇编、Python实现/测试、构建/复现脚本及Markdown报告。无新增依赖、环境、权重、图表、磁盘checkpoint或持久图缓存；测试全部退出，公开source-release未改，未推送GitHub。

