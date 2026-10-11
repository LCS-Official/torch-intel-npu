# SmolLM2-135M：真实权重的 NPU 单层与混合生成

2026-10-04，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，
驱动 32.0.100.5540，图编译器 8.3。结果：`passed_with_limitations`。

此文保留融合 RMSNorm 前的首轮记录。当前适配器已默认使用融合内核，最新结果见
`../rmsnorm-vector/REPORT.md`；串行融合阶段在 `../rmsnorm/REPORT.md`，
原计时/profile/汇总快照在 `../rmsnorm/before/`。
相同 demo 的 case JSON/log 路径现保存最新运行，源码与内核 hash 以该次记录为准。

已下载原始 safetensors，跑通 CPU 完整模型参考、真实 decoder layer 的 FP16/W8A16
NPU 推理，以及第 0 层 NPU、其余 29 层 CPU 的短文本生成。新增
`torch_npu_smollm.py` 和离线检查 `torch_npu_smollm_demo.py`；复用已有权重编码、
weight_graph、FrozenTransformer 图执行/缓存/ELF 检查与真正 FP32 SHAVE 算术。
基础设备后端、原 Transformer、低位 Linear、C++/DLL 和 resident 内核均未修改。

当前单层比 CPU 慢，测量定位到分散的 FP32 RMSNorm 图。后续优先融合 RMSNorm。
本轮没有实现整网 NPU 推理、KV cache、INT4/NF4 LLM 或新的训练功能。

## 模型、依赖与可复现性

来源：[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac)。
固定 revision：`12fd25f77366fa6b3b4b768ec3050bf629380bac`。
本地目录：`E:\Pytorch_xfire_NGPU\models\SmolLM2-135M-Instruct`。
只下载 model.safetensors、config/generation_config、tokenizer 文件及 README；
未下载 GGUF、ONNX、pickle 权重，也未启用远程 Python 代码。

model.safetensors 为 269,060,552 字节，SHA256：
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`，
与该 revision 的下载 LFS 元数据一致。`download-manifest.json` 保存所有下载文件
的大小和 SHA256，本轮重新验证全部与 manifest 相符。

[该 revision 的配置](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json)
为 Llama 类结构，30 层、hidden 576、FF 1536、9 个 query head、3 个 KV head、
head_dim 64、SiLU/SwiGLU、RMSNorm eps=1e-5、RoPE theta=100000，词表 49152，
无 attention/MLP bias，embedding 和输出权重绑定。原始参数 BF16；本轮冻结为 FP16
或再按输出通道量化 INT8，激活、attention 矩阵和图输出保持 FP16。

环境为项目 `.conda\npunlock`：Python 3.12.14、Torch 2.14.1+cpu、NumPy 2.5.3。
本轮新增 safetensors 0.8.0、tokenizers 0.23.2、huggingface_hub 1.33.0、
transformers 5.18.0 及其依赖；Torch/NumPy 版本未变。未安装 OpenVINO runtime。
适配器只额外需要 safetensors；Transformers/tokenizer/Hub 用于准备和官方 CPU 参考。
demo 设置 HF_HUB_OFFLINE，local_files_only=True；测试阶段不访问网络。

各 case JSON 记录执行时源码 SHA256、包版本、模型 manifest、结果与清理统计。
`before/` 保留原核心/README；`after/` 保留最终适配器、demo 和文档。
`summary.json` 汇总本轮，原 Transformer 三种模式的 probe 回归亦通过。

## 支持和调用

```python
import torch
import torch_intel_npu
from torch_npu_smollm import FrozenSmolLayer

layer = FrozenSmolLayer.from_checkpoint(
    r"E:\Pytorch_xfire_NGPU\models\SmolLM2-135M-Instruct",
    layer_index=0, mode="w8a16")  # 也可 mode="fp16"
# hidden_states 是相应层的 CPU FP16 隐藏状态，不是整数 token ID。
with torch.no_grad():
    output = layer(hidden_states.to("intel_npu"))
```

构造校验精确结构和全部九个参数的 dtype/shape/有限值，并拒绝 FP16 参数溢出。
输入 FP16 `(B,S,576)`，B<=8、S<=64，所有中间张量遵守原 1,048,576 元素限制。
接受完整密集排列，经 NPU DMA materialize；拒绝部分 storage、偏移、错误 dtype、
active autograd 和训练模式。保存原 CPU checkpoint，再构造冻结层；其权重不是
可训练 Parameter，普通 state_dict 不包含冻结字节快照。

forward 为：真正 FP32 RMSNorm → attention 图 → 真正 FP32 RMSNorm → FFN 图。
RMSNorm 在 FP32 中求均方、加 eps、rsqrt、归一化，然后转 FP16、乘 FP16 gamma，
与本地 Transformers LlamaRMSNorm 的顺序一致。RoPE 采用非交错 rotate_half，
GQA 将每个 KV head 复制三份；causal mask 禁止看未来 token。
SwiGLU 为 SiLU(gate projection) × up projection → down projection，遵守
[Swish-4 IR 规范](https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/activation/swish-4.html)。
运行图由 Intel 驱动编译器生成，SHAVE 算术使用既有自定义内核，没有 CPU 算术回退。

权重读入/编码和位置 0..63 的 RoPE 常量是显式 CPU 准备；norm gamma 首次上传并缓存。
热连续调用没有测得显式主机载荷复制或新编译。计数不包含驱动 DMA 或编译器内部复制。
ELF 检查证明存在 INT8 矩阵权重操作数，不声称逐一映射所有投影和 DPU 任务；
attention/布局也可能产生非逐元素 FP16 DPU 任务。

接口要求从零开始的完整前缀，没有 KV cache、任意位置、padding/custom mask、
一般 Llama ATen dispatch 或普通模型 `.to('intel_npu')` 接入。S=1 检查是单 token
完整序列，不能理解成有历史 KV cache 的自回归 decode 吞吐。
只验证有限输入的模型路径，未建立 attention/非线性对 NaN/Inf 的通用语义保证。

## 验证与精度门槛

| Case | 检查与结果 |
|---|---|
| cpu | CPU FP32 整网贪心生成 12 个新 token，原始权重/分词正常 |
| probe | 第 0 层 B1/S16，FP16/W8A16；保存生产 attention/ffn XML/bin/blob/NPZ |
| numeric | 第 0/10/20 层，B1/2、S1/17/32/64，共 8 组 |
| boundary | 最后一层（index 29）B8/S64，两个模式，最大 batch/长度通过 |
| guards | 16 项：结构、权重溢出/非有限值、设备/dtype、部分存储、训练、梯度、缓存及快照 |
| causality | 两个模式，改变后 8 个 token，前 9 个输出逐位不变 |
| hybrid | 两条提示 × 两个模式，共 4 次真实短生成，各有 9 次单层 NPU forward |
| performance | 三个形状 × 两个模式，先数值检查，再热调用交错计时 |
| profile | B4/S64 W8A16 固定 NPU 输入，分开测两次 norm、attention、FFN |

合计 19 组层级数值检查，另验证每组两次 RMSNorm 和固定实际输入的两段图。
CPU 输入取自真实分词和官方整网的相应层隐藏状态，并非仅用随机激活。
最后一层参考直接调用官方 layer，未误用已施加 final norm 的最后 hidden_state。
CPU量化参考用 FP64 点积计算 INT8 码×FP16 scale；不提前重建舍入后的 FP16 权重。
浮点矩阵参考使用 Torch，避免 NumPy BLAS 与 Torch OpenMP DLL 冲突。

门槛分别为：

- 未量化的分阶段 CPU 参考与官方 FP16 层，relative L2<0.1%。
- 同一实际输入的 RMSNorm，rtol=.002、atol=.001；其内部计算真正 FP32。
- 整层 NPU 与相同权重格式的 CPU 参考，relative L2<0.3%。
- 每个输出元素 `abs(error)/(abs(reference)+该 token 输出的 RMS)<1%`。
- 固定相同实际 residual/normalized 输入的 attention 和 FFN，L2<0.1%、逐项尺度误差<1%。
- 量化 NPU 输出与官方未量化 FP16 层，relative L2<3%。

所有层检查的最大执行 L2 为 **0.06784%**，最大逐项尺度误差 **0.42664%**；
固定输入段最大 L2 **0.05146%**；CPU 分阶段对官方层最大 L2 **0.01975%**。
W8A16 相对官方浮点层最大 L2 **1.13399%**（最后一层 B8/S64）。
这些是层级误差指标，不等价于完整 30 层模型任务质量。

初始全图采用 FP16 RMSNorm，随后改为真正 FP32 RMSNorm；layer0 FP16 执行误差
从约 0.0876% 降到 0.0471%。但 **原全层逐元素 rtol=.01/atol=.005 仍有失败**：
layer0 W8A16 B1/S16 的一个接近零输出，reference=0.05126953，actual=0.05853271，
差值 0.00726318；输出 RMS 约 1.944。末尾 down Linear 独立执行时在该位置可与
CPU 参考一致，但上游相同输入的门控/FFN 仍有小幅舍入偏差；拆开 down 并未改善
最终输出，拆图实验已撤回。诊断支持误差沿整层传播，未证明一个可单独修复的
驱动错误，也未证明已消除全部融合语义差异。

因此最终采用上述尺度化门槛，原固定-atol mismatch 数逐项保留在 JSON，不能称为
通过了旧门槛。深层激活尺度更大：最后一层 FP16/W8A16 的最大绝对误差分别
0.5/0.75，尺度误差分别 0.164%/0.304%；原固定-atol 失败 2040/3672 项。
`fused-rms-failure.*`、`fp32-rms-nearzero-failure.*`、`fp32-rms-before-ffn-split.*`、
`ffn-split-experiment.*` 保留原失败和源码。无段名的早期 XML/blob 属于原型；
最终正常 forward 对应带 `-attention`/`-ffn` 的文件，诊断全图不是生产路径。
既有真正 FP32 设备运算保证未改变，本适配器整体是浮点近似而非逐位保证。

## 真实生成接入

demo 的 Bridge 仅将第 0 层的 FP16 hidden states 上传 NPU、执行冻结层、下载结果，
其余 29 层、embedding、final norm、词表 logits 明确在 CPU。使用 `use_cache=False`，
每个新 token 重算完整前缀；检查位置和普通 causal mask，拒绝 padding/KV cache。
CPU/NPU 传输在此演示中是显式的，不能把热单层的零显式复制说成整网零复制。

| 提示 | 模式 | 首个 next-token logits L2 | 短生成与 CPU |
|---|---|---:|---|
| What is two plus two? | FP16 | 0.08143% | 8 个新 token 相同 |
| What is two plus two? | W8A16 | 0.18382% | 8 个新 token 相同 |
| Complete: Paris is the capital of | FP16 | 0.08821% | 8 个新 token 相同 |
| Complete: Paris is the capital of | W8A16 | 0.24476% | 8 个新 token 相同 |

例如第二条提示均生成 `Paris is the capital of France.`。next-token 分布 KL 最大
0.000179。这里只比较两条短提示，不能作为一般语言能力、困惑度或所有层同时量化
的质量结论。CPU 完整模型参考和混合生成也不代表完整 NPU 推理。

## 性能与下一步

同进程 3 次预热、21 轮四条路径循环交错，CPU 8 线程。CPU rotary/mask 已在计时
前准备；NPU 同步完成后返回。排除模型读取、量化、首次编译、输入上传和输出下载。
CPU FP32 层由同一 CPU FP16 参数转为 FP32，输入也是同一 FP16 隐藏状态转为 FP32。

| 输入 (B,S,576) | NPU FP16 ms | NPU W8A16 ms | CPU FP16 ms | CPU FP32 ms | 单层热图次数 |
|---|---:|---:|---:|---:|---:|
| 1,1,576 | 13.413 | 13.016 | 1.088 | 0.949 | 22 |
| 1,32,576 | 19.357 | 19.044 | 2.331 | 1.669 | 26 |
| 4,64,576 | 127.780 | 129.097 | 7.041 | 6.010 | 120 |

两段图的 ELF 常量区，B1/S32：FP16 7,205,376 字节，W8A16 3,691,008 字节，
约下降 49%；B4/S64 为 7,558,912 / 4,019,968 字节。此数不包含 gamma 缓冲区、
其他归一化会话、驱动开销或主机冻结快照，不等价于进程总体内存降低相同比例。
冷 compile+run 为约 0.5–4.6 秒，包含该次首次 norm 图/gamma 准备；FP16 先执行，
W8A16 可复用 norm 会话，不能把它们当作完全独立冷编译速度对照。

B4/S64 W8A16 的 9 轮分段 profile：

| 段 | 中位数 ms | 图提交次数 |
|---|---:|---:|
| RMSNorm1 | 59.790 | 59 |
| Attention（含残差） | 0.855 | 1 |
| RMSNorm2 | 59.360 | 59 |
| FFN（含残差） | 0.711 | 1 |

两次 norm 占分段中位数之和约 98.7%。该比例用于定位瓶颈，各段中位数相加并非
另一轮整层时延。真正 FP32 norm 目前由转换、平方、mean、eps、rsqrt、广播、乘法、
转换和 affine 等独立操作组成，大输入还会分块；每段同步 fence 和布局操作成本高。
在本轮形状上，多次提交抵消了 DPU 的计算收益；量化减小常量空间，但并未带来
显著整层提速。这是一次未控制桌面负载的测量，不是功耗测试或 NPU 理论性能上限。

推荐下一步：为固定 width=576 融合一个 FP16 输入/输出、内部真正 FP32 的 SHAVE
RMSNorm，将目前每次 59 个图降到一个有界图，保留官方 half-rounding/affine 顺序，
先按现有真实层/边界/生成用例回归，再测性能。之后再将更多 decoder 层放入 NPU；
完整模型还需处理 49152×576 embedding/lm_head 超出现有张量/权重元素限制的问题，
以及 KV cache、长上下文和整网量化质量。当前没有实现 NPU/GPU 共同训练或采样。

## 复现

在 `E:\Pytorch_xfire_NGPU` 下运行，模型已下载，环境已准备：

```powershell
foreach ($smolCase in 'cpu','probe','numeric','boundary','guards','causality','hybrid','performance','profile') {
    .\run_torch_npu.ps1 -Script torch_npu_smollm_demo.py -ScriptArguments '--case',$smolCase
}

# 原 Transformer 回归
.\run_torch_npu.ps1 -Script torch_npu_transformer_demo.py -ScriptArguments '--case','probe'
```

每项在独立子进程中离线执行，超时 300 秒，NPU 测试必须串行。正常 NPU case 的
shutdown 后 live_buffers=0、cached_sessions=0、context_failed=false，原三种模式
Transformer probe 也通过。源码/依赖 hash 和调用总数以 `summary.json` 及各 case
JSON 为准；每项保存运行时版本，demo 增加独立 case 时已有记录不会重写其源 hash。
