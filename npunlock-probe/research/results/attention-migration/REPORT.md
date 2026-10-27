# S41：Attention边界与原版Hugging Face模型迁移

日期：2026-10-26。本地开发版，尚未发布或修改公开source-release。平台为Windows x64、Core Ultra 7 258V / Lunar Lake NPU `8086:643e`、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu、Transformers5.18.0。

## 本轮改动

目标是让普通PyTorch/Hugging Face调用更容易迁移到`intel_npu`。沿用SHAVE算术及设备DMA，不引入OpenVINO运行时，没有下载权重、tokenizer或数据集，没有修改已安装的PyTorch/Transformers源码，也没有重建DLL/PYD/ELF。

- `torch.arange(..., device="intel_npu")`补功能式入口及start/end/step重载。旧版只有out入口，功能式组合会先产生空out，随后被精确输出形状保护拒绝，阻断HF位置索引生成。新入口先在Meta设备验证dtype/长度/1GiB字节预算，再从标量元数据生成CPU范围并上传。这是位置/索引工厂，不能称作原生NPU arange内核。
- `torch.cat`忽略用于初始化KV cache的rank1空Tensor，但仍计入其dtype提升和字节预算；反向为空输入生成空梯度。正常输入的形状、存储和输出别名保护保持。
- 浮点逐元素计算接受有界部分视图及零stride只读输入。输出别名检查在整理输入之前完成，再由设备DMA整理有效载荷，解决Llama RoPE对半切片取负的入口问题。
- 整数/BOOL输入的`cumsum`支持指定整数输出dtype与精确预分配out；默认INT64。用设备整数加法的并行前缀组合，保持INT64大于2^53及溢出模运算语义。按行复用既有连续范围DMA提取片段，避免本轮小尺寸字节Gather图编译超时的组合。非连续、标量、空输入已有对照；浮点cumsum仍不支持。
- `isin`补Tensor/Tensor、Tensor/Scalar、Scalar/Tensor及out入口，支持invert。用设备比较和逻辑归约，保留大INT64原始值和NaN不相等行为。最多16,777,216对比较，按有界块扫描；不是高效哈希集合。整数Tensor与浮点操作数混合、BOOL输入明确拒绝。
- 整数/BOOL constant pad走原始存储复制，支持合法裁剪及空结果，解决HF causal-LM labels末尾补`ignore_index`的问题；不会经过FP32近似转换INT64标签。
- native MHA及EncoderLayer推理入口补空batch/空序列。参数和mask保护先执行；空MHA返回空输出及None权重，对齐本机CPU原生入口，Encoder保留输入空形状。

实现位于根目录`torch_npu_sequence.py`、`torch_intel_npu.py`、`torch_npu_autograd.py`、`torch_npu_compat_extended.py`及`torch_npu_vision_extended.py`。序列和布局改动适用于一般调用，HF模型保持原类和原forward/generate实现。

## 权重共享与迁移方式

GPT-2的embedding与lm_head共享同一Parameter。本机Torch默认PrivateUse1转换会替换Parameter，可能将两个注册位置拆成独立参数：前向仍相同，训练梯度却被分开。这个问题不能靠放宽数值误差解决。

验收使用PyTorch已有的`set_swap_module_params_on_conversion(True)`，在转换期间保留Parameter身份，随后在finally恢复原设置。模型结构和共享关系均逐项核对。没有全局永久开启该选项或给所有Module._apply安装补丁；普通`.to("intel_npu")`本身仍不自动修复共享参数，迁移这类模型时须采用此方式。该选项是进程级状态，示例在单线程串行转换模型。

```python
import torch
import torch_intel_npu

old = torch.__future__.get_swap_module_params_on_conversion()
torch.__future__.set_swap_module_params_on_conversion(True)
try:
    model = model.to("intel_npu")
finally:
    torch.__future__.set_swap_module_params_on_conversion(old)

# 创建优化器应在转换后；CPU输入先整理成连续存储再上传。
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.5,
                            foreach=False, fused=False)
```

`torch_npu_attention_migration_demo.py`的`migrate_to_npu`封装这一验收步骤，便于复现；没有新增后端设备名或宣称默认Torch转换已无条件兼容所有共享存储。

## 验收范围和参考

采用Transformers原版`LlamaForCausalLM`和`GPT2LMHeadModel`，从Config随机初始化：不依赖任何预训练模型或联网资源。固定torch seed419、CPU线程1；CPU attention显式使用SDPA数学参考，模型默认选择`sdpa`。

| 项目 | Llama | GPT-2 |
|---|---|---|
| 层数、隐藏宽度 | 2层、16 | 2层、16 |
| attention | 4个Q头、2个KV头，RoPE/GQA | 4头MHA，学习位置嵌入 |
| FFN、词表、位置上限 | 32、32、32 | 32、32、32 |
| 参数量 | 5,712 | 5,504，含共享embedding/lm_head |
| Dropout | 0 | 0 |

inference覆盖batch1普通prompt和batch2左padding：prefill、3步teacher-forced DynamicCache decode，每步logits同时对照CPU cache及CPU完整前缀重算；每层K/V留在NPU并核对cache长度。每种prompt还分别开启/关闭cache，用原版generate贪心生成3个token，核对输出token及所有步骤的scores。

sampling使用top_k=4、top_p=1、temperature=1的原版generate：恢复NPU RNG后token/scores逐位重放，最终RNG状态相同，CPU RNG不变；每个采样步的处理后scores对照CPU对实际采样前缀的独立teacher forcing。没有要求NPU与CPU随机数流相同。

training使用原版`labels=input_ids`和内置causal-LM loss、use_cache=False，batch2/sequence4；FP32两步SGD动量训练。逐项核对独立CPU模型的loss、所有参数梯度和更新后参数；第一步模型/优化器状态显式转CPU，在新NPU模型恢复后，第二步loss、模型状态和全部动量逐位重放。没有把原始NPU Tensor直接交给torch.save。

误差标准固定为FP32 rtol=5e-5/atol=3e-6；FP16推理rtol=0.005/atol=0.001。整数、token、BOOL及指定逐位重放检查使用零误差。测试使用小词表随机权重，只验证算子与控制流语义，不评价语言质量、低位整网质量、吞吐或长期收敛。

## 正式验收结果

最终17组独立进程验收全部passed，共43,958次NPU计算图提交。数量包含准备、比较和读回所需操作，不是吞吐基准；重复开发轮不计入。

| 入口 / case | 精度 | NPU提交 |
|---|---|---:|
| HF Llama / inference | FP32 | 7,110 |
| HF Llama / inference | FP16 | 9,090 |
| HF GPT-2 / inference | FP32 | 6,671 |
| HF GPT-2 / inference | FP16 | 9,371 |
| HF Llama / sampling | FP32 | 2,047 |
| HF GPT-2 / sampling | FP32 | 2,069 |
| HF Llama / training | FP32 | 2,574 |
| HF GPT-2 / training | FP32 | 2,588 |
| 新序列/布局 primitives | FP32 | 193 |
| 新序列/布局 primitives | FP16 | 202 |
| native MHA/Encoder空输入 | FP32 | 0 |
| native MHA/Encoder空输入 | FP16 | 0 |
| 原standard_modules / inference | FP32 | 328 |
| 原standard_modules / inference | FP16 | 565 |
| 原layout / cat | FP16/FP32/INT64 | 539 |
| 原SDPA extended / gqa | FP32 | 393 |
| 原SDPA extended / dropout | FP16 | 218 |

两组native空输入每组覆盖4种MHA配置（空batch/空序列，含/不含padding mask）及4种Encoder配置（空batch/序列，pre/post norm），总16种。空计算不需要提交计算图，NPU_calls=0符合语义。原非空推理回归每精度包含plain/merged/all-padding MHA及两种Encoder，保留CPU原生全屏蔽行的NaN行为；没有关闭全局MHA fastpath。

两类模型、两种推理精度、两种prompt、两种cache设置共16次greedy generate均通过；每次3个新token，所有scores对照通过。每个模型/精度有6个cached decode比较点，同时匹配CPU cache及完整前缀参考，每层KV均为NPU Tensor。

| 模型 | FP32 loss第1→2步 | 恢复 |
|---|---|---|
| Llama | 3.4469473362 → 3.2790396214 | loss、全部模型状态和SGD动量逐位重放 |
| GPT-2 | 3.5423476696 → 3.3871634007 | 同上，embedding/lm_head共享Parameter保留 |

plain prompt `[1,4,7,9]`的greedy续写：Llama为`[24,30,7]`，GPT-2为`[9,9,9]`，两种精度均匹配CPU。相同NPU RNG下的top-k采样续写为Llama `[7,14,25]`、GPT-2 `[5,5,11]`；这是随机小模型的控制流样本，不评价文字质量。

离线共186项通过：149项根目录检查（含新增6项序列/布局/输出别名契约）、32项npunlock Python API、5项源码发布工具检查。CI已纳入新离线测试，不要求NPU或Transformers。原内存范围/形状图复用的两个AST测试夹具已跟随只读整理依赖更新，检查内容保留。

所有最终进程shutdown均closed=True、context_failed=False，live_buffers、cached_buffer_blocks、cached_scalar_buffers、cached_sessions及quarantined_buffer_bytes为0。没有留下后台NPU测试任务。

## 复现

已有S38–S40环境同步Python源码、重启Python即可，无需重建原生组件。HF测试使用现有可选`requirements-models.txt`；核心后端仍只要求torch/numpy，纯离线单测不需要Transformers。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=''
foreach ($precision in @('float32','float16')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','primitives','--precision',$precision)
    .\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','native','--precision',$precision)
}
foreach ($kind in @('llama','gpt2')) {
    foreach ($precision in @('float32','float16')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','inference','--kind',$kind,'--precision',$precision)
    }
    .\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','training','--kind',$kind)
    .\run_torch_npu.ps1 -Script .\torch_npu_attention_migration_demo.py -ScriptArguments @('--case','sampling','--kind',$kind)
}
```

## 保留的限制

SDPA仍保存完整分数，GQA仍重复KV；未新增FlashAttention、NestedTensor、分块在线softmax、DPU训练或异步提交。原有单buffer、score存储、行长及算子工作区约束保持，本轮没有扩大最大上下文或测性能优势。

本轮仅覆盖这两个模型族的上述小配置。不同架构、预训练大模型、混合/低位权重、FP16训练、AMP模型训练、长期训练、beam search、top-p采样、静态/滑动窗口cache、任意generate组合及通用torch.compile均未由本轮认证。特别是浮点cumsum尚缺，不能从top-k采样成功推广到top-p。

新增out重载要求精确预分配形状和无别名，不自动resize空out。arange来自标量元数据的CPU范围工厂；模型激活、cache、算术、梯度和优化器更新走NPU，但HF的Python生成控制、动态结果/完成标志仍同步主机。

开发过程中发现的小尺寸整数Gather编译超时与默认GPT-2共享权重梯度错误已分别通过改变前缀累计片段传输方式和显式保留Parameter转换解决；不能据此宣称所有Gather编译器形状或默认Module.to共享关系已修复。早期单层probe和失败调试轮不计入最终验收数量。

## 来源与指纹

按resident manifest验证64枚ELF、64份汇编、26份C源码，并按build.json验证C++/PYD/session DLL，157项指纹全部一致。新增代码复用既有内核，没有移植新的外部实现。下列为交付时SHA-256；安装包文件只核对，没有打包或改写。

```text
torch_intel_npu.py
69325d449e3e7c8cb4774d8292a06e1b87747be1be6b686f17d645ed1294099e
torch_npu_autograd.py
cc2e66c02685b8ad5a1a1c58f4cadb8dbbfc0bc09693051aa414892458aeebdd
torch_npu_sequence.py
57a971e63ef6ee963eaf30802353e4aae7d1b7e23c2fee2aff595028fb777184
torch_npu_compat_extended.py
e7753f4c535bfa359da21cb940222292831f19e465a2d891c194f0b7c82476d9
torch_npu_vision_extended.py
7040001f36f09dc461779fa0c750faf59a370ef387e36953395c6a98bd91f8c7
torch_npu_attention_migration_demo.py
1c598934020b844bcbbfdc007e52146b7701e282e098e6492ac279aaba688dac
test_torch_npu_attention_migration.py
e973574800c11db3fbab3fd22452ac09654208a436aea6ee6a7569463aa11d64
test_torch_npu_memory_scale.py
581b8862240728be1c013b605eb3e900ea8cc0a9a9fa0ba885e7e8d1752433ec
.github/workflows/offline.yml
c1636dc7dcb3b7a0defdda03fa37ffcdde1a2e930b69bf8bea3712fe6130957e
npunlock-probe/torch-device/backend.cpp
9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba
npunlock-probe/torch-device/_torch_intel_npu.pyd
433757698bbc704a44da3ca9c3c3355d3fac49ab19b82d06f0d09790aa4c6ea0
npunlock-probe/session-runtime/npunlock_session.dll
1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90
npunlock/tests/fixtures/npu4000/resident/manifest.json
d7b35afdcd6270c435f1b644af2c3fec5d3e0e9a16cb44c8860a3316356eb6bb
.conda/npunlock/Lib/site-packages/transformers/models/llama/modeling_llama.py
0108c9ddea94cadf55de878eb0e78bba0a8cf26349b3253f6d2f0c63d1805dd0
.conda/npunlock/Lib/site-packages/transformers/models/gpt2/modeling_gpt2.py
f7eee865bfac4f0d636dd174c435e50fb6d3728d3a7f194c1ebd941e8d49829a
.conda/npunlock/Lib/site-packages/transformers/cache_utils.py
136b3d85b4496b6c4d614a676bdfd4f32e926b88eafe9a88c7cf1e173e8390b5
.conda/npunlock/Lib/site-packages/transformers/generation/utils.py
7c24fb4503d2fadf5aff1bb7b0f253e2e7442b9e2775ea5fa703eea5e47d2b1b
.conda/npunlock/Lib/site-packages/transformers/masking_utils.py
100c7fce03f65deeb83031152d5cafd3030c2173b3a8bec1803b6739996d682f
.conda/npunlock/Lib/site-packages/torch/nn/modules/module.py
800bb3aa4f59b8efd49a46381b465873067f2bbc86c3a1b5172e2c9ba0981bd7
.conda/npunlock/Lib/site-packages/torch/__future__.py
c1c16cd3b2ecdf01dd6b9f73e7126b314eff0b0eece4f69b4dd09517063b2b9d
```

## 清理与交付

最终验收后删除17份正式JSON及2份早期probe，共19份、118464 字节。删除前后896份保留源码/内核/原生组件/报告及安装源文件SHA-256一致，随后补充本节和空间台账。无新增依赖、环境、权重、图表、磁盘checkpoint、持久图缓存或原生构建。测试进程全部退出，source-release未改，未推送GitHub。

