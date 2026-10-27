# S43：分块 Attention 热路径与自动分块

日期：2026-10-26。平台：Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），Windows，驱动32.0.100.5540，NPU编译器8.3，独立 `.conda/npunlock` 环境，PyTorch 2.14.1+cpu。沿用[S42实现](../flash-nested/REPORT.md)，没有新增依赖、环境或模型权重。

这轮把实验 flash 路径的预热前向加Q/K/V反向，在固定小测例上提速约2倍，主要收益来自SHAVE内核热路径，随后是减少分块数量。优化直接写在C源码和后端中，标准SDPA显式选择flash后自动生效。此测例仍比本项目的math路径约慢一倍，保持flash为显式选择；没有整网训练加速、硬件峰值算力或通用FlashAttention性能结论。

## 先测量，再选方向

使用新增[复现脚本](../../../../torch_npu_flash_optimization_demo.py)，相同随机种子431、相同Q/K/V与反向seed。Q为`[1,2,128,16]`，K/V为`[1,2,256,16]`；无mask、无dropout，所有Q/K/V均求一阶梯度。每个模式独立进程，先运行1轮预热，再计3轮中位数，计时涵盖完整前向和反向，梯度每轮清空。FP16/FP32分别验证输出及全部梯度与独立CPU math SDPA一致。

预热测量区间没有新建图、LRU淘汰或显式CPU/NPU上传下载。Python/驱动/系统仍有正常波动，没有固定功率、频率或排除全部后台负载。本报告是单平台短测，不外推其他尺寸、长时间吞吐或整网效果。

脚本按`_run`、范围提交等入口累计宿主墙钟时间，包括Python检查、会话查找、驱动提交、等待和内核执行。它不是硬件计数器，不能直接拆出纯计算周期、DMA带宽或NPU利用率。

原版FP32的3轮累计前后向为10.784秒，两个flash计算图累计9.551秒，约占88.6%；byte-pack累计仅0.060秒。因此本轮优先优化计算图内的重复标量运算，而没有改动原生提交/chain或引入异步队列。

## 源码改动

1. [SHAVE C内核](../../../../npunlock/tests/fixtures/npu4000/resident_flash_attention.c)：把常规FP32加乘除的简单分支局部内联，异常、下溢及次正规数继续走已有IEEE辅助函数；没有把整套64位软件边界处理在每个调用点展开。`-ffp-contract=off`保持不变。
2. 每个query行的带符号`sqrt(abs(scale))`缩放只计算一次，同一行的各key dot和dK累积复用该结果。乘法及求和顺序保持原有定义，没有降为FP16/TF32或移除补偿求和。
3. [后端自动分块](../../../../torch_npu_flash_attention.py)：默认优先query32/key128，按实际尾块、QK/V维数以及mask字节数检查完整反向packet，必要时缩小自动选择的块。head dimension 128、长于128的本轮测例自动选32/64。用户显式指定的块不会被静默缩小，超预算仍报错。公共packet上限仍是32,768个FP32载体word，未放宽驱动/CMX/代码窗口保护。
4. 新增自动分块离线测试；真机增加head dimension 128、Q33/K129的跨块尾部因果梯度对照，以及FP32次正规数前反向检查。宿主C oracle仅适配GNU内联属性的MSVC拼写，算术体不变。

只重建原有两枚flash ELF及其汇编、manifest；旧64枚resident内核、其余C源、Torch C++/PYD和session DLL不变。没有借此改动NestedTensor的能力边界，也没有添加K/V全序列缓存或完整score矩阵。

## 预热性能与内存

下表前向、反向、总计各自取3轮中位数，因此前两列之和不必严格等于总计列。

| 路径 | 精度 | 前向/秒 | 反向/秒 | 前后向/秒 | 每轮图调用 | Tensor峰值/字节 |
|---|---|---:|---:|---:|---:|---:|
| S42 flash，16/64 | FP32 | 1.413430 | 2.173911 | 3.586110 | 1,552 | 308,232 |
| 仅C热路径，16/64 | FP32 | 1.030672 | 1.069659 | 2.100331 | 1,552 | 308,232 |
| C热路径，手动32/128 | FP32 | 0.798696 | 0.856148 | 1.653764 | 472 | 319,496 |
| 最终自动分块 | FP32 | 0.836185 | 0.883980 | 1.718339 | 472 | 319,496 |
| S42 flash，16/64 | FP16 | 1.482404 | 2.232484 | 3.713418 | 1,864 | 275,464 |
| 最终自动分块 | FP16 | 0.903301 | 0.938070 | 1.845218 | 568 | 286,728 |
| 当前math对照 | FP32 | 0.365838 | 0.486757 | 0.855006 | 537 | 2,236,424 |
| 当前math对照 | FP16 | 0.388710 | 0.535867 | 0.925092 | 550 | 2,203,656 |

相对S42，最终FP32为2.09倍、FP16为2.01倍速度；图调用减少69.6%/69.5%。手动32/128与自动选择32/128相同，独立短测的约4%耗时差属于本轮观测波动，不能解释为手动接口具有额外优化。

更大的块使峰值比S42增加11,264字节（约11KiB），但相对math仍降低85.7%/87.0%。保存供反向使用的唯一storage（包括原始Q/K/V）仍分别为100,352/59,392字节；math为393,216/352,256字节。峰值口径只覆盖后端Tensor分配，不含驱动、编译图blob、CMX、分配缓存或CPU参考Tensor。

第1轮含建图与编译：原flash FP32/FP16为17.449/19.391秒，最终为8.381/9.669秒；新建会话由96/111降至43/53。当前math为6.016/6.891秒。这些是进程内第1次调用，不是包含Python导入或机器重启的启动时间，也不能混入预热吞吐。

当前flash的FP16存储仍转换为FP32分块执行与累积。它不是DPU原生FP16融合attention；这轮不会自动加速所有FP16模型。

## 数值与兼容性口径

| 最终路径 | 输出最大绝对误差 | dQ最大绝对误差 | dK最大绝对误差 | dV最大绝对误差 |
|---|---:|---:|---:|---:|
| FP32 对CPU math | 3.353e-8 | 9.313e-9 | 4.773e-9 | 4.098e-8 |
| FP16 对CPU math | 3.815e-6 | 3.815e-6 | 9.537e-7 | 3.052e-5 |

普通FP32逐元素比较仍用`rtol=5e-5, atol=5e-6`，FP16用`rtol=.01, atol=.002`，没有放宽旧验收容差。FP32额外对照独立CPU FP64参考：tiny V、tiny Q（乘`2^-125`）、tiny scale（`2^-132`）的前向与全部梯度，使用`rtol=2e-4, atol=32*2^-149`，并明确检查相应次正规数结果非零，防止普通绝对容差掩盖FTZ。

宿主编译实际C算术体，16组/288个packet对照独立FP64 SDPA及autograd，覆盖正/负/零scale、Bool与float mask、全屏蔽行、因果偏移及尾块。

### 使用与复现

默认math路径不变。选择实验flash：

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch_intel_npu

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
out.backward(grad)
```

也可显式指定`torch.intel_npu.flash_attention(q,k,v,query_block=16,key_block=64)`；省略的block参数由后端选择。旧的显式分块参数仍可用，数值、mask、first-order autograd和storage安全检查继续生效。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=''
.\run_torch_npu.ps1 -Script .\torch_npu_flash_optimization_demo.py -ScriptArguments @('--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_flash_optimization_demo.py -ScriptArguments @('--precision','float16')
.\run_torch_npu.ps1 -Script .\torch_npu_flash_optimization_demo.py -ScriptArguments @('--mode','math','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_flash_nested_demo.py -ScriptArguments @('--case','underflow','--precision','float32')
```

构建仍使用`npunlock-probe/research/build_flash_attention_kernels.py`；宿主C oracle、原验收脚本和本轮预热脚本均保留。脚本会输出原始JSON，后续可按空间策略在记录结论后删除。

## 尚未完成与后续方向

flash仍限FP16/FP32、head dimension 1..128、dropout=0、常量mask及一阶Q/K/V梯度；不支持可微mask。NestedTensor仍是strided推理实验支持，jagged、Nested训练未新增。没有实现CUDA运行时兼容、FlashAttention的CUDA内核或无条件后端替换。

最终自动分块的计算图仍占绝大部分宿主墙钟时间；不宜把当前瓶颈归结为PCIe IO或仅靠减少同步就能解决。后续若继续追求速度，可单独评估SHAVE的dot/PV调度与指数实现，或为FP16引入有明确数值边界的DPU分块融合路径。先保留当前math默认，不因本轮相对旧flash提速而声称已超过math。

## 验收与构建指纹

本轮最终23份真机记录全部通过，合计记录64,220次图提交：8份性能/阶段对照，15份功能或整网检查。所有记录在shutdown后均满足context正常、live buffers/分配缓存/scalar缓存/图会话/quarantine全部归零。开发期首版benchmark因循环变量保留两个Tensor触发清理断言；修复引用释放后重新取得正式基线，未出现设备context失败。阶段对照使用当时的源码/ELF，不把它们误算成最终实现的整网性能。

| 检查 | 进程记录 | 图提交 | 结果 |
|---|---:|---:|---|
| 同条件性能、分步优化与math对照 | 8 | 30,268 | 通过 |
| 小型flash前后向smoke | 1 | 221 | 通过 |
| 两精度mask/GQA/因果/128维与尾块梯度 | 2 | 6,711 | 通过 |
| 参数拒绝、空输入、RNG不推进、保存KV版本 | 2 | 142 | 通过 |
| 两精度NaN/Inf/全屏蔽行 | 2 | 520 | 通过 |
| FP32次正规数前向与全部梯度 | 1 | 315 | 通过 |
| Nested SDPA、MHA及CPU Nested autograd代理 | 2 | 928 | 通过 |
| Nested构造/copy/LayerNorm/默认Encoder推理 | 2 | 2,193 | 通过 |
| 原版两层Llama/GPT-2 FP32训练与恢复 | 2 | 8,348 | 通过 |
| 原版两层Llama FP16 cache/generate | 1 | 14,574 | 通过 |

Llama为5,712参数，GPT-2为5,504参数，均为原版Transformers类的小随机配置，沿用已有可选Transformers5.18.0。各做两步训练，全部梯度/参数对照CPU，CPU checkpoint恢复的下一步bitwise重放，GPT-2共享参数保持；Llama loss为3.446947→3.279040，GPT-2为3.542348→3.387163。FP16 Llama包含batch1和batch2左padding、3次cached decode，以及cache开/关的贪心generate全部分数/Token对照。它们验证迁移正确性，不认证预训练大模型质量、长期收敛或模型级提速。

离线194项通过：根目录157项（含新增自动分块边界）、npunlock API32项、源码发布工具5项。实际C宿主oracle另有16组/288个packet通过。旧64枚ELF/汇编和26份旧C源与S42一致；当前66枚ELF/汇编及27份C源共159项内容指纹通过，再核对3个原生组件，共162项。

| flash产物 | S42 ELF字节 | S43 ELF字节 | 当前ELF SHA-256 |
|---|---:|---:|---|
| forward | 9,072 | 11,832 | `5023ad58e6325b251dc753d78f1e6c62d6dd3fe8b1d045685be1c61dff58f0d3` |
| backward | 10,116 | 13,780 | `8d2b608c6c110adb8d181d77e869ffa7074ddf5025dc8628a53b4055354c7016` |

保留原代码窗口规则，宿主C++/PYD和session DLL未重建。当前关键源码/组件指纹：

```text
torch_npu_flash_attention.py
bda9f5bd9aa3273b0acf1defed5c5541d7e0d76ff2bb5c4488bdd77df7929630
torch_npu_flash_nested_demo.py
171ff4ffa2458813559fa9f47004144b4e88d2cc27ec2eb3e33908449882470b
test_torch_npu_flash_nested.py
13b0b7f006677b099319225a324ce3d1b5b9cd09c3a6091e410309357ff6167a
torch_npu_flash_optimization_demo.py
5683de681101746921d47df843b4f08a923af031dc074db4e6f7cf3d510c5af5
npunlock-probe/research/verify_flash_attention_host.py
a2facfd0f757f8460e46e99f0b83f6c6dc75a641bdd7eff7e97759537b5cfa49
npunlock-probe/research/build_flash_attention_kernels.py
56ede9f7a60a413db463d5d03920f480376e406bf43c7702b58c68d77c6c8c0b
npunlock/tests/fixtures/npu4000/resident_flash_attention.c
a348e531ee19c8e84f111dc45a1b6a672a978412e70757fa0b229038e923ff14
npunlock/tests/fixtures/npu4000/resident/manifest.json
3f695f0ad9e1a1f80604cadc9a15598838bf0ce75ffb4d4d513760f547ac4e36
npunlock-probe/torch-device/backend.cpp
9c3035281a9e1f6547c2d145a63cda495c0b2478ba2691e999d9548154970fba
npunlock-probe/torch-device/_torch_intel_npu.pyd
433757698bbc704a44da3ca9c3c3355d3fac49ab19b82d06f0d09790aa4c6ea0
npunlock-probe/session-runtime/npunlock_session.dll
1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90
```

## 清理与交付

删除本轮23份正式真机JSON、1份宿主oracle JSON、12份Movi阶段JSON/response及5份临时宿主C/DLL/obj/lib/exp，共41份、614,734字节；空host目录一并移除。清理前后1,949份保留源码、文档、内核及原生组件内容指纹一致，随后补充本节及空间台账。保留C源码、两组更新的ELF/汇编、manifest、Python实现/验收与构建/复现脚本，以及Markdown报告。

无新增依赖、权重、环境、图表或磁盘checkpoint/图缓存。全部测试进程已退出；公开source-release仍干净，未更新或推送GitHub。S42报告保留原始历史数据，本报告为S43当前性能和边界说明。

