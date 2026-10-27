# S16：离散随机数、原生Generator与采样训练兼容性

日期：2026-10-20。Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、图编译器8.3、Python3.12.15、Torch2.14.1+cpu、NumPy2.5.3。
本地开发版，未推送GitHub；source-release快照未改。本轮优先补模型迁移接口，没有做性能排名、AMP或低位质量调优。

## 补齐了什么

改动前真机调用randint和multinomial分别在aten.random_.from和aten.multinomial.default被明确拒绝；不是CPU回退。
现支持标准INT64 randint/randint_like/random_、FP16/FP32 multinomial，以及真正的torch.Generator(device="intel_npu")。
原生Generator由C++ GeneratorImpl与PrivateUse1的getNewGenerator/getDefaultGenerator hook接入，没有替换torch.Generator或冒充CPU Generator。
默认Generator和显式Generator均支持seed、manual_seed、initial_seed、offset、state及clone_state，既有24字节NPURNG01 checkpoint格式继续兼容。

浮点rand/randn/uniform_/normal_及标量概率bernoulli_也能使用独立NPU Generator。CPU Generator明确拒绝；普通Dropout继续使用设备默认Generator。
新建Generator采用PyTorch Generator的默认种子；不同实例需要不同随机流时由调用方分别manual_seed或seed。

## 执行与数值

新增resident_discrete_rng.c，复用本项目arithmetic.c的IEEE辅助函数、32位分肢Philox4x32-10及已有图载体和DMA。主机仅维护种子/计数器、范围和形状元数据，不生成随机payload。
两个SHAVE ELF及汇编随源码提供，manifest校验源码/ELF/汇编哈希；Movi4000xx构建使用-Os、-ffp-contract=off、-fno-jump-tables，不绕过代码窗口/ABI限制。

| 新ELF | 字节 | 用途 |
|---|---:|---|
| integer-rng-packet | 2,540 | 完整64位整数与拒绝采样 |
| multinomial-packet | 7,880 | 权重校验、CDF采样及不放回scratch |

整数使用64位Philox输出、阈值2^64 mod span和局部重试counter，避免直接取模偏差。full-2^64跨度用span=0编码；上下界和结果通过原始整数位模式传输，绝不经过FP32数值转换。
每个结果预留一个全局counter块；拒绝重试使用该块的独立高64位子计数器，不额外推进全局offset。分块与调用分段不改变序列定义；不承诺与CPU/CUDA相同seed产生相同序列，也不是密码学随机源。

multinomial使用FP32归一化与累计和、24位均匀数，允许FP16输入经NPU转换。每次抽取按剩余原始权重的最大值重缩放；不放回时清零选中项，避免先统一归一化导致极小剩余权重下溢丢失。正权重检测/最大值选择按浮点位模式，保留最小subnormal边界。
该有限精度CDF不保证任意动态范围下的理论分布逐位精确；小于24位抽样分辨率的概率尤其不能据此宣称高质量保真。

所有输入行先在NPU校验，再预留随机流、产生结果和写out。每个校验包同步读取一个4字节错误标志；类别权重、随机结果和INT64索引不回到CPU参与计算。CPU数值参考、checkpoint与输出展示下载在独立检查区间之后。
坏概率（负数、NaN、Inf、全零）、后一个分包才出现的坏行、output别名、错误shape等均检查拒绝；拒绝不能消耗RNG或改写调用方out。

## 明确支持边界

- randint/random_仅INT64，rank0–4、非空、最多524,288元素；原地目标须完整连续存储。random_默认范围[0,2^63)，允许显式low配合high=None覆盖到INT64_MAX；randint仍遵守PyTorch的有符号SymInt参数范围。
- randint.out只接入已有精确shape的输出。标准工厂会调用resize_，故新增仅同shape/连续format的无操作入口；真正扩容、改shape、INT32/浮点random_仍拒绝。
- multinomial仅rank1/2 FP16/FP32、非负有限权重、每行正质量。支持replacement及不放回；不放回严格要求samples<=正权重类别数。CPU Torch2.14在某些零概率情形会额外返回零权重类别，本项目选择明确拒绝该情形，不能声称完全同语义。
- 输出最多524,288个INT64，扫描工作rows*categories*samples<=1,048,576；完整类别行、结果及scratch须容纳于65,536个32位word。按完整行分包，不拆类别；49,152类别实测通过，不代表所有形状都能同时达到各上限。
- multinomial输入可只读整理布局；out须已有精确shape、完整连续、INT64且无输入存储别名。输入requires_grad可采样，输出不求导，权重梯度保持None。
- 没有新增randperm、Tensor概率Bernoulli、top-k/top-p、HF generate、动态布尔索引、AMP/GradScaler或异步stream/event；native Generator不提供图捕获状态接口。

## 真机验收

| 专项 | FP16图提交 | FP32图提交 |
|---|---:|---:|
| INT64范围、独立Philox参考、65,537元素尾块、out/like | — | 85 |
| multinomial分布、极值、out与分包 | 100 | 84 |
| 七类显式Generator操作、clone/offset/state隔离 | 132 | 114 |
| 元数据、坏权重、后包错误、别名、状态耗尽拒绝 | 40 | 30 |
| 随机token训练、采样反馈与checkpoint恢复 | 522 | 303 |

九个正式独立进程共1,410次NPU图提交，全部通过。每个进程shutdown后live/cache/scalar/session/quarantine均为0，context_failed=False、closed=True。
统计不含初始缺失探测、resize_缺失的失败试跑及最终hook构建前的探索性重复检查。

整数对照覆盖跨符号范围、INT64_MIN附近、INT64_MAX附近、单值范围、大于2^63的跨度、全64位跨度，以及跨2^32计数器进位；与独立Python整数Philox/拒绝采样参考完全一致。
multinomial在权重[1,3,6]下抽10,000次得到[979,3061,5960]，两种精度一致；每类频率偏差低于0.025。另检查完整16类别不放回排列、零权重/最小subnormal/最大值混合、49,152类别one-hot、4×20,000类别按行分包、requires_grad输入及精确状态重放。这是接口/有限统计检查，不是完整随机质量认证。

训练模型为Embedding(17,4)→mean→Linear(4,17)，153参数；NPU生成2×3 token和2个硬标签，显式FP32 CrossEntropy、一阶反向及SGD(momentum=.5,lr=.02,foreach=False,fused=False)。随后softmax→multinomial结果直接反馈Embedding推理。
两步独立CPU/NPU损失、全部参数梯度、更新及采样反馈推理均在原容差内：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005。
FP32两步loss为3.086181641、2.758208513；FP16为3.086323261、2.758356571。数据/标签每步变化，不将loss下降当作收敛证明。
在第1步后将CPU权重/优化器及Generator状态存入内存BytesIO，新模型/新优化器恢复后逐位重放第2步loss、采样索引、参数、动量及Generator状态；默认NPU流和CPU RNG保持独立。
训练检查允许Embedding前后向及NLL合计3个校验标志下载；采样反馈允许multinomial及Embedding合计2个标志下载，无host payload计算回退。

74项离线检查通过：核心/几何/低位/mask/RNG/SDPA/标准模块/Transformer/数学37项、npunlock Python API32项、source exporter5项。新增完整整数范围/拒绝阈值、packet容量边界及Philox零key/counter已知向量三项检查。

原RNG/Dropout十组回归共942次提交全部通过，覆盖FP16/FP32均匀/正态/Dropout、种子/24字节状态、fork_rng、调用分段、counter进位及相同mask训练/checkpoint。原始状态bytes可直接被新原生Generator读取；没有更换既有浮点RNG算法。

首轮设备Quick到旧unsupported_integer_rng拒绝项时失败：默认INT64 randint已被本轮支持，旧用例仍要求报错。将该项改为仍未支持的FP32整数随机入口，并增加拒绝前后RNG状态一致断言；没有删除其他拒绝检查、放宽数值容差或改回已实现功能。正式设备回归结果另见下文。

重跑设备Quick全部33组通过，共4,162次NPU提交；43项拒绝未新增图提交且RNG状态不变，最终资源全部归零、context_failed=False。合计本轮正式硬件验收52组、6,514次图提交。旧回归仅修改过时的拒绝项，计算内核及数值阈值未为其调整；首次失败试跑不计入正式合计。

## 复现与构建

旧PYD与新Python派发不兼容，需要先运行build_torch_npu_device.ps1重编；build_npu.ps1完整构建已包含此步骤。没有新增pip依赖、下载模型或安装监控驱动。普通matmul/DPU路由及低位量化未改。

```powershell
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_discrete_demo.py
.\run_torch_npu.ps1 -Script .\torch_npu_rng_demo.py
.\run_torch_npu.ps1 -Device -Quick
```

专项可传-ScriptArguments @('--case','generators','--precision','float32')。入口按独立进程串行执行；原始JSON/编译响应/日志验收后清理，保留本报告和源代码/必要内核。
修改SHAVE源码另需外部MoviTools，构建入口npunlock-probe/research/build_discrete_rng_kernels.py，完整resident构建会调用它。

支持范围内的标准调用方式：

```python
import torch
import torch_intel_npu
g = torch.Generator(device="intel_npu").manual_seed(42)
tokens = torch.randint(100, (2, 8), device="intel_npu", generator=g)
logits = torch.randn(2, 100, device="intel_npu", generator=g)
next_token = torch.multinomial(logits.softmax(-1), 1, generator=g)
saved_rng = g.get_state()  # CPU ByteTensor，可与显式CPU模型checkpoint一起保存
```

这轮补齐了随机输入与采样的一段模型迁移链路。通用CUDA运行时、任意模型即改设备名运行和低位完整模型质量均仍未完成；W8混合Up/Down FP16的历史质量仍是129/142位置，INT4/NF4完整语言模型尚未认证。本轮不能用来宣布通用CUDA或低位兼容性完成。

## 本轮代码指纹（SHA-256）

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | 0b4c25c02a87424ea08d9413e673723e1fb14e95fa46de0857a9048886409bf3 |
| torch_npu_autograd.py | b31d5a05cf0379e0ba22dfbc6a4eec91a6c770dc228ecd51bc121c5b1a56c52e |
| torch_npu_discrete_demo.py | 30263ef256d5a63b24cfdea75f9aaebe01b5903481b80563c9a3af0677aca6bd |
| torch_npu_device_demo.py | a552f8d08f0df3ae6398a45dc2b66a91a4a9a41464dc19f5255d7fc498de0378 |
| npunlock/python/npunlock/npu4000.py | 1cfe036ad211d9cfe6e095087b7a65311037d14e5cf96b08190bd49f138ee087 |
| npunlock/tests/fixtures/npu4000/resident/manifest.json | 1cd769a7521ce48192bbc68bab4679d0d5e2bf1340470da7d5623071be054415 |
| npunlock-probe/torch-device/backend.cpp | 2ac3cd00d410be6c7976c176dd2dc87bb5a3bdb843ec7edf623bdc16a83b7f1a |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | 29fbf23ab1b17fb8912ddea5f19aeea738face355780a66446b6c665e5362430 |
| integer-rng-packet.elf | 9b9cd46b5154c0c91a5570c5c17d0ab3c9f19795c529112c35cb360967bbf7cf |
| multinomial-packet.elf | f07858bcc3dca7379f1fdf7e58715d534d0ffc7ce44cd71cbebeb2064a43155a |

原session DLL未重编，仍为0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9。二进制哈希用于本轮运行 provenance；其他机器从源码重建不要求得到完全相同字节。

## 清理记录

验收后删除本轮42份原始JSON/日志、Movi worker响应及可重建backend.obj/导出表，共6,113,158字节（5.830MiB）。逐项校验实际绝对路径位于工作区的指定结果目录或两个明确构建输出，没有递归删除。
清理前后777份保留源码、报告、ELF/汇编、manifest与必要运行时文件SHA-256完全一致，随后有意补充本节与存储台账；旧import lib保留。source exporter已检查771份源码/报告/内核文件的路径与内容，含本轮全部新源码和两个ELF/汇编；没有生成额外快照、修改source-release、推送GitHub或启用磁盘图缓存。
