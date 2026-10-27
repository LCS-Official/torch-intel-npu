# 兼容性优先：公开数学、激活、RMSNorm与Decoder

日期：2026-10-20；Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu。
本地开发版，未推送GitHub，source-release快照未修改。本轮按用户要求优先兼容性，没有做效率调优或性能排名。

## 实际补齐的缺口

改动前在真实NPU上逐项调用前向和backward，七项全部被阻塞：exp/log/tanh/sigmoid/SiLU/rsqrt缺少公开入口，sqrt已有优化器前向但拒绝autograd。
现接通七项FP16/FP32公开前向、out及一阶梯度，标准torch/nn.RMSNorm，以及使用RMSNorm、SiLU、Tanh、Sigmoid的门控网络训练。
另扩展标准TransformerDecoder的兼容性验收；Decoder复用已有算子，本轮没有为它新增专用数值内核。

新增五个SHAVE内核，仅处理FP32计算；FP16通过已有设备内IEEE转换及DMA压紧路径运行。sqrt/rsqrt复用原内核；反向以现有设备算术组合完成，没有CPU数值fallback。
内核源为resident_unary_math.c，复用本项目arithmetic/classification/GELU helpers，并扩展exp的正数范围、log的负数/零/subnormal和激活特殊值处理。构建使用隔离Movi worker、4000xx、-Os、-ffp-contract=off；ELF/汇编附带manifest哈希校验。

| 新增ELF | 字节 |
|---|---:|
| exp-f32 | 3,288 |
| log-f32 | 3,752 |
| tanh-f32 | 6,168 |
| sigmoid-f32 | 4,052 |
| silu-f32 | 4,728 |

宿主PYD/DLL、普通矩乘路由、图会话上限1,536及低位量化实现均未修改；没有新增依赖、下载模型、启用磁盘图缓存或保存磁盘checkpoint。

## 数值与契约

- FP16/FP32、rank0–4、非空，继承1,048,576元素存储及65,536元素计算分块。只读密集转置、部分view、零stride广播先通过已有DMA整理。
- out须匹配shape/dtype、完整连续、无输入别名且无梯度；本轮没有接入sigmoid_/tanh_/SiLU(inplace=True)或二阶导。
- NaN/Inf类别、正负零、subnormal与边界行为对照CPU；不认证NaN payload、通用libm正确舍入或逐输入位精确的超越函数。
- RMSNorm使用FP32 opmath与已有可微NPU算子。None epsilon按本机Torch2.14的FP32计算dtype epsilon，而非half epsilon；支持0及非负有限FP32可表示eps，拒绝负数/NaN及不匹配的normalized_shape/weight。结果回到输入dtype；FP16输入+FP32权重的梯度单独保持FP32严格阈值。
- 输入和输出的saved-tensor version保护保留；错误out、非法RMSNorm及未支持inplace均在新增NPU提交前拒绝。

沿用模型精度比较阈值：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=0.003/atol=0.0005；FP32权重梯度单独采用FP32阈值。没有放宽既有模型的验收阈值。

### 定位并修复的问题

最初使用Abs一输入载体时，较大的tanh替换代码触及code-window保护。没有扩大窗口或绕过检查，改用项目已验证的Power双输入载体，第二个只读绑定指向同一输入且内核不读取它；保留原调度、ABI与代码边界校验。

rsqrt反向最初按FP32数学公式重排乘法，普通输入通过，但不能匹配PyTorch的result.pow(3)溢出边界。CPU检查确认：FP16输入约1e-4时，输出约100，立方先溢出half，再乘较小gradient，梯度仍为Inf。
最终显式保留立方结果回到输入dtype的舍入/溢出，以及-.5*gradient的dtype边界，再相乘；FP32也先算立方，保持其溢出顺序。没有将CPU的Inf改成“数学上更有限”的值。

版本保护专项最初只匹配一种报错文案；half输出为custom Function中的view，PyTorch更早报“view被modified inplace”。补充认可这一正确保护分支，并仍断言拒绝时没有NPU提交，没有为测试复制输出以隐藏版本问题。

## 算子与模型验收口径

values：七项算子各覆盖scalar、2D、3D、4D，前向/一阶梯度及非有限值、正负零。
layouts：七项均检查转置、部分view、broadcast读取、out及65,537元素尾块。
edges：FP16遍历全部65,536种编码；FP32用32,768种固定随机位模式，加13个exp溢出/下溢、rsqrt梯度溢出、log近1等边界，共32,781输入。前向和反向的NaN/Inf mask均与CPU一致。
norms：FP16有27组，FP32有18组，覆盖多个归一化轴、无/有weight、None/0/1e-5 eps，以及half输入+FP32 weight与其梯度。
contracts：每种精度21项提交前拒绝和7项saved-tensor版本保护。
计算区间检查没有host_download/device copy，上传只允许显式标量缓存；Decoder另允许Dropout每步12个32字节RNG元数据头。CPU数值比较和checkpoint传输在这些区间之后。

edges中仅对参考绝对值>=1e-3的有限前向统计最大相对误差：

| 算子 | FP16 | FP32 |
|---|---:|---:|
| exp | 0 | 1.141e-7 |
| log | 0 | 2.120e-7 |
| tanh | 7.093e-4 | 4.823e-7 |
| sigmoid | 0 | 1.297e-7 |
| SiLU | 7.284e-4 | 2.001e-7 |
| sqrt/rsqrt | 0 | 0 |

零误差只适用于该统计范围和样本，不推广为完整数域逐位一致；所有更小有限值仍参与原容差检查。

门控网络使用标准nn.RMSNorm、Linear、SiLU、Tanh、Sigmoid，355参数，batch2/length3/width8、hidden12。三步独立CPU/NPU训练对照损失、所有参数梯度及SGD动量更新；第2步checkpoint在新模型/优化器中重放第3步，权重/动量/loss逐位一致，随后eval对照。
Decoder为标准nn.TransformerDecoder的两层小配置，1,947参数，width8/heads2/FFN16、batch2、target3/memory4、GELU、Dropout0.25，含tgt因果mask与memory padding。两步同mask CPU/NPU参考比较全部参数梯度与更新，checkpoint在新模型中逐位重放第2步的权重/动量/loss/RNG；eval不消耗RNG。没有关闭MHA全局fastpath。

## 本轮真机结果

以下每项均通过CPU数值/梯度或明确拒绝检查，最终shutdown后live/cache/scalar/session/quarantine为0，context_failed=False。

| 专项 | FP16提交 | FP32提交 |
|---|---:|---:|
| values | 419 | 159 |
| layouts/out/尾块 | 388 | 178 |
| edges/编码与溢出 | 85 | 33 |
| RMSNorm | 918 | 468 |
| contracts/version | 77 | 28 |
| 门控网络训练/恢复/eval | 1,668 | 795 |
| 标准Decoder训练/恢复/eval | 9,159 | 4,832 |

14组正式验收合计19,207次NPU图提交，不含最初失败探测、载体窗口拒绝、版本文案检查失败和探索性重复检查。
门控网络FP16三步loss为0.319876522、0.297842324、0.267398387；FP32为0.319836348、0.297852576、0.267289966。
Decoder FP16两步loss为0.093982689、0.060718793；FP32为0.173803210、0.165812299。各精度的CPU随机输入随dtype不同，不把两列loss直接比较为精度优劣。

部分探索性进程曾短暂重叠，本轮未作计时结论；正式FP32门控训练另外独立重跑，随后优化器回归按顺序执行。数值及资源状态须分别通过，不能把进程返回0当作唯一验收标准。
71项离线检查通过：核心/几何/低位/mask/RNG/SDPA/标准模块/Transformer及新CPU控制34项、npunlock Python API32项、source exporter5项。新增CPU控制核对RMSNorm默认epsilon、门控全部参数梯度及Decoder自/交叉attention同mask参考的648个RNG元素消耗。

既有AdamW回归全部通过：primitives两组1,279次提交，contracts一组12次，optimizer十三组2,856次，共16组/4,147次。optimizer覆盖FP16/FP32默认betas、无decay、自定义betas、maximize、AMSGrad及组合配置，保留原有FP16默认epsilon边界。sqrt-autograd旧拒绝项改为本轮正向/反向验收，其余旧拒绝检查保留。
新增正式专项与既有AdamW回归合计23,354次NPU提交；此外独立重复FP32门控训练通过。没有做CPU/GPU速度对比或功耗结论。

## 后续兼容性边界

本轮不是通用CUDA完成声明；仍没有CUDA runtime/PTX、自定义CUDA扩展、NCCL、异步stream/event或torch.xpu合并。通用AMP、显式Generator、离散采样/部分随机API、动态索引、一般pow及更多公开算子仍有缺口。
Decoder仅认证上述稠密小配置，不代表任意encoder-decoder模型、HF模型、长上下文或完整LLM训练。短程门控训练也不是长期收敛认证。
低位完整模型质量保持原状态：W8/混合方案尚未全部通过既定质量门槛，INT4/NF4完整LLM未认证。本轮没有修改量化器、权重策略或降低其质量要求；接下来仍先推进兼容性。

## 复现与来源

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_unary_math_demo.py -ScriptArguments '--label','acceptance'
.\run_torch_npu.ps1 -Script .\torch_npu_adamw_demo.py -ScriptArguments '--case','primitives'
.\run_torch_npu.ps1 -Script .\torch_npu_adamw_demo.py -ScriptArguments '--case','contracts'
.\run_torch_npu.ps1 -Script .\torch_npu_adamw_demo.py -ScriptArguments '--case','optimizer'
```

主验收使用独立进程隔离各组。改动前probe需恢复旧后端才能复现；当前probe应全部通过。
RMSNorm及激活反向语义核对本机Torch的torch/_decomp/decompositions.py及CPU执行结果；数值辅助函数沿用本项目既有实现，详见REFERENCES.md。新三项CPU控制已接入既有离线CI列表，不需要NPU即可运行。

## 源码指纹

```text
torch_intel_npu.py
e012ed74d5ba3f2f00a1227a6cf8cd7a6983a827eaabe37922c60833ade640c7
torch_npu_autograd.py
bcfa907399a5680540d0e2730a96791af3b496268395252e46ca72348f06b680
torch_npu_unary_math_demo.py（最终版本保护文案断言版）
9c503162c7c646eea259092a521de8be60b8b87eccf4044abc6252da81cd8c49
test_torch_npu_unary_math.py
f6d87ffd5a5d17908b16d0baf834b50de77aa20582dbefcadc7515d9a1fb467b
build_unary_math_kernels.py
e3b909182d5c9a8cc61f2d4714dfd74c755909f362bab5e61740565986222f7d
```

旧/新resident源、所有ELF与汇编均重新核对manifest哈希一致；宿主PYD与session DLL保持此前验收版本。源码导出清单包含新C、五个ELF/汇编、构建脚本、验收/CPU测试与本报告。

## 清理与交付

删除本轮56份原始JSON及Movi响应，共325,231字节；包括探测/失败记录、正式数值与模型验收、既有优化器回归和五个新内核的编译中间响应，结论已归纳于此。
清理前后767份保留源码、报告、内核和必要运行时SHA-256一致；随后有意补充本节、存储台账及兼容性清单中的旧内部rsqrt表述。source exporter包含763份文件，覆盖新增源码和全部五个ELF/汇编。
新ELF合计21,988字节；保留构建代码与汇编，未生成图表、磁盘模型/缓存或新的发布快照。source-release的Git状态干净，未推送GitHub，未声称GitHub CI已运行。
