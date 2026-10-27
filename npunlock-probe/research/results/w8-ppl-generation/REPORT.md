# 低位完整模型质量：新文本PPL与自由生成

2026-10-15，Windows / Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，
驱动1005540、图编译器8.3、Torch 2.14.1+cpu；本地SmolLM2-135M-Instruct。

本轮把评价从固定生成轨迹的logits误差扩展到真实文本的目标token损失和各精度自主生成。
真实NPU的PPL为：FP16 27.93748、全W8 28.03998、原混合50% 27.93820、逐层37.5% 27.92814。
当前小样本中两个混合配置未显示明显PPL退化，8条自由生成中它们彼此完全相同；这不足以证明通用质量提高或升级默认。
评测开始前冻结语料、滑窗、生成规则和两个既有候选，没有根据新结果调整精度、权重、scale或候选范围。
生产接口、量化器、SHAVE/C++、DLL/PYD和默认配置没有改动，未更新GitHub。

## 固定数据与方法

数据为[Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)
的 `wikitext-2-raw-v1/test`，版本 `b08601e04326c79dfdd32d625aee71d232d685c3`。
通过官方[rows接口](https://huggingface.co/docs/dataset-viewer/rows)读取8个固定范围，各32行：
起始行为0、500、1000、1500、2000、2500、3000、3500。检查每次HTTP响应的 `x-revision`、行号、未截断标志及文本指纹。
这是本项目新采用的评测数据，与校准提示不相同；不声称它未被模型预训练见过。
原数据许可见其数据卡；本项目不打包WikiText原文或token IDs。

每份范围的text按原行顺序用两个换行拼接，tokenizer不添加特殊token，取前257个token。
每段从目标位置1到256，共256个真实目标；8段合计 **2,048个目标token**。
文本不来自本模型或本助手生成的续写。

### 短上下文滑窗与损失

- NPU输入固定64个token，步长32；每段窗口起点为0/32/64/96/128/160/192，共7窗，总56窗。
- 用64个输入位置的logits预测其后一个token：65-token片段中输入0..63，目标1..64。
  第一个窗口计64个目标，后续窗口只计最后32个，保证目标1..256恰好各计一次。
  最后一个目标用于评分，不消费进KV，因此未扩展NPU的64位置容量。
- 每窗重新开始相对位置；后续计分token具有33..64个输入token的上下文，首窗则为1..64。
- 用FP64 log-softmax累加真实目标的负对数似然，先求全部目标的总损失，再除以2,048得到NLL；PPL为 `exp(NLL)`。
  不平均窗口均值，也不平均各段PPL。
- CPU与NPU使用完全相同的目标、窗口、tokenizer、半精度checkpoint及两端计算。
  NPU的全部30层decoder执行在设备，embedding、final norm、lm_head和损失显式在CPU。

固定上下文与步长会影响PPL，参见[Transformers的说明](https://huggingface.co/docs/transformers/perplexity)。
本项目自行实现计分，不复制其示例；另与Transformers内置shifted-label loss交叉核对。
本轮是 **64输入token、2,048目标token的小样本PPL**，不是完整WikiText-2测试集、长上下文成绩或通用模型排名。
没有预设或事后新增“PPL通过即可发布”的认证门槛，也未更改旧logits质量门槛。

### 冻结的四种配置

| 名称 | decoder投影权重策略 |
|---|---|
| FP16 | 全部7类投影使用原FP16 |
| W8 | 全210个投影使用同一份既有逐行GPTQ校准W8 |
| 原混合50% | 全30层Up/Down保留FP16，其他5类W8 |
| 逐层37.5% | 复用上一轮已选中的45个Up/Down保护项，其他165个投影W8 |

比例指decoder投影权重元素中保留FP16的比例，不是进程内存节省比例。
45项策略逐项核对[逐层研究报告](../w8-layer-sensitivity/REPORT.md)中的保留范围，没有重新运行选择。
校准重新生成一份供全部配置共享，仍只用旧32条轨迹的2,048行，不接触新WikiText文本或新生成提示。
逐层配置仍是研究脚本通过既有单层API组合，不是新增公开stack参数。

## CPU控制组

| 配置 | NLL/token | 短上下文PPL | 自由生成与CPU FP16完全相同/8 |
|---|---:|---:|---:|
| FP16 | 3.329926874 | 27.936298767 | 8 |
| W8 | 3.333585489 | 28.038694120 | 6 |
| 原混合50% | 3.329820428 | 27.933325220 | 6 |
| 逐层37.5% | 3.329525094 | 27.925076776 | 6 |

CPU低位数值参考仍按整数code与FP16 scale在FP64中相乘；其余部分使用官方FP16网络。
FP16的56个窗口与未替换Linear之前的官方模型输出逐位相同。
两个混合候选的小样本PPL略低于FP16，幅度很小，不能解释为模型能力或量化质量显著提高。

## 自由生成规则与解释

8条全新聊天提示，固定system为 `You are a helpful assistant.`，使用本地chat template。
输入长度依次为27/28/26/29/28/30/27/32；逐token cached处理提示，然后greedy argmax、最多32个新token，遇tokenizer EOS停止。
每个配置沿自己生成的token继续推理，不强制喂CPU参考轨迹，不加重采样、温度、惩罚或补救规则。
各配置最多消费63个token；截断由32新token上限造成，不能把截断文本当成完整回答。
CPU控制也使用同一逐token cached调度，不宣称与所有Transformers.generate默认配置逐位相同。

FP16真实NPU的PPL为27.937479644，与CPU相差约0.00423%（NLL差0.000042269）。
但它的自由生成只有7/8条与CPU FP16完整token序列相同：divide在第10个生成token处分歧（0起始index9）。
小幅数值差异在greedy决策后可能改变后续整条路径；PPL接近不保证自由生成逐位相同。

CPU FP16本身已有明显能力限制。例如它把最接近太阳的行星回答为Venus，
对“只返回blue”的提示生成了“The word ... is not in the sentence.”，而非按格式回答。
数学提示出现不正确的解题方向，翻译提示也未给出预期的常用问候。
这些问题在CPU FP16中已存在，不能统归因于低位量化或NPU执行。
本轮不把与CPU一致率当作任务正确率，不建立8题总体准确率或自动语义评分。

## 真实NPU汇总与验收

| 配置 | NLL/token | 短上下文PPL | PPL相对NPU FP16变化 | NLL相对同配置CPU差值 |
|---|---:|---:|---:|---:|
| FP16 | 3.329969144 | 27.937479644 | 基线 | +0.000042269 |
| W8 | 3.333631410 | 28.039981705 | +0.366898% | +0.000045921 |
| 原混合50% | 3.329995022 | 27.938202635 | +0.002588% | +0.000174594 |
| 逐层37.5% | 3.329634763 | 27.928139441 | -0.033433% | +0.000109668 |

每段256个目标，下面为各段NLL相对NPU FP16的变化（nats/token）：

| 起始行 | 全W8 | 原混合50% | 逐层37.5% |
|---:|---:|---:|---:|
| 0 | +0.010267163 | +0.002400941 | +0.002026267 |
| 500 | +0.004887261 | -0.001922721 | -0.001812954 |
| 1000 | +0.002348136 | -0.001022467 | -0.000584654 |
| 1500 | -0.000089259 | -0.003566473 | -0.005061725 |
| 2000 | +0.004558909 | +0.003176654 | +0.002582492 |
| 2500 | +0.002023164 | +0.000275304 | +0.000841015 |
| 3000 | +0.006088256 | +0.000884658 | -0.001887083 |
| 3500 | -0.000785503 | -0.000018868 | +0.001221593 |

存在段落间正负抵消，均值的微小优势不能解读为所有文本上都更好。
8段固定前缀不是随机抽样置信区间，2,048个相关token也不能当成2,048个独立样本来宣称统计显著。
逐层配置虽然在旧原始L2门槛上更差，这里PPL却略好，说明两类指标不等价；本轮仍保留旧结果和门槛。

### 生成一致率与分歧

下表比较完整生成token序列（含EOS），不是语义准确率：

| NPU配置 | 与CPU FP16相同/8 | 与NPU FP16相同/8 | 与同配置CPU相同/8 |
|---|---:|---:|---:|
| FP16 | 7 | 8 | 7 |
| W8 | 6 | 5 | 8 |
| 原混合50% | 7 | 6 | 7 |
| 逐层37.5% | 7 | 6 | 7 |

相对NPU FP16，全W8在divide/9、shadow/19、clock/13首次分歧；两个混合配置均在divide/9、clock/13首次分歧（生成位置从0开始）。
两个混合配置彼此8/8完整生成相同。全W8与同精度CPU则8/8相同，不能把与FP16的差异全视为NPU实现错误。
所有配置生成长度都是32/10/12/32/32/32/10/32，planet、format、spanish因EOS停止，其他5条达到32新token上限。

额外以普通CPU `model.generate(do_sample=False,max_new_tokens=32)` 核对divide/planet/format，
三条均与逐token CPU FP16结果完全相同；模型原有答错/格式问题不只是本轮提示调度造成的。
不对截断的数学、代码或故事回答计算“完成率”。本报告末尾保留全部提示与生成文本，供人工检查。

### 检查与执行证据

- 自检覆盖唯一目标计数、token加权、uniform logits PPL、EOS停止、不消费最后生成token及64位置容量。
- 用官方CPU模型65-token片段的内置shifted-label loss交叉核对首窗64个目标，差为-0.000016060 nats/token，
  在预设0.0003容差内。这个65-token计算只用于CPU计分检查，不代表NPU支持65个输入位置。
- CPU因果性检查：改变第64个输入token后，前63个位置的logits逐位不变。
- 每种NPU配置首个64-token窗口逐层与CPU局部参考比较，共120个局部检查。
  全部最大相对L2为0.067936%、最大scaled error为0.481984%，均通过原0.3%/1%局部门槛。
- 每种配置90个实际图证据（30层 × forward64/prefill1/decode1），共360份。
  三种低位配置的所有层保留INT8矩阵任务，检查为FP16激活 `amode=0`、INT8权重 `wmode=2`；FP16控制没有该类整数任务。
- 每种配置8次热重复decode，输出/KV/位置91份数组逐位一致；90份种子数组未变，区间内不新增会话或显式payload复制。
  哈希检查的下载在计数区间外；计数不涵盖驱动内部DMA。
- 每种配置14,370次NPU调用：56窗×30层 + 411个生成阶段消费token×30层 + 8次热重复×30层 + 120次gamma准备。
  总计 **57,480次**；全部退出时context健康，live/cached/quarantined buffer字节为0。
- 审计复核所有来源绑定、56窗顺序、2,048个目标、生成停止/长度、逐层保护范围和上述设备证据。
  未修改生产代码；原50项核心/可选模型回归不重复运行，本轮不重新认证其他算子或训练。

没有测速度、能耗或全进程峰值内存；这些质量对照不能充当性能结论。

验收后清理11份JSON/NPZ，共107,044,464字节（102.09MiB），包括临时WikiText语料、token IDs、校准权重和原始测量记录。
保留3份新研究脚本、来源条目、下方生成文本和本报告。没有安装依赖、下载新模型、保存原始logits/激活/图blob/图片或启用磁盘图缓存。

### 指纹

| 对象 | SHA256 |
|---|---|
| `w8_ppl_data.py` | `309b3505a9171308286ceb8925b4448f11f220806de6f60ca3812615c5ac7e9e` |
| `w8_ppl_generation.py` | `0394416160cd7d646d7c79952a8567f748d90051ceee2d1d5d851197287c9f6a` |
| `w8_ppl_audit.py` | `6ff649f1463dc8a0a3a96e5d96acfd16545b1ab63211224107ecfca211c649f8` |
| `torch_npu_smollm.py`，未改 | `d95e571675972d9605fb431e76512671a0619619fbf8dc28e98d5e98757c6e8c` |
| `torch_npu_lowbit.py`，未改 | `e9f5b12b09275f73933ef1ab325346776c75a905d48486e3aad5256d08a46f21` |
| `torch_intel_npu.py`，未改 | `bd147eb2adbf49d092084ae4f1de767b395424016e9f8862eb850ab571b55cce` |
| C++ `backend.cpp`，未改 | `d6acf5d765b1024ef783e4370f30064e7c67aa4a26f7a10e9eba8c0864fe6d09` |
| session DLL，未改 | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |
| Torch PYD，未改 | `3325f761ea55485271e26ecc3667b820f428d31f665e4c31fb916e8c30aefc8b` |
| 模型权重，未改 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 校准权重，已清理 | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| freeze.json，已清理 | `c26537c7eeb8461dc54fb75453e474ee40b74ae55d34f2c170e08e5e9956bd32` |
| selection.json，已清理 | `1c02fd45e1c22af14374c5b93c4b784dba1fa123554ba9270c6cc74975e1c300` |

## 复现与保留策略

使用已有模型环境；数据下载仅用Python标准库，不需要datasets/pyarrow/pandas。
先重新生成校准权重，再下载固定小样本、冻结输入；其后NPU命令串行执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = $null
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python;$PWD\npunlock-probe\research"
# 旧prepare实现重定向到本轮目录，保持拟合方式不变。
.\.conda\npunlock\python.exe -B -c "import w8_ppl_generation as e; e.layers.mixed.OUT.mkdir(parents=True, exist_ok=True); e.torch.set_grad_enabled(False); e.torch.set_num_threads(8); e.layers.mixed.prepare()"
.\.conda\npunlock\python.exe -B .\npunlock-probe\research\w8_ppl_data.py
.\.conda\npunlock\python.exe -B .\npunlock-probe\research\w8_ppl_generation.py --case freeze
.\.conda\npunlock\python.exe -B .\npunlock-probe\research\w8_ppl_generation.py --case cpu
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_ppl_generation.py -ScriptArguments '--case','npu-fp16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_ppl_generation.py -ScriptArguments '--case','npu-w8'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_ppl_generation.py -ScriptArguments '--case','npu-mixed'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_ppl_generation.py -ScriptArguments '--case','npu-selected'
.\.conda\npunlock\python.exe -B .\npunlock-probe\research\w8_ppl_audit.py
```

本轮准备校准时使用 `torch.no_grad()`；复现同样应关闭梯度，避免无谓占用内存。
评测所需原始语料、token IDs、权重和JSON按用户策略在报告验收后清理；源码与文字报告保留。
下载器若收到其他数据版本会拒绝，不会悄悄换用新数据。旧版本服务不可用时需显式恢复同内容，不能混报历史成绩。

## Fixed text fingerprints

| Row offset | SHA256 of the 32 text rows |
|---:|---|
| 0 | `f71c8d864dd0058f32d56a720d44e320c77d97dd2dcc2d100a3f656beb826752` |
| 500 | `3b1c301d2dd4b06b4424aa4b18c9a4cd5c6a06c3008e30c2df0233721e2c20fb` |
| 1000 | `d9919dcec4aad831a6c1772cfec7b5fe2e21bea9d64a45d57a528cc6ae03b5fd` |
| 1500 | `55adc86cb6be873932bc0d5e2bb2db3103f39433c5947b56b28424c166795705` |
| 2000 | `bd4c18ce5d7e9a6177f9bc7c22e5c81ae2d3e683306f726ad13dcebc948b359b` |
| 2500 | `c65bed5746bf13f0308d0868052b15d154d1a659a40a01b48ac3f26571b9fae9` |
| 3000 | `4403ba3395627b87c228c47aa9920c1cef97a1f21c4be8e90176962750294b0c` |
| 3500 | `1ec54ee2bae6c66c0654e8e9f3a26ea36eb3394bab0d0189b1ade075c903d3f2` |

## Generation samples

Full visible completions are grouped only when their complete token sequences match. EOS is included in sequence comparisons and omitted from decoded text. Newlines below use HTML line breaks; model code is displayed as text. These are model outputs, not instructions or endorsed answers.

### divide

Prompt: What is twelve divided by three?

- **CPU fp16, CPU w8, CPU mixed, CPU selected, NPU w8, NPU mixed, NPU selected** (32 new tokens, max_new_tokens): To divide twelve by three, we need to first find the least common multiple (LCM) of 12 and 3. The LCM of 
- **NPU fp16** (32 new tokens, max_new_tokens): To divide twelve by three, we need to find a common denominator. The least common multiple of 3 and 12 is 18. So,

### planet

Prompt: Name the planet closest to the Sun.

- **CPU fp16, CPU w8, CPU mixed, CPU selected, NPU fp16, NPU w8, NPU mixed, NPU selected** (10 new tokens, eos): The planet closest to the Sun is Venus.

### format

Prompt: Return only the word blue.

- **CPU fp16, CPU w8, CPU selected, NPU fp16, NPU w8, NPU mixed, NPU selected** (12 new tokens, eos): The word &quot;blue&quot; is not in the sentence.
- **CPU mixed** (12 new tokens, eos): The word &quot;blue&quot; is not in the list.

### square

Prompt: Write a Python function that squares a number.

- **CPU fp16, CPU w8, CPU mixed, CPU selected, NPU fp16, NPU w8, NPU mixed, NPU selected** (32 new tokens, max_new_tokens): &#96;&#96;&#96;python<br>def square(n):<br>    return n ** 2<br>&#96;&#96;&#96;<br><br>This function uses the built-in &#96;**&#96; operator to square the

### paperclip

Prompt: List three uses of a paperclip.

- **CPU fp16, CPU w8, CPU mixed, CPU selected, NPU fp16, NPU w8, NPU mixed, NPU selected** (32 new tokens, max_new_tokens): Here are three uses of a paperclip:<br><br>1. To hold things together: A paperclip can be used to hold items together, such as a book

### shadow

Prompt: Explain why a shadow changes length during a day.

- **CPU fp16, CPU mixed, NPU fp16, NPU mixed, NPU selected** (32 new tokens, max_new_tokens): A shadow changes length during a day because the Earth&#x27;s rotation is not uniform, and the Earth rotates on its axis. The Earth rotates at a rate of approximately
- **CPU w8, NPU w8** (32 new tokens, max_new_tokens): A shadow changes length during a day because the Earth&#x27;s rotation is not uniform, and the Earth&#x27;s rotation is not uniform. The Earth rotates at a rate of
- **CPU selected** (32 new tokens, max_new_tokens): A shadow changes length during a day because the Earth&#x27;s rotation is not uniform. The Earth rotates on its axis, but it does not rotate at the same rate

### spanish

Prompt: Translate good morning into Spanish.

- **CPU fp16, CPU w8, CPU mixed, CPU selected, NPU fp16, NPU w8, NPU mixed, NPU selected** (10 new tokens, eos): Estoy muy buena hora.

### clock

Prompt: Write two sentences about a clockmaker who loses a key.

- **CPU fp16, NPU fp16** (32 new tokens, max_new_tokens): One day, a clockmaker named Thomas stumbled upon a hidden keyhole in his workshop. As he carefully turned the keyhole, he noticed that it was slightly
- **CPU w8, CPU mixed, CPU selected, NPU w8, NPU mixed, NPU selected** (32 new tokens, max_new_tokens): One day, a clockmaker named Thomas stumbled upon a hidden key hidden in a dusty old shop. As he carefully opened the door, he found a small,
