# 完整行 attention 扩容与较长序列标准模型

日期：2026-10-20；本地开发版，未推送 GitHub，既有 source-release 快照未修改。
平台：Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu。

## 结论与范围

softmax/log_softmax、SDPA及默认native attention的总分数存储从65,536扩至1,048,576元素，采用完整行分块；单行/key宽度仍<=65,536。
复用既有softmax数值内核，新增统一行调度，并扩展SDPA因果元数据以保留切块前的query位置。保留完整分数矩阵，不是FlashAttention、在线softmax或不受限上下文支持。

新增算子专项六组通过，共5,519次图提交，含最大1,048,576元素softmax/log_softmax及其梯度、较大SDPA的Q/K/V/浮点mask梯度、相同mask Dropout、native FP16 masked-softmax和明确拒绝。
完整标准模型的FP32与FP16均通过两步训练、全部参数梯度/更新、逐位断点重放和推理对照；具体配置和回归见下。

本轮仅使用既有Movi工具重建sdpa-softmax的C/ELF/汇编和manifest。ELF为7,680字节，目标4000xx、-O2、-ffp-contract=off。
没有新增依赖、下载模型、重建宿主PYD/DLL或启用磁盘图缓存。

## 第一个实际阻塞：attention分数总量

修改前标准模型batch=4、heads=4、sequence=96，分数总量147,456。
它能上传并进入前向，但被旧guard拒绝：

```text
NPU SDPA attention scores exceed 65536 elements
```

权重与激活均在现有张量容量内；问题是旧softmax一次调用的总量限制。
新增softmax_row_tiles，每个packet至多65,536个FP32载体元素，按照完整行的边界切分。
普通softmax/log_softmax的前向、反向、native FP16 masked-softmax及SDPA使用同一调度；逐行exp、求和、补偿及half舍入公式不变。
单个归约行没有被拆分，因此不会因跨块合并改变该行的累加顺序。

### 因果位置

原SDPA内核以packet内的row%query_length计算query位置，在任意行切块后会错误地从零开始。
新packet的第二输入只携带元数据：b[0]=query_length、b[1]=start_row%query_length。内核使用(row+b[1])%query_length。
仍校验元数据为有界整数；旧uniform query_length元数据经取模仍表示偏移零。
一元素packet只有一个key，因果mask始终开放，不访问不存在的b[1]。

专项的(2,4,129,129)分数共有1,032行，分成508、508、16行，后两块的query偏移分别为121、113，故边界落在head内部。
这一配置实际对照CPU输出/梯度，能检出把新块错误地当成query第0行的实现。
另测query=65,537、key=1的非方形因果attention，包含65,536行主块及单元素尾块。

Dropout仍在完整概率张量上按原有扁平顺序申请RNG计数器，不随softmax行切块重新抽样。backward使用保存的mask，不推进RNG。

## 新增算子真机结果

| 专项 | NPU图提交 | 结果 |
|---|---:|---|
| rows FP16 | 1,026 | 通过 |
| rows FP32 | 304 | 通过 |
| SDPA FP16 | 2,090 | 通过 |
| SDPA FP32 | 2,020 | 通过 |
| native masked-softmax FP16 | 79 | 通过 |
| contracts FP32 | 0 | 四项元数据拒绝通过 |

rows的两种精度各覆盖(257,257)、(256,4096)、(513,129)上的softmax与log_softmax，六种组合均检查前向、梯度及out。
(256,4096)达到总存储1,048,576；(513,129)在切块前后分别放置全-inf、NaN和+inf行，保持原始特殊值语义。

SDPA两种精度各四种配置：较大广播因果、BOOL全屏蔽行、FP32浮点mask梯度、单key尾块。
前三种分数133,128元素且dropout=0.25；尾块配置65,537元素且dropout=0。输出、Q/K/V及mask梯度与独立同mask CPU参考对照，检查CPU RNG不受影响、NPU计数器准确推进及backward不再抽样。
native专项采用133,128元素均匀分数及不均匀mask，有限结果与CPU原生half API逐位一致，并覆盖masked NaN和跨块全屏蔽行。

所有数值专项计算区间均无host_download_copies/host_device_copies增加；CPU比较读取在区间之后，不参与NPU数值计算。
contracts在提交前拒绝超过1,048,576个分数或单行/key宽度65,537的输入，且不推进RNG。

## 第二个实际阻塞：静态编译图数量

扩大模型完成前向后，首次反向触及512项会话上限，在4,345次调用处被拒绝；context_failed=False，安全关停后资源归零。
这不是NPU设备丢失，也不是把数学错误用宽容差绕过。

较长序列使现有展开矩乘与静态DMA切片产生更多shape/offset专用图。
FP32在1,024项策略下完成，训练保留614图、eval后635图。FP16有额外精度转换图，在首次反向末尾的Embedding梯度处达到1,024项/25,924次调用；该次context_failed=False，峰值进程工作集645,079,040字节，关停正常。
最终将_MAX_SESSIONS有界调整为1,536，保留原有容量拒绝和保留至shutdown的策略。未恢复LRU或运行中图淘汰，因为此前本机的运行中淘汰曾可重复触发设备丢失。
这提高的是内存中的编译图会话数量，不是512MiB的磁盘图缓存设置。

上述六组新增算子验收在512项策略下完成，FP32完整模型使用1,024项；最终实现仅修改这一容量常量，行调度和数值内核完全相同。FP16完整模型和后续回归使用1,536项最终策略。

## 标准模型验收

固定两层标准nn.TransformerEncoder、width=32、heads=4、FFN=128、vocabulary=256，含Embedding、可学习位置、最终LayerNorm、padding池化、分类头和CrossEntropy，共36,835参数。
本轮针对sequence=96、batch=4验证，单次attention分数147,456；参数规模不是本轮的扩展轴。
GELU、Dropout=0.25，损失显式使用FP32。padding推理为enable_nested_tensor=False；未关闭全局MHA fastpath或修改PyTorch模块源码。

| 项目 | FP32 | FP16模型/FP32损失 |
|---|---:|---:|
| 状态 | 通过 | 通过 |
| 第1步loss | 1.055941820 | 1.055916309 |
| 第2步loss | 1.037641048 | 1.037853479 |
| 全部验收NPU图提交 | 22,209 | 81,885 |
| 训练后保留图数 | 614 | 1,091 |
| eval后保留图数 | 635 | 1,146 |
| 进程峰值工作集，字节 | 563,884,032 | 700,891,136 |
| 后端活跃缓冲区峰值，字节 | 9,773,545 | 8,624,097 |

两步各独立比较logits、损失、全部参数梯度和SGD动量更新；第1步保存显式CPU checkpoint，在新模型/新优化器中恢复后重放第2步，完整权重/动量/RNG和loss逐位一致，再对照同权重CPU推理。
保留FP32 rtol=3e-5/atol=3e-6、FP16 rtol=0.003/atol=0.0005；没有放宽阈值。
CPU同mask参考额外与本机ATen数学SDPA的输出及三个输入梯度交叉校验，测试入口不参与设备运行时。

计算区间每步56次各32字节的RNG元数据上传，另有首次标量缓存上传；10次标量状态下载共37字节，来自四块Embedding前后向、因果检测及标签检查。
没有下载激活到CPU完成计算。模型日志、CPU比较及checkpoint转移均在单独验收区间。

FP32首次前后向219.672秒、FP16为369.273秒，包含首次编译；同进程第2步分别为7.130和10.307秒。该观察不含优化器/比较/checkpoint，不是稳定吞吐或CPU/GPU对等性能benchmark。
普通FP16路径需额外转换/压紧图，每步约26,000次提交，FP32约6,600次；存储更小不代表这条SHAVE实现更快。
进程工作集包含Torch、编译器、图和CPU参考等，后端活跃缓冲区峰值则不含全部驱动与空闲缓存；二者都不能直接代替任务管理器的NPU内存读数。

## 回归、限制与后续方向

68项离线检查通过：核心/几何/低位/mask/RNG/SDPA/标准模块控制31项、npunlock Python API32项、source exporter5项。现有CI已包含相关测试文件，无新工作流或依赖。
新增离线控制检查完整行覆盖、容量/单行拒绝、跨head因果偏移、65,537行尾块，以及较长序列参考与官方数学路径的梯度。

原有SDPA数值回归FP16/FP32两组通过，分别700/583次图提交，共1,283次；覆盖原65,536分数边界、非方形因果、负/零缩放及预缩放避免溢出，逐项检查输出和梯度。
原contracts组通过，34次图提交；元数据错误无提交，保存输入的version guard继续生效，score越界用例更新为1025×1024以检验新的存储上限。
接受的六组新增算子、两个完整模型和两项原回归合计110,930次图提交；不含原65,536拒绝探测和两次会话数量限制失败。
全部验收正常shutdown，context_failed=False，live/cache/scalar/session/quarantine归零。普通活跃Tensor在关停前均已释放，仅余标量缓存。

后续仍有明确边界：

- 分数矩阵和梯度仍完整存储，最多1,048,576元素；单行/key<=65,536。通用Tensor的rank、非空、4MiB allocation和元素容量未变。
- NLL/CrossEntropy的N*(C+4)<=65,536限制未变，不能把log_softmax扩容当作分类损失同步扩容。
- 编译图会话有1,536项上限，不承诺任意模型；静态切片/转换专用图数量仍多，首次编译开销明显。下一步优先评估按元素数共享转换图、减少精度往返和小图提交，保持布局与half位语义后接入。
- 普通FP16矩乘仍用FP32 SHAVE算术再转换，没有接入通用DPU训练路径。不是FlashAttention、AMP或CUDA运行时兼容。
- 合成两步训练及重放不验证真实数据集质量、长期收敛、encoder-decoder或任意模型迁移。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_attention_rows_demo.py
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_scale.py -ScriptArguments '--profile','long','--batch','4','--precision','float32','--label','rows-after'
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_scale.py -ScriptArguments '--profile','long','--batch','4','--precision','float16','--label','rows-after'
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments '--case','sdpa'
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments '--case','contracts'
```

代码位于torch_intel_npu.py、torch_npu_geometry.py和resident_sdpa_softmax.c；构建复用build_sdpa_softmax_kernel.py。
新验收脚本torch_npu_attention_rows_demo.py，完整模型复用torch_npu_transformer_scale.py并新增long配置/batch选择和进程内存记录。
更新原SDPA score拒绝用例至新容量边界、三个CPU测试文件及双语README/兼容性/来源说明。

## 构建指纹

逐文件核对正式验收记录与最终源码；早期512项/FP32的1,024项策略仅有_MAX_SESSIONS常量这一行不同，将最终源码该行回溯替换后SHA-256与原记录一致，其他实现文件逐字节一致。
以下为最终1,536项策略指纹：

```text
torch_intel_npu.py
83947da59f531e3d8734b743d807e91796805ecd6455f77c4c32033eab4ec2c6
torch_npu_geometry.py
14eba9444304ad6aa994c4f6a1b7691f537819c315d4ae4b451a1a889e5869eb
torch_npu_attention_rows_demo.py
5b7e10ba22aecf4cb74df8c252b775a8ddbcb80930eb5de21f07e938db293600
torch_npu_transformer_scale.py
18722f10d9d7ab3d76245b67f8e5a4b585908b88634e40e3336e0289b48a085b
resident_sdpa_softmax.c
ef3c542ce4a1444ff640b144bcf8ec59e8a6f843357ebd3d6bb50a2c9ad777f0
resident/sdpa-softmax.elf
5c9a57243cc4f02de56016efa1d085b0020b4426a5666a2cc9eabc06f9a8704f
resident/manifest.json
a44acce1b3b2ccf80219de9dcefe08b902efa2cf29e935f2d6ff6fefaf193a11
```

## 清理与交付

删除本轮19份原始JSON/Movi响应，共147,771字节；清理前后750份保留源码/报告/内核/必要运行时文件SHA-256一致，随后补充本记录和存储台账。
源码导出选择746个文件，包含新增验收脚本、修改后的C/ELF/汇编/manifest及本报告；未创建新发布快照。
attention-rows目录最终仅保留REPORT.md；没有图表、模型下载或磁盘checkpoint。
