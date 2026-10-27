# FP16/FP32转换图按元素数复用

日期：2026-10-20；本地开发版，未推送GitHub，source-release快照未修改。
平台：Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu。

## 实现与范围

torch_intel_npu._cast原先以输入dtype及完整shape缓存转换图，即使元素数相同，rank或维度不同也会重复编译。
现改为：先执行原有dtype、连续布局、完整allocation、offset-zero及out检查，再将绑定视为(numel,)并共享转换图，最后恢复调用方shape。FP16→FP32提供out时仍返回原out对象。

大于65,536元素时仍由原有_map_tiles分块；FP32→FP16仍以FP32 carrier承载binary16编码，并由halfcode_slice在NPU内通过DMA压紧。
arithmetic.c、转换ELF、编译器ABI、IEEE舍入/非规格化数路径没有改变。没有新增依赖、宿主构建、模型下载或磁盘图缓存。
图缓存上限仍为1,536项，保留至shutdown，未引入运行中淘汰。

这是PyTorch模型在NPU上的首次准备与缓存压力优化，不新增CUDA运行时、PTX、异步stream、torch.xpu或通用DPU训练支持。每次转换的图提交数没有减少。

## 本轮真机专项前后比较

独立新进程先运行旧实现，再运行新实现。同一组数据生成、shape顺序、out复制、9次热调用及布局/边界检查；修改后脚本另外增加“每组只新建3张转换图”的断言，并简化计数代码，数值工作负载未改变。
图磁盘缓存默认关闭。冷调用只测source.to(target_dtype)，包括其首次编译，后续out复制与CPU对照不计入该计时。

两组形状：

- 96元素：(96,)、(8,12)、(2,3,16)、(2,2,4,6)。
- 1元素：scalar、(1,)、(1,1)、(1,1,1)、(1,1,1,1)。

两种转换方向均覆盖；每个shape换用不同payload，防止图复用错误地保留上次绑定。

| 指标 | 旧实现 | 新实现 |
|---|---:|---:|
| 96元素组新增转换/压紧图 | 9 | 3 |
| scalar及1元素组新增转换/压紧图 | 9 | 3 |
| 两组冷调用耗时之和，秒 | 6.187282 | 2.185899 |
| 完整专项保留图数，含布局/切片等 | 79 | 65 |
| 完整专项NPU提交 | 427 | 427 |
| 活跃设备缓冲区峰值，字节 | 10,878,990 | 10,878,990 |
| 进程峰值工作集，字节 | 381,009,920 | 372,301,824 |

每个元素数只需一张f16→f32、一张f32→half编码图及一张DMA压紧图。两组图数18→6，冷调用合计减少约64.7%。
这是各一次新进程的观察，未做交替多轮性能统计。9次热调用中位数的跨shape均值：96元素336.7→365.2微秒、1元素321.5→321.9微秒，没有证据表明热转换更快。
进程工作集含Torch、图编译器等，不等于NPU专属内存；活跃设备缓冲峰值不变，符合未减少计算载荷的预期。

专项检查全部通过：

- 所有65,536种half编码的拓宽与再压缩，含正负零、非规格化数、Inf及NaN。NaN比较类别，有限输出与CPU逐位一致，不认证NaN payload。
- 65,536、65,537和1,048,576元素的转换，包含一个元素的尾块和多整块。
- scalar至rank4的输出shape、显式out对象身份；两个方向的公开转置/部分view读取及密集转置目的copy。
- 私有原生绑定继续拒绝非连续/部分allocation、错误out shape/dtype，共8项，无提交。
- 转换/布局计算区间无主机upload/download/device/pack复制增加；对照数据读取在区间之后。全专项显式复制计数前后相同。

## 完整标准模型回归

复用S13同一long配置：两层标准TransformerEncoder，36,835参数，width32/heads4/FFN128/vocabulary256，sequence96、batch4，attention分数147,456；GELU、Dropout=0.25，FP16模型与显式FP32分类损失。
模型、验收脚本、seed227、全部参数比较容差均未改变；harness SHA-256与S13报告相同。

| 指标 | S13旧实现 | 本轮新实现 |
|---|---:|---:|
| 第1步loss | 1.0559163093566895 | 1.0559163093566895 |
| 第2步loss | 1.037853479385376 | 1.037853479385376 |
| 训练保留图数 | 1,091 | 1,072 |
| eval后保留图数 | 1,146 | 1,124 |
| 全验收NPU提交 | 81,885 | 81,885 |
| 首次前向/反向，含编译，秒 | 369.273220 | 363.037925 |
| 同进程第2步前向/反向，秒 | 10.307484 | 9.968283 |
| 进程峰值工作集，字节 | 700,891,136 | 687,587,328 |
| 活跃设备缓冲区峰值，字节 | 8,624,097 | 8,624,097 |

模型中图数降幅比同元素数专项小：训练少19张（约1.7%）、eval少22张（约1.9%）。大量图属于矩乘展开、offset专用DMA及其他算子，未被本轮合并。
模型基线取自[S13报告](../attention-rows/REPORT.md)，本轮没有再跑旧实现的完整模型；不是同期交替多轮A/B计时。单次耗时只供参考，不能据此认定冷/热训练有稳定加速，也不能把专项约64.7%的冷耗时下降推广为整网收益。

两步均比较logits、损失、每个参数梯度及SGD动量更新；第1步CPU checkpoint在新模型/优化器中恢复后，权重、动量、RNG及loss与第2步逐位一致。相同权重CPU推理对照通过，保持默认MHA fastpath、padding推理enable_nested_tensor=False。
FP16仍使用rtol=0.003、atol=0.0005；两步loss与S13记录完全相同，但这不等于对旧进程所有中间Tensor做了逐位重放。

每步计算区间仍有56次各32字节的RNG元数据上传及10次标量状态下载（四块Embedding的前后向状态、因果检测、标签检查），无激活载荷回CPU计算；checkpoint和数值比较的CPU读取另计。原有统计/RNG推进断言通过。
最终context_failed=False，shutdown后live/cache/scalar/session/quarantine均为0。

## 回归与验收口径

原设备quick验收33组全部通过，4,162次NPU提交，含8,192种随机FP32转换样本、全部half编码、FP16/FP32算术/归约/矩乘、out、版本与布局拒绝、多线程及shutdown后存储寿命。进程正常返回，context_failed=False，关停后资源归零。
该脚本120秒的faulthandler诊断在编译阶段打印了一次线程栈；它是定时栈采样，没有触发外层300秒终止，也没有数值/设备失败，最终JSON的status=passed。
补充reduction-cast专项检查sum/mean的两种显式dtype转换及一阶梯度。初次误传其他脚本使用的--case，被argparse在导入设备前拒绝；改用本脚本已有--child入口后通过，未修改受测代码。

68项离线检查通过：几何/低位/mask/RNG/SDPA/标准模块及Transformer控制31项、npunlock Python API32项、source exporter5项。没有添加依赖或CI工作流。
硬件转换专项为新验收入口；原设备和模型harness未改。未复测完整FP32 long模型、真实数据集长期训练或GitHub CI，不扩大这些方面的结论。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_cast_reuse_demo.py
.\run_torch_npu.ps1 -Script .\torch_npu_transformer_scale.py -ScriptArguments '--profile','long','--batch','4','--precision','float16','--label','cast-after'
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_compat_demo.py -ScriptArguments '--child','reduction-cast'
```

专项当前断言每组3张图；重现旧基线需恢复原_cast的按shape缓存实现，不能只把label改为before。
首次编译时间、后台负载和驱动状态会影响计时，重新运行应另开进程。

## 来源及边界

本轮为本项目自编缓存键/元数据视图调整，复用既有npunlock载体编译与本项目IEEE转换代码；没有借用新的外部数值实现或增加OpenVINO运行依赖。
不同元素数仍会新建静态图，其他算子的形状/切片专用图未合并。普通FP16矩乘仍使用SHAVE FP32中间计算，不能据此声称原生DPU FP16训练加速。
完整模型短程数值验收不验证真实数据集质量、长期收敛或任意模型支持。

## 源码指纹

```text
旧torch_intel_npu.py
83947da59f531e3d8734b743d807e91796805ecd6455f77c4c32033eab4ec2c6
新torch_intel_npu.py
e4ae5a4e7e8f79b5b6187eadeb83ff88aab04d61fa6b9209f7d24827c5d18680
torch_npu_cast_reuse_demo.py（最终断言版）
570d41a12e75a3967d9bc1ad4b1915bb8050d327fd58fd39a3a20e31177c239c
torch_npu_transformer_scale.py（未改）
18722f10d9d7ab3d76245b67f8e5a4b585908b88634e40e3336e0289b48a085b
arithmetic.c（未改）
579fc8a26844182bcdaf15d3bb03408a3fcd587d32a777b6c4f91d1fa7d680b8
precision/cast-f16.elf（未改）
84944fa7ac6bf57124d0fe6743df86fd39c1c2e1a5b0cb7db3f2e91036f3327f
precision/cast-f32.elf（未改）
0e71653de2c5cb0cb2cfdc63671ab867ea5246ae2323b6b823463e23356aa0d0
```

## 清理与交付

已删除本轮6份原始JSON/日志，共38,573字节；转换基线/新实现、完整模型、设备quick及归约转换的结果均已归纳至本报告。
清理前后752份保留源码、报告、内核与必要运行时文件SHA-256一致，随后有意补充本节及WORKSPACE_STORAGE台账。source exporter选择748份文件，包含新验收脚本和本报告；本轮没有生成新的发布快照或推送。
宿主PYD与session DLL哈希仍为此前验收版本，source-release的Git状态干净。
