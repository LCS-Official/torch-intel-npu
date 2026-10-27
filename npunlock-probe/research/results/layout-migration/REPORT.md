# 拼接、切片整理与布局梯度验收

日期：2026-10-19。Windows x64，Core Ultra 7 258V，Intel NPU `8086:643e`，驱动API 1005540、编译器8.3，Torch 2.14.1+cpu；沿用项目`.conda/npunlock`环境。

本轮继续补普通PyTorch模型迁移的算子与存储语义：`cat/stack`、部分view的安全读取，以及slice/select、split/unbind的一阶反向。**新增14组验收全部通过，共2,260次真实NPU图调用；原功能40组回归通过，共4,487次调用。** 当前为本地开发版，没有推送GitHub或改动`source-release`。

这不是CUDA运行时、PTX或自定义CUDA扩展兼容，也没有把NPU并入`torch.xpu`。

## 接口范围

| 接口 | 本轮支持范围 |
|---|---|
| `torch.cat` | 同设备、同dtype FP16/FP32/INT64；输入rank 1–4，合法正负dim，除拼接轴外shape一致；输出连续并独立拥有storage，单输入也不返回别名 |
| `torch.stack` | 同设备、同dtype，输入rank 0–3且shape相同；合法正负dim；输出rank<=4、连续、独立storage |
| cat/stack的`out=` | 无梯度、精确shape/dtype、完整连续目的存储，拒绝输入别名；不自动resize或做混合dtype promotion |
| cat/stack反向 | FP16/FP32一阶梯度；支持单侧梯度、重复输入、非连续输出梯度和部分view输入；INT64仅存储/搬运 |
| slice、select、narrow | 保留PyTorch原生view及共享storage/version语义；正step、合法整数选择。支持的读取入口将其整理成完整缓冲区后再绑定NPU |
| clone、复制只读源、CPU下载 | 新增部分/偏移/带孔洞的常见正stride切片与零stride广播；FP16/FP32/INT64。clone独立分配，preserve_format沿用PyTorch的轴顺序与孔洞压紧规则 |
| slice/select反向 | 通过设备字节Gather把梯度放回原位置，未选位置填精确零；不依赖CPU数据scatter |
| split/unbind反向 | 复用新增cat/stack；包括未使用部分输出的零梯度分支 |
| slice_copy/select_copy/narrow_copy | 三者前向及无梯度out；slice_copy/select_copy的一阶梯度已测，narrow_copy未认证反向。仅新增缺失的narrow_copy.out注册，其余复用PyTorch复合实现 |

每个浮点Tensor仍限制1–1,048,576元素，INT64限制1–524,288元素，单allocation最多4MiB；无空Tensor。只读布局采用保守的逐轴跨度检查并允许零stride广播，不保证任意交错/重叠`as_strided`布局。

普通算术的输入契约没有整体放开。特别是连续的部分切片，`.contiguous()`可能直接返回原view，**需要完整独立存储时使用`.clone()`**。部分/重叠目的写入仍拒绝，INT64 `copy_`目的仍须完整连续；INT64与浮点之间没有数值cast。

## 实现方式

主要改动在根目录的`torch_intel_npu.py`、`torch_npu_autograd.py`，新增`torch_npu_layout_demo.py`。已有device/vector检查中与新支持冲突的负向断言同步更新。

1. 保留原生`buffer_for`/完整allocation绑定检查。新增只读view校验，与可直接绑定的Tensor校验分开。
2. CPU只根据shape、stride、offset生成静态搬运元数据；Tensor数据由已有byte-Gather、Concat、Transpose/DMA图在设备执行。
3. INT64按两个原始32位字搬运。F32仅作为传输载体，没有INT64到浮点的数值转换。
4. cat先把目标轴移到前面、拼接再恢复轴序；stack复用cat。反向分割输出梯度，并让原生slice/select反向继续传播。
5. clone用PyTorch的empty_like计算输出布局，避免自行推断preserve_format时漏掉转置后步进切片的轴顺序。

没有新增OpenVINO Python/runtime依赖，没有改动SHAVE C/ELF、宿主C++、DLL/PYD或驱动。普通矩乘的SHAVE/DPU分工和精度策略沿用原实现。

计算区间的上传/下载/设备复制计数在测试指定边界核对：cat/stack及view复制/反向没有额外宿主Tensor payload搬运。初始化上传、显式CPU下载及CPU参考比较在此统计区间外；计数不涵盖驱动内部DMA，不表示没有数据搬运。

## 新增硬件验收

| 组别 | 覆盖 | NPU调用 |
|---|---|---:|
| cat / FP16、FP32、INT64 | 每种20组主shape/轴组合，另含单输入独立storage、重复输入、转置切片、非连续梯度、out | 203 / 200 / 140 |
| stack / FP16、FP32、INT64 | 每种20组主shape/轴组合，覆盖标量至三维输入，其余附加契约同上 | 152 / 149 / 91 |
| views / FP16、FP32、INT64 | 每种8类slice/select/narrow/转置/标量view；clone stride、复制、stack；3种copy API及out；浮点含slice/select、split/unbind、部分view dtype转换及广播clone梯度 | 264 / 234 / 132 |
| edges / FP16、FP32、INT64 | 原始位模式、65路输入、70,000元素源、跨度拆分、密集转置clone、广播只读及原生绑定保护 | 38 / 19 / 21 |
| contracts | 20项metadata/别名/容量/非法out拒绝，以及共享version计数 | 5 |
| training | 普通FP32分支网络，四步独立CPU对照 | 612 |
| **合计** | **14组全部通过** | **2,260** |

cat与stack各有60组主shape/轴组合，正负轴分别执行。浮点的数值搬运与梯度使用精确对照；训练按下述数值阈值对照。

FP16 edges包含全部65,536种half编码的切片后重排，按原始字节比较，涵盖NaN payload、负零和次正规数。FP32包含特殊位模式。INT64包含`2**53+1`、有符号64位上下界和可呈现为NaN字的原始模式，未转成浮点计算。

跨度检查使用`(10000,7)`源的`[1::5,1::3]`切片：首个候选Gather窗口的物理跨度超过65,536，实际触发窗口拆分；INT64同时覆盖双字偏移。65个输入还覆盖原生64参数上限下的分层拼接。

20项非法调用均在NPU图提交前拒绝：非法轴/shape/rank、混合dtype/device、输出超限、out shape/dtype/别名/部分存储、narrow out别名/shape、部分目的写入、正stride重叠、不匹配的反向参数与可微out。原生`_C.copy`仍直接拒绝偏移或部分allocation绑定。

每组无live buffer净增量；每个验收进程关闭后live/cache/scalar/quarantine/session全零，`context_failed=False`。

## 普通分支网络训练

模型为两路`Linear(6,4) → ReLU`，分别取步进切片后cat；另将两路stack后求mean；再拼接原分支并接`Linear(12,3)`与CrossEntropy。共95个参数，不使用冻结模型适配器或自定义模型执行器。

CPU和NPU从相同初始状态分别训练，SGD `lr=.03, momentum=.5, foreach=False, fused=False`。四步逐项检查损失、全部参数梯度及更新后的参数，FP32阈值`rtol=3e-5, atol=3e-6`。

| 步 | NPU损失 |
|---|---:|
| 1 | 1.0790500641 |
| 2 | 1.0767973661 |
| 3 | 1.0734847784 |
| 4 | 1.0697249174 |

四步CPU轨迹对照全部通过；这证明本例的组合训练兼容性，不构成真实数据集精度、长期收敛或速度结论。本轮没有测性能。

## 原功能回归与修复过程

| 回归 | 组数 | NPU调用 |
|---|---:|---:|
| device Quick | 33 | 4,162 |
| SDPA layouts，含两个线程的clone反向与共享排列图 | 2 | 79 |
| Embedding contracts / edges | 1 / 2 | 33 / 209 |
| vector contracts / gaps | 1 / 1 | 4 / 0 |
| **合计** | **40** | **4,487** |

device Quick用时130.19秒，最终状态通过。其120秒faulthandler堆栈是诊断快照，不是最终超时失败。Embedding的部分索引拒绝、INT64非法值及version保护仍有效；部分向量计算和部分标量直接读取仍按原契约拒绝。迁移gap检查更新为4项：设备RNG、训练Dropout、异步复制、原始NPU序列化；已支持的cat从负向列表移除。

离线geometry/lowbit共8项、Python API共32项通过。新增脚本的all进程隔离及子进程失败传播用无NPU的mock断言检查，覆盖14/6个正常任务和其中一个失败的传播；默认all将不同case/精度分到独立进程，遵守512条图缓存上限。

初始探针确认slice/select/split能创建view，但部分view clone及cat/stack不能完成。开发中修复了三个实际遗漏：转置步进切片的clone布局、原生split/unbind未用分支的零stride梯度读取，以及narrow_copy.out缺少派发。原device测试把offset.cpu()拒绝写成旧契约，现改为检查安全materialize成功并保留直接原生偏移绑定的拒绝检查。未放宽数值容差或原生内存检查来通过验收。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_layout_demo.py
# 或选定一组
.\run_torch_npu.ps1 -Script .\torch_npu_layout_demo.py -ScriptArguments @('--case','views')
.\run_torch_npu.ps1 -Script .\torch_npu_layout_demo.py -ScriptArguments @('--case','cat','--precision','float32')
.\run_torch_npu.ps1 -Script .\torch_npu_layout_demo.py -ScriptArguments @('--case','training')
.\run_torch_npu.ps1 -Quick -Device
.\run_torch_npu.ps1 -Script .\torch_npu_sdpa_demo.py -ScriptArguments @('--case','layouts')
```

入口在源码根目录执行。测试生成的原始JSON/日志按用户约定在验收后清理，仅保留源码和本报告；需要复测时重新运行。报告中的独立用例结果来自对应子集执行，没有为了报告重复全部默认all任务。

## 剩余边界与下一步

尚未补BOOL存储、比较/where/masked_fill、布尔SDPA mask、通用高级索引、部分目的写入、空Tensor和混合dtype拼接。没有认证二阶梯度、通用AMP、异步stream/event或任意CUDA模型迁移，也未穷举极限容量下的全部轴/布局组合。

下一步优先补BOOL与mask路径：先定义存储/复制及比较语义，再接where/masked_fill和布尔SDPA mask，配合普通attention模型验收。设备RNG/Dropout随后单独实现种子、状态恢复和反向契约。

## 产物清理

已删除本轮28份原始JSON/日志，共91,322字节；清理前后675份保留文件指纹一致，随后补充本记录与存储台账。逐文件核对路径位于实验目录内并拒绝reparse point，没有递归删除目录。此实验目录只保留本报告，源码、当前内核及环境保留。未下载模型、安装依赖或启用磁盘图缓存；公开发布快照的Git工作区仍干净。

## 来源与源码指纹

API语义参照PyTorch官方[cat](https://docs.pytorch.org/docs/2.14/generated/torch.cat.html)、[stack](https://docs.pytorch.org/docs/2.14/generated/torch.stack.html)与[Tensor views](https://docs.pytorch.org/docs/2.14/tensor_view.html)。实现复用本项目已有npunlock设备DMA编译/提交路径，没有引入新的第三方源码片段。

最终源码SHA-256：

```text
torch_intel_npu.py
a6ef3895f6bedf5177e83b73515a8f033de037cfdbc2be3287b12632e44cb4b0
torch_npu_autograd.py
37fddb452efc67893d7b1416c3d1bc44476a5e041be181e4cbdd5a59f2f197ba
torch_npu_layout_demo.py
58cb8e0902f3c534adeadc562ba8c22f3a2b2b464dfc8d95ae63c914040a65cf
torch_npu_device_demo.py
1a6f20e1cafd159930f0a4b3d61d796ee69a22aa37d7b903e283058989ebdc64
torch_npu_vector_demo.py
b76ea8454d8e0d9c545469744362c06276f4f42dbc1d956abcfb1beb4407ac5a
```

cat/stack和训练验收时主实现为`7e38211ddf05bfa77bc438a1cff4d3048273689f3ac254034f23b0955fff128e`，随后仅添加独立的narrow_copy.out函数及注册，形成上述最终主实现；views/out/contracts、edges及原功能回归在补齐后完成。

沿用且未修改的原生文件：backend.cpp SHA-256 `006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924`；PYD `b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e`；session DLL `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9`。
