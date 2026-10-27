# 常用PyTorch模型迁移接口扩展（S19）

2026-10-21，Windows x64 / Core Ultra 7 258V / Lunar Lake NPU `8086:643e`，
驱动32.0.100.5540、图编译器8.3、Torch2.14.1+cpu、现有`.conda/npunlock`环境。

本轮以普通PyTorch程序迁移到`intel_npu`为目标，补齐六组有界接口，
没有新增CUDA运行时、PTX或第三方CUDA扩展兼容层。实现主要改动在
`torch_intel_npu.py`和`torch_npu_autograd.py`，复用已有SHAVE计算、设备DMA和图会话。
没有新增OpenVINO依赖、Python包、模型、SHAVE ELF或宿主C++构建。
普通FP16矩乘/卷积仍是原有SHAVE/FP32中间计算路径，不宣称新增DPU训练或性能收益。
本地修改未同步至公开v0.1.0，`source-release`发布快照未改。

## 新增范围

| 接口 | 行为与实现 | 保留边界 |
|---|---|---|
| 分组/深度Conv2d | 按通道拆分输入、权重和bias，由设备DMA整理，串行调用现有单组前后向；拼接结果/梯度。包含depthwise通道倍率 | 仅非转置、合法整数组数；保持输入/输出、分配和工作区边界，无分组并行或新DPU内核 |
| Conv1d | NCL/三维权重映射到高度1的Conv2d，前后向复用上述分组路径 | 只覆盖已有FP16/FP32和合法stride/padding/dilation；无Conv3d或转置卷积 |
| 任意合法轴softmax/log_softmax | 归一化轴交换到末轴、设备整理、原完整行内核计算、恢复原轴；反向沿相同布局执行 | rank1–4非空、总量<=1,048,576、归一化轴宽<=65,536；内部half_to_float=True仍拒绝 |
| 多维NLL/CrossEntropy | `(C)`、`(N,C)`、`(N,C,L)`、`(N,C,H,W)`的硬标签损失；保持类轴为dim1，整理成二维送入原NLL内核 | 同设备连续INT64标签，weight=None；目标数*(C+4)<=65,536，无soft targets或spatial原生forward/out重载 |
| Label smoothing | CE的[0,1]系数；三种reduction及ignore_index，使用设备log_probs、比较、where和归约组合；全部忽略时保留mean的NaN值及零梯度 | 仅上述无类别权重的硬标签路径；不是自动选择精度或量化质量修复 |
| rsub | Tensor/Scalar输入、广播、alpha和一阶反向；修复旧实现不能接收ATen位置参数alpha的问题 | 继承sub的dtype/alpha/容量契约；公开torch.rsub不接受out，out验证使用ATen重载 |

卷积保存原始输入/权重以保留原地修改的版本检查，而非只保存分组副本。
大工作区卷积继续由已有流式路径处理；分组不会把完整组的工作区误当作整个输入通道数。
非末轴softmax和多维损失的复制只整理设备布局，不下载payload计算。
NLL标签检查仍每次读取一个4字节错误标志；合法参数的形状/类别权重/系数错误尽可能在提交前拒绝。

## 正式硬件验收

19个专项进程全部通过，共 **29,636次NPU图提交**。下面每行的两个精度分别在独立进程运行；
AMP保持FP32主参数，另起进程。首次四项旧实现探针及其后回归探针不计入此总数。

| 专项 | FP16调用 | FP32调用 | 覆盖 |
|---|---:|---:|---|
| operators | 916 | 422 | 每精度36组softmax/log_softmax（rank2–4、全部正负轴）、特殊值/转置、8组rsub前后向及ATen out |
| conv | 826 | 563 | 每精度6组：groups2、depthwise倍率2、逐通道pointwise、有/无bias、各参数梯度及权重version guard |
| conv1d | 509 | 355 | groups1/2/4、stride2、padding2、dilation2、bias，前向/全部梯度和无梯度推理 |
| losses | 889 | 483 | 每精度24组NLL/CE × rank1–4 × none/mean/sum，另3组全部忽略 |
| smoothing | 2,822 | 1,358 | 每精度24组smoothing0.1/1 × rank1–4 × reduction，另6组全部忽略及精确零梯度 |
| axis-tiled | 270 | 212 | 非末轴、69,904元素、正stride裁切读及完整基底梯度，跨65,536计算块 |
| contracts | 22 | 18 | 非法组数/转置卷积/维度/系数/标签形状/类别权重及大INT64越界标签 |
| spatial模型 | 2,544 | 1,576 | 95参数、两层深度卷积和pointwise输出，四步独立CPU轨迹及恢复 |
| 分组流式卷积 | — | 3,415 | 输入(1,4,82,82)、groups2、9×9卷积；单组展开/折叠工作量1,089,288项，触发流式路径 |
| spatial AMP | — | 3,026 | 同一95参数网络，FP32参数/梯度、专用scaler、label smoothing及四步训练/恢复 |
| Transformer AMP | — | 9,410 | 标准两层Encoder token分类网络，443参数、width4/heads2、batch2/length3、padding/因果mask、dropout0；四步训练/恢复 |

原四项探针的分组卷积、通道softmax、空间CE、rsub alpha均从拒绝变为可执行。
正式算子测试在前后向区间检查显式上传/下载计数：没有CPU/NPU payload搬运；
分类损失仅允许一次4字节错误标志读取。误差比较、checkpoint转换和version检查所需的诊断下载在计数区间外。
统计不覆盖驱动内部DMA，不能据此声称零数据移动。
所有专项进程退出后context健康，live/cache/scalar/session/quarantine均为零。

## 模型数值口径

所有训练控制都从相同初始权重独立运行四步，没有逐步把CPU权重重置为NPU权重。
输入与标签为固定种子的小型合成数据，不表示真实任务收敛或准确率验证。
SGD显式foreach=False/fused=False、lr0.01、momentum0.5；CE均显式FP32并使用label_smoothing0.1。

| 模型/精度 | NPU第一步→第四步loss | 四步最大参数梯度绝对差 | CPU比较容差 |
|---|---|---:|---|
| spatial FP32 | 1.095738649→1.095608234 | 1.4901161e-8 | rtol5e-5 / atol6e-6 |
| spatial FP16 | 1.095734000→1.095623016 | 7.6293945e-6 | rtol0.006 / atol0.0005 |
| spatial AMP | 1.095734000→1.095602036 | 7.6293945e-6 | rtol0.006 / atol0.0005 |
| Transformer AMP | 1.514511824→1.441683054 | 4.8828125e-4 | rtol0.006 / atol0.0005 |

AMP控制使用标准CPU FP16 autocast与CPU GradScaler；NPU使用标准NPU autocast和专用GradScaler。
两侧scaler增长状态一致。梯度和主参数在AMP中为FP32，计算包含FP16舍入，
上表AMP容差不等同于S17/S18的严格FP32梯度容差。
Transformer的最大梯度差4.8828125e-4没有被记成严格FP32轨迹通过；S17/S18原有严格失败记录仍保留。
专项算子比较采用FP32 rtol4e-5/atol5e-6、FP16 rtol0.004/atol0.0005，并检查NaN一致。
上述均为相应样本下的检查，不保证任意值域逐位一致。

四个模型均在第三步后把detached参数/优化器/scaler状态显式转CPU，用内存BytesIO进行torch.save/load(weights_only=True)，
创建新模型/优化器恢复第四步。恢复后的loss、参数与momentum逐位相同，AMP scaler状态相同。
未保存磁盘checkpoint，也没有新增原始NPU Tensor序列化支持。

## 既有路径回归与离线检查

7个既有路径进程、42组检查全部通过，共6,658次NPU提交：

| 既有入口 | 检查组数 | NPU提交 |
|---|---:|---:|
| device quick | 33 | 4,162 |
| attention softmax | 2 | 327 |
| attention contracts | 1 | 19 |
| classification logsoftmax | 2 | 335 |
| classification nll | 2 | 1,490 |
| classification contracts | 1 | 37 |
| S18局部卷积FP32/FP64参考 | 1（含6种几何） | 288 |

所有既有进程正常shutdown。合并新增专项共26个正式硬件进程、36,294次NPU提交；
未将基线/修复过程中的试跑重复计入。
82项离线检查通过：45项核心/参考、32项npunlock Python API、5项源码导出检查。
旧softmax/log_softmax拒绝检查改用真正越界的轴，label_smoothing拒绝检查改用1.1；
没有删除合法性检查或把原本失败的数值阈值放宽为通过。

## 复现

在已有构建及上述环境中，硬件测试串行执行。没有启用磁盘图缓存：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = $null
foreach ($precision in @('float16','float32')) {
    foreach ($case in @('operators','conv','conv1d','losses','smoothing','axis-tiled','models','contracts')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_portability_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
foreach ($case in @('conv-stream','conv-amp','transformer-amp')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_portability_demo.py -ScriptArguments @('--case',$case)
}
```

代码入口保留，验收原始JSON/日志按用户保留策略在报告完成后清理。
复核历史数值需重新运行，不打包实验输出或图表。

## 剩余缺口

本轮扩大算子与模型语义覆盖，不提供CUDA生态全面替代或兼容率百分比。
动态索引/nonzero/scatter/gather、更多随机/数学接口、类别权重/soft targets、通用池化、
Conv3d/转置卷积、更多dtype/空Tensor/更大存储、二阶梯度、异步stream/event、
通用torch.compile、原始设备序列化和广泛第三方模型仍未覆盖。
通用torch.amp.GradScaler("intel_npu")及更大模型的AMP/长期收敛仍需后续工作。
低位整网质量、GPU+NPU协同和性能优化未在本轮改动或重测。

## 代码与运行时指纹

| 文件 | SHA256 |
|---|---|
| `torch_intel_npu.py` | `c674745394f875a04296d2844b20b12d3719bc2a107bfa24abe1a05c84bc6feb` |
| `torch_npu_autograd.py` | `65076b1bedd07d7afe0dfdb60c724f85ea9244ba161c73a8f3f835722320d576` |
| `torch_npu_portability_demo.py` | `85cec8b7a05b77d6058f8d5492044ad3612f043b9b7462c0053de95711c422ae` |
| `torch_npu_attention_demo.py` | `93b148c717f2309db167c236fc54e91c084412dc2cf556d0e1d7bbeffb9dd6c9` |
| `torch_npu_classification_demo.py` | `0132109ec9aaaaeecf6f907f6b0baa6d88364eeebc385db86608a91255142733` |
| `npunlock-probe/torch-device/backend.cpp` | `4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |

C++、PYD和session DLL与本轮初始构建一致。所有19项正式专项的两个生产Python模块指纹均与当前源码一致。

## 存储清理

验收后仅删除本轮29份原始JSON/日志，共89,342字节。清理前后786份保留源码、报告、内核及必要运行时文件SHA256一致；核对后追加本段文字与存储台账。源码导出选择包含779份文件，新增入口与本报告均被纳入，未生成另一份发布副本。
保留全部实现、测试入口和本报告；没有下载模型、安装依赖、保存实验数组/图表或磁盘checkpoint，没有启用磁盘图缓存，GitHub与source-release未改。
