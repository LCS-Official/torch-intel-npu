# MaxPool、可微原地ReLU与topk（S22）

2026-10-21，Windows x64 / Core Ultra7 258V / NPU4000 `8086:643e`，驱动32.0.100.5540，
图编译器8.3，Torch2.14.1+cpu；沿用`.conda/npunlock`，未安装依赖或下载模型。
目标为普通PyTorch模型迁移，不提供CUDA runtime/PTX/第三方CUDA扩展兼容。

## 新接口与边界

| 接口 | 实现 | 本轮边界 |
|---|---|---|
| MaxPool1d/2d | FP16/FP32，CL/NCL、CHW/NCHW；kernel/stride/padding/dilation/ceil_mode，可选return_indices；2D原生双out和backward入口；值及空间INT64索引均在NPU产生 | 输出<=524,288元素，核面积<=32,765，输出总量×核面积<=8,388,608；每个输出窗口至少有一个合法采样点；不含3D、自适应最大池化或Unpool |
| relu_/nn.ReLU(inplace=True) | 同一对象/存储原地写入，version递增；非view非叶的可微训练；no_grad叶张量更新与inference_mode | 完整密集FP16/FP32存储；可微叶/view在提交前拒绝，可先clone；不含任意部分目的写入、原地ReLU6/Hardtanh或一般可微inplace |
| torch.topk | FP16/FP32、rank0–4、任意合法单轴、largest=True/False，值一阶梯度/设备INT64索引；双out、只读partial输入、长行分块 | 1<=k<=min(轴长,128)，输入元素数×k<=16,777,216，输出<=524,288；k=0与其他空Tensor、整数输入、更大预算、sort/argsort未支持 |

沿用全局非空/rank<=4、FP16/FP32最多1,048,576元素、INT64最多524,288元素的软件契约。
这些不是机器共享内存容量；out仍须预分配精确shape/dtype且不与输入或另一输出别名。
sorted=False允许返回有序的合法top-k，当前复用同一有序选择实现，不声明更快的无序路径。
相等值的topk索引和梯度路由可能与CPU不同；新实现按原位置确定性选择，不把CPU的某一平局排序当作通用保证。
不承诺所有NaN payload/signed-zero值顺序逐位相同；topk负零上游梯度另有逐位检查。

## 实现与已修复问题

**池化。** CPU仅依据形状/步长/池化参数生成窗口地址，复用现有设备DMA gather读取payload。
每个窗口包上传一份空间INT64几何表，非法padding项为-1，SHAVE跳过它们；不会用零padding改变负值池化。
正常平局取首个合法点，多个NaN取最后一个，与本机CPU控制分别核对；不能复用S21 argmax的首个NaN规则。
最多32,768个采样格组成一个选择包；具体包大小还受header/输出容量限制。

**选择。** 新SHAVE有序插入选择使用IEEE原始位序，保留subnormal比较，直接输出INT64原始高低字。
长行按65,530个值分块，再在NPU合并带原始索引的候选；几何offset在header中，索引不经过数值FP32转换。
这是有界兼容实现，不是通用高性能partition/sort或融合采样内核。

**反向。** 线性索引内核将全局输出位置映射回输入，输入梯度按65,535元素块、上游按21,842元素包分块。
每次先校验完整INT64索引，再在NPU回填/累加；池化重叠梯度按逻辑输出次序相加，Half每次写入后舍入。
topk索引不重复，采用覆盖回填以保留上游负零；不能共用“从+0开始加一次”的路径，后者会丢失负零符号。
该细节已修复，FP16/FP32各用原始整数字检查梯度位模式。
每个反向包只读取4字节校验标志；前向无payload回读，池化另有上述几何表上传。
计数不含驱动内部DMA，不称为零搬运或异步实现。

**原地语义。** 自定义Function保存更新后的ReLU输出并mark_dirty；no_grad路径显式increment_version。
最初的no_grad覆盖写入没有递增version，旧计算图可能漏报输入被修改；现已修复并检查恰好一次版本递增及保存张量保护。
叶/可微view在提交前检查，不先修改再报错。
Binary、Clone、Linear、Conv2d、BatchNorm2d及普通ReLU的已知新分配结果，在Function.forward内部仅移除内部reshape的view元数据，
由自定义Function建立梯度边；存储/版本计数仍共享，没有额外payload复制。
这使标准Conv2d/BN2d/Linear后接原地ReLU可用，不是让调用者在计算图外detach输入；调用者创建的可微view仍需clone。
更多复合入口形成的view（例如未认证的Conv1d→原地ReLU组合）不能由这些测试自动推广。

## 正式验收

**15个最终新增独立进程全部通过**：7项×FP16/FP32，另有一个FP32主参数/FP16 autocast训练进程。开发轮失败与try不计入正式通过数。

| 专项 | FP16提交 | FP32提交 |
|---|---:|---:|
| 原地ReLU/别名/版本/标准模块 | 221 | 125 |
| topk轴/out/布局及一阶梯度 | 1008 | 735 |
| MaxPool1d/2d几何/索引/out/backward | 600 | 486 |
| NaN/平局/负零梯度/重叠/版本 | 283 | 196 |
| 元数据拒绝及非法INT64索引 | 12 | 11 |
| 长轴与大池化输入分块 | 183 | 144 |
| 四步CNN训练及恢复 | 1516 | 964 |
| AMP CNN训练及恢复 | — | 1749 |

新增专项共8233次记录NPU图提交，所有进程shutdown后live/cache/scalar/session/quarantine归零、context_failed=False。

**87项离线检查通过**：根目录50项几何/参考检查、npunlock Python API 32项、源码导出5项。新增选择packet/池化几何两项单测进入既有CI列表。

| CNN模式 | 首步→第四步NPU loss | 四步最大梯度绝对差（裁剪后） | 恢复第四步 |
|---|---|---:|---|
| FP32 | 1.041151047→1.037012458 | 7.4505806e-09 | 逐位一致 |
| FP16 | 1.041106701→1.036978960 | 3.81469727e-06 | 逐位一致 |
| AMP | 1.041106701→1.037050605 | 6.35068864e-06 | 逐位一致 |

**8个既有回归进程、50组检查/训练通过**，含旧Transformer及AMP数据流：

| 回归 | 组数 | NPU提交 |
|---|---:|---:|
| 设备quick | 33 | 4162 |
| 基础训练quick | 11 | 2774 |
| 极值除法梯度 | 1 | 6 |
| 密集转置叶/别名 | 1 | 151 |
| 单目梯度 | 1 | 35 |
| 归约/转换梯度 | 1 | 30 |
| 标准attention模型训练 | 1 | 3101 |
| S21范数AMP训练 | 1 | 1882 |

旧路径共12141次记录提交；23个正式进程合计20374次。这是功能覆盖计数，不是性能测量，所有旧进程同样正常shutdown。

覆盖scalar至rank4、正负轴、最大k=128、65,537长轴（含大于65,535的原始索引）、67,600元素池化输入跨梯度块；池化partial view、dilation/ceil/padding、首合法tie/末NaN；原地ReLU精确一次version递增、叶/view无提交拒绝和clone恢复训练。

普通数值控制为本机CPU PyTorch。索引、池化/选择值和指定边界梯度按零容差比较；
非选择性模块FP32用rtol6e-5/atol8e-6，FP16用rtol0.006/atol0.0007。
topk平局时分别核对选出值、索引对应源值、唯一性及按实际返回索引构造的CPU梯度，不强行要求CPU相同索引顺序。

训练网络含Conv2d、两次原地ReLU（含Binary结果）、MaxPool2d、flatten/topk及Linear。
固定相同CPU/NPU初态独立四步，显式FP32 CE、SGD lr0.02/momentum0.5、梯度范数裁剪0.1；不逐步重置CPU权重。
FP32梯度/参数用rtol6e-5/atol8e-6，FP16及AMP用rtol0.01/atol0.001。
AMP保留FP32主参数/梯度、使用NPU专用GradScaler；CPU关闭oneDNN使用原生Half反向。
第三步后保存显式CPU参数/momentum及scaler到BytesIO，恢复第四步检查loss/参数/momentum逐位一致、AMP scaler状态一致。
模型每步计算区间核对3个标志读取（NLL/池化反向/topk反向，AMP再加scaler）及4份池化几何表上传，没有payload下载做算术。
这不替代真实任务质量、任意模型或长期收敛认证；历史S17/S18严格数值差异与S21长范数CPU差异仍保留。

## 复现与来源

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=$null
foreach ($precision in @('float16','float32')) {
    foreach ($case in @('relu','topk','pools','edges','contracts','training','tiled')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_pool_topk_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
.\run_torch_npu.ps1 -Script .\torch_npu_pool_topk_demo.py -ScriptArguments @('--case','amp')
```

已有MoviTools环境中，修改C后运行`npunlock-probe/research/build_selection_kernels.py`。
正常运行使用有源码/指纹的三枚ELF，不要求新Python依赖、OpenVINO包、C++/PYD或session DLL重建。
CPU数值/梯度来自本机Torch2.14.1；窗口地址、SHAVE插入选择、原始INT64合并与线性索引反向由本项目实现，
复用既有IEEE helper/设备DMA，未复制外部选择或池化内核。

收尾交付文件SHA-256；本轮最后的覆盖回填修正只作用于S22选择反向，既有回归不调用该新内核；修改后的S22完整专项均重新验收。

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `9d9cbf259c20b13ce551b551547d801ccb0d87838591c0a11252d490bdbf2db3` |
| `torch_npu_autograd.py` | `e6cbe1b4886e949850c4d6912963bc3948572e98a454356153be5e2d8e3ef985` |
| `torch_npu_geometry.py` | `e95850c61399d50583bf6d71bebb5aadddd0d528872e23267b280b7b7db2b67f` |
| `torch_npu_pool_topk_demo.py` | `c02233bbe63a21267b2ee192f541ce3659bccf66335dbc7d41e81ed200201289` |
| `test_torch_npu_geometry.py` | `31405056a2627eefdc5fe659a958629189eb4b9e3ed471327fa7eac600fd9cc0` |
| `npunlock-probe/research/build_selection_kernels.py` | `90be7598215c954143c4694e450e42f3144ec878b500f90164cc046cc1f605e2` |
| `npunlock/tests/fixtures/npu4000/resident_selection.c` | `bea7de2bcacb09b674d30ac3d13f3eeb2f88c9904eed51574c1a24b438b7477d` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `bfca632f67e924c1ffb372ae9e38cda5d10af35c2a752565d0e2aa2cfc9c04cb` |
| `npunlock-probe/torch-device/backend.cpp` | `4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |

新ELF共16,436字节，源C、汇编、构建及manifest均保留：

| ELF | 字节 | SHA-256 |
|---|---:|---|
| `topk-f32` | 4684 | `a9acfa8ef01ab5da8b3b0fb71ce0994e4bcaad09bf0d36e596940d11544aa2f0` |
| `pool-select-f32` | 4604 | `7e64095220071155c839c23d20be55ebf414a047cd5b494f8f86740910bbdaae` |
| `selection-backward-f32` | 7148 | `639ed7917bdb84cb04d4c65b6167266a60a24b2d358b2a8ef277beaa23bbcbef` |

已检查源码导出选择801个文件，包含新源码、ELF/汇编、构建/验收脚本和报告；未生成重复发布快照。

## 剩余待修复/扩展兼容性

| 方向 | 仍缺条目 |
|---|---|
| 视觉 | interpolate/上采样、ConvTranspose/Conv3d、MaxPool3d/AdaptiveMaxPool/Unpool；GroupNorm/InstanceNorm与默认完整视觉模型仍未认证 |
| 排序/采样 | sort/argsort、max/min的值/索引二元接口、更大k/工作量、k=0、完整top-p/HF generate；topk平局不承诺CPU相同索引路由 |
| 索引/张量 | 更大的gather/scatter/Embedding、nonzero/动态布尔索引、覆盖式scatter、部分写入、更多可微view原地语义；空Tensor、rank>4、更大容量/任意stride/完整resize行为 |
| 数学/训练 | 通用pow、其他p/矩阵范数、混合dtype/foreach梯度裁剪、CE类别权重/soft targets、更多优化器、原地ReLU6/其他激活、二阶导 |
| 数值/精度 | 既有FP16/AMP严格轨迹及长范数CPU FP32差异；BF16等更完整dtype算术、通用GradScaler和任意模型/长期质量认证 |
| 随机/运行时 | randperm/Tensor概率Bernoulli、GQA/NestedTensor/长上下文、完整设备torch.compile、异步stream/event/non_blocking、原始NPU序列化、GPU+NPU协同 |
| CUDA生态 | CUDA runtime/PTX、自定义CUDA扩展、NCCL未实现，不会随普通ATen兼容自动获得 |

本轮未优化性能/能效或低位整网质量，不提供通用兼容率百分比。source-release/GitHub未更新。

按用户保留策略，删除本轮44份原始JSON/日志/Movi响应，共251799字节；清理前后808份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与台账。保留新C、三枚ELF/汇编/manifest、构建/验收、设备派发/autograd/几何、离线检查与双语文档。无依赖安装、模型下载、C++/PYD/session DLL重建、图表或磁盘checkpoint/图缓存；source-release与GitHub未改。
