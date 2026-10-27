# 索引、平均池化与限幅兼容性（S20）

2026-10-21，Windows x64 / Core Ultra7 258V / Intel NPU4000 `8086:643e`，驱动32.0.100.5540，
图编译器8.3，Torch2.14.1+cpu；复用已有`.conda/npunlock`环境。

本轮扩展普通PyTorch模型迁移到`intel_npu`时的接口及一阶梯度，不提供CUDA运行时/PTX或第三方CUDA扩展兼容。
新增入口为`torch_npu_index_pool_demo.py`；生产改动在设备派发、autograd、几何规划及原索引SHAVE源码。
无新Python依赖、OpenVINO依赖、模型下载或C++/PYD/session DLL重建；新增/重建的索引ELF、汇编及manifest保留。
公开v0.1.0和`source-release`未更新。

## 本轮范围

| 接口 | 新增行为 | 软件边界 |
|---|---|---|
| gather | 动态INT64索引、任意合法轴、FP16/FP32前向和稠密一阶梯度；重复索引累加；精确无别名out | rank1–4非空，非索引轴尺寸须完全相同；最多128条独立索引行，每行继承原索引packet和R*L工作量界限；不含sparse_grad |
| scatter_add | 同设备src/self，src.shape=index.shape，FP16/FP32；self和src梯度；no_grad完整可写存储的scatter_add_ | 继承同一行/工作区限制，不是覆盖式scatter；普通可微inplace及更小src/index子集未认证 |
| take_along_dim | 指定dim时支持非索引轴广播，正负及超范围完整INT64按本机Torch规范化；dim=None展开时严格检查范围 | 继承gather/INT64存储界限；规范化不是通用公开remainder算子 |
| AvgPool1d/2d | CL/NCL、CHW/NCHW；kernel/stride/padding、ceil_mode、count_include_pad；2D支持divisor_override | FP16/FP32，非零divisor的绝对值<=2^24；输入、补零后输入、权重、输出及工作区仍有界；不是MaxPool或3D池化 |
| AdaptiveAvgPool1d/2d | 上/下采样、重叠bin、无batch/有batch，一阶梯度 | 正输出且空间bin总数<=64；保留原4D到(1,1)路径，其余复用FP32窗口均值 |
| minimum/maximum | 广播、浮点提升、部分只读布局、一阶梯度；相等平分、NaN导数及half平分舍入 | 保留maximum原AMSGrad精确out别名路径，其余out无别名；不保证NaN payload/signed-zero ties逐位一致 |
| clamp/clip/clamp_min/clamp_max | 实数或NPU Tensor边界、广播及Tensor边界梯度；标量边界的no_grad inplace；clip_grad_value_(foreach=False) | 不混用Tensor与Python数值的两个边界；不是clip_grad_norm_、一般可微inplace或所有dtype |
| Hardtanh/ReLU6 | 标准非inplace模块及一阶梯度 | 沿用上述浮点/布局/容量边界，不提供inplace ReLU/ReLU6 |

所有新算术与梯度计算在NPU完成。CPU只处理形状/步长、公开的几何常量、scaler控制状态及测试参考。
每个索引行每次前向或反向同步读一个4字节校验标志；索引与浮点payload不下载做运算。
有dim的take_along_dim规范化每32765个索引一包，各额外读取4字节标志。
这些计数不覆盖驱动内部DMA，不能称为零搬运或异步实现。

## 数值与实现细节

**scatter从self开始累加。** 不能先把src按索引求和再加self，后者会改变消减顺序。
扩展原backward packet的第二输入以携带FP32初始值，按原索引顺序add32，最后才转换为输出dtype。
原index_select/Embedding的零初始值、padding及逐次half舍入模式保留。
用self=65504、src=[-65504,0.01]的FP16/FP32消减样本精确检查；模型反向仍对self返回grad、对src返回gather(grad)。

**INT64规范化不经过数值FP32。** 新`index-wrap-i64`用原始高低32位字及64次整数余数迭代处理正负值，
包括-2^63/2^63-1；输出仍为INT64。FP32只作为旧字节packet载体。
32,823个索引横跨两个packet，包含广播、负轴、单元素packet和dim=None展开对照。
本机Torch2.14在指定dim时会规范化超范围索引，dim=None则报越界；这两条路径分别实现，不把该行为推广到其他版本。

**平均池化复用现有算子。** AvgPool用FP32深度卷积求和、按合法窗口除数归一，再转换到输入dtype；
ceil尾部补零不改变count_include_pad的除数定义。
只有在除数随位置变化时上传一个由几何决定的FP32元数据网格，常量除数复用标量缓存。
AdaptiveAvgPool的非全局路径按floor/ceil窗口边界裁切、FP32 mean、拼接；重叠梯度由现有设备切片/累加完成。
这扩大兼容性，未提供专用高性能池化内核。

**限幅梯度按本机版本分开核对。** Torch2.14标量clamp、Tensor clamp、单Tensor边界及Hardtanh的边界/NaN导数并不相同。
新测试精确覆盖相等、倒置、NaN、broadcast及partial view，不用一个统一的“>=/<=”公式代替所有入口。
相等平分梯度先保留上游dtype的half舍入，再做需要的归约/参数dtype转换；混合FP16/FP32标量边界的subnormal梯度另作精确检查。

## 正式验收

正式新增 **22个独立进程全部通过**，记录49791次NPU图提交。FP16/FP32各10个，另有两个FP32主参数/FP16 autocast组合训练进程；probe、早期try及两次超时进程不计为通过。

| 专项 | FP16提交 | FP32提交 |
|---|---:|---:|
| gather/scatter_add | 11081 | 8861 |
| minimum/maximum | 451 | 343 |
| clamp/Hardtanh/clip_grad_value_ | 1274 | 1106 |
| AvgPool1d/2d | 967 | 832 |
| AdaptiveAvgPool1d/2d | 728 | 650 |
| 特殊值、view、out、INT64边界 | 727 | 664 |
| 拒绝契约 | 59 | 49 |
| 完整INT64规范化跨包 | 2183 | 1796 |
| CNN四步训练/恢复 | 2946 | 2419 |
| 索引网络四步训练/恢复 | 3199 | 2536 |
| CNN AMP | — | 3269（混合精度） |
| 索引网络 AMP | — | 3651（混合精度） |

**旧路径回归：6个进程、70组全部通过**，共13374次NPU图提交。

| 回归入口 | 检查组 | NPU提交 |
|---|---:|---:|
| 设备quick | 33 | 4162 |
| Embedding全组 | 10 | 2291 |
| AdamW primitives | 2 | 1279 |
| AdamW optimizer | 13 | 2856 |
| AdamW contracts | 1 | 12 |
| training quick | 11 | 2774 |

上述28个正式进程共63165次记录提交；这些是功能覆盖计数，不是性能/吞吐测量。所有进程shutdown后live_buffers、cached_buffer_blocks、cached_scalar_buffers、cached_sessions、quarantined_buffer_bytes均为0，context_failed=False。

**离线84项通过**：根目录47项几何/参考检查、npunlock Python API 32项、源码导出5项；使用最终文件再次执行。

测试覆盖rank1–4、正负轴、重复索引、越界高位INT64、索引version guard、特殊值、partial view及out。
算子比较容差：FP32 rtol5e-5/atol6e-6，FP16 rtol0.004/atol0.0005；索引值、特定舍入样本和边界导数另用零容差。
多种平均池化几何/除数还在离线测试中与独立CPU FP64输出交叉核对，含ceil尾部。

## 组合训练与恢复

两种固定种子合成网络：91参数CNN含分组卷积、ReLU6、带ceil/不计padding的平均池化、自适应池化及Linear；
63参数索引网络含Linear、动态gather、重复scatter_add、可训练Tensor限幅边界及Linear。
两侧从相同初始状态独立运行四步，没有逐步重置CPU权重。
损失为显式FP32 CE、label_smoothing0.1；SGD lr0.01/momentum0.5、foreach=False/fused=False；更新前使用标准梯度值限幅。

| 网络/模式 | 首步→第四步NPU loss | 四步最大梯度绝对差（限幅后） | 第四步恢复 |
|---|---|---:|---|
| CNN / FLOAT32 | 1.044274688→1.039297342 | 2.98023224e-08 | 逐位一致 |
| CNN / FLOAT16 | 1.044254780→1.039338827 | 6.10351562e-05 | 逐位一致 |
| CNN / AMP | 1.044254780→1.039324045 | 2.28881836e-05 | 逐位一致 |
| 索引网络 / FLOAT32 | 1.157764316→1.146983504 | 2.98023224e-08 | 逐位一致 |
| 索引网络 / FLOAT16 | 1.157741666→1.146984816 | 0 | 逐位一致 |
| 索引网络 / AMP | 1.157741666→1.146980166 | 0 | 逐位一致 |

普通FP32模型按rtol6e-5/atol7e-6比较；纯FP16/AMP模型按rtol0.008/atol0.0006比较。
这是新模型的半精度验收口径，不把FP32主参数误当作严格FP32计算，也没有改写S17/S18的历史严格失败记录。
CPU对照关闭oneDNN以使用原生CPU半精度反向；关闭CPU oneDNN不改变NPU路径。
AMP保留FP32参数/梯度，使用NPU专用GradScaler；scale增长状态与CPU GradScaler一致。
训练区间显式核对复制计数：CNN仅NLL标志（AMP再加scaler标志）和一个几何网格上传；
索引网络每步仅9个校验标志（AMP为10个）及标量缓存上传，没有payload下载做模型计算。

第三步后用BytesIO保存显式CPU参数/优化器/scaler状态，新模型恢复第四步。
loss、参数、momentum逐位相同，AMP scaler状态一致；未写磁盘checkpoint，未增加原始NPU Tensor序列化。
这些短合成任务不代表真实任务准确率、长期收敛或任意模型支持。

## 冷编译超时记录

正式试跑中index-wrap FP32与CNN FP32各出现一次ir2blob.compile超时，均在DMA Slice图编译阶段；
当时默认编译预算为20秒。失败进程仍正常shutdown，context健康、资源归零，没有把它们记为数值通过。
index-wrap新进程复验通过；再次遇到CNN同类超时后，将本后端DMA Slice冷编译预算有界调整到60秒，
不增加自动重试，不修改数值内核或图执行超时。
后续完整复验的成功结果见上表；这不能证明驱动偶发慢编译的根因已消除，也未证明成功编译实际需要超过20秒。

## 复现

已有构建环境中硬件测试串行运行，默认不启用磁盘图缓存：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=$null
foreach ($precision in @('float16','float32')) {
    foreach ($case in @('indexed','extrema','clipping','pools','adaptive','edges','contracts','index-wrap','cnn','index_model')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_index_pool_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
foreach ($case in @('cnn_amp','index_amp')) {
    .\run_torch_npu.ps1 -Script .\torch_npu_index_pool_demo.py -ScriptArguments @('--case',$case)
}
```

正常运行直接使用有源码及SHA的三个索引ELF。修改C后，使用本机已有MoviTools运行
`npunlock-probe/research/build_index_kernels.py`，无需重编C++或添加Python依赖。
旧设备quick、Embedding全组、AdamW primitives/optimizer/contracts、training quick为本轮回归入口。

## 指纹与保留

收尾交付文件SHA-256：

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `082afa54e32d16e286576b3999a53076c9460524a2c5c4388602d5805fe51b58` |
| `torch_npu_autograd.py` | `bff25da6a1ecae342e3a3de6f98dc36d2f8b7bfe026799e19c1e24055eac56c6` |
| `torch_npu_geometry.py` | `dd69664929effd871ac6a029f40826497bf05a9258abb827ab63069f6c5f379d` |
| `torch_npu_index_pool_demo.py` | `5c0e27bc4240cd05db0a06cda9f556538e430bf73566716c961493eb870a65f7` |
| `test_torch_npu_geometry.py` | `56cc1a7dd49d5bb9e50f5f4dcaa345f8c7575242ded4b44187e533043f27ae73` |
| `torch_npu_adamw_demo.py` | `ff2de204aaf0cdc65ab05d61785b9b6229d39e0bc7bcda1b856b5f4e84fc9c64` |
| `torch_npu_training_demo.py` | `d63261b7cbbc03f1b286ac8e30bbe5140665c3c98fb8148a863adc57fdd5735b` |
| `npunlock-probe/research/build_index_kernels.py` | `3ab665a4c60602c0a8c5935b3899870d94455e6b046c7a825db51e8a2b6940d2` |
| `npunlock/tests/fixtures/npu4000/resident_index.c` | `8ad8c30704803e3c002e98a38d3193849399b7608a45836e6e8e14bcba0ec61e` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `393ffa0cb74dbfd720fab8b987e936cc6a4af299bdbcda05f438616a19554722` |
| `npunlock-probe/torch-device/backend.cpp` | `4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |

索引ELF合计16,084字节（含旧内核重建；不是净增体积）：

| ELF | 字节 | SHA-256 |
|---|---:|---|
| `index-gather-f32` | 4956 | `e2d4f0c93d6fc4dde16a2c8849d3806d216c6602649ad9d16ed53fec908f0078` |
| `index-backward-f32` | 7020 | `ea134236ae3d9353b1a0c469692399f243587ec60b9aa0029eb50207f52cece8` |
| `index-wrap-i64` | 4108 | `e2ebd775feb83f98c2c53d4336db7933c41055c41cd24513f567843744b55f2b` |

早期专项完成后曾补充边界验收及调整Slice编译预算；表中指纹描述最终交付文件，不声称每次早期运行都来自同一整文件hash。完整INT64规范化、单元素/展开边界、out、受影响的组合训练及旧路径回归均已在最终生产实现上复验。源码导出选择783个文件，已确认新增源码、报告、ELF/汇编和构建入口均包含；未生成另一份发布快照。

按用户保留策略，删除本轮68份原始JSON/日志/Movi响应，共345132字节；清理前后790份保留源码、报告、内核和必要运行时文件SHA-256一致，随后补充本段与存储台账。保留源C、三个索引ELF/汇编/manifest及构建脚本、设备派发/autograd/几何、验收与离线测试和双语文档；无依赖安装、模型下载、C++/PYD/DLL重建、图表或磁盘checkpoint/图缓存。source-release与GitHub未改。

MaxPool、Conv3d/转置卷积、inplace激活、更多dtype/空Tensor、128行之外的动态索引、nonzero/布尔动态索引、
覆盖式scatter、通用范数梯度裁剪、通用GradScaler、异步stream/event、torch.compile及第三方CUDA扩展仍有缺口。
本轮未优化低位整网质量、GPU+NPU协同或评估算力/能效/延迟；不提供通用兼容率百分比。
