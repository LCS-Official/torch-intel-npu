# 极值归约、向量范数与标准梯度范数裁剪（S21）

2026-10-21，Windows x64 / Core Ultra7 258V / NPU4000 `8086:643e`，驱动32.0.100.5540、
图编译器8.3、Torch2.14.1+cpu；复用`.conda/npunlock`，不安装依赖或下载模型。
目标为普通PyTorch模型迁移的算子/训练语义，不提供CUDA runtime/PTX/第三方CUDA扩展。

## 接口与边界

| 新接口 | 本轮实现 | 尚未覆盖 |
|---|---|---|
| argmax/argmin | FP16/FP32，rank0–4；None按逻辑顺序展开或任一合法单轴、keepdim；返回设备INT64，重复极值/NaN取首个；只读partial/broadcast布局整理、精确无别名out | 整数/BOOL输入；topk、sort、max/min的values/indices二元入口 |
| amax/amin | FP16/FP32，多轴/全归约/keepdim；一阶梯度对相等极值平分，NaN梯度单独对照；out仅无梯度 | 二阶导数；signed-zero极值输出不保证逐位相同 |
| linalg.vector_norm、普通torch.norm/Tensor.norm的向量入口 | ord=0/1/2/+inf/-inf，多轴、keepdim；FP16/FP32，允许FP16升FP32，拒绝缩窄；前向/一阶梯度及精确无别名out，p=0返回None梯度 | 任意p、矩阵谱/核范数、通用pow、二阶导；不把普通Frobenius向量化入口推广为全部矩阵范数 |
| isnan/isinf/isfinite | 浮点分类在NPU执行，返回不可微BOOL；公开ATen已有out重载 | INT64/BOOL输入未认证 |
| 标准clip_grad_norm_/get_total_norm/clip_grads_with_norm_路径 | 不改PyTorch工具函数；补齐其调用的范数、浮点谓词及同dtype零维NPU Tensor乘法系数。foreach=None/False，完整密集且同dtype的FP16/FP32梯度；无grad参数跳过 | 混合dtype参数梯度、foreach=True/fused、稀疏梯度及任意范数阶数；无梯度参数集沿用标准CPU零标量返回，不新增空设备Tensor |

所有接口保留全局rank/非空/存储边界：FP16/FP32最多1,048,576元素，INT64最多524,288元素。
arg归约在提交前额外检查INT64输出容量；这些是软件契约，与机器17.9GB共享内存上限不同。
新行极值内核不继承S20 gather的128行限制；每个packet最多65,536个FP32 word，超过后按行/长行分块。

## 实现与数值

两枚源C/汇编/ELF由现有MoviTools构建。packet为4个几何元数据word加FP32 payload，
第二输入可携带原始INT64候选索引，输出为行极值和原始INT64索引。
按IEEE位序比较有限值，正负零视为相等；首个NaN胜过数值，已有NaN不被后续NaN替换。
避免普通浮点比较的FTZ影响subnormal次序；索引不经浮点数值转换。

长行以65,532元素分块，再在NPU合并候选值及原始索引；相同值/NaN保持原块和块内顺序。
验证中修复了INT64候选拼接使用错误内部接口、第二输入刚好填满时生成零长度padding的问题。
均使用本项目原有字节DMA及支持INT64的cat整理，不放开全局空Tensor契约。

L1/L2使用FP32 abs/平方和、已有补偿sum/sqrt后转回输出dtype；不是稳定缩放范数。
多元素L2的FP32平方和可溢出为Inf或下溢为0，这些样本分别与本机CPU行为核对；单值归约用abs避免无谓平方溢出。
Inf范数复用行极值；大量并列极值的导数先在FP32按计数平分，再舍入到梯度dtype，
避免把65,537等计数先转Half变为Inf。0范数按非零比较计数，NaN/Inf也计为非零，CPU对照的一阶梯度为None。

**保留的严格CPU数值差异：** 65,537个FP32的0.01值，单线程CPU L1返回655.698486328125，
NPU补偿归约返回655.3699340820312，独立FP64参考为655.369985351339。
原CPU FP32比较rtol6e-5/atol8e-6失败，未放宽原阈值；长向量L1/L2另按FP64参考转回结果dtype验收，
同时记录原CPU FP32绝对差。它证明该有限样本的高精度参考通过，不证明CPU轨迹逐位/严格一致。
CPU累加也受线程数影响；不能把这一归因推广到任意模型差异或所有极值。

新增算术、极值、索引与梯度在NPU执行。CPU只使用几何元数据、Python参数及测试参考；
正常归约/范数/裁剪不下载payload。error_if_nonfinite=True沿用PyTorch条件分支，
每次判断读取一个BOOL标量；False路径无需该读取。计数不覆盖驱动内部DMA，不能解释为零搬运或异步。
仍仅扩展no_grad乘法的同dtype零维设备系数，其他Tensor广播/混合dtype原地更新仍拒绝。

## 正式验收

**15个正式新增进程按下列参考验收通过**，共25,459次记录NPU图提交；7项×FP16/FP32，另加一个FP32主参数AMP进程。原始probe/try及失败开发轮不作为正式通过数。长向量L1/L2的FP64参考验收与原CPU FP32差异分别记录，不能称为全部严格CPU等价通过。

| 专项 | FP16提交 | FP32提交 |
|---|---:|---:|
| arg/extreme归约及一阶梯度 | 2852 | 2165 |
| 160组向量范数及dtype/out/legacy入口 | 4273 | 3121 |
| 特殊值、布局、谓词及大量并列梯度 | 1803 | 1518 |
| 标准梯度裁剪/非有限控制 | 1615 | 1216 |
| 拒绝、版本保护及dtype入口 | 14 | 10 |
| 百万元素分块/长范数（L1/L2按FP64） | 1316 | 1057 |
| 四步训练及恢复 | 1654 | 963 |
| AMP四步训练及恢复 | — | 1882 |

**85项离线检查通过**：48项根目录几何/参考、32项npunlock Python API、5项源码导出；新增packet几何单测纳入既有CI列表。所有正式新进程shutdown后live_buffers、cached_buffer_blocks、cached_scalar_buffers、cached_sessions、quarantined_buffer_bytes归零，context_failed=False。

**既有回归4个进程、50组全部通过**：

| 入口 | 检查组 | NPU提交 |
|---|---:|---:|
| 设备quick | 33 | 4162 |
| 基础训练quick | 11 | 2774 |
| 更新/原地契约 | 5 | 649 |
| GradScaler | 1 | 350 |

旧路径共7935次图提交；连同新增专项，19个正式进程合计33394次记录提交。仅用于功能覆盖，未测吞吐/延迟；旧进程同样context健康且shutdown资源归零。

| 网络模式 | 首步→第四步loss | 四步最大梯度绝对差（裁剪后） | 恢复第四步 |
|---|---|---:|---|
| FP32 | 1.101503849→1.100301147 | 3.0733645e-08 | 逐位一致 |
| FP16 | 1.101493239→1.100321174 | 4.57763672e-05 | 逐位一致 |
| AMP | 1.101493239→1.100329876 | 4.27579507e-05 | 逐位一致 |

三种模式各四步的裁剪前范数均大于0.1，实际触发缩放。标准裁剪专项每种dtype另有30组有限数据检查（5种ord×2种foreach选项×3种阈值），并检查Inf/NaN的异常前不修改梯度、False路径传播及有限True路径。

| 长向量原CPU差异 | FP16绝对差 | FP32绝对差 |
|---|---:|---:|
| p=1（65537个0.01） | 0.5 | 0.328552246 |
| p=2（65537个0.01） | 0 | 3.69548798e-05 |

测试覆盖scalar、rank1–4、正负/多轴、keepdim、partial/广播只读view、out、版本保护、
NaN/Inf/正负零、首个tie、FP32最小subnormal、跨65,532块边界、INT64大于65,535的结果索引，
最多1,048,576元素输入以及2,051/65,537个并列极值的精确梯度。
常规算子FP32比较rtol6e-5/atol8e-6，FP16比较rtol0.006/atol0.0007；索引/极值前向及指定梯度样本另用零容差。
新网络FP32梯度/参数用rtol8e-5/atol8e-6，FP16/AMP用rtol0.008/atol0.0008。
这些是有限验收阈值，不放宽S17/S18既有严格失败记录，不宣称任意模型长期收敛或任务准确率。

组合网络为Linear4→6/tanh/向量归一化/amax残差/Linear6→3，含动态前向argmax预测与标准clip_grad_norm_。
相同固定初态CPU/NPU各独立运行四步，显式FP32 CE、SGD lr0.02/momentum0.5、裁剪阈值0.1，未逐步重置CPU权重。
第三步后用BytesIO存显式CPU参数、momentum及可选scaler，恢复第四步并逐位核对loss/参数/momentum；AMP再核对scaler状态。
CPU关闭oneDNN以使用原生Half反向；NPU没有CPU payload算术回退，也未增加原始设备序列化。

## 复现与来源

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=$null
foreach ($precision in @('float16','float32')) {
    foreach ($case in @('reductions','norms','edges','clipping','contracts','tiled','training')) {
        .\run_torch_npu.ps1 -Script .\torch_npu_reduction_norm_demo.py -ScriptArguments @('--case',$case,'--precision',$precision)
    }
}
.\run_torch_npu.ps1 -Script .\torch_npu_reduction_norm_demo.py -ScriptArguments @('--case','amp')
```

只修改C时才需已有MoviTools及`npunlock-probe/research/build_row_extrema_kernels.py`。
无新OpenVINO依赖、C++/PYD/session DLL重建或新Python依赖。
CPU数值/导数和标准裁剪状态来自已安装Torch2.14.1；行极值/候选合并为本项目自写，未复制外部归约内核。

收尾交付指纹（SHA-256）；测试脚本的额外边界在对应专项复验，未要求开发轮全部整文件hash相同。

| 文件 | SHA-256 |
|---|---|
| `torch_intel_npu.py` | `4a2fb160e75ba37d73e6474a433d1c38aec0b1c9f3b4ddf2df33ca52c8b2e59f` |
| `torch_npu_autograd.py` | `cad8868a5485852eba2c8f23a3c365cefc1a4b7f60a8ff2d779a25da06598454` |
| `torch_npu_geometry.py` | `c3b1916e59c1c58bf9535b976430e458154b047f2593366195f27ebe6e7c7af7` |
| `torch_npu_reduction_norm_demo.py` | `7c1fb78548e7f7b4810420dca6e8d0ba37a8f1b9a1ab2f3b5b71ab2aecc5539e` |
| `test_torch_npu_geometry.py` | `48855440ae07207ff1c3ae1ac91433866072699598288d1dc2d11099e0ba511d` |
| `npunlock-probe/research/build_row_extrema_kernels.py` | `377d231594146a76ba95e995b64dfa45485a8addf4f58ed8df51c550974d50ec` |
| `npunlock/tests/fixtures/npu4000/resident_row_extrema.c` | `b835665a02a005e8cc193b07cf38b97ffabe871f0e8ac26579e340ac22623f77` |
| `npunlock/tests/fixtures/npu4000/resident/manifest.json` | `fb05fccfd8a9b63371fb22bf26d2fbec83b7ecc2d5e6e7d1ea92359e055efa7c` |
| `npunlock-probe/torch-device/backend.cpp` | `4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83` |
| `npunlock-probe/torch-device/_torch_intel_npu.pyd` | `3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd` |
| `npunlock-probe/session-runtime/npunlock_session.dll` | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |

新增两枚ELF共11,192字节，另保留对应汇编：

| ELF | 字节 | SHA-256 |
|---|---:|---|
| `row-max-f32` | 5596 | `c8824b5d48fdcb402c8f5c19f3a214bae33672d3d3dc17c67c1da0a5935f255a` |
| `row-min-f32` | 5596 | `1660462f8ce7385e6cce06084871230d7876dcc070c73a3576ed86b37fc565ee` |

源码导出选择791个文件，已检查包含新C、ELF/汇编、构建脚本、验收脚本及报告；本轮不生成另一份发布快照，不更新source-release或推送GitHub。

## 剩余兼容性缺口

| 优先方向 | 仍缺条目 |
|---|---|
| 常见视觉模型 | MaxPool、上采样/插值、ConvTranspose/Conv3d；GroupNorm/InstanceNorm及默认模型尚无正式支持/验收；inplace ReLU等 |
| 排序与预测 | topk/sort/argsort、max/min二元返回；未认证完整top-k/top-p或HF generate |
| 索引与容量 | 更大的gather/scatter/Embedding、nonzero/动态布尔索引、覆盖式scatter、部分写入；空Tensor、rank>4、任意stride和更大存储 |
| 数学与训练 | 通用pow、其他p及矩阵范数、混合dtype/foreach梯度裁剪、CE类别权重/soft targets、更多优化器、可微inplace、二阶导 |
| 数值认证 | 长向量范数的CPU FP32严格轨迹差异、原S17/S18半精度严格差异仍保留；独立高精度/短网络通过不等于长期或任意模型质量认证 |
| 精度/随机 | BF16等设备算术、randperm/Tensor概率Bernoulli；通用GradScaler与任意模型AMP仍缺 |
| 模型/运行时 | GQA/NestedTensor、任意模型/长上下文、真正设备后端完整torch.compile、异步stream/event/non_blocking、原始NPU序列化、GPU+NPU协同 |
| CUDA生态 | CUDA runtime/PTX、自定义CUDA扩展、NCCL未实现，不随普通ATen兼容自动获得 |

低位整网质量、性能/能效和上述其他缺口本轮未修改；未给出通用兼容率百分比。

按用户保留策略，删除本轮35份原始JSON/日志/Movi响应，共199672字节；清理前后798份保留源码、报告、内核与必要运行时文件SHA-256一致，随后追加本段与存储台账。保留新C、两枚ELF/汇编/manifest、构建与验收入口、几何/派发/autograd、离线测试和双语文档。未安装依赖、下载模型、重建C++/PYD/session DLL、生成图表或磁盘checkpoint/图缓存；source-release与GitHub未改。
