# 完整模型低位质量：敏感投影保留 FP16

2026-10-15，Windows / Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`，
驱动 `1005540`、图编译器8.3、Torch 2.14.1+cpu；本地 SmolLM2-135M-Instruct。

**本轮找到了一项可复现的质量改善，但没有通过全部质量门槛。**
内部选择得到：30层中的 `mlp.up_proj.weight` 与 `mlp.down_proj.weight` 使用原始FP16，
其余Q/K/V/O/Gate五类投影继续使用校准后的逐行W8。真实NPU独立检查从全W8的118/142提高到129/142，
平均logits相对L2下降34.29%，最大KL下降82.90%。最差相对L2仍有8.413%，因此不能设为通用默认。

已在本地适配器中加入默认关闭的 `fp16_projections=()` 实验选项，默认FP16/W8图和参考输出保持一致。
修改位于Python图构造与显式CPU参考中；宿主C++、SHAVE内核、DLL/PYD和公开量化算法没有改动。
本轮没有更新GitHub发布快照，公开v0.1.0不包含这个新增参数。

## 方法与数据边界

沿用[上一轮研究](../w8-sequential-calibration/REPORT.md)的40条固定提示，重新生成每条64个消费token的CPU greedy轨迹，不在EOS停止。
32条（编号 `i % 5 != 4`）拟合2,048行；8条（`i % 5 == 4`）只用于选择策略，合计290个预测位置。
全210个矩阵均保持原公开GPTQ算法：FP64求解、1%阻尼、128列块、激活排序、原FP16逐行scale、逐行RTN回退。
所有矩阵校准重建SSE均不大于RTN；合计SSE为RTN的1.386742%。拟合耗时102.157秒，不是NPU推理计时。

本轮只搜索跨所有30层使用相同投影类型的保护策略，最多保护50%的decoder投影权重元素。
预设候选为全W8、7种单投影保护、全部attention保护及3种MLP两投影组合，共12种。
按失败位置数、最大L2、平均L2、保护元素数排序。预留的两轮贪心追加没有产生新候选，因为最佳项已用满预算。
全FP16仅作控制，不参与候选排名。没有根据独立集继续调整策略。

CPU候选参考把W8的整数code与FP16 scale在FP64中精确相乘，矩乘结果转FP16；保护项直接调用原始FP16 Linear。
其余部分使用官方Transformers FP16网络。这与旧研究的逐阶段CPU参考不同，不能混用两者的CPU数字。
控制组与原官方CPU模型逐位相同；CPU筛选结果仍须由真实NPU复核。

独立回归集与40条校准提示不重合，但仍是历史16条提示，并非新采集的大规模盲测集。
bird/garden/cat各32个预测位置，其余13条各1个；另外重复前缀案例33个位置，合计17个案例、142个位置，覆盖KV容量64。
采用相同的CPU greedy token进行teacher forcing；没有测试各精度自由生成后的轨迹分歧，也没有PPL/任务准确率评估。
质量门槛沿用 **logits相对L2 < 3%、KL(reference || actual) < 0.02**；argmax一致率单独记录。

## 内部选择结果

下表的权重比例只以decoder的7类矩阵为分母，不包含embedding、lm_head、norm或KV。

| FP16保护项 | 保护元素比例 | 通过/290 | 平均相对L2 | 最大相对L2 |
|---|---:|---:|---:|---:|
| 无，全W8 | 0% | 189 | 2.8033% | 11.6684% |
| Q | 9.375% | 193 | 2.7522% | 11.7013% |
| K | 3.125% | 195 | 2.7760% | 12.4062% |
| V | 3.125% | 207 | 2.5424% | 10.2143% |
| O | 9.375% | 198 | 2.6261% | 10.2638% |
| Gate | 25% | 200 | 2.6128% | 9.4808% |
| Up | 25% | 209 | 2.4261% | 9.8878% |
| Down | 25% | 232 | 1.9973% | 9.5301% |
| Q/K/V/O | 25% | 216 | 2.3122% | 10.0935% |
| Gate + Up | 50% | 224 | 2.2157% | 7.0718% |
| Gate + Down | 50% | 254 | 1.7571% | 5.8653% |
| **Up + Down（选中）** | **50%** | **259** | **1.5953%** | **8.1314%** |
| 全FP16控制，未参与选择 | 100% | 290 | 0 | 0 |

Up+Down按预定“先减少失败位置”规则胜出；Gate+Down的最差L2更低，不能称选中策略在每个指标上均最佳。
保护60个矩阵、53,084,160个元素；其余150个矩阵、53,084,160个元素仍为W8。

## 独立检查

同批固定token，参照为完整官方CPU FP16模型：

| 执行路径 | 策略 | 通过/142 | 平均相对L2 | 最大相对L2 | 最大KL | argmax一致/142 |
|---|---|---:|---:|---:|---:|---:|
| CPU数值参考 | 全W8 | 117 | 2.243806% | 10.133561% | 0.011365214 | 140 |
| CPU数值参考 | Up+Down FP16，其余W8 | 129 | 1.470607% | 8.115854% | 0.002540685 | 140 |
| CPU数值参考 | 全FP16 | 142 | 0 | 0 | 0 | 142 |
| 真实NPU | 全W8 | 118 | 2.237747% | 10.905755% | 0.011397425 | 140 |
| **真实NPU** | **Up+Down FP16，其余W8** | **129** | **1.470534%** | **8.413030%** | **0.001948558** | **140** |
| 真实NPU | 全FP16 | 142 | 0.348106% | 1.219016% | 0.000181764 | 142 |

全W8与FP16的真实NPU控制结果复现上一轮。混合策略相对本轮全W8：平均L2下降34.285%，最差L2下降22.857%，最大KL下降82.904%。
argmax一致数仍为140/142，不能声称生成token准确率提高。
全30层decoder及KV在NPU；embedding、final norm和lm_head显式在CPU，整个模型并非全部迁移到NPU。

混合策略未通过的位置（位置从0开始）：

| 案例 | 未通过位置 | 通过数 | 最差相对L2 |
|---|---|---:|---:|
| bird | 37、54 | 30/32 | 7.345284% |
| garden | 无 | 32/32 | 2.5448% |
| cat | 34、38、51、53 | 28/32 | 8.413030% |
| repeat32 | 34、39、43、47、52、56、60 | 26/33 | 5.521991% |
| 其余13条 | 无 | 13/13 | <3% |

这13个失败均来自L2门槛，KL均未超限。cat位置34的L2为8.413030%，但KL仅0.000001543，argmax仍相同。
反过来，language位置26与cat位置50通过L2/KL门槛却改变argmax。
全W8的bird argmax分歧消失，但混合策略新出现language分歧；不能把整体平均改善解读为所有提示均改善。
KL由FP32计算，极小负数是数值舍入，没有修改或放宽原门槛。

## 实际压缩与执行证据

汇总30个单token decode图的编译常量区：

| 路径 | 常量字节 | 常量MiB | INT8矩阵任务数 |
|---|---:|---:|---:|
| 全W8 | 113,441,280 | 108.1860 | 840 |
| 混合 | 166,525,440 | 158.8110 | 600 |
| 全FP16 | 219,240,960 | 209.0845 | 0 |

混合常量区比全FP16小24.0446%，比全W8多50.625MiB。保护项的IR常量为原始FP16；其余5个投影仍带W8解压属性。
所有30层均有非elementwise INT8 DPU任务，检查得到 `amode=0, wmode=2`（FP16激活、INT8权重）。
编译器分块产生多个任务，因此600不是矩阵数；也未把FP16布局任务误判为低位失败。

以上仅为一组decode图的常量区，不能当成全进程峰值内存、完整模型文件大小或推理速度。
适配器仍保留原FP16权重与packed权重；实验同时驻留prefill/decode图和CPU参考。
本轮没有公平的延迟/吞吐/功耗benchmark，不宣称混合策略加速。

## 回归与资源检查

- 三种NPU配置分别18,330次调用：592个输入token × 30层，17次重复decode × 30层，加60次gamma准备。
- 每种配置17次热重复：不新增会话或显式payload上传/下载，输出/KV/位置的91份数组哈希逐位一致；90份种子KV/位置哈希保持不变。
  这些复制计数不包含驱动内部DMA；哈希检查自身会下载数据，位于热调用计数区间之外。
- 另有30层混合完整前缀prefill局部检查，共90次调用；每层以实际NPU输入计算对应CPU参考。
  最大局部相对L2为0.0782347%，最大scaled error为0.434400%，均通过原局部门槛0.3%/1%。
- 参数错误在新增会话、驱动分配、上传和图提交之前拒绝；验证30层策略传递及 `start_layer=3,num_layers=2` 的绝对层编号。
- 三组模型检查与局部检查共 **55,080次NPU调用**。全部退出时context健康，live/cached/quarantined buffer字节为0。
- 45项核心离线回归、5项新增可选模型回归全部通过。新增项需要safetensors，不加入仅安装核心依赖的CI。
- 发布快照原适配器与修改后的默认入口比较：FP16/W8 × prefill/decode四组XML与常量逐字节一致，CPU参考逐位相同。
  新选项不同策略得到不同IR/权重缓存标识，颠倒同一tuple的次序不改变图。
- `w8_projection_audit.py` 复核来源/权重/选择哈希、划分、候选预算、原质量门槛、全部142个位置及执行证据。

## 本地实验入口

```python
from torch_npu_smollm import FrozenSmolStack

# packed_by_layer 必须包含每层全部7个校准后的逐行INT8 PackedWeight。
# 本轮的生成及加载示例见 w8_projection_mixed.prepare()/weights()。
stack = FrozenSmolStack.from_checkpoint(
    model_dir, num_layers=30, mode="w8a16", packed_weights=packed_by_layer,
    fp16_projections=("mlp.up_proj.weight", "mlp.down_proj.weight"),
)
# 输入仍是已有适配器要求的FP16 intel_npu Tensor。
hidden, kv = stack.prefill(embedded_tokens)
hidden, kv = stack.decode(next_embedded_token, kv)
```

保护项始终使用checkpoint原始FP16权重，而非W8解压权重。策略对所有请求层相同，属性返回只读frozenset。
参数须为不重复的合法投影名称tuple；默认空tuple保持原行为，非空仅用于w8a16，保护全部7类应改用 `mode="fp16"`。
未提供packed权重时仍会走原RTN准备，**不能把本报告的校准质量数字套用到RTN**。
本轮实际NPU认证仅覆盖上述两类保护、batch1、容量64及默认融合路径；其他策略的CPU排名不是NPU认证。
修改私有融合开关、保护整个attention/MLP子段的诊断分段执行未认证，保留原INT8执行拒绝条件。
这不是AMP、低位训练、自动选精度或任意模型支持。

复现使用已有环境、本地模型与当前运行时；无联网下载：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPUNLOCK_GRAPH_CACHE = $null
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_mixed.py -ScriptArguments '--case','prepare'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_mixed.py -ScriptArguments '--case','select'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_mixed.py -ScriptArguments '--case','independent'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_mixed.py -ScriptArguments '--case','npu-mixed'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_mixed.py -ScriptArguments '--case','npu-w8'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_mixed.py -ScriptArguments '--case','npu-fp16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_projection_checks.py
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python;$PWD\npunlock-probe\research"
.\.conda\npunlock\python.exe -B .\npunlock-probe\research\w8_projection_audit.py
```

审计默认图比较依赖本工作区未修改的 `source-release/torch_npu_smollm.py` 基线，公开副本不包含这个嵌套目录。
按用户存储要求，验收后删除本轮10份JSON/NPZ，共101,772,466字节（97.06MiB），仅保留源码与本报告。
复现须先重新prepare；没有新增依赖、下载模型、保存原始logits/激活/图blob/图片或启用磁盘图缓存。

## 指纹与后续方向

| 对象 | SHA256 |
|---|---|
| 修改前/发布快照适配器 | `a9ce3949141ebd3fae258a73578d9f13ed7bd0da4e0d3809a97ca0c46296c869` |
| 当前 `torch_npu_smollm.py` | `d95e571675972d9605fb431e76512671a0619619fbf8dc28e98d5e98757c6e8c` |
| `w8_projection_mixed.py` | `2159948d48957d960946f90ac36567390368bdab21667eaff6d90724a6dc498e` |
| `w8_projection_checks.py` | `da4532f759f8deb92c502a680d4b30fafe06309af3bed343042e70f26413ca35` |
| `w8_projection_audit.py` | `e449a39cfe0f2cb9d1e224ce78644c63eb934c872c5bd5163fde2f4bb8f2eaa2` |
| 新增模型策略测试 | `f15f2f90c623a0ac5cbbb1d932af64868839c221cceecb5d9d15e6eb538c7a5f` |
| `torch_npu_lowbit.py`，未改 | `e9f5b12b09275f73933ef1ab325346776c75a905d48486e3aad5256d08a46f21` |
| `torch_intel_npu.py`，未改 | `bd147eb2adbf49d092084ae4f1de767b395424016e9f8862eb850ab571b55cce` |
| `backend.cpp`，未改 | `d6acf5d765b1024ef783e4370f30064e7c67aa4a26f7a10e9eba8c0864fe6d09` |
| session DLL，未改 | `0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9` |
| Torch PYD，未改 | `3325f761ea55485271e26ecc3667b820f428d31f665e4c31fb916e8c30aefc8b` |
| 模型权重，未改 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 本轮校准权重NPZ，已清理 | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 内部选择JSON，已清理 | `15dddcc52a4f6b8f54d882e84acb1431b375b28bf91ddf31e1e8885c7b06dd19` |

当前结论只支持“敏感投影保留FP16可以改善完整模型数值质量，并保留部分W8执行”。
下一阶段应在拟合集内做逐层敏感度定位，寻找更少FP16权重的策略，并用新增的独立语料、自由生成和分段PPL复核。
旧独立集可继续作为回归集，不能边看其失败位置边调参数后再称独立验证。
在质量通过之前，不把该配置升级为默认，也不扩散到INT4/NF4。
