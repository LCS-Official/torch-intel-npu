# NPU KV decode：位置更新与样本选取合并

2026-10-05，Lunar Lake 8086:643e、驱动 1005540、编译器 8.3。
本轮仅改变冻结 SmolLM2 的图和 Python 调度，复用上一轮原生多输出扩展。
原生 session DLL、C++/PYD、SHAVE RMSNorm 内核、量化格式、模型权重和普通 forward 算术保持原版本。

## 实现与调用

`prefill`／`decode` API 和不可变快照约定不变。注意力图现在输出第四项 FP32 位置，
消除独立位置加法提交。图内使用已有 FP16 位置加一再转回 FP32；1..64 均可精确表示，
这是有界整数元数据更新，不更改模型精度或基础真正 FP32 算术保证。
B>1 时整批输入、归一化结果和 K/V 直接绑定，设备标量 sample 经 Convert/Gather 选取一个样本，
消除每个样本外部四次 DMA 切片提交。同一层／精度／批量形状的样本／位置复用一张图。
sample 常量在 prefill 准备并复用；暖 decode 不上传／下载 Tensor 或按样本／位置新建图。

编译器 8.3 的批量动态权重矩乘限制仍存在；注意力依次执行 B 次，随后在设备上拼接三个输出。
两次 RMSNorm 和 FFN 仍各自提交。B1 为四次提交，B>1 为 B+6。

| 每层暖 decode | 上一轮多输出版 | 本轮 |
|---|---:|---:|
| B1 | 5 | 4 |
| B2 | 17 | 8 |
| B4 | 27 | 10 |
| B8 | 47 | 14 |

前轮文档的 B8=49 来自首次 decode，包含两次 gamma 广播准备；本轮旧／新重复调用确认
真正暖 B8 为 47→14。首次 decode 的准备和编译不包含在暖性能测量中。

## 硬件验证

9 个独立进程串行通过，累计 26,517 次 NPU 调用；每个进程
shutdown 后 live_buffers=0、context_failed=false。底层扩展及基础算术源码哈希与前轮一致，
本轮没有重跑未改动的通用设备全套；改动覆盖下列模型测试。

| case | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
| probe | 8 | 0 | 136 |
| numeric | 40 | 0 | 2304 |
| boundary | 14 | 0 | 952 |
| guards | 23 | 0 | 119 |
| positions | 44 | 0 | 2600 |
| hybrid | 8 | 0 | 816 |
| paired | 24 | 0 | 1666 |
| regression | 8 | 0 | 88 |
| performance | 0 | 6 | 17836 |

102 条数值记录、336 项固定实际输入 K/V 检查。
覆盖起点 0/10/26、1/2/4 层、不同样本 B2/B8、63→64 与完整 64-token prefill。
单层／四层 × FP16/W8A16 共连续 252 步，固定图复用且无暖主机复制／新增编译。
旧快照与已有 K/V 逐位不变，空槽保持零，重复解码一致。22 项错误模型缓存用法在任何层执行前拒绝，
inference_mode 通过。8 组普通 forward 与旧版逐位一致；24 组连续旧／新 decode 的输出、K/V 和位置也逐位一致，
涵盖 B1/B2/B4/B8、单层／四层和 1→4、32→35、61→64 位置。

执行门槛保持 L2<0.3%、`abs(error)/(abs(reference)+RMS)<1%`；输出 RMS 按 token，K/V 按 head。
最大执行 L2 0.2003%，最大归一化误差 0.6472%；
相对官方 CPU 浮点层最大 L2 2.7445%，保持已有 3% 质量门槛。
固定输入 K/V 最大 L2 0.0323%，归一化误差 0.1356%。
原固定 atol 失败计数保留。8 组前 2／4 层 NPU、其余 CPU 的混合生成 token 与 CPU 全部一致，
首步 logits 最大 L2 0.5888%；属于短生成 smoke，不代表全 30 层／广泛质量。

## 同进程性能

CPU 8 线程、21 轮交错测量、轮换顺序。旧适配器从 before/ 原样加载，仅重定位内核路径；
双方复用同一个原生扩展和权重。排除初始准备、编译、预热、输入上传／输出下载和 CPU mask/RoPE 准备。
时间包含 Python、分配、设备 DMA、提交和 fence；CPU FP16 decode 包含 DynamicCache crop，非最优 CPU 基准。
桌面负载和频率未控制，跨轮历史绝对时间不作配对比较。

单位 ms，中位数。旧→新为上一轮多输出适配器→本轮位置／样本选取合并。

| B / 前缀→总长 | 层数 | FP16 旧→新 | W8A16 旧→新 | W8 延迟减少 | W8 重算前缀 | CPU FP16 decode |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 32→33 | 1 | 4.399 → 3.967 | 3.955 → 3.266 | 17.4% | 3.822 | 1.385 |
| 4 / 63→64 | 1 | 14.907 → 9.357 | 14.970 → 8.636 | 42.3% | 9.917 | 2.110 |
| 1 / 32→33 | 2 | 8.291 → 7.677 | 8.477 → 6.723 | 20.7% | 7.766 | 2.395 |
| 4 / 63→64 | 2 | 30.158 → 18.614 | 29.698 → 17.139 | 42.3% | 18.511 | 4.084 |
| 1 / 32→33 | 4 | 17.388 → 15.722 | 17.087 → 14.202 | 16.9% | 16.977 | 5.664 |
| 4 / 63→64 | 4 | 63.536 → 37.569 | 63.012 → 37.986 | 39.7% | 42.025 | 7.566 |

W8A16 相比上一轮延迟减少 16.9–42.3%，
FP16 为 7.4–40.9%。本轮六组 W8A16 缓存 decode
均快于重算短前缀，延迟低 7.4–16.3%；
所有缓存路径仍慢于对应 CPU FP16 decode。这是本轮暖层段结果，不代表端到端 LLM 或更长上下文优势。

重算前缀／CPU 对照与全部原始 21 样本保存于 performance.json；速度结论以同进程配对结果为准。
Tensor 存储上限 1,048,576、计算分块 65,536、缓存 512 图未放宽。显式复制计数不包含驱动 DMA；
显式峰值内存不包含编译常量、CMX、驱动内部和 CPU 模型。保留多个快照会额外占用内存。
仍限冻结 FP16/W8A16、B<=8、总长<=64；尚未增加 W4A16、训练、padding、通用 generate 或更长上下文。
混合生成保留 CPU embedding/final norm/logits、剩余层及段间传输，未实现 GPU/NPU 并行。

## 复现与来源

模型 revision：12fd25f77366fa6b3b4b768ec3050bf629380bac；safetensors SHA256：5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c。
离线加载，不执行远程模型代码，不导入／安装 OpenVINO runtime；仍需驱动图编译器。
无需重建原生扩展或 SHAVE 内核。本轮 before/after、源码 diff、JSON/log/NPZ、summary.json 保留来源及原始数据。

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','paired'
# 依次运行 probe / numeric / boundary / guards / positions / hybrid / regression / performance。
# 所有 NPU 测试必须串行；before/ 用于旧版对照。
& .\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_smollm_fused.py
```
