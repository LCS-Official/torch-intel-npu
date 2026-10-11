# NPU KV cache：原生多输出

2026-10-05，Lunar Lake 8086:643e、驱动 1005540、编译器 8.3。
底层 session DLL 已有多输出能力；本轮扩展 PyTorch C++ 接入层，复用其原生 infer 接口。
注意力输出、K/V 直接绑定三个独立设备分配，删除旧版 Concat 打包和后续 DMA 拆分。
无需新增依赖，session DLL、SHAVE RMSNorm 内核、权重量化与基础 FP32 算术保留原版本。

## 接入与提交

`stack.prefill(prefix)`／`stack.decode(one_token,cache)` API 不变。输入为 FP16 隐藏状态，
模式 FP16／W8A16，B<=8、总长<=64、从零开始的 causal 前缀。K/V 按三个 GQA head
留在设备上，固定 64 槽位。旧快照不被覆盖，调用方不得编辑 state Tensor。
多输出要求完整、连续、独立的分配，不得和输入或其他输出共享存储。

C++ `run_many` 将多个 Tensor 绑定到一次原生 infer；旧 `run` 保留签名并委托该执行器。
同步 fence、异常及失败隔离策略保留。Python `_run_many` 检查输入、输出个数／shape／dtype、
存储和别名。本轮没有启用异步、原地缓存更新或任意子缓冲区绑定。

| 每层暖调用 | 旧打包版 | 多输出版 |
|---|---:|---:|
| B1 decode | 8 | 5 |
| B2 decode | 23 | 17 |
| B4 decode | 39 | 27 |
| B8 decode | 73 | 49 |

B1 为两次 RMSNorm、一次注意力／K/V、一次 FFN、一次 FP32 位置增量。
B>1 仍通过设备样本切分、B1 注意力图和设备拼接绕过编译器对批量动态权重矩乘的限制；
减少的是各样本的输出拆分。B8 还包含超过 65,536 计算分块的拼接。
暖 decode 区间没有显式主机复制／按位置新增编译，计数器排除驱动 DMA。
普通 forward 仍每层 4 次提交。

## 验证

10 个独立进程通过，累计 42,497 次 NPU 提交；
各进程 shutdown 后 live_buffers=0，context_failed=false。

| case | 检查记录 | 计时组 | NPU 提交 | 最终 live buffers |
|---|---:|---:|---:|---:|
| probe | 8 | 0 | 154 | 0 |
| numeric | 40 | 0 | 3132 | 0 |
| boundary | 14 | 0 | 1966 | 0 |
| guards | 23 | 0 | 131 | 0 |
| positions | 44 | 0 | 3240 | 0 |
| hybrid | 8 | 0 | 984 | 0 |
| performance | 0 | 6 | 28588 | 0 |
| regression | 8 | 0 | 88 | 0 |
| bindings | 17 | 0 | 54 | 0 |
| device | 33 | 0 | 4160 | 0 |

102 条数值记录、336 条固定实际输入 K/V 检查，
覆盖起点 0／10／26、1／2／4 层、不同样本 B2／B8、63→64 边界与完整 64-token prefill。
单层／4 层 × 两种精度各追加 63 步，总计 252 步；
每步复用图且无显式主机复制，10 个位置抽查数值。旧快照和已有 K/V 逐位不变，
未用槽位为零，同输入／快照重复解码逐位一致。22 项模型缓存错误用法在执行前拒绝，
inference_mode 正向执行通过。

执行门槛未变：L2<0.3%，每项 `abs(error)/(abs(reference)+RMS)<1%`。
层输出按 token 的 576 项算 RMS，K/V 按 head 的 64 项算 RMS。
最大输出执行 L2 0.2003%，最大归一化误差
0.6472%；相对官方 CPU 浮点层最大 L2
2.7445%，通过已有 3% 质量门槛。
固定实际输入 K/V 最大 L2 0.0323%，
最大归一化误差 0.1356%。
整段 CPU 输入的累积舍入差异另记诊断，旧固定 atol 失败计数保留；比较口径见前轮报告。

通用绑定检查覆盖三个混合形状／FP16/FP32 输出；FP32 位模式包括 70000、1 ULP、
次正规数、负零和 infinity。此项为 DMA/IO 检查，不作为 FP32 矩乘能力证明。
50 次复用输出无新增分配／主机复制／编译，改变输入后观察到新值。13 项 Python／C++
错误绑定拒绝后可继续有效调用，旧单输出真正 FP32 add 通过。
设备 quick suite 覆盖 FP16/FP32 算术、转换、归约、矩阵、Linear、视图、标量缓存、
线程调用与 shutdown 后存储寿命；8 组普通 Smol forward 与改动前输出逐位一致。

8 组显式混合生成 token 全部与 CPU 参考一致：True。
前 2／4 层使用 NPU KV，其余层／embedding／final norm／logits 在 CPU，保留段外传输。
首步 logits 最大 L2 0.5888%，最大 KL
0.002058。这是短生成 smoke test，未验证全 30 层或广泛质量。

## 同进程性能

CPU 8 线程、21 轮交错计时、轮换顺序。旧打包适配器原样从 before/ 加载，仅重定位
内核路径；旧／新适配器使用同一个本轮 C++ 扩展，隔离输出图／拆分路径的改动。
编译、初始准备／预热、输入上传／输出下载和 CPU mask／RoPE 准备排除。
NPU 时间包含 Python、分配、提交、设备 DMA 和同步 fence，不是纯内核时间。
CPU FP16 为官方浮点参考，重复同一 decode 时 crop 成本包含，不代表 CPU 最优实现。

单位 ms，中位数；旧→新为缓存 decode 打包版→多输出版。

| B / 前缀→总长 | 层数 | FP16 旧→新 | W8A16 旧→新 | W8 延迟减少 | W8 重算前缀 | CPU FP16 decode |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 32→33 | 1 | 4.684 → 4.484 | 4.687 → 3.652 | 22.1% | 3.677 | 1.328 |
| 4 / 63→64 | 1 | 18.290 → 15.358 | 18.202 → 14.616 | 19.7% | 10.938 | 2.174 |
| 1 / 32→33 | 2 | 9.146 → 8.268 | 9.236 → 7.797 | 15.6% | 7.458 | 2.595 |
| 4 / 63→64 | 2 | 33.967 → 28.595 | 34.315 → 26.997 | 21.3% | 19.099 | 3.092 |
| 1 / 32→33 | 4 | 18.670 → 15.662 | 18.423 → 14.615 | 20.7% | 14.752 | 4.596 |
| 4 / 63→64 | 4 | 74.120 → 61.967 | 74.892 → 59.911 | 20.0% | 41.640 | 8.424 |

W8A16 六组延迟减少 15.6–22.1%，
FP16 为 4.3–16.4%。提交数减少不等于同等比例的延迟改善。
B1 W8 解码基本接近重算短前缀，尚不能认为建立了稳定 KV 加速优势；B4 仍慢于重算。
所有缓存路径仍慢于相应 CPU FP16 decode。桌面负载和频率未控制，小差异不宜过度解读。
全部原始 21 样本、各路径提交和零暖主机复制／新增编译记录均保存于 performance.json。

## 来源与后续

计时首次失败是 runner 把另一个模块定义的旧 SmolKVCache 当作层组；改为判断缓存是否
为 tuple 后通过。失败 JSON/log／runner 保存于 `failures/packed-cache-type/`，
不计入最终 10 进程统计。各 case 的生产哈希一致，runner 两版本均有快照和按 case 哈希。

K/V 本体 B1／4 层 192 KiB，B8／4 层 1.5 MiB；保留快照会额外占用内存。
显式 Tensor 峰值排除编译常量、CMX、驱动内部和 CPU 模型。Tensor 1,048,576 元素、
计算分块 65,536 和缓存 512 图的上限未放宽。未增加 W4A16、训练、padding、通用 generate、
任意位置／更长上下文或 TF32 路径。下一步可先消除独立位置更新小图，减少 batch
切分／拼接，再推进 W4A16；长上下文收益需另测。

本轮只重建 `_torch_intel_npu.pyd`；session DLL、RMSNorm ELF 和检查点哈希与前轮一致。
模型 revision：12fd25f77366fa6b3b4b768ec3050bf629380bac；safetensors SHA256：
5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c。离线加载，不执行远程模型代码，不安装／导入 OpenVINO runtime。

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_multi_output_demo.py
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','probe'
# 依次运行 numeric / boundary / guards / positions / hybrid / performance / regression。
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_multi_device_probe.py
# NPU 测试必须串行；before/ 快照用于旧版性能和普通 forward 对照。
& .\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_smollm_multi.py
```

before/after、源码 diff、build.json、各 JSON/log/NPZ 与 summary.json 保存来源及原始数据。
前轮报告和记录保留于 `smollm-kv/`。
