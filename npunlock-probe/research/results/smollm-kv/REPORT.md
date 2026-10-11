# SmolLM2-135M：NPU 常驻 KV cache

2026-10-05，在本机 Lunar Lake NPU（8086:643e，驱动 1005540、编译器 8.3）
完成单层及 2／4 层 FP16／W8A16 的有界 prefill/decode。K/V、RoPE 位置选择、
槽位更新、causal attention、RMSNorm 与 FFN 在 NPU 运行。暖 decode 区间没有显式
主机上传／下载或按位置重新编译；层间激活仍常驻。权重读取、编码与常量准备在 CPU。
没有新增 OpenVINO runtime 或第三方运行依赖，基础设备后端／原生 DLL／RMSNorm 内核未改动。

## 范围与 API

```python
with torch.inference_mode():
    output, cache = stack.prefill(prefix_hidden_states.to("intel_npu"))
    output, cache = stack.decode(next_hidden_states.to("intel_npu"), cache)
```

输入为层段入口的 FP16 隐藏状态，输出为层段末端隐藏状态。Prefill 从位置 0 开始，
decode 仅接收 `(B,1,576)`，B<=8，总长度<=64。每层 K/V 固定 `(B,3,64,64)`，
不会为重复的 9 个 query head 保存三倍缓存；注意力内部仍展开至 9 head。
返回新快照，旧缓存不会被覆盖，可重复使用或分支；调用方不得修改 state Tensor。
`forward` 保留原来的整段重算 API。多层会先检查全部缓存，再执行第一层。

仍为特定 SmolLM2 冻结推理适配器：没有 padding／任意位置／训练／W4A16，
没有普通 Transformers `generate()` 自动接入，也没有验证完整 30 层全 NPU 生成。
FP16 与 W8A16 投影继续经 DPU 检查；RMSNorm 使用真正 FP32 SHAVE 后按官方顺序舍入。
设备上的 FP32 位置值用于选择 RoPE 和缓存槽位，不改变矩乘精度模式。

## 检查结果

最终 8 个独立进程全部通过，累计 32,739 次 NPU 提交，
每个进程 shutdown 后 live_buffers=0，context_failed=false。

| case | 检查记录 | 计时组 | NPU 提交 | 最终 live buffers |
|---|---:|---:|---:|---:|
| probe | 8 | 0 | 220 | 0 |
| numeric | 40 | 0 | 4320 | 0 |
| boundary | 14 | 0 | 3070 | 0 |
| guards | 23 | 0 | 203 | 0 |
| positions | 44 | 0 | 5190 | 0 |
| hybrid | 8 | 0 | 1560 | 0 |
| performance | 0 | 6 | 18088 | 0 |
| regression | 8 | 0 | 88 | 0 |

共 102 条数值记录、336 条固定实际输入的 K/V 检查。
覆盖层段起点 0／10／26、1／2／4 层，短前缀到 63→64 边界、完整 64-token prefill，
以及 B2／B8 中各不相同的样本。连续位置测试为 1／4 层 × 两种精度，各追加 63 步，
总计 252 步；每一步均无新增编译或显式主机复制，10 个位置抽查数值。
旧快照、已有有效 K/V 逐位不变，未使用槽位为零，重复同一解码输出逐位一致。
错拥有者／层顺序／缓存类型／batch／dtype／存储／长度、混合前缀或解码分支、
满缓存、训练模式与 active autograd 均在执行前拒绝；inference_mode 正向执行通过。
普通 forward 与改动前版本的 8 组回归输出逐位相同，包括深层及 B8/S64。

执行门槛继续为 L2<0.3%、逐元素 `abs(error)/(abs(reference)+token_RMS)<1%`。
最大输出执行 L2 0.2003%，最大归一化误差
0.6472%；相对官方 CPU 浮点层最大 L2
2.7445%，通过已有 3% 质量门槛。
固定实际输入的 K/V 最大 L2 0.0323%，
最大归一化误差 0.1356%，通过相同执行门槛。
K/V 的归一化分母按每个 head 的 64 个元素计算 RMS；层输出按每个 token 的 576 个元素计算。
旧 `rtol=.01, atol=.005` 失败计数继续保留，不宣称逐位等价或通过旧固定绝对门槛。

## 失败与诊断

`failures/` 保留原始 JSON/log，源码版本在对应失败目录或 before 快照中保存：

- initial-scalar-upload：首次 decode 上传了 4-byte 步长常量；已移到 prefill 准备。
- semantic-token-scope：旧 CPU staged／official 0.1% 门槛原本比较整个前缀，
  首次 runner 误用于单 token。现保持原比较范围，单 token 误差另记；
  每个真实 decode 输出仍独立接受 0.3%／1% 执行检查，质量门槛仍为 3%。
- batched-decode／batched-attention-unroll：编译器将批量动态权重矩乘降低为卷积，
  `ConvertBatchedLayerTo1N` 转换失败；同图逐样本展开也失败。已采用设备样本切分，
  B1 注意力计算和设备拼接。没有 CPU 算术兜底，但提交数和 DMA 增多。
- distinct-batch-kv-accumulation：不同输入 B8、4 层 FP16 的 V 相对整段 CPU 参考，
  最大归一化误差为 1.1559%，L2 为 0.1440%。固定该层实际 NPU 输入后，
  L2 为 0.0017%、最大归一化误差为 0.0565%；整个输出通过原门槛。
  原差异来自前层舍入累积。最终同时记录 accumulated `cache_reference` 诊断，
  并对固定实际输入的 `cache_local_reference` 做数值检查，未修改生产算术来掩盖该差异。
- regression-fixture-path：旧源码快照在 before/ 下，首次加载时找不到相对路径的内核。
  回归 loader 将模块文件位置重设到项目根目录，旧算术源码及默认内核均未改动。

`local-cache-probe.json` 保留局部诊断，`before-distinct-batch/` 保留同输入 batch 的早期成功，
`before-local-cache-check/` 保存补充局部检查前的成功数值。首轮静态图原型另存
`static-probe.json`，上述探索记录不计入最终 8 进程统计。

## 实际混合生成

两条提示 × 2／4 层 × FP16／W8A16，共 8 组显式混合生成。NPU 层段使用本次 KV，
剩余 CPU 层使用标准 Transformers DynamicCache，embedding／final norm／logits 在 CPU。
每步层段入口上传、出口下载仍存在。8 组 token 全部与 CPU 参考一致：
True；最大首步 logits L2
0.5888%，最大 KL 0.002058。
这是短生成 smoke test，不是广泛质量评估。hybrid 日志的冷路径耗时包含首次编译、
准备和输入上传，不能与下表的暖运行直接比较。

## 同进程暖运行

CPU 8 线程，21 轮交错计时、轮换顺序。一次 decode 与重算相同完整前缀的最后一个
token 做对照。编译、准备、初始预热、输入上传／输出下载、CPU mask／RoPE 准备排除；
NPU 包含 Python、分配、设备 DMA、提交和同步 fence，不是纯内核时间。
CPU 是官方 FP16 浮点层，CPU decode 为重复同一 token 会在每次调用后 crop 回前缀，
crop 成本包含在计时中。没有把此 CPU FP16 对照当作 CPU 最优实现或 FP32 对照。

单位 ms，中位数；各“decode / full”分别为缓存解码与重算完整前缀。

| B / 前缀→总长 | 层数 | NPU FP16 decode / full | NPU W8A16 decode / full | W8 full/decode | CPU FP16 decode / full |
|---|---:|---:|---:|---:|---:|
| 1 / 32→33 | 1 | 5.455 / 3.730 | 5.014 / 3.693 | 0.74× | 1.357 / 2.009 |
| 4 / 63→64 | 1 | 18.883 / 9.881 | 18.237 / 10.011 | 0.55× | 2.337 / 6.791 |
| 1 / 32→33 | 2 | 10.508 / 7.859 | 10.011 / 7.683 | 0.77× | 2.411 / 4.491 |
| 4 / 63→64 | 2 | 37.471 / 19.360 | 37.194 / 18.895 | 0.51× | 3.911 / 13.418 |
| 1 / 32→33 | 4 | 20.824 / 16.046 | 19.537 / 15.524 | 0.79× | 4.866 / 7.329 |
| 4 / 63→64 | 4 | 77.626 / 45.239 | 77.067 / 40.673 | 0.53× | 7.110 / 26.287 |

full/decode 大于 1 才表示缓存解码更快。原始 21 样本、实际提交次数与复制／编译
计数保存在 performance.json。B1 每层 decode 8 次提交，B2 为 23，B4 为 39，
B8 为 73（后者含超过计算分块大小的拼接）。普通 forward 仍为每层 4 次。
本轮全部 6 组 W8A16 缓存解码都比重算前缀更慢，耗时约为重算的
1.26–1.97 倍，均慢于相应 CPU FP16 缓存参考。
固定 64 槽位的缓存拷贝及批量切分可能抵消省下的矩乘，不能仅凭实现 KV 就宣称加速。
桌面后台负载、功耗与频率未控制，毫秒差异应按本轮原始样本理解。

## 内存与下一步

缓存本体为 `B * 层数 * 2 * 3 * 64 * 64 * 2` bytes：B1／4 层为 192 KiB，
B8／4 层为 1.5 MiB。每步返回新快照，因此保留旧快照会额外占用内存。
显式共享 Tensor 峰值排除编译常量、CMX、驱动内部与 CPU 模型，不能推断 NPU 总内存。
1,048,576 存储元素、65,536 计算分块及最多 512 张缓存图的限制均未放宽。

功能路径已经有真实缓存与连续解码。根据本轮计时，下一步优先减少单输出打包／拆分
及批量样本切分，再在这个基础上验证 W4A16 权重，按测量决定是否扩展更长上下文；
不要把 64-token 的小模型结果外推为大模型吞吐优势。

## 复现与来源

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','probe'
# 依次运行 numeric / boundary / guards / positions / hybrid / performance / regression。
# 不要同时运行多个 NPU 测试进程。
& .\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_smollm_kv.py
```

`summary.json`、各 case JSON/log、NPZ、before/after 源码快照及 adapter.diff 保留证据。
最终 8 个 case 的生产哈希一致。runner 后期只补充旧快照的路径重定位，
修改前 runner 在 before-regression-path/ 保存；每个 case 的 runner 哈希均可追溯。
检查点为 HuggingFaceTB/SmolLM2-135M-Instruct，固定 revision
12fd25f77366fa6b3b4b768ec3050bf629380bac，safetensors SHA256：
5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c。全程离线，不执行远程模型代码。
