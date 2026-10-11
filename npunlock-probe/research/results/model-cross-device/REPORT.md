# SmolLM2 CPU / NPU / Arc GPU 模型基准

2026-10-10，本机 Core Ultra 7 258V，Arc 140V，Intel AI Boost NPU 8086:643e。
这是一轮当前软件实现的推理基准，用于决定后续优化方向，不是硬件峰值测试。
没有修改生产内核、模型图或默认精度。

## 相同的工作负载

- 使用本地 `SmolLM2-135M-Instruct` 完整 30 层、相同 checkpoint、FP16 权重/激活。
  checkpoint SHA256 为 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。
- batch=1，24-token 提示词，greedy 连续生成 32 token，不提前遇 EOS 停止。
  共处理 55 个输入位置，生成后序列长 56，符合当前 NPU 的 64-position 边界。
- CPU / GPU 使用 Transformers 5.18.0 的 Llama eager attention、动态 KV cache。
  NPU 使用当前 `FrozenSmolStack` 的完整 30 层图、固定 64-position KV cache、group4 decode。
- CPU / GPU 的完整模型分别在对应设备。NPU 的 embedding、final RMSNorm、lm_head、argmax
  在 CPU，30 层 decoder 与全部层的 KV 在 NPU；没有 CPU decoder fallback。
- 三种设备各在独立进程串行测试。CPU 测 1/8 线程；GPU 环境也运行同样的 CPU 对照。
  native 环境为 Python 3.12.14 / Torch 2.14.1+cpu，XPU 环境为 Python 3.11.15 / Torch 2.12.0+xpu。
  两种环境的差异保留在原始报告中，不把它单独归因于 Torch 版本。
- GPU 缺少的 Transformers 依赖装在项目的 `research/model-benchmark-deps`，不替换已有 GPU Torch。
  Transformers / safetensors / tokenizers / huggingface-hub 与 NPU 环境同版本；NumPy 分别为 2.5.3 / 2.4.6。

## 计时方法

每个进程记录第一次完整生成，随后额外预热两次，再测七次完整生成。
热速度包含 embedding、CPU/NPU 传输、decoder、输出头、argmax、token 返回主机和循环维护。
GPU 每步在 token 返回主机后同步；异步提交不能作为完成时间。
CPU/GPU 和 NPU 的输出头均计算 prefill 的完整位置，随后只选最后位置 logits，
没有单方面应用只算最后位置输出头的优化。

进程启动、库导入、模型加载/初始设备迁移、准确性检查、保存证据、最终 KV 对象释放和
NPU shutdown 均在热计时之外。模型准备、backend 初始化和冷调用另存。
GPU 的最终全量 logits 下载用于校验，在计时之外；实际生成的 token 每步已在计时中返回主机。

表中每列为七轮独立中位数，不应把不同列的中位数相加。
总速度为 `32 / 完整生成秒数`，decode 速度为 `31 / 首步之后耗时秒数`。

## 热结果

| 路线 | Torch | CPU 线程 | 首 token ms | 32 token 总 ms | 总 token/s | decode token/s |
|---|---|---:|---:|---:|---:|---:|
| CPU native 配套 | 2.14.1+cpu | 1 | 118.10 | 1549.54 | 20.65 | 21.80 |
| CPU native 配套 | 2.14.1+cpu | 8 | 61.46 | 955.29 | 33.50 | 34.29 |
| CPU XPU 配套 | 2.12.0+xpu | 1 | 127.05 | 1627.22 | 19.67 | 20.74 |
| CPU XPU 配套 | 2.12.0+xpu | 8 | 62.62 | 1005.80 | 31.82 | 32.87 |
| NPU decoder + CPU 两端 | 2.14.1+cpu | 8 | 23.71 | 466.80 | 68.55 | 69.97 |
| GPU 全模型 eager | 2.12.0+xpu | 8 | 42.34 | 1288.76 | 24.83 | 25.06 |

总生成耗时 Q1–Q3：CPU native 8线程 940.51–998.54 ms，CPU XPU 配套8线程
991.63–1055.65 ms，NPU 462.58–472.50 ms，GPU 1259.15–1373.96 ms。
GPU 七个样本的速度从约 22.47 增至 25.85 token/s，有趋势和桌面波动，不能用这组数据认证微小收益。
七轮原始数据、每步时间和各项分位数保存在六个 JSON 文件中。

NPU 本轮总速度约为 native 配套 CPU 8线程的 2.05 倍、GPU eager 的 2.76 倍。
这里比较的是已经融合的 NPU 图与 CPU/GPU eager 实现，不能据此判断三种硬件的最终排名。

## 冷启动与运行计数

NPU 第一次完整生成 **68.124 秒**，其中首次 prefill 含编译约 26.499 秒，
首次 decode 含编译约 41.147 秒；模型准备另计约 0.894 秒。
GPU 首次完整生成约 3.065 秒，首次 token 约 1.649 秒，模型准备另计约 0.940 秒。
native CPU 8线程首次完整生成约 1.025 秒，模型准备约 0.515 秒。
这些冷调用只测一次，包含首次初始化和执行，不能解释为纯编译耗时。
热 token/s 不代表启动一次仅生成几个 token 的体验。

每次热 NPU 生成均为 960 次 layer graph 执行、930 个 chain graph、248 个 chain group，
显式上传/下载各 32 次，新增图编译和驱动分配均为零。
本轮 NPU 总计 10,680 次执行；结束时 live/cache/quarantine 分配与 session 全部归零，无 context failure。
NPU decoder 总时间中位数为 387.79 ms；按每轮的比例再取中位数，约占完整生成的 **83.36%**。
剩余时间包括 CPU 两端、复制和 Python 管理，不能把整个 decoder 时间解释成纯硬件计算。

GPU Torch allocator 的峰值 allocated 为约 269.82 MiB。
NPU allocator 统计不包括驱动管理的编译图常量，因此没有给出跨设备内存效率排名。
统一物理内存也不会消除驱动内部 DMA 或显式数据转换。

## 准确性

先用 native CPU FP16 建立共同的 32-step logits / token 参考。
每条路线在完整生成之外重放相同的固定 token，避免轨迹变化干扰 logits 比较。

| 路线 | 32步中最大 logits relative L2 | 固定回放 argmax 相同 | 实际 greedy 32 token 相同 |
|---|---:|---:|---|
| 四种 CPU 配置 | 0 | 32/32 | 是 |
| NPU | 1.1196% | 32/32 | 是 |
| GPU | 0.8359% | 32/32 | 是 |

全部路线通过现有 3% 相对 L2 门槛；每条路线的两次额外预热与七次热生成，
全部 logits / token 均与自身第一次生成逐位相同。
这只是当前一个提示词的数值和输出验证，不是完整语言模型质量评测。
W8A16 的既有质量限制没有因本轮 FP16 结果改变。

最早一次 CPU 1线程试跑在计时后调用 NumPy BLAS 计算误差时触发第二套 OpenMP runtime。
已将误差计算改为普通逐元素求和，未启用 `KMP_DUPLICATE_LIB_OK`，重新生成参考并重跑正式 CPU 数据。
失败时的原始 runner、参考和不完整报告保留在 `pilot/openmp`，不计入正式结果。

## 结论与下一步

当前短上下文 FP16 推理路线中，NPU 具备实际的热延迟优势；优先建立更强的模型对照，
比继续只看单算子变化更有价值。

1. 补测 CPU FP32/BF16 和 GPU SDPA 等实际可用的快路径，保留当前同精度 eager 基线。
   本轮未测 torch.compile，不能把 GPU eager 当作已优化 GPU 上限。
2. 扩展到 batch4/8、更多提示词和不同 prefill 长度，再判断吞吐/延迟的适用范围。
3. 把首次编译和模型形状变化的等待纳入下一轮；本轮 68秒冷调用远高于热生成，
   应评估编译结果缓存和形状复用，而不只缩短热内核时间。
4. NPU 内部继续以剩余 RMSNorm/SHAVE 热点为候选；已有剖析支持这一方向，
   本轮的 decoder 占比也说明模型仍主要花时间在 NPU 路径，尚不需要优先转向通用 FP32 GEMM。
5. 训练另用已有小 ResNet / Conv FP32 工作负载测 forward/backward/optimizer；
   `FrozenSmolStack` 仍是冻结推理路径，本轮不能证明完整 LLM 训练性能。

测试后快照为 AC 供电、Balanced 电源方案，NPU 驱动 32.0.100.5540。
未锁定频率、温度和用户后台负载；未测功耗/能效、长上下文、大 LLM 或 GPU+NPU 同时运行。

## 复现与证据

已有 GPU Python 不必更换。若项目内模型依赖尚未安装：

```powershell
$gpuPython = 'C:\Users\LC\miniconda3\envs\blood\python.exe'
& $gpuPython -m pip install --no-cache-dir --target .\npunlock-probe\research\model-benchmark-deps -r .\npunlock-probe\research\results\model-cross-device\gpu-model-requirements.txt
```

完整串行测试：

```powershell
.\npunlock-probe\research\run_model_cross_device.ps1
# 只重新核验证据；不执行模型
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\model_cross_device.py -ScriptArguments '--device','audit'
```

完整测试会重写本目录的同名当前结果，保留历史结果时应先归档。
复现入口位于 `research`；本目录 `runner.py` 及 PS1 是证据副本，Python 副本不从结果目录直接执行。
`summary.json` / `summary.csv` 为汇总，六组 JSON 保存原始计时、版本、源码/权重/参考指纹，
`*-outputs.npz` 保存全部 replay / greedy logits，`hardware.json` 保存测试后的硬件快照。
离线 audit 已重新计算误差、统计与 SHA256，通过全部六组，质量门槛总状态为 true。
新增模型依赖约 207.6 MiB，本目录含失败证据约 46.3 MiB，没有复制 checkpoint 或生成大型 graph blob。
