# NPU 冷首 token：方向诊断与缓存加载优化

2026-10-11，Core Ultra 7 258V / Intel AI Boost 8086:643e，驱动32.0.100.5540，图编译器8.3。
结论：优先缩短已有编译图缓存命中后的准备开销。首个已接入优化为免解压图缓存，默认仍关闭磁盘缓存。

## 计时定义与范围

- 同一SmolLM2-135M-Instruct、FP16、完整30层、batch1、24-token提示词，只测prefill到首token。
- 冷首token：模型加载完成后，从embedding开始到首token argmax可用，包括上传、图准备、NPU prefill、下载及CPU完整prefill输出头。
- 脚本入口到首token：包含Torch/Transformers导入、NPU初始化、模型和tokenizer加载；从Python脚本开始计时，不含PowerShell/解释器启动。
- 每个独立进程只有一次冷请求；额外预热2次再测7次热请求。NPU调用同步，未提前预热要测的图/session。
- 新进程测试没有清空Windows文件系统页缓存；属于进程冷启动。后台负载/频率未锁定。
- 上轮6.229秒是首次完整生成32 token，不能与本轮首token秒数直接比较。
- 临时hook记录包含式时间，各项可能嵌套，不能简单相加；所有模式采用相同hook。没有提前转移工作来美化首token。

## 缓存加载格式对照

两种格式来自相同的30份blob和manifest，重打包时逐项验证字节一致；缓存键、模型图和运行路径相同。
各取3个新进程，执行顺序为压缩1、免压缩1、压缩2、免压缩2、免压缩3、压缩3。

| 新进程 | 冷首token ms | 读缓存 ms | 脚本入口到首token ms | 热首token中位 ms |
|---|---:|---:|---:|---:|
| load-1 | 3036.135 | 1296.043 | 7674.601 | 27.125 |
| load-2 | 2620.930 | 947.413 | 7632.968 | 28.095 |
| load-3 | 2637.122 | 957.064 | 7358.667 | 27.471 |
| stored-1 | 1840.972 | 169.029 | 6572.535 | 27.224 |
| stored-2 | 1905.378 | 175.177 | 6593.897 | 27.655 |
| stored-3 | 1837.245 | 174.557 | 6504.337 | 28.123 |

冷首token中位数 **2.637→1.841秒**，降低约30.2%。
脚本入口到首token中位数 **7.633→6.573秒**，降低约13.9%。
这是小样本本机结果，不是跨负载速度保证；没有声称热内核或tokens/s提升。

| 冷prefill准备阶段 | 压缩缓存三次中位 ms | 免压缩缓存三次中位 ms |
|---|---:|---:|
| graph_build_serialize | 794.296 | 786.967 |
| cache_read | 957.064 | 174.557 |
| inspect_blob | 293.622 | 291.378 |
| session_create | 110.153 | 108.839 |
| norm_parameter | 106.034 | 112.948 |
| layer_run_many | 178.288 | 177.144 |

## 正式接入与恢复验收

生产代码仅将缓存写入从`ZIP_DEFLATED(level=1)`改成`ZIP_STORED`。AST还原检查确认无其他计算逻辑改动。
图/权重/内核/驱动/编译器身份检查、SHA256、大小检查、原子写入、容量限制和失败回退全部保留。
读取器仍能读取压缩格式；因缓存键保守包含实现源码哈希，本次代码更新会使旧键失效，首次需重建。

| 正式生产入口 | 冷首token ms | 脚本入口到首token ms | 命中 | 融合图编译 | 写入 |
|---|---:|---:|---:|---:|---:|
| promoted-fill | 26828.107 | 31849.417 | 0 | 30 | 30 |
| promoted-load-1 | 2350.983 | 7747.161 | 30 | 0 | 0 |
| promoted-load-2 | 2079.275 | 7150.436 | 30 | 0 | 0 |
| promoted-corrupt | 2757.322 | 7414.965 | 29 | 1 | 1 |

首次写缓存累计耗时由旧格式试跑的4.738秒降到正式入口的0.200秒；两次首次编译未配对，不把整个调用差值都归因于格式。
损坏一个临时条目后29命中、1拒绝、1重编译并恢复，无设备故障重试。
19项缓存保护/格式兼容检查通过；11个进程全部首token logits和prefill hidden逐位一致，
与完整CPU FP16前向的logits相对L2为0.4840%，argmax一致，仍使用3%门槛。
每进程7次热请求均与本进程冷结果逐位一致，最终live/cache/quarantine/session归零，无context failure。

## 空间代价与使用

30份prefill图：旧格式160.75MiB，新格式216.88MiB，增加约56.13MiB。
总缓存上限仍512MiB，保留256MiB磁盘余量；关闭缓存不新增这些文件。一个模型的不同batch/prefix形状不保证都能放入上限。
本轮只生成30份prefill图，未测decode完整生成的正式新格式缓存容量。上一轮60图原生blob总量约438MiB，仅作容量参考。
**按用户约定，本轮验收后清理所有临时数组、计时JSON和图缓存，仅保留源码和报告。**
实际使用时要保留缓存才能获得下次新进程的命中速度；每次删除缓存会再次编译。本轮正式首次编译首token仍需约27秒。

```powershell
$env:NPUNLOCK_GRAPH_CACHE=Join-Path $env:LOCALAPPDATA 'torch-intel-npu\smollm-graph-cache'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\startup_first_token.py -ScriptArguments '--label','first'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\startup_first_token.py -ScriptArguments '--label','next-process'
Remove-Item Env:NPUNLOCK_GRAPH_CACHE
```

测试完成、报告生成后，使用`research/clean_experiment_artifacts.ps1 -Apply`清理可重建实验产物和默认缓存目录。

## 接下来的优化顺序

1. 缓存命中时仍有约0.8秒的图构建/权重序列化：研究复用经过严格身份校验的I/O合同，跳过重建完整IR；需防止权重/形状变更误命中。
2. 减少启动依赖导入和不需要的CPU decoder权重加载。当前仅用CPU embedding/norm/head，但参考驱动仍加载完整HF模型。
3. 再优化blob检查重复解析及常量gamma准备；保留矩阵精度证明和异常保护。
4. 热首token/prefill计算与batch8重复分配另开受控对照；它们不解释本轮秒级冷启动等待。
尚未实现1–4，未测试更多提示词/上下文、功耗、完整32-token新格式生成或LLM训练。
首次试跑库导入21.56秒，后续约3–4秒；系统页缓存及后台原因未独立确认，保留该异常，不把它算作NPU内核改进。

生产缓存源码SHA256：`e0a0c19dd5baf94ba85a4cb63a051e2d4edf0c1a335d77c00fc4fa2938ec467e`。
模型实现SHA256未变：`026d604effcdd55d3682e1a76acd773df8fb01d563edc882efb68e3d3a25f00e`。
原始数据在清理前已完成上述离线验证；清理后历史audit不能原样复算，复测需重新生成数据。
