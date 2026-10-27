# 较大 Attention 模型持续训练与真实 NPU 忙碌率

日期：2026-10-18。本地开发版。Core Ultra 7 258V / Intel AI Boost 8086:643e / driver API 1005540 / 图编译器8.3 / Python 3.12.14 / Torch 2.14.1+cpu。

标准PyTorch Module、SDPA、CrossEntropy和SGD，设备intel_npu:0。每层为独立Q/K/V、因果注意力、输出投影、两次LayerNorm、GELU前馈与残差，最后mean pooling及四分类头。固定合成数据用于负载与数值稳定测试，不代表真实数据集质量或LLM训练。

## 实测

| 配置 | 参数 | 稳态秒数 | 步数 | 平均计算秒/步 | NPU进程忙碌率 | 稳态loss |
|---|---:|---:|---:|---:|---:|---|
| calibrate-64 | 200,196 | 13.792 | 2 | 6.895 | 88.61% | 1.149852 → 1.062795 |
| power-load-128 | 793,604 | 60.363 | 2 | 30.181 | 92.87% | 1.027548 → 0.869179 |
| sustained-128 | 793,604 | 182.227 | 6 | 30.370 | 91.67% | 1.027548 → 0.460670 |

### sustained-128

- 4层、hidden=128、4头、FFN=512；batch=4、sequence=16、float32，793,604参数。
- 稳态训练182.227秒，6步；0.0329步/秒，2.107序列位置/秒。包含反向及SGD更新，非LLM生成tokens/s。
- NPU进程/整机引擎平均忙碌率：91.67% / 91.67%；约1秒区间min/median/max：55.56% / 92.80% / 94.84%；>=95%的采样区间合计0.00秒。
- 180份独立定时遥测；进程CPU时间13.906秒，约7.63%单核当量。
- 本次冷启动首步125.379秒，第二次热身30.201秒；两者不计入上述稳态时长。
- loss（含热身）1.46401882 → 0.46067005；最终全部参数和梯度有限。
- 稳态NPU调用70,758次，11793次/步；会话固定279个，稳态新建图0。驱动allocation仍增加972次：块缓存达到256块上限，存在分配/回收周转，但live与缓存字节没有增长。
- 参数3.027MiB；步末设备live 6.086MiB且无漂移，张量分配峰值7.224MiB，步末缓存3.367MiB。这些数值不含驱动内部图/临时区。
- 进程RSS开始/结束415.80/416.31MiB；本进程RSS峰值417.44MiB。
- Windows DXCore进程共享committed步末43.020MiB，进程resident字段为0字节、采样resident峰值0.000MiB；整NPU共享resident步末43.090MiB，dedicated 0字节。进程resident读数持续0，不能据此说未使用NPU内存；用committed与整适配器resident交叉核对。
- 稳态每步仅显式读取一次4字节分类错误标志，并额外下载4字节loss用于健康检查；输入、参数与训练算术留在NPU。计数不包含驱动内部DMA。

## 校验与解释

- 每个配置先做两步独立CPU训练，逐项核对loss、全部参数梯度和更新；CPU单线程、禁用MKLDNN、SDPA MATH，随机种子103。FP32 rtol=3e-5/atol=3e-6（FP16入口为.003/.0005）。没有放宽原有容差。
- 热身后连续执行训练循环，无人工sleep、无CPU算术回退；稳态结束才检查最终全部参数/梯度。所有运行关闭后live/cache/scalar/quarantine/session为0，无失败context。
- DXCore按vendor/device选出AI Boost，读取专属LUID及引擎0:0，分别统计进程和整机累计运行微秒差/墙钟差。核显Arc 140V的Neural引擎被排除。监控线程约每秒采样；端点差得到全程平均，分段峰值不当成全程平均。
- 该指标表示Windows调度引擎的忙碌比例，含图内DMA/布局和SHAVE工作，不代表DPU阵列、全部SHAVE流水线利用率、40 TOPS达成率或实际TFLOPS。当前训练路径仍是SHAVE FP32数学实现。
- 驱动不支持DXCore温度或频率查询，三分钟原测试未测功耗/温度、不能据此断言没有降频；后续用户授权的独立功耗采集另列。
- 大量展开乘加、分块、布局搬运及逐图同步仍限制吞吐。高引擎忙碌率本身不能证明当前算子高效；后续优先专用矩乘/批量矩乘和减少中间Tensor/提交次数，同时处理256块缓存上限造成的allocation周转。
- 新增离线自检覆盖DXCore ABI、进程/整机占用分离、无效进程计数及计数重置。生产Python后端、C++/PYD和SHAVE ELF未改。

## 内存读数与短测误判

79万参数模型FP32权重约3.03MiB，梯度同量级；逐算子工作区复用，不预占17.9GB。任务管理器以GB显示一位小数时，几十MiB以内可能显示0.0。引擎持续高忙碌与小内存占用不矛盾；也不通过多分配闲置内存制造压力。DXCore的驱动共享resident、后端张量分配和进程RSS是不同口径，不相加当作显存。

初次width128短测只设10秒，而一步约30秒，只完成一次稳态更新。旧脚本将这一步loss与自身比较，误报“Loss did not decrease”；两步CPU对照均通过、loss实际继续下降、资源正常关闭，但该短测不计为通过。已修正为与最后热身loss比较，随后完整三分钟运行使用新脚本通过；没有追改旧结果为成功。

## NPU独立功耗实测（2026-10-19）

经用户授权使用Windows UAC，Intel Unified Telemetry v0.2.0-beta1.1 + SoCWatch；ut.exe与socwatchdrv.sys签名均为有效Intel Corporation。仅开启npu-pwr/pkg-pwr；这是独立PMT NPU能量计，不是从整包功耗相减估算。

| 阶段 | 功率域 | 完整采样数 | 采样秒数 | 平均W | 区间最小/中位/最大W |
|---|---|---:|---:|---:|---|
| 空闲 | PMT-NPU-PWR | 23 | 20.707 | 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| 空闲 | PKG-PWR | 23 | 20.723 | 9.0646 | 5.2450 / 8.3643 / 16.6596 |
| 相同模型稳态训练 | PMT-NPU-PWR | 65 | 58.594 | 0.5251 | 0.4779 / 0.5198 / 0.6563 |
| 相同模型稳态训练 | PKG-PWR | 65 | 58.737 | 9.8173 | 8.3262 / 9.4656 / 20.2445 |

- 功耗重跑相同793,604参数FP32网络，热身后60.363秒、2步，NPU平均忙碌92.87%；数值/生命周期检查通过。原三分钟测试仍单独保留，不把本次采样冒充当时功耗。
- 工具输出能量mJ与Duration纳秒，按 `W = mJ × 10^6 / duration_ns` 换算，时间加权平均为总能量/总采样时间；未将mJ直接当成W。空闲0表示工具在当前分辨率下报告0，并非证明物理漏电绝对为0。
- 单调时钟/UTC两端偏移漂移0.5320ms；仅选完整落在稳态窗口内的样本，并保守剔除边界窗口。原始约1秒采样，区间峰值不能捕捉更短瞬时峰值。
- 整包读数包含CPU、其他片上域和后台工作；不等于NPU功耗，也未将各功率域相加。高调度忙碌率下的功耗仍只代表当前SHAVE实现，不能外推DPU满吞吐功耗或能效。
- 工具正常结束后再次检查socwatchdrv服务与UT进程均不存在，临时驱动由工具自动卸载，未手工修改系统NPU驱动。官方工具只用于外部监控，没有加入后端运行依赖；下载件、解包目录和原始采集数据在审计后清理。
- 功耗较低本身不能证明任务能效领先；该训练仍约30秒/步，尚未对相同任务的CPU/GPU能量或外置功率计做对照。

复现功耗采集：从下方Intel页面下载便携工具，在管理员PowerShell执行（ToolRoot指向解包根目录；运行前可加-PrepareOnly审阅）：

```powershell
.\npunlock-probe\research\measure_npu_power.ps1 -ToolRoot <Intel-UT-root> -Seconds 260 -Label load
```

原始日志的稳态筛选与单位换算实现见`npu_power_report.py`。本轮官方下载件指纹：

- `ut-power-check-20261018.zip` SHA256 `f78151fffa89a05b7551da3cbfaf53fcc22d6fffdb97605af00706d6393b6da1`
- `ut.exe` SHA256 `47ff26a6e509ff753a7049d39aca35b4d4b78172290cbeef2c37eb26d2e7ddce`
- `socwatchdrv.sys` SHA256 `41e964516f39198f44f448fe820e04918cc6176213413c5bcbd4efc2b9e55627`

[Intel Unified Telemetry功能与Lunar Lake支持说明](https://www.intel.com/content/www/us/en/developer/tools/intel-unified-telemetry.html)

## 复现

```powershell
$env:NPUNLOCK_GRAPH_CACHE=$null
.\run_torch_npu.ps1 -Script .\torch_npu_sustained_train.py -ScriptArguments @('--width','128','--blocks','4','--seconds','180','--label','sustained-128')
.\.conda\npunlock\python.exe -B -m unittest test_torch_npu_sustained_train -v
.\.conda\npunlock\python.exe -B npunlock-probe/research/sustained_train_report.py
```

报告生成脚本须在原始JSON仍保留时执行；完成审计后按用户约定清理原始JSON，只保留脚本与报告。未修改source-release，未推送GitHub。

## 监控API参考

- [Microsoft DXCore设备枚举](https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-enum-adapters)
- [DXCoreAdapterState运行时间/温度/频率契约](https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterstate)
- [DXCoreAdapterProperty设备LUID/引擎属性](https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterproperty)

### 本轮实现指纹

- `torch_npu_sustained_train.py` SHA256 `f7ebf90d349eec365252a0e6d8d0a9a8554f513de4dc0e9dd2e486e773c09a33`
- `tools/npu_dxcore.py` SHA256 `aba7bd580529b78503641849054f2d0a8e84f3dfcc0838b55e3b8ccb0931a3b0`
- `torch_intel_npu.py` SHA256 `97b87be9ee0d8bc9e8632fb708e2963be94f45acd81ad4a2bc335a93d3f5d1c5`
- `torch_npu_autograd.py` SHA256 `76218869984dedfe75e4b4a4b75a54c43eb8b40814224dc3bdb28c77734a5470`

### 原始记录指纹（审计后清理）

- `calibrate-128.json` 18449字节，SHA256 `7c35a5d29c8076013f9c4ca12ca0f8a4d104e7adf8c86432acdac8e9a2bfc442`
- `calibrate-64.json` 13064字节，SHA256 `367c4e0ceadcd646a7355625f0b863cd88f8f24f81128691ee52221a2e9c6edf`
- `clock-after.json` 95字节，SHA256 `ea97a3ed07f37eb10479d16927433e42359905a9de22335a0480b97138bc40c3`
- `clock-before.json` 96字节，SHA256 `3eb587a50e1d48e78b6ee01723e176afee04d099eb0f27bf66a2a612b3a8f155`
- `driver-state.json` 134字节，SHA256 `97381a2b8d180f0e08bb25f5d93d66620f1b1c81df602dd156d0a5a8e370c6fc`
- `power-complete.json` 166字节，SHA256 `2073fb2eb1e49f9e1e510ec644af5cf9ae18324891b6d46ba9cf1b67f8efcbab`
- `power-config.json` 832字节，SHA256 `7eaea5ac333369ad277620c99177e9e6d69d34c9b37d8365e3bf4d799048a462`
- `power-output.txt` 270598字节，SHA256 `b9bf5e99625212e1c9adf34fcf7c17c332fc26666f20303ad6ab08df8eb8920c`
- `power-complete.json` 165字节，SHA256 `df301e56d27a37b6e1cbefc7735e5c74d1f82155062f37037da40b0324b56f3e`
- `power-config.json` 832字节，SHA256 `7eaea5ac333369ad277620c99177e9e6d69d34c9b37d8365e3bf4d799048a462`
- `power-output.txt` 22464字节，SHA256 `aba2ba7be6ff990ae409072b8fa0559ae80268c842138639d0a0d0401540ce83`
- `tool-metadata.json` 779字节，SHA256 `2059dc4335bc36a695ff40b28c66b0df60a48e1d8fbc185ecf5296e71602f778`
- `power-load-128.json` 53883字节，SHA256 `6c271ac753857e443817c593da061f94b30f93ff39614c2770fbfa78d57b182c`
- `sustained-128.json` 146767字节，SHA256 `ad735531337b0062f026736506a7749deebe0aab9115ae216bc3f1b042edde62`

## 清理与保留（2026-10-19）

清理本轮14份原始训练/功耗记录与临时Intel工具（含下载ZIP、解包内容及工具临时文件），共572个文件、59838403字节（57.066MiB）。删除前逐项校验路径、文件类型、哈希及非重解析点；删除前后313份源码、全部NPU4000 fixtures/内核、原生组件与历史报告哈希一致，随后补记本节。

六项离线检查通过；公开source-release保持干净，没有推送GitHub。临时socwatchdrv与UT进程在清理前已确认退出。仅保留源码、验证/换算脚本及报告，没有增加项目运行依赖。

