# S59：结束单层输出因子校准分支

2026-10-30，Windows / Core Ultra 7 258V / NPU4000 8086:643e，驱动32.0.100.5540、编译器8.3。

按用户要求，本轮只做最后一次有界的实际NPU主投影拟合对照。直接采集原48条轨迹的实际gated、reduced和main，验证拟合输入逐项及顺序一致，再复用S58原固定ridge／alpha=0.5方法，仅改变末层down的576×32输出矩阵。没有新学习率、rank或位置扫描。

原生参照主投影与观测值逐项一致，消除这部分代理误差；实际局部校准SSE仍只由3750.838184降至3564.807261，约4.96%。原精确代理对照3564.082254，与S58复现。

| 内部配置 | 通过 | 平均L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 0.0204191343 | 0.1339174956 | 0.0040496844 | 286/290 |
| exact-a50 | 243/290 | 0.0204286227 | 0.1333402842 | 0.0039781677 | 286/290 |
| native-a50 | 243/290 | 0.0204151943 | 0.1333571076 | 0.0042041303 | 286/290 |

原生候选通过数未增加、最大KL上升，按原晋级规则保留baseline。没有再查看新提示挑参数，也没有追加单层变体。**停止该单层校准方向，主线转到INT4/NF4整网入口、模型覆盖与通用兼容性。**

4个正式NPU进程共2686次提交，120项局部检查、4项原图／保留输入控制通过，全部健康shutdown。三项新离线契约检查通过；完整回归随本轮主线功能一并记录于S60。生产后端、原模型参考、参数量及精度门槛均未改变。

复现：已有项目环境下依次运行 `w8_native_down_check.py prepare`、`fit`、三个profile的`internal --profile baseline/exact-a50/native-a50`、`select`，硬件步骤使用项目run_torch_npu.ps1串行包装。中间激活仅驻留内存；清理后需重新prepare。

## 身份与审计

| 文件／记录 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_native_down_check.py | `0e6fd475751424d3f9572ccc65ea1b6aa0010552b41f152b12655fa5e3a9fee6` |
| test_torch_npu_w8_native_down.py | `c7248d23dda08898dfe410747f5f18815ed812e4ecfb5842d586413ae5d227f3` |
| npunlock-probe/research/w8_decode_proxy.py | `df085c82ef3b901b66ee460c7bd2bb15c5387d359aa5fd5f2def0cafbeba9d89` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| 临时decode-fit.json | `080a567b8f85c477e3ca8cd517405f20f9c7113c1f4711e7ca30218ba060dcfe` |
| 临时internal-baseline.json | `cceaa236de9944e7751dc5658d54ef5ea870ba8d8a535b5d59df93fb52f78232` |
| 临时internal-exact-a50.json | `f521cabf0a0364a694a1d9969ce32339683a0c0f731b27887cb587bed609fff8` |
| 临时internal-native-a50.json | `1ff26789d9683abb3b29abb9e30ba0e8a27d9a3876ea7854c344cd2e111d71d6` |

## 清理

本阶段删除19份临时NPZ／JSON／日志，共115,969,864字节（110.597MiB）；清理前后2,005份保留文件SHA一致。结果目录只留REPORT.md，模型、环境及下一阶段默认图指纹保留。随后更新报告和存储台账。
