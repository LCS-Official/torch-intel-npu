# v0.3.0-alpha.1 累计发布集成验收

日期：2026-10-30。把v0.2.0后的损失修复、S47–S59研究与S60四位decoder入口整合为面向main的预览源码。保留原提交及分支历史；未将任何新校准候选设为默认。

## 集成审查

原功能链没有包含独立的S46修复及S47分支。本轮把两者以merge commit纳入release/v0.3.0-alpha.1，随后统一发布首页、安装说明、兼容性及release notes。冲突仅位于文档，保留两侧报告引用；四个生产模块最终与此前已测的开发源码逐字节一致。

相对v0.2.0没有C/C++、SHAVE ELF／汇编、原生构建脚本或requirements变化。六份旧FP16/W8 forward／prefill／decode图指纹不变；230份跟踪NPU4000 fixture与v0.2.0及开发副本相同。本轮没有扩大硬件／模型合同，也没有新效率数字。

## 验收环境与方法

Windows x64、Core Ultra 7 258V、NPU4000 8086:643e、驱动32.0.100.5540、编译器8.3；Python3.12、Torch2.14.1+cpu、NumPy2.5.3，现有可选模型依赖。所有新真机检查从累计公开checkout导入后端与npunlock；冒烟日志的npunlock路径指向公开目录。

临时复用匹配的现有Torch PYD、session DLL及build.json，外部native库通过NativeDirectory明确指定；不重新编译宿主、不创建环境、不下载／复制模型。这不是干净机器安装验证。检查结束后删除公开checkout中的三个临时宿主文件；原开发运行组件保留。

## 本轮实际真机检查

7个独立进程顺序执行，共24,900次NPU调用。各阶段断言均通过，关闭后context未失败且无残留缓存会话／live buffers／隔离内存。无CPU decoder回退；CPU参考及模型加载是显式边界。

| 检查 | 内容 | NPU调用 | 结果 |
|---|---|---:|---|
| 安装／FP16模型冒烟 | FP32尾块、FP16矩乘、Linear/Conv2d梯度与SGD、单层FP16 prefill/decode | 77 | 通过 |
| 分类损失版本 float16 | 122项，CPU/NPU拒绝／接受行为无差异 | 14,539 | 通过 |
| 分类损失版本 float32 | 122项，CPU/NPU拒绝／接受行为无差异 | 10,240 | 通过 |
| 冻结单层 fp16 | forward、8-token prefill、9th-token decode、旧KV及热重放 | 11 | 3项局部通过 |
| 冻结单层 w8a16 | forward、8-token prefill、9th-token decode、旧KV及热重放 | 11 | 3项局部通过 |
| 冻结单层 w4a16 | forward、8-token prefill、9th-token decode、旧KV及热重放 | 11 | 3项局部通过 |
| 冻结单层 nf4a16 | forward、8-token prefill、9th-token decode、旧KV及热重放 | 11 | 3项局部通过 |

分类损失版本检查共244项，四种格式另有12项局部数值检查。冒烟确认没有导入OpenVINO。四位单层门槛沿用L2<0.003、scaled max<0.01，未放宽；本轮没有再次跑完整30层质量校准或长期训练。

## 离线及包装检查

累计公开目录自动发现349项（348通过、1既有跳过），Python API32通过，导出器6通过；共387项检查，386通过、1跳过、0失败。可选模型依赖已安装。旧嵌套source-release对照在独立checkout中缺失而跳过，未冒充通过。

完整发布清单在删除临时宿主／JSON／日志后重建；源包仅包含源码、许可／来源、配套内核文件和Markdown报告。最终逐文件集／SHA以SOURCE_MANIFEST.json为准，并验证实际Git归档内的清单和payload一致。云端CI只验证离线源码，不能替代上述真机过程；最终main／tag运行状态以GitHub对应检查为准。

## 质量与发布边界

S60的8提示／257位置原模型质量结果作为已测证据保留：FP16 257/257，普通逐行RTN W8 96/257，INT4和NF4均0/257。W8控制未使用可选校准因子，不能与旧142或290位置研究直接比较。四位入口是FP16 DPU计算前的压缩权重解压，不是原生四位乘加或通用低位训练。

NF4单行投影内部补行规避8.3编译错误；S60完整栈热路径用显式1GiB图预算，默认512MiB可能淘汰／重编译。生产默认仍512MiB。四位整网质量、任意HF模型、长上下文、grouped decode、4位训练和GPU＋NPU协同没有本轮新增认证。S59已结束单层校准分支，主线继续围绕实际模型兼容和整网质量。

v0.1.0／v0.2.0 tag保持原值。v0.3.0-alpha.1使用新tag指向累计合并结果，发布为prerelease；模型、环境、宿主二进制及原始数据不随源码发布。旧提交和分支保留，不用删除历史文件实现清理。

## 复现

先依[安装说明](../../../../INSTALL.md)准备匹配宿主及本地checkpoint。安装冒烟使用tools/smoke_install.py --model；分类损失使用torch_npu_loss_extended_demo.py --case versions --precision float16/float32，分别启动进程。

单层四种格式复用smollm_lowbit_coverage.py既有run(mode,1)与同一冻结语料；临时适配器只把MODEL指定到已有外部checkpoint、OUT指定到本轮目录，并断言生产后端从公开checkout导入，不改变计算、门槛或权重策略。使用PYTHONPATH加入公开根目录、npunlock/python及npunlock-probe/research；每个格式独立进程。完整30层方法与数值边界见[S60报告](../smollm-lowbit-coverage/REPORT.md)。

## 当前源码与运行组件身份

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| torch_npu_autograd.py | `283f52d7b5939b5c8b79569fc51b7a4cc54495a4aecaf4be713953f7111afe79` |
| torch_npu_smollm.py | `d82bbfb90241b851edcf0ef10669676b8f112fb911eb5931588030dc594a80f6` |
| torch_npu_transformer.py | `4d5ef74adc3460d1aac234b2b2dd546008ab6db17bf1f6af5168c7155dd06792` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |

| 本轮临时原始记录 | SHA-256 |
|---|---|
| corpus.json | `e6b4f93c94a93a45557a322897ddff447808bf009be5dae37e41ea19b5942e80` |
| fp16-1.json | `0c2cacf22d04ec818b765d11eb838dcebd6bb85779861c498bc6ed2c0272b4f4` |
| integrity.json | `268120794d87f0d7ca5c5ca2c1b73c5b578344055f5dac20edacd4a0c39c824e` |
| nf4a16-1.json | `77c92734e9d174a729142cb5f581fa896d19ab4e1e4d1d23f08e755b68340ba5` |
| versions-float16.json | `623d521d97a6aaba1582e9241197454aad75c6563eaeb21628e4fd73fb9bda2c` |
| versions-float32.json | `3bf500dd22f8e63adc94d3bc34ea887e57760f106fc50e9f07b1577186a4b476` |
| w4a16-1.json | `e93f6c1cf9edc66e4cd5be7b2cce2037c26fd1bfdd5e50d962a74df6eef509fe` |
| w8a16-1.json | `51bd60f780f599d1df6eba98f90416d1370b35bc81c5dbe6325096a0e6f81a78` |

## 清理

本阶段删除19份临时适配脚本／JSON／日志及3份公开目录宿主副本，共1,321,165字节（1.260MiB）；清理前后2,027份保留文件SHA一致。随后仅更新清理说明、存储台账和发布清单。结果目录只留REPORT.md，原开发宿主、模型与环境保留。
