# S17：原生FP16 autocast与NPU专用GradScaler

日期：2026-10-20。Windows x64、Core Ultra 7 258V、NPU 8086:643e、驱动32.0.100.5540、编译器8.3、Python3.12.15、Torch2.14.1+cpu。
本地开发版，未推送GitHub、未修改source-release。按用户要求先补兼容性，没有做速度排名或低位整网质量调优。

## 结果与接口

原生`torch.autocast("intel_npu",dtype=torch.float16)`和NPU专用`torch.intel_npu.amp.GradScaler()`已接通，模型主参数与优化器状态保留FP32。
MLP三步损失、全部梯度、参数/SGD动量更新及checkpoint通过严格CPU对照。卷积网络可以训练和逐位恢复，但严格FP32梯度对照仍失败，单列为passed_with_limitations，不能称为通用AMP完成。

修改前autocast因支持dtype列表为空而警告并禁用，Linear输出FP32；标准`torch.amp.GradScaler("intel_npu")`在INT32增长状态分配时报错。
本轮没有给这个通用GradScaler入口伪造FP64/INT32支持；使用明确的NPU专用入口。该类继承PyTorch GradScaler的step/update、OptState与checkpoint，主机仅维护scale/growth/found-inf控制元数据。
FP64倒数只作用于CPU scale标量；模型、梯度、有限性归约和反缩放均由现有NPU算子执行。

```python
import torch
import torch_intel_npu

model = torch.nn.Linear(4, 3).to("intel_npu")  # 保持FP32主参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False, fused=False)
scaler = torch.intel_npu.amp.GradScaler()
inputs = torch.randn(2, 4, device="intel_npu")
labels = torch.tensor([0, 2], dtype=torch.int64, device="intel_npu")

optimizer.zero_grad(set_to_none=True)
with torch.autocast("intel_npu", dtype=torch.float16):
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
# 与显式CPU模型/优化器checkpoint一起保存scaler.state_dict()。
```

## 精度策略及实现

使用本机PyTorch的ATen/autocast_mode.h、KERNEL_PRIVATEUSEONE、CastPolicy与cached_cast，在AutocastPrivateUse1注册28项重载及fallthrough。转换先于autograd记录，低精度计算图可回传FP32主参数梯度；权重缓存由PyTorch按autocast上下文管理。

| 策略 | 本轮注册范围 |
|---|---|
| FP16 | linear、mm、matmul、mv、addmm、bmm、baddbmm、conv2d、convolution、scaled_dot_product_attention |
| FP32 | exp、log、reciprocal、rsqrt、layer_norm、native_layer_norm、rms_norm、mse_loss、nll_loss、nll_loss_nd、nll_loss_forward、cross_entropy_loss |
| 未指定dtype时FP32 | softmax.int、log_softmax.int、sum/default与dim_IntList、mean/default与dim |
| 保持原dtype | 其他算子、out与inplace；不自动声明未实现算子可用 |

这是项目的有界NPU精度策略，不声称完整复刻CUDA的autocast算子列表。BF16按PyTorch规则警告并禁用；嵌套disabled上下文、线程局部状态、显式dtype及out保持原语义。
普通FP16矩乘/卷积仍复用已有SHAVE计算与FP16/FP32转换，没有新增DPU训练、内核融合或训练加速结论。

专用GradScaler接收Tensor或非空list/tuple的浮点NPU输出，以FP32进行loss缩放；要求FP32主参数和完整密集FP32梯度、单NPU、明确foreach=False/fused=False的普通优化器。
每个优化器先验证全部参数/梯度dtype、存储和别名，再计算有限性与反缩放，最后回读一个BOOL标志并写入梯度。标志为逻辑1字节，通过已有同步scalar extraction读取；未下载梯度payload。scale、INT32 tracker及CPU found-inf控制值不属于模型计算回退。
支持动态增长/回退、Inf/NaN跳步、梯度累积、多优化器、手动正有限Python数值scale和state_dict。稀疏/FP16主梯度、梯度/参数存储别名、fused/capturable/differentiable优化器、Tensor新scale、无限迭代器及图捕获不在支持范围。scale和倒数须可由正有限FP32表示。

## 真机检查与数值边界

| 专项 | 状态 | NPU图提交 |
|---|---|---:|
| autocast dtype、主参数梯度、显式dtype/out、嵌套/线程状态与缓存 | passed | 255 |
| 缩放器有限/Inf/NaN、默认scale真实反向溢出、累积与多优化器 | passed | 350 |
| dtype/别名/优化器预检、stage与checkpoint错误保护 | passed | 32 |
| Linear→LayerNorm→SiLU→Linear 三步训练/恢复 | passed | 1,372 |
| Conv2d→SiLU→AvgPool→Linear 三步训练/恢复 | passed_with_limitations | 1,000 |

五组共3,009次正式图提交，其中四组通过、一组保留限制。各进程shutdown后live/cache/scalar/session/quarantine归零，context_failed=False、closed=True。初始基线、缺少fallthrough的失败探测、初次卷积严格失败及探索性重复不计入正式合计。
梯度有限性在输入梯度上检查，与PyTorch GradScaler一致；不把有限输入在极端反缩放后新产生的溢出宣称为另一次自动检测。

policy检查使用FP32输入与参数，断言linear/mm/matmul/mv/addmm/bmm为FP16、损失/归一化/归约/数学敏感项为FP32，显式half dtype及FP32 out不被强制转换；不将注册列表直接等同于所有配置均已测试。
同一autocast上下文重复Linear，第一次18次提交、第二次14次，确认权重/bias转换缓存生效；退出上下文后更新权重，重新进入能读到新值。只核对图提交和数值，没有计时加速声明；不支持在同一缓存上下文内部修改权重后仍假定缓存自动失效。

缩放器与独立CPU标准GradScaler交叉对照七步（有限、有限、Inf、有限、NaN、有限、有限），scale依次为8、16、8、8、4、4、8，参数与动量一致。默认65536的真实FP16 Linear反向溢出会跳过更新并降到32768；双优化器时一个更新、另一个溢出跳过，汇总后scale回退。梯度累积、disabled身份路径和scaler状态恢复通过。

两个模型都保留FP32主参数/梯度，前向最后logits为FP16、CrossEntropy为FP32；SGD lr=.02/momentum=.5，init_scale128、growth_interval2。独立CPU参考显式将Linear/Conv输入与权重转FP16，LayerNorm和损失转FP32；关闭本机不支持FP16卷积反向的oneDNN路径。
每个训练检查区间只允许NLL的4字节标签校验和unscale的BOOL标志各一次下载；step/update没有模型payload下载。CPU梯度对照、局部诊断和checkpoint传输在这些区间之外。

MLP有83参数，三步loss为1.362824678、1.329084396、1.281943798；全部损失/梯度/更新通过rtol=3e-5、atol=3e-6。
卷积有29参数，三步loss为1.288599730、1.282931805、1.274547219。损失与参数更新仍通过严格FP32阈值，但first.weight梯度分别有3.814697e-6、7.629395e-6、1.525879e-5最大绝对差，三步严格梯度检查均保留为失败。另行按既有FP16计算界限rtol=.003/atol=.0005检查通过，没有把它改记为严格FP32成功。
局部诊断取实际保存的NPU上游FP16梯度，CPU用FP32 conv2d_weight累加、舍入FP16再按scale反缩放；三步与NPU权重梯度最大差均为0。该诊断只隔离卷积局部累加，不是独立整网CPU轨迹替代品，也不能证明任意模型的梯度都正确。
第2步后将CPU权重、优化器和scaler状态存入BytesIO，在新模型/优化器恢复第3步；两模型loss、参数、动量及scale状态均逐位重放。三步有限结果不证明长期收敛、完整Transformer或更大网络训练质量。

## 构建、回归与来源

本轮只新增C++派发策略、torch_npu_amp.py和验收/离线测试；既有SHAVE ELF、session DLL、普通非AMP精度规则与低位量化实现不变。无新增pip依赖、模型下载、监控驱动或图表。

```powershell
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_amp_demo.py
.\run_torch_npu.ps1 -Device -Quick
```

Torch PYD须重编，不能将旧扩展与新Python入口混用；build_npu.ps1已包含设备扩展构建。纯离线CI已加入test_torch_npu_amp，79项检查通过：核心/模型42项、npunlock Python API32项、source exporter5项。
新增五项离线控制覆盖scaler配置/checkpoint原子验证、disabled的CPU工作流、控制元数据CPU dtype、独立CPU溢出增长序列及两个显式混合精度模型的FP32主梯度。
既有设备Quick全部33组通过，共4,162次提交；43项拒绝检查未新增图提交且不消耗RNG，shutdown后全部资源归零。非AMP路径沿用原数值阈值和原生Generator，不因autocast注册而改变默认计算dtype。
FP16/FP32显式Generator两组回归246次提交通过，覆盖随机数、采样、state/clone/offset及默认/CPU流隔离。全轮正式硬件检查合计40组、7,417次提交：39组passed，一组卷积passed_with_limitations；该合计不能表述成40组严格数值全部通过。
专用scaler是PyTorch2.14.1接口上的有限适配，未来升级Torch要重新核对其控制状态机、autocast宏及API；不是对其他版本的认证。来源说明见根目录REFERENCES.md。

仍缺通用torch.amp.GradScaler入口、BF16、广泛模型/优化器与动态索引；低位完整模型质量本轮未改，W8历史结果仍为129/142检查位置，INT4/NF4完整语言模型尚未认证。不能据此宣布通用CUDA兼容性或低位质量完成。

## 本轮代码指纹（SHA-256）

| 文件 | SHA-256 |
|---|---|
| torch_intel_npu.py | 51d4779778d3a10617fdf9bb936c26ec044a48dd4cbb154d6722a91eafb94ae6 |
| torch_npu_amp.py | c93123ef60bf21a8596a4af99740b1ba3032e7fba2ab9dfc9f4744cf25cb7cea |
| torch_npu_amp_demo.py | 4d427c226ec04d5a7d1d6cc154f0142db75cc2a5e98f75134b5b52e27489da8f |
| test_torch_npu_amp.py | 97f6b51529fe5698c12390220b3c0db80c09fd07f9a06551daa27dc5fa98be3f |
| npunlock-probe/torch-device/backend.cpp | 4c64ae12ae8e013e23f1b93667bcbb95366af508ae40c481204aea112af10d83 |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | 3592af43edfc638f761de57d1b5f2502a982638aed495895363032128596ffbd |

五组正式AMP结果的全部源码/原生产物指纹与最终文件一致。session DLL仍为0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9，SHAVE manifest/ELF沿用S16；没有重编数值内核。哈希用于本轮运行provenance，其他机器源码重建不要求产生相同二进制字节。

## 清理记录

验收后按明确绝对路径删除14份本轮原始JSON/日志及可重建backend.obj/导出表，共7,533,636字节（7.185MiB），没有递归删除。清理前后781份保留源码、报告、ELF/汇编与必要运行时SHA-256一致，随后有意补充本节与存储台账；原import lib保留。
source exporter检查775份源码/报告/内核，覆盖新增AMP模块、硬件/离线测试与报告，所有SHAVE ELF/汇编哈希仍有效。没有生成额外源码快照、改动source-release、推送GitHub、启用磁盘图缓存或保存磁盘checkpoint；仅保留源码、必要运行时和文字报告。
