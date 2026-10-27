# DPU压力负载与CPU/GPU稳态能效实测

日期：2026-10-19；本地开发版，未改生产后端/原生组件/内核，未推送GitHub。

平台：Core Ultra 7 258V，NPU 8086:643e / graph compiler 8.3 / driver API 1005540，Arc 140V。CPU和GPU主对照Torch 2.12.0+xpu，另设Torch 2.14.1+cpu FP32控制；CPU均8线程。

## 两轮结果

下表为两轮均值。功耗来自Intel Unified Telemetry v0.2.0-beta1.1的独立能量计；主能效指标为整包mJ/次1024³矩乘，越低越好。

| 路径 | 有效TFLOPS/TOPS | 整包平均W | 整包mJ/矩乘 | 两轮mJ范围 | NPU域W | VCCGT核显域W |
|---|---:|---:|---:|---:|---:|---:|
| cpu-f16 | 0.454 | 30.737 | 144.0727 | 131.4954–156.6500 | 0.000 | 0.031 |
| cpu-f32 | 0.464 | 27.603 | 127.6812 | 118.0145–137.3479 | 0.000 | 0.026 |
| cpu-native-f32 | 0.461 | 26.150 | 121.6377 | 118.6793–124.5962 | 0.000 | 0.025 |
| gpu-f16 | 21.798 | 26.165 | 2.5773 | 2.5485–2.6061 | 0.000 | 13.701 |
| npu-f16 | 10.385 | 26.297 | 5.4380 | 5.3358–5.5402 | 15.157 | 0.031 |
| npu-int8 | 14.376 | 26.120 | 3.9021 | 3.8713–3.9329 | 9.307 | 0.028 |

FP16 DPU相对最低整包能耗CPU路径（cpu-native-f32），每次矩乘能耗比CPU为0.0447；相对GPU FP16为2.1099。GPU/DPU吞吐比2.099。这些是本次稳态矩乘结果，不能外推所有模型。

## 是否达到DPU满吞吐

本轮最高INT8稳态均值为14.376 TOPS，相当于258V官方47 TOPS规格数值的30.6%。这一比值是含提交/DMA的有效吞吐与规格的比值，不是DPU利用率。
已扫描M=128/512/1024、单会话、四个独立会话合并提交、四输出融合图，以及六tile编译配置；选用通过数值验证的最佳四矩阵配置测功耗。
六tile图的ELF包含task.dpu.invariant.0.0至5.0；FP16非逐元素DPU amode/wmode全为0/0，INT8全为1/1。未强制Turbo或更改功率上限。
因此只验证到当前边界内的高负载DPU功耗，不能称为47 TOPS满吞吐功耗。没有硬件DPU忙碌/频率计数证明所有流水线饱和，也没有扩大现有单Tensor存储边界或测试多队列。

- FP16 NPU域平均15.157W，完整采样区间最大21.968W；INT8平均9.307W，区间最大11.334W。峰值为约一秒区间平均，非瞬时峰值。
- r1-npu-f16的Windows NPU引擎忙碌率：88.28%。
- r2-npu-f16的Windows NPU引擎忙碌率：88.84%。
- r1-npu-int8的Windows NPU引擎忙碌率：86.17%。
- r2-npu-int8的Windows NPU引擎忙碌率：85.73%。

## 工作与计量口径

- 四组不同的(1024,1024)输入，共享固定权重，逻辑运算为四次A@W.T；一次矩乘按2*1024³有效操作计数。没有把padding、量化、复制或重复输出当成额外有效运算。
- 所有设备读取相同fixture指纹。输入/权重来自非零随机INT8网格，按1/128和1/256精确缩放为FP16。CPU参考先用FP32计算完整输出，另用NumPy INT64核对每组首/中/末三行；各路径在热身后和计量后验证全部输出。
- 主对照为相同FP16逻辑输入/输出；CPU FP32先计算再转FP16，这次转换计入耗时。GPU/CPU用等价的4096x1024扁平GEMM共享权重，避免人为增加Python循环。DPU四输出各自满足既有1,048,576元素界限。
- INT8 DPU另列压力结果，使用相同网格得到相同逻辑输出，但不把INT8 TOPS与FP16 TFLOPS直接当成同精度硬件对比；没有用该网格宣称真实模型量化无损。
- 先预热三次大块提交、验证输出，再连续测量；每块完成64次逻辑矩乘。GPU每块末同步，NPU等待完成fence，CPU同步执行。排除编译、权重打包、输入上传/最终下载；包含驱动提交、图内DMA与布局。
- 两轮采用正序/反序，各配置至少25秒；CPU/GPU/NPU串行运行。稳态NPU无新编译、驱动allocation或显式Tensor上传/下载；关闭后live/cache/scalar/quarantine/session归零，无失败context。
- 能量单位mJ，Duration单位ns，W=mJ*10^6/duration_ns。只取完整落在计量窗口内的约1秒区间，剔除边界；整包mJ/矩乘=平均W/该阶段完成矩乘数每秒*1000，是稳态估算。域功耗边界不同，主结论使用同一整包计量，未把各域相加。
- 桌面后台、温度和电源策略未锁定；两次测量的范围不是置信区间。未测整机插座能量，未把空闲功耗相减；短任务还需计入冷编译和传输成本。
- before空闲整包平均7.689W，9个完整区间。
- after空闲整包平均13.750W，9个完整区间。

空闲整包漂移较大，故不将这些数字称为受控硬件能效认证。仅作敏感性估算：若各阶段基线落在上述空闲范围，分别扣除最高/最低基线后，稳态mJ/矩乘范围如下；这不是独立测得的动态能量：

- cpu-native-f32：57.7004–85.9041mJ/矩乘。
- gpu-f16：1.2230–1.8201mJ/矩乘。
- npu-f16：2.5944–3.8477mJ/矩乘。

这项粗略敏感性计算不改变本轮GPU相对当前DPU FP16路径的能效排序，但完整模型、其他形状和更好的DPU执行策略仍需重测。

## 逐轮记录

| 配置 | 实测秒 | 有效TFLOPS/TOPS | 整包W | 整包mJ/矩乘 | NPU域W | VCCGT域W |
|---|---:|---:|---:|---:|---:|---:|
| r1-cpu-f16 | 25.031 | 0.500 | 36.448 | 156.6500 | 0.000 | 0.030 |
| r1-cpu-f32 | 25.299 | 0.473 | 30.228 | 137.3479 | 0.000 | 0.026 |
| r1-cpu-native-f32 | 25.200 | 0.469 | 27.213 | 124.5962 | 0.000 | 0.023 |
| r1-gpu-f16 | 25.001 | 22.022 | 26.726 | 2.6061 | 0.000 | 14.049 |
| r1-npu-f16 | 25.001 | 10.357 | 26.719 | 5.5402 | 15.127 | 0.034 |
| r1-npu-int8 | 25.006 | 14.570 | 26.266 | 3.8713 | 9.581 | 0.023 |
| r2-npu-int8 | 25.003 | 14.182 | 25.974 | 3.9329 | 9.033 | 0.034 |
| r2-npu-f16 | 25.010 | 10.414 | 25.875 | 5.3358 | 15.187 | 0.029 |
| r2-gpu-f16 | 25.004 | 21.575 | 25.604 | 2.5485 | 0.000 | 13.352 |
| r2-cpu-native-f32 | 25.130 | 0.454 | 25.087 | 118.6793 | 0.000 | 0.027 |
| r2-cpu-f32 | 25.098 | 0.455 | 24.978 | 118.0145 | 0.000 | 0.027 |
| r2-cpu-f16 | 25.221 | 0.409 | 25.026 | 131.4954 | 0.000 | 0.032 |

## 吞吐校准

| 精度 | M | 布局/提交 | 编译配置 | 有效TFLOPS/TOPS |
|---|---:|---|---|---:|
| float16 | 128 | single | default | 1.593 |
| float16 | 128 | chain4 | default | 2.304 |
| float16 | 128 | fused4 | default | 3.676 |
| float16 | 512 | single | default | 3.959 |
| float16 | 512 | chain4 | default | 5.201 |
| float16 | 512 | fused4 | default | 6.507 |
| float16 | 1024 | single | default | 5.221 |
| float16 | 1024 | chain4 | default | 6.814 |
| float16 | 1024 | fused4 | default | 7.411 |
| float16 | 1024 | single | tiles6 | 6.278 |
| float16 | 1024 | chain4 | tiles6 | 8.009 |
| float16 | 1024 | fused4 | tiles6 | 9.547 |
| int8 | 1024 | single | tiles6 | 7.828 |
| int8 | 1024 | chain4 | tiles6 | 9.673 |
| int8 | 1024 | fused4 | tiles6 | 12.912 |
| int8 | 1024 | chain4 | default | 8.461 |
| int8 | 1024 | fused4 | default | 10.415 |

首轮校准错误地在同一提交组重复会话，受到已有运行时保护而拒绝；当时脚本闭包仍持有Tensor，退出断言也失败。测试脚本改为四个独立会话，并在关闭前释放引用。融合图在一个图内处理四组不同输入，没有关闭保护或修改生产运行时。正式17项校准和全部NPU测量进程均通过关闭检查后才计入结果。

### 扁平大矩乘候选核对

另验证NPU图内Concat→4096x1024矩乘→Split的等价路径；外部仍为四个合法的1024²输入/输出。FP16短测8.268TFLOPS、INT8 10.002TOPS，均未超过原四矩乘融合图的相应校准值；因此主能效测量没有仅因NPU外部Tensor上限而放弃等价大矩乘。
该候选全部输出和资源关闭检查通过，但没有单独做功耗采集，不能据其吞吐断言能效更差。首次Split多输出未显式命名而被绑定类型检查拒绝，改为已有命名Reshape包装后重跑通过；生产绑定代码未改。

## 复现与限制

入口torch_npu_gemm_energy.py、run_gemm_energy.ps1；外部功耗采集measure_npu_power.ps1。Intel工具需要管理员权限，本轮沿用用户授权并通过UAC。仅启用SoCWatch的npu-pwr/pkg-pwr/vccgt-pwr，没有加入项目运行依赖。
本轮最佳DPU六tile/多输出图仅在基准中生成，不代表普通torch.mm已自动改走DPU，也不是完整模型/训练能效结论。

本机复现（功耗命令在管理员PowerShell执行；按实际解包位置设置utRoot）：

```powershell
.\.conda\npunlock\python.exe -B torch_npu_gemm_energy.py --prepare
.\run_torch_npu.ps1 -Script .\torch_npu_gemm_energy.py -ScriptArguments @('--calibrate','--label','calibration-complete')
$utRoot = 'E:\path\to\ut-tool-ext-v0.2.0-beta1.1'
.\npunlock-probe\research\measure_npu_power.ps1 -ToolRoot $utRoot -Seconds 600 -IncludeGpu -Label paired -OutputRoot "$PWD\npunlock-probe\research\results\gemm-energy\power" -Application "powershell.exe -NoProfile -File $PWD\npunlock-probe\research\run_gemm_energy.ps1"
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\dpu_flat_gemm_probe.py
.\.conda\npunlock\python.exe -B npunlock-probe/research/gemm_energy_report.py
```

run_gemm_energy.ps1中的GPU环境路径为本机C:\Users\LC\miniconda3\envs\blood\python.exe，其他机器需更换。报告生成须在原始记录仍保留时运行，审计后仅保留源码与Markdown报告。

[Intel 258V官方47 TOPS规格](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html)；[NPU tile配置定义](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/intel_npu/properties.hpp)；[Intel Unified Telemetry](https://www.intel.com/content/www/us/en/developer/tools/intel-unified-telemetry.html)。

## 原始记录指纹（审计后清理）

- `calibration-complete.json` 423715字节 SHA256 `e0f48bb09d3a36c14609b8baa5af815c024f738999270de147ccd0f22b7d542e`
- `calibration-final.json` 260286字节 SHA256 `9e31f2b5e0ee21cd9c11ef2d00e0b7089ebb43a0e8c8cc8710c5dba5008fa831`
- `calibration.json` 11405字节 SHA256 `c258c462d2cee0acdc99aa9ca2b22b180dfe1b15a5e62ca56bcce6167ad33ed2`
- `fixture.json` 219字节 SHA256 `053c1e1ccf3b66c50fde60932b04a21771f9055e3a283a51e8f388f7e8148b40`
- `fixture.npz` 12346650字节 SHA256 `61e4088b0e9f9382b3a144da319f4781966f29c10ba3b8f27a56596297da7e9b`
- `flat-probe-names-failed.json` 1409字节 SHA256 `9aab7f1e28ca6a7c968cf84b7135dac2f7db2e153ad7e4eea6693d4775323cea`
- `flat-probe.json` 126905字节 SHA256 `219ebcaf86964b2a2963019f4b75e5b3d5703cc3f5565e3f7e538c6f2ab52356`
- `measurement-plan.json` 237907字节 SHA256 `5be1951f598ff9abe032dbd374e4f10e5a085d12b7753b8fdc8b8f20af4dceb3`
- `pilot-cpu-f16.json` 1594字节 SHA256 `54c46d5120c44cde7a01eca36df370c400f04ed3edad8ad9412180ebdb880745`
- `pilot-gpu-f16.json` 11998字节 SHA256 `4fc214ae0fe847f22a0dbab8c01b97fa3b341429b66e17a8c360827f793bedd3`
- `power\paired\power-complete.json` 166字节 SHA256 `8847bf521ae3cbbd9fcea8b778194a14312873d5e8793616cf44e45f151b728a`
- `power\paired\power-config.json` 866字节 SHA256 `ca0dff08aedf483c5d28dfcac624a46de3c255108ad0bece83027557c27ae2a8`
- `power\paired\power-output.txt` 550154字节 SHA256 `a4cface5a0d34b41a22f46a82e8193cb30dfa669d443087fa1c00fea9c46cd89`
- `r1-cpu-f16.json` 4035字节 SHA256 `d712b21f4a5315878d24f4e69b4b508efb8cb8808b078a44c59f64f41489be42`
- `r1-cpu-f32.json` 3911字节 SHA256 `197a421c31212540f17ef40690cabcd76252902de9f4511dd6843506f8f566e3`
- `r1-cpu-native-f32.json` 3901字节 SHA256 `29030fb8fdedb525e2dd1cb424a4f231db263a1f3f6c0d4fe7475a7578974279`
- `r1-gpu-f16.json` 125741字节 SHA256 `bddec747dcb4f1b7a5f0106368619a1972fb05330d0ba77fd56c0d438a3fcbc0`
- `r1-npu-f16.json` 159469字节 SHA256 `611a1f1d9f4afed2069d615ef9e6841598b485ecc47fad94581c2f5b0839558e`
- `r1-npu-int8.json` 112368字节 SHA256 `4fd540cea6e396308c45ee0cb0252db1b1a54b3557f9b2a4bcdf7f889c6be681`
- `r2-cpu-f16.json` 3564字节 SHA256 `c33134457563d3fa6133e63ed786d11030f8fa8825783b1628bfd154cfdda3d9`
- `r2-cpu-f32.json` 3793字节 SHA256 `2664cb4e2404caa3974f16c55f1d6f9a8b6f7570960254e0ac8bc178c542d002`
- `r2-cpu-native-f32.json` 3801字节 SHA256 `d61763391f948b5fdc30de6b39180081f208a6806298f11f9a222b35495ced45`
- `r2-gpu-f16.json` 123249字节 SHA256 `2cfe29f66d081bd869817f7b5ecdd824002b42aa3e7d6fc9d8bad9dc4711d89d`
- `r2-npu-f16.json` 159885字节 SHA256 `cb0fcfc093bcbdee4cb8132f5420756d88e4ec439180efc9405cbc4935f2507e`
- `r2-npu-int8.json` 110243字节 SHA256 `80d3a641cb425207e9bccc3dbf48ac972b974e8487a162adb09334edfef591b4`

## 验收与保留

9项离线监控/计量测试通过。正式12个测量阶段均通过完整输出、工作计数、源码与输入指纹审计；NPU资源关闭检查通过。源代码发布快照未修改，本轮未推送GitHub。

按用户保留源码和报告的要求，审计后删除本轮原始记录、输入fixture、Intel便携监控工具与下载包，共583个文件、74,097,313字节（70.665MiB）。清理前后668份项目源码、原生组件、内核及报告哈希一致，随后补充本记录与存储台账。此结果目录仅保留REPORT.md；原始记录指纹保留在上文，重新审计需按复现步骤重新采集。

采集进程已退出，临时socwatchdrv监控驱动已正常卸载；未引入项目运行依赖。
