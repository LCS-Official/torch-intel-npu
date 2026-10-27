"""Audit matched work and energy units; keep the measured conclusions in Markdown."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from torch_npu_sustained_train import engine_interval
from npu_power_report import parse, read_log, summarize
OUT = ROOT/'npunlock-probe/research/results/gemm-energy'
METRICS = ('PMT-NPU-PWR', 'PKG-PWR', 'PMT-VCCGT-PWR')


def mj_per_gemm(watts, throughput):
    assert watts >= 0 and throughput > 0
    return watts/throughput*1000


def verify_work(row):
    meta = row['meta']
    assert all(meta[k] == 1024 for k in ('m','k','n')) and meta['groups'] == 4
    assert row['seconds'] > 0 and row['completed_gemms'] == len(row['chunks'])*row['gemms_per_chunk']
    assert row['gemms_per_chunk'] == 64
    assert abs(row['gemms_per_second']-row['completed_gemms']/row['seconds']) < 1e-7
    assert abs(row['teraops']-2*1024**3*row['gemms_per_second']/1e12) < 1e-9
    assert row['validation']['relative_l2_max'] < .001
    if row['stats_before']:
        before, after = row['stats_before'], row['stats_after']
        graphs = row['completed_gemms']//4 if meta['layout'] == 'fused4' else row['completed_gemms']
        assert after['npu_calls']-before['npu_calls'] == graphs
        for field in ('host_upload_copies','host_download_copies','host_device_copies',
                      'session_creations','driver_allocations','live_buffers','live_buffer_bytes'):
            assert before[field] == after[field], field
        wanted = (1,1) if meta['precision'] == 'int8' else (0,0)
        assert all((v['amode'],v['wmode']) == wanted for v in meta['dpu_modes'])


def main():
    plan = json.loads((OUT/'measurement-plan.json').read_text(encoding='utf-8-sig'))
    assert plan['status'] == 'passed' and len(plan['cases']) == 12
    complete = json.loads((OUT/'power/paired/power-complete.json').read_text(encoding='utf-8-sig'))
    assert complete['exit_code'] == 0
    records = parse(read_log(OUT/'power/paired/power-output.txt'), accepted=METRICS)
    sha = hashlib.sha256((OUT/'fixture.npz').read_bytes()).hexdigest()
    cases = {}
    for label in plan['cases']:
        data = json.loads((OUT/f'{label}.json').read_text())
        assert data['status'] == 'passed' and data['fixture_sha256'] == sha
        assert len(data['rows']) == 1
        row = data['rows'][0]
        verify_work(row)
        for name, digest in data['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        if data['device'] == 'npu':
            final = data['final_stats']
            assert final['closed'] and not final['context_failed']
            assert all(final[k] == 0 for k in ('live_buffers','live_buffer_bytes','cached_sessions',
                'cached_scalar_buffers','cached_buffer_blocks','quarantined_buffer_bytes'))
        powers = {}
        for metric in METRICS:
            powers[metric] = summarize([r for r in records if r['metric'] == metric],
                                       row['start_unix_ns'], row['end_unix_ns'])
            assert powers[metric]['count'] >= 15, (label,metric)
        cases[label] = dict(data=data, row=row, powers=powers,
            mj_per_gemm=mj_per_gemm(powers['PKG-PWR']['average_w'], row['gemms_per_second']))
    names = ('cpu-f16','cpu-f32','cpu-native-f32','gpu-f16','npu-f16','npu-int8')
    aggregate = {}
    for name in names:
        values = [cases[f'r{round}-{name}'] for round in (1,2)]
        aggregate[name] = dict(teraops=statistics.mean(v['row']['teraops'] for v in values),
            watts=statistics.mean(v['powers']['PKG-PWR']['average_w'] for v in values),
            mj=statistics.mean(v['mj_per_gemm'] for v in values),
            minimum=min(v['mj_per_gemm'] for v in values), maximum=max(v['mj_per_gemm'] for v in values),
            npu_w=statistics.mean(v['powers']['PMT-NPU-PWR']['average_w'] for v in values),
            npu_w_max=max(v['powers']['PMT-NPU-PWR']['max_w'] for v in values),
            gpu_w=statistics.mean(v['powers']['PMT-VCCGT-PWR']['average_w'] for v in values))
    best_cpu = min(names[:3], key=lambda n: aggregate[n]['mj'])
    npu, gpu, cpu = (aggregate[n] for n in ('npu-f16','gpu-f16',best_cpu))
    rawrows = []
    for label, c in cases.items():
        rawrows.append(f'| {label} | {c["row"]["seconds"]:.3f} | {c["row"]["teraops"]:.3f} | '
            f'{c["powers"]["PKG-PWR"]["average_w"]:.3f} | {c["mj_per_gemm"]:.4f} | '
            f'{c["powers"]["PMT-NPU-PWR"]["average_w"]:.3f} | {c["powers"]["PMT-VCCGT-PWR"]["average_w"]:.3f} |')
    calibration = json.loads((OUT/'calibration-complete.json').read_text())
    assert calibration['status'] == 'passed' and len(calibration['rows']) == 17
    curves = []
    for r in calibration['rows']:
        if r['status'] != 'passed':
            continue
        m = r['meta']
        curves.append(f'| {m["precision"]} | {m["m"]} | {m["layout"]} | {m["profile"]} | {r["teraops"]:.3f} |')
    idle = []
    idle_watts = []
    for window in plan['idle']:
        power = summarize([r for r in records if r['metric'] == 'PKG-PWR'],
                          window['start_unix_ns'],window['end_unix_ns'])
        idle.append(f'- {window["name"]}空闲整包平均{power["average_w"]:.3f}W，{power["count"]}个完整区间。')
        idle_watts.append(power['average_w'])
    low, high = min(idle_watts), max(idle_watts)
    sensitivity = {}
    for name in (best_cpu,'gpu-f16','npu-f16'):
        v = aggregate[name]
        throughput = v['teraops']*1e12/(2*1024**3)
        sensitivity[name] = (mj_per_gemm(max(0,v['watts']-high),throughput),
                             mj_per_gemm(max(0,v['watts']-low),throughput))
    assert sensitivity['npu-f16'][0] > sensitivity['gpu-f16'][1], 'Sensitivity ordering needs a fresh review'
    flat=json.loads((OUT/'flat-probe.json').read_text())
    assert flat['status']=='passed' and len(flat['rows'])==2
    assert flat['source_sha256']==hashlib.sha256((ROOT/'npunlock-probe/research/dpu_flat_gemm_probe.py').read_bytes()).hexdigest()
    assert flat['final_stats']['closed'] and not flat['final_stats']['context_failed']
    assert all(flat['final_stats'][key]==0 for key in ('live_buffers','live_buffer_bytes','cached_sessions',
        'cached_scalar_buffers','cached_buffer_blocks','quarantined_buffer_bytes'))
    for row in flat['rows']:
        verify_work(row)
    flat_values={r['meta']['precision']:r['teraops'] for r in flat['rows']}
    util = []
    for name in ('npu-f16','npu-int8'):
        for round in (1,2):
            row = cases[f'r{round}-{name}']['row']
            util.append(f'- r{round}-{name}的Windows NPU引擎忙碌率：'
                f'{engine_interval(row["npu_counter_before"],row["npu_counter_after"])["0:0"]["process_percent"]:.2f}%。')
    text = '\n'.join([
        '# DPU压力负载与CPU/GPU稳态能效实测', '', '日期：2026-10-19；本地开发版，未改生产后端/原生组件/内核，未推送GitHub。', '',
        '平台：Core Ultra 7 258V，NPU 8086:643e / graph compiler 8.3 / driver API 1005540，Arc 140V。CPU和GPU主对照Torch 2.12.0+xpu，另设Torch 2.14.1+cpu FP32控制；CPU均8线程。', '',
        '## 两轮结果', '',
        '下表为两轮均值。功耗来自Intel Unified Telemetry v0.2.0-beta1.1的独立能量计；主能效指标为整包mJ/次1024³矩乘，越低越好。', '',
        '| 路径 | 有效TFLOPS/TOPS | 整包平均W | 整包mJ/矩乘 | 两轮mJ范围 | NPU域W | VCCGT核显域W |',
        '|---|---:|---:|---:|---:|---:|---:|',
        *[f'| {name} | {v["teraops"]:.3f} | {v["watts"]:.3f} | {v["mj"]:.4f} | {v["minimum"]:.4f}–{v["maximum"]:.4f} | '
          f'{v["npu_w"]:.3f} | {v["gpu_w"]:.3f} |' for name,v in aggregate.items()], '',
        f'FP16 DPU相对最低整包能耗CPU路径（{best_cpu}），每次矩乘能耗比CPU为{npu["mj"]/cpu["mj"]:.4f}；'
        f'相对GPU FP16为{npu["mj"]/gpu["mj"]:.4f}。GPU/DPU吞吐比{gpu["teraops"]/npu["teraops"]:.3f}。这些是本次稳态矩乘结果，不能外推所有模型。', '',
        '## 是否达到DPU满吞吐', '',
        f'本轮最高INT8稳态均值为{aggregate["npu-int8"]["teraops"]:.3f} TOPS，相当于258V官方47 TOPS规格数值的'
        f'{aggregate["npu-int8"]["teraops"]/47*100:.1f}%。这一比值是含提交/DMA的有效吞吐与规格的比值，不是DPU利用率。',
        '已扫描M=128/512/1024、单会话、四个独立会话合并提交、四输出融合图，以及六tile编译配置；选用通过数值验证的最佳四矩阵配置测功耗。',
        '六tile图的ELF包含task.dpu.invariant.0.0至5.0；FP16非逐元素DPU amode/wmode全为0/0，INT8全为1/1。未强制Turbo或更改功率上限。',
        '因此只验证到当前边界内的高负载DPU功耗，不能称为47 TOPS满吞吐功耗。没有硬件DPU忙碌/频率计数证明所有流水线饱和，也没有扩大现有单Tensor存储边界或测试多队列。', '',
        f'- FP16 NPU域平均{npu["npu_w"]:.3f}W，完整采样区间最大{npu["npu_w_max"]:.3f}W；'
        f'INT8平均{aggregate["npu-int8"]["npu_w"]:.3f}W，区间最大{aggregate["npu-int8"]["npu_w_max"]:.3f}W。峰值为约一秒区间平均，非瞬时峰值。',
        *util, '',
        '## 工作与计量口径', '',
        '- 四组不同的(1024,1024)输入，共享固定权重，逻辑运算为四次A@W.T；一次矩乘按2*1024³有效操作计数。没有把padding、量化、复制或重复输出当成额外有效运算。',
        '- 所有设备读取相同fixture指纹。输入/权重来自非零随机INT8网格，按1/128和1/256精确缩放为FP16。CPU参考先用FP32计算完整输出，另用NumPy INT64核对每组首/中/末三行；各路径在热身后和计量后验证全部输出。',
        '- 主对照为相同FP16逻辑输入/输出；CPU FP32先计算再转FP16，这次转换计入耗时。GPU/CPU用等价的4096x1024扁平GEMM共享权重，避免人为增加Python循环。DPU四输出各自满足既有1,048,576元素界限。',
        '- INT8 DPU另列压力结果，使用相同网格得到相同逻辑输出，但不把INT8 TOPS与FP16 TFLOPS直接当成同精度硬件对比；没有用该网格宣称真实模型量化无损。',
        '- 先预热三次大块提交、验证输出，再连续测量；每块完成64次逻辑矩乘。GPU每块末同步，NPU等待完成fence，CPU同步执行。排除编译、权重打包、输入上传/最终下载；包含驱动提交、图内DMA与布局。',
        '- 两轮采用正序/反序，各配置至少25秒；CPU/GPU/NPU串行运行。稳态NPU无新编译、驱动allocation或显式Tensor上传/下载；关闭后live/cache/scalar/quarantine/session归零，无失败context。',
        '- 能量单位mJ，Duration单位ns，W=mJ*10^6/duration_ns。只取完整落在计量窗口内的约1秒区间，剔除边界；整包mJ/矩乘=平均W/该阶段完成矩乘数每秒*1000，是稳态估算。域功耗边界不同，主结论使用同一整包计量，未把各域相加。',
        '- 桌面后台、温度和电源策略未锁定；两次测量的范围不是置信区间。未测整机插座能量，未把空闲功耗相减；短任务还需计入冷编译和传输成本。', *idle, '',
        '空闲整包漂移较大，故不将这些数字称为受控硬件能效认证。仅作敏感性估算：若各阶段基线落在上述空闲范围，分别扣除最高/最低基线后，稳态mJ/矩乘范围如下；这不是独立测得的动态能量：', '',
        *[f'- {name}：{bounds[0]:.4f}–{bounds[1]:.4f}mJ/矩乘。' for name,bounds in sensitivity.items()], '',
        '这项粗略敏感性计算不改变本轮GPU相对当前DPU FP16路径的能效排序，但完整模型、其他形状和更好的DPU执行策略仍需重测。', '',
        '## 逐轮记录', '',
        '| 配置 | 实测秒 | 有效TFLOPS/TOPS | 整包W | 整包mJ/矩乘 | NPU域W | VCCGT域W |',
        '|---|---:|---:|---:|---:|---:|---:|', *rawrows, '',
        '## 吞吐校准', '',
        '| 精度 | M | 布局/提交 | 编译配置 | 有效TFLOPS/TOPS |', '|---|---:|---|---|---:|', *curves, '',
        '首轮校准错误地在同一提交组重复会话，受到已有运行时保护而拒绝；当时脚本闭包仍持有Tensor，退出断言也失败。测试脚本改为四个独立会话，并在关闭前释放引用。融合图在一个图内处理四组不同输入，没有关闭保护或修改生产运行时。正式17项校准和全部NPU测量进程均通过关闭检查后才计入结果。', '',
        '### 扁平大矩乘候选核对', '',
        f'另验证NPU图内Concat→4096x1024矩乘→Split的等价路径；外部仍为四个合法的1024²输入/输出。FP16短测{flat_values["float16"]:.3f}TFLOPS、INT8 {flat_values["int8"]:.3f}TOPS，均未超过原四矩乘融合图的相应校准值；因此主能效测量没有仅因NPU外部Tensor上限而放弃等价大矩乘。',
        '该候选全部输出和资源关闭检查通过，但没有单独做功耗采集，不能据其吞吐断言能效更差。首次Split多输出未显式命名而被绑定类型检查拒绝，改为已有命名Reshape包装后重跑通过；生产绑定代码未改。', '',
        '## 复现与限制', '',
        '入口torch_npu_gemm_energy.py、run_gemm_energy.ps1；外部功耗采集measure_npu_power.ps1。Intel工具需要管理员权限，本轮沿用用户授权并通过UAC。仅启用SoCWatch的npu-pwr/pkg-pwr/vccgt-pwr，没有加入项目运行依赖。',
        '本轮最佳DPU六tile/多输出图仅在基准中生成，不代表普通torch.mm已自动改走DPU，也不是完整模型/训练能效结论。', '',
        '本机复现（功耗命令在管理员PowerShell执行；按实际解包位置设置utRoot）：', '',
        '```powershell',
        '.\\.conda\\npunlock\\python.exe -B torch_npu_gemm_energy.py --prepare',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_gemm_energy.py -ScriptArguments @(\'--calibrate\',\'--label\',\'calibration-complete\')',
        '$utRoot = \'E:\\path\\to\\ut-tool-ext-v0.2.0-beta1.1\'',
        '.\\npunlock-probe\\research\\measure_npu_power.ps1 -ToolRoot $utRoot -Seconds 600 -IncludeGpu -Label paired -OutputRoot "$PWD\\npunlock-probe\\research\\results\\gemm-energy\\power" -Application "powershell.exe -NoProfile -File $PWD\\npunlock-probe\\research\\run_gemm_energy.ps1"',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\dpu_flat_gemm_probe.py',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/gemm_energy_report.py',
        '```', '',
        'run_gemm_energy.ps1中的GPU环境路径为本机C:\\Users\\LC\\miniconda3\\envs\\blood\\python.exe，其他机器需更换。报告生成须在原始记录仍保留时运行，审计后仅保留源码与Markdown报告。', '',
        '[Intel 258V官方47 TOPS规格](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html)；'
        '[NPU tile配置定义](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/intel_npu/properties.hpp)；'
        '[Intel Unified Telemetry](https://www.intel.com/content/www/us/en/developer/tools/intel-unified-telemetry.html)。', '',
        '## 原始记录指纹（审计后清理）', '',
        *[f'- `{p.relative_to(OUT)}` {p.stat().st_size}字节 SHA256 `{hashlib.sha256(p.read_bytes()).hexdigest()}`'
          for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz','.txt')], ''])
    (OUT/'REPORT.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(status='passed',aggregate=aggregate,ratio_npu_to_gpu=npu['mj']/gpu['mj']),indent=2))


if __name__ == '__main__':
    main()
