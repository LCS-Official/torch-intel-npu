"""Convert Intel UT energy intervals to watts and align with steady training."""
import json
from pathlib import Path
import re
import statistics

LINE = re.compile(r'Metric:\s+(\S+).*?Value:\s+([0-9]+(?:\.[0-9]+)?)\s+mJ\s+\|\s+'
                  r'Timestamp:\s+(\d+)\s+\|\s+Duration:\s+(\d+)')


def parse(text, accepted=('PMT-NPU-PWR', 'PKG-PWR')):
    rows = []
    for line in text.splitlines():
        if 'Metric:' not in line:
            continue
        match = LINE.search(line)
        if match is None:
            raise ValueError(f'Unexpected energy record: {line[:200]}')
        name, energy, timestamp, duration = match.groups()
        assert name in accepted, name
        rows.append(dict(metric=name, mj=float(energy), ns=int(timestamp), duration_ns=int(duration)))
    assert rows, 'No energy records'
    return rows


def summarize(rows, begin=None, end=None):
    selected = [r for r in rows if r['duration_ns'] >= 500_000_000 and
                (begin is None or r['ns']-r['duration_ns'] >= begin) and
                (end is None or r['ns']+r['duration_ns'] <= end)]
    assert selected, 'No complete energy intervals in this phase'
    assert all(b['ns'] > a['ns'] for a, b in zip(selected, selected[1:])), 'Repeated/reversed energy timestamps'
    watts = [r['mj']*1e6/r['duration_ns'] for r in selected]
    return dict(count=len(selected), sampled_seconds=sum(r['duration_ns'] for r in selected)/1e9,
                energy_j=sum(r['mj'] for r in selected)/1000,
                average_w=sum(r['mj'] for r in selected)*1e6/sum(r['duration_ns'] for r in selected),
                min_w=min(watts), median_w=statistics.median(watts), max_w=max(watts))


def read_log(path):
    data = path.read_bytes()
    return data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')


def report_lines(out):
    power = out/'power'
    idle = json.loads((power/'power-complete.json').read_text(encoding='utf-8-sig'))
    load = json.loads((power/'load/power-complete.json').read_text(encoding='utf-8-sig'))
    assert idle['exit_code'] == load['exit_code'] == 0
    state = json.loads((power/'driver-state.json').read_text(encoding='utf-8-sig'))
    assert state['monitor_drivers_remaining'] == 0 and state['collector_processes_remaining'] == 0
    metadata = json.loads((power/'tool-metadata.json').read_text(encoding='utf-8'))
    first = json.loads((power/'clock-before.json').read_text())
    last = json.loads((power/'clock-after.json').read_text())
    drift = last['unix_offset_ns']-first['unix_offset_ns']
    assert max(first['uncertainty_ns'], last['uncertainty_ns']) < 2_000_000
    assert abs(drift) < 10_000_000, f'UTC/QPC alignment drift: {drift}ns'
    offset = (first['unix_offset_ns']+last['unix_offset_ns'])//2
    training = json.loads((out/'power-load-128.json').read_text())
    assert training['status'] == 'passed'
    begin = round(training['begin_counter']['time']*1e9)+offset
    end = round(training['end_counter']['time']*1e9)+offset
    idle_rows = parse(read_log(power/'power-output.txt'))
    load_rows = parse(read_log(power/'load/power-output.txt'))
    lines = ['## NPU独立功耗实测（2026-10-19）', '',
             '经用户授权使用Windows UAC，Intel Unified Telemetry v0.2.0-beta1.1 + SoCWatch；ut.exe与socwatchdrv.sys签名均为有效Intel Corporation。仅开启npu-pwr/pkg-pwr；这是独立PMT NPU能量计，不是从整包功耗相减估算。', '',
             '| 阶段 | 功率域 | 完整采样数 | 采样秒数 | 平均W | 区间最小/中位/最大W |',
             '|---|---|---:|---:|---:|---|']
    measurements = {}
    for phase, records, start, finish in (('空闲', idle_rows, None, None),
                                         ('相同模型稳态训练', load_rows, begin, end)):
        for metric in ('PMT-NPU-PWR', 'PKG-PWR'):
            values = summarize([r for r in records if r['metric'] == metric], start, finish)
            measurements[phase, metric] = values
            lines.append(f'| {phase} | {metric} | {values["count"]} | {values["sampled_seconds"]:.3f} | '
                         f'{values["average_w"]:.4f} | {values["min_w"]:.4f} / {values["median_w"]:.4f} / {values["max_w"]:.4f} |')
    assert measurements['相同模型稳态训练', 'PMT-NPU-PWR']['average_w'] > 0, 'NPU energy counter did not respond to load'
    lines.extend(['',
        f'- 功耗重跑相同793,604参数FP32网络，热身后{training["elapsed_seconds"]:.3f}秒、{len(training["steps"])}步，NPU平均忙碌{training["engine_utilization"]["0:0"]["process_percent"]:.2f}%；数值/生命周期检查通过。原三分钟测试仍单独保留，不把本次采样冒充当时功耗。',
        '- 工具输出能量mJ与Duration纳秒，按 `W = mJ × 10^6 / duration_ns` 换算，时间加权平均为总能量/总采样时间；未将mJ直接当成W。空闲0表示工具在当前分辨率下报告0，并非证明物理漏电绝对为0。',
        f'- 单调时钟/UTC两端偏移漂移{drift/1e6:.4f}ms；仅选完整落在稳态窗口内的样本，并保守剔除边界窗口。原始约1秒采样，区间峰值不能捕捉更短瞬时峰值。',
        '- 整包读数包含CPU、其他片上域和后台工作；不等于NPU功耗，也未将各功率域相加。高调度忙碌率下的功耗仍只代表当前SHAVE实现，不能外推DPU满吞吐功耗或能效。',
        '- 工具正常结束后再次检查socwatchdrv服务与UT进程均不存在，临时驱动由工具自动卸载，未手工修改系统NPU驱动。官方工具只用于外部监控，没有加入后端运行依赖；下载件、解包目录和原始采集数据在审计后清理。',
        '- 功耗较低本身不能证明任务能效领先；该训练仍约30秒/步，尚未对相同任务的CPU/GPU能量或外置功率计做对照。', '',
        '复现功耗采集：从下方Intel页面下载便携工具，在管理员PowerShell执行（ToolRoot指向解包根目录；运行前可加-PrepareOnly审阅）：', '',
        '```powershell',
        '.\\npunlock-probe\\research\\measure_npu_power.ps1 -ToolRoot <Intel-UT-root> -Seconds 260 -Label load',
        '```', '',
        '原始日志的稳态筛选与单位换算实现见`npu_power_report.py`。本轮官方下载件指纹：', '',
        *[f'- `{Path(name).name}` SHA256 `{digest}`' for name, digest in metadata['sha256'].items()], '',
        '[Intel Unified Telemetry功能与Lunar Lake支持说明](https://www.intel.com/content/www/us/en/developer/tools/intel-unified-telemetry.html)', ''])
    return lines
