"""Audit sustained-training raw records and retain a compact Markdown report."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from torch_npu_sustained_train import engine_interval
OUT = ROOT/'npunlock-probe/research/results/sustained-training'


def audit(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status'] == 'passed', result.get('error')
    assert len(result['warmup']) >= 2
    assert result['elapsed_seconds'] >= result['config']['seconds']
    before, after = result['stats_before'], result['stats_after']
    assert before['session_creations'] == after['session_creations']
    assert before['live_buffers'] == after['live_buffers']
    assert before['live_buffer_bytes'] == after['live_buffer_bytes']
    assert result['final_parameters_and_gradients_finite']
    assert not result.get('telemetry_error')
    final = result['final_stats']
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'live_buffer_bytes', 'cached_buffer_blocks',
                                      'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    assert result['counter']['npu']['vendor'] == 0x8086 and result['counter']['npu']['device'] == 0x643e
    assert result['engine_utilization'] == engine_interval(result['begin_counter'], result['end_counter'])
    for row in result['steps']:
        assert row['compute_copies'] == 1  # No payload copies; four-byte CE validation flag only.
    assert after['host_upload_copies'] == before['host_upload_copies']
    assert after['host_download_copies']-before['host_download_copies'] == 2*len(result['steps'])
    for name, digest in result['source_sha256'].items():
        # Calibration predates the single-step guard and memory-telemetry fix.
        if path.stem.startswith('sustained') or name in ('torch_intel_npu.py', 'torch_npu_autograd.py'):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    return result


def main():
    paths = sorted(OUT.glob('*.json'))
    assert paths
    rejected = [path for path in paths if json.loads(path.read_text(encoding='utf-8'))['status'] != 'passed']
    for path in rejected:
        r = json.loads(path.read_text(encoding='utf-8'))
        assert path.stem == 'calibrate-128' and len(r['warmup']) == 2 and len(r['steps']) == 1
        assert r['steps'][0]['loss'] < r['warmup'][-1]['loss']
        assert r['error'].endswith('AssertionError: Loss did not decrease\n')
        assert r['final_stats']['closed'] and not r['final_stats']['context_failed']
        assert all(r['final_stats'][k] == 0 for k in ('live_buffers', 'cached_buffer_blocks',
            'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    paths = [path for path in paths if path not in rejected]
    results = [audit(path) for path in paths]
    rows = []
    for path, r in zip(paths, results):
        c, steps = r['config'], r['steps']
        util = r['engine_utilization']['0:0']['process_percent']
        rows.append(f'| {path.stem} | {c["parameters"]:,} | {r["elapsed_seconds"]:.3f} | {len(steps)} | '
                    f'{statistics.mean(s["compute_seconds"] for s in steps):.3f} | {util:.2f}% | '
                    f'{steps[0]["loss"]:.6f} → {steps[-1]["loss"]:.6f} |')
    sustained = [r for r in results if r['config']['seconds'] >= 180]
    assert sustained, 'No three-minute steady training run'
    details = []
    for r in sustained:
        c, steps = r['config'], r['steps']
        samples = [r['begin_counter'], *r['telemetry'], r['end_counter']]
        intervals = [(samples[i+1]['time']-samples[i]['time'],
                      engine_interval(samples[i], samples[i+1])['0:0']['process_percent'])
                     for i in range(len(samples)-1)]
        assert all(0 <= value <= 101 for _, value in intervals), intervals
        seconds95 = sum(duration for duration, value in intervals if value >= 95)
        sample_values = [value for _, value in intervals]
        before, after = r['stats_before'], r['stats_after']
        memory = r['npu_memory_after']
        peak_shared = max(row['npu_memory']['process_shared']['resident'] for row in r['telemetry']
                          if row['npu_memory'].get('process_shared') is not None)
        details.extend([
            f'### {c["label"]}', '',
            f'- {c["blocks"]}层、hidden={c["width"]}、{c["heads"]}头、FFN={c["width"]*4}；batch={c["batch"]}、sequence={c["sequence"]}、{c["dtype"]}，{c["parameters"]:,}参数。',
            f'- 稳态训练{r["elapsed_seconds"]:.3f}秒，{len(steps)}步；{len(steps)/r["elapsed_seconds"]:.4f}步/秒，{len(steps)*c["batch"]*c["sequence"]/r["elapsed_seconds"]:.3f}序列位置/秒。包含反向及SGD更新，非LLM生成tokens/s。',
            f'- NPU进程/整机引擎平均忙碌率：{r["engine_utilization"]["0:0"]["process_percent"]:.2f}% / {r["engine_utilization"]["0:0"]["all_percent"]:.2f}%；约1秒区间min/median/max：{min(sample_values):.2f}% / {statistics.median(sample_values):.2f}% / {max(sample_values):.2f}%；>=95%的采样区间合计{seconds95:.2f}秒。',
            f'- {len(r["telemetry"])}份独立定时遥测；进程CPU时间{r["process_cpu_seconds"]:.3f}秒，约{r["process_cpu_seconds"]/r["elapsed_seconds"]*100:.2f}%单核当量。',
            f'- 本次冷启动首步{r["warmup"][0]["seconds"]:.3f}秒，第二次热身{r["warmup"][1]["seconds"]:.3f}秒；两者不计入上述稳态时长。',
            f'- loss（含热身）{r["warmup"][0]["loss"]:.8f} → {steps[-1]["loss"]:.8f}；最终全部参数和梯度有限。',
            f'- 稳态NPU调用{after["npu_calls"]-before["npu_calls"]:,}次，{(after["npu_calls"]-before["npu_calls"])/len(steps):.0f}次/步；会话固定{before["cached_sessions"]}个，稳态新建图0。驱动allocation仍增加{after["driver_allocations"]-before["driver_allocations"]}次：块缓存达到{after["cached_buffer_blocks"]}块上限，存在分配/回收周转，但live与缓存字节没有增长。',
            f'- 参数{c["parameter_bytes"]/1048576:.3f}MiB；步末设备live {before["live_buffer_bytes"]/1048576:.3f}MiB且无漂移，张量分配峰值{after["peak_buffer_bytes"]/1048576:.3f}MiB，步末缓存{after["cached_buffer_bytes"]/1048576:.3f}MiB。这些数值不含驱动内部图/临时区。',
            f'- 进程RSS开始/结束{r["memory_before"]["rss"]/1048576:.2f}/{r["memory_after"]["rss"]/1048576:.2f}MiB；本进程RSS峰值{r["memory_after"]["peak_rss"]/1048576:.2f}MiB。',
            f'- Windows DXCore进程共享committed步末{memory["process_shared"]["committed"]/1048576:.3f}MiB，进程resident字段为{memory["process_shared"]["resident"]}字节、采样resident峰值{peak_shared/1048576:.3f}MiB；整NPU共享resident步末{memory["adapter_shared"]["resident"]/1048576:.3f}MiB，dedicated {memory["process_dedicated"]["resident"]}字节。进程resident读数持续0，不能据此说未使用NPU内存；用committed与整适配器resident交叉核对。',
            '- 稳态每步仅显式读取一次4字节分类错误标志，并额外下载4字节loss用于健康检查；输入、参数与训练算术留在NPU。计数不包含驱动内部DMA。', ''])
    power_lines = []
    if (OUT/'power/clock-after.json').exists():
        from npu_power_report import report_lines
        power_lines = report_lines(OUT)
    text = '\n'.join([
        '# 较大 Attention 模型持续训练与真实 NPU 忙碌率', '',
        '日期：2026-10-18。本地开发版。Core Ultra 7 258V / Intel AI Boost 8086:643e / driver API 1005540 / 图编译器8.3 / Python 3.12.14 / Torch 2.14.1+cpu。', '',
        '标准PyTorch Module、SDPA、CrossEntropy和SGD，设备intel_npu:0。每层为独立Q/K/V、因果注意力、输出投影、两次LayerNorm、GELU前馈与残差，最后mean pooling及四分类头。固定合成数据用于负载与数值稳定测试，不代表真实数据集质量或LLM训练。', '',
        '## 实测', '',
        '| 配置 | 参数 | 稳态秒数 | 步数 | 平均计算秒/步 | NPU进程忙碌率 | 稳态loss |',
        '|---|---:|---:|---:|---:|---:|---|', *rows, '', *details,
        '## 校验与解释', '',
        '- 每个配置先做两步独立CPU训练，逐项核对loss、全部参数梯度和更新；CPU单线程、禁用MKLDNN、SDPA MATH，随机种子103。FP32 rtol=3e-5/atol=3e-6（FP16入口为.003/.0005）。没有放宽原有容差。',
        '- 热身后连续执行训练循环，无人工sleep、无CPU算术回退；稳态结束才检查最终全部参数/梯度。所有运行关闭后live/cache/scalar/quarantine/session为0，无失败context。',
        '- DXCore按vendor/device选出AI Boost，读取专属LUID及引擎0:0，分别统计进程和整机累计运行微秒差/墙钟差。核显Arc 140V的Neural引擎被排除。监控线程约每秒采样；端点差得到全程平均，分段峰值不当成全程平均。',
        '- 该指标表示Windows调度引擎的忙碌比例，含图内DMA/布局和SHAVE工作，不代表DPU阵列、全部SHAVE流水线利用率、40 TOPS达成率或实际TFLOPS。当前训练路径仍是SHAVE FP32数学实现。',
        '- 驱动不支持DXCore温度或频率查询，三分钟原测试未测功耗/温度、不能据此断言没有降频；后续用户授权的独立功耗采集另列。',
        '- 大量展开乘加、分块、布局搬运及逐图同步仍限制吞吐。高引擎忙碌率本身不能证明当前算子高效；后续优先专用矩乘/批量矩乘和减少中间Tensor/提交次数，同时处理256块缓存上限造成的allocation周转。',
        '- 新增离线自检覆盖DXCore ABI、进程/整机占用分离、无效进程计数及计数重置。生产Python后端、C++/PYD和SHAVE ELF未改。', '',
        '## 内存读数与短测误判', '',
        '79万参数模型FP32权重约3.03MiB，梯度同量级；逐算子工作区复用，不预占17.9GB。任务管理器以GB显示一位小数时，几十MiB以内可能显示0.0。引擎持续高忙碌与小内存占用不矛盾；也不通过多分配闲置内存制造压力。DXCore的驱动共享resident、后端张量分配和进程RSS是不同口径，不相加当作显存。', '',
        '初次width128短测只设10秒，而一步约30秒，只完成一次稳态更新。旧脚本将这一步loss与自身比较，误报“Loss did not decrease”；两步CPU对照均通过、loss实际继续下降、资源正常关闭，但该短测不计为通过。已修正为与最后热身loss比较，随后完整三分钟运行使用新脚本通过；没有追改旧结果为成功。', '',
        *power_lines,
        '## 复现', '', '```powershell',
        '$env:NPUNLOCK_GRAPH_CACHE=$null',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_sustained_train.py -ScriptArguments @(\'--width\',\'128\',\'--blocks\',\'4\',\'--seconds\',\'180\',\'--label\',\'sustained-128\')',
        '.\\.conda\\npunlock\\python.exe -B -m unittest test_torch_npu_sustained_train -v',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/sustained_train_report.py',
        '```', '',
        '报告生成脚本须在原始JSON仍保留时执行；完成审计后按用户约定清理原始JSON，只保留脚本与报告。未修改source-release，未推送GitHub。', '',
        '## 监控API参考', '',
        '- [Microsoft DXCore设备枚举](https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-enum-adapters)',
        '- [DXCoreAdapterState运行时间/温度/频率契约](https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterstate)',
        '- [DXCoreAdapterProperty设备LUID/引擎属性](https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterproperty)', '',
        '### 本轮实现指纹', '',
        *[f'- `{name}` SHA256 `{digest}`' for name, digest in results[-1]['source_sha256'].items()], '',
        '### 原始记录指纹（审计后清理）', '',
        *[f'- `{path.name}` {path.stat().st_size}字节，SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`'
          for path in sorted(OUT.rglob('*')) if path.is_file() and path.suffix in ('.json','.txt')], ''])
    (OUT/'REPORT.md').write_text(text, encoding='utf-8')
    print(f'Audited {len(results)} runs; wrote {OUT/"REPORT.md"}')


if __name__ == '__main__':
    main()
