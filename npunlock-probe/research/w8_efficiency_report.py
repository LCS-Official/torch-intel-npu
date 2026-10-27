"""Audit temporary efficiency evidence and retain a self-contained Markdown report."""
import json
import statistics as st
from pathlib import Path

import w8_efficiency as bench


def median(rows, key): return st.median(row[key] for row in rows)
def mib(value): return value/(1024*1024)
def f(value): return f'{value:.3f}'


def audit(data, sample_count):
    assert data['status'] == 'passed' and set(data['ready']) == set(bench.LABELS)
    assert data['freeze'] == bench.read('freeze')
    for name, digest in data['sources'].items(): assert digest == bench.sha(bench.ROOT/name)
    for label in bench.LABELS:
        ready = data['ready'][label]; rows = data['samples'][label]
        assert len(rows) == sample_count and len(ready['graphs']) == 60
        expected = {'fp16':0, 'w8':0, 'mixed':60, 'selected':45}[label]
        assert len(ready['protected']) == expected
        assert len(set(map(tuple, ready['protected']))) == expected
        if label == 'mixed':
            assert set(map(tuple, ready['protected'])) == {(i, 'mlp.'+name+'_proj.weight')
                for i in range(30) for name in ('up', 'down')}
        elif label == 'selected':
            assert set(map(tuple, ready['protected'])) == set(map(tuple, bench.read('selection')['selected']))
        for kind in ('prefill', 'decode'):
            graphs = [g for g in ready['graphs'] if g['key'][-1] == kind]
            assert len(graphs) == 30 and {g['layer'] for g in graphs} == set(range(30))
        for row in [ready['cold'], *ready['warmups'], *rows]:
            assert len(row['steps']) == 32
            assert row['logits_sha256'] == ready['cold']['logits_sha256']
            assert abs(row['decode_ms']-sum(sum(v.values()) for v in row['steps'][1:])) < 1e-8
            assert row['elapsed_ms'] >= sum(sum(v.values()) for v in row['steps'])
            assert abs(row['decode_tokens_per_second']-31000/row['decode_ms']) < 1e-8
            assert row['memory']['peak_commit_bytes'] >= row['memory']['private_bytes'] > 0
            assert row['memory']['peak_working_set_bytes'] >= row['memory']['working_set_bytes'] > 0
            assert row['quality']['max_relative_l2'] < .03 if label == 'fp16' else True
        for row in [*ready['warmups'], *rows]:
            delta = row['delta']
            assert delta['npu_calls'] == 960 and delta['chain_groups'] == 248 and delta['chain_graphs'] == 930
            assert delta['session_creations'] == 0 and delta['driver_allocations'] == 0
            assert delta['host_upload_copies'] == delta['host_download_copies'] == 32
        closed = data['closed'][label]
        assert closed['kind'] == 'closed' and not closed['stats']['context_failed']
        for key in ('live_buffers', 'live_buffer_bytes', 'cached_buffer_bytes', 'quarantined_buffer_bytes', 'cached_sessions'):
            assert closed['stats'][key] == 0
        assert closed['stats']['npu_calls'] == sum(r['delta']['npu_calls'] for r in [ready['cold'], *ready['warmups'], *rows])


def main():
    data = bench.read('benchmark'); profile = bench.read('profile'); startup = bench.read('startup')
    audit(data, 7); audit(profile, 3)
    assert tuple(map(tuple, data['orders'])) == tuple(bench.orders())
    assert startup['status'] == 'passed' and startup['caches_removed']
    assert startup['script_sha256'] == bench.sha(bench.ROOT/'npunlock-probe/research/w8_efficiency_startup.py')
    for label in bench.LABELS:
        assert profile['ready'][label]['cold']['logits_sha256'] == data['ready'][label]['cold']['logits_sha256']
        assert len(startup['variants'][label]['rows']) == 3
        for i, entry in enumerate(startup['variants'][label]['rows']):
            ready = entry['ready']; cache = ready['cache']
            assert ready['cold']['logits_sha256'] == data['ready'][label]['cold']['logits_sha256']
            assert cache['compiles'] == (60 if i == 0 else 0) and cache['hits'] == (0 if i == 0 else 60)
            assert cache['ir_skipped'] == (0 if i == 0 else 30)
            assert cache['rejected'] == cache['write_skipped'] == 0
            stats = entry['closed']['stats']
            assert stats['closed'] and not stats['context_failed']
            assert all(stats[k] == 0 for k in ('live_buffer_bytes', 'cached_buffer_bytes', 'quarantined_buffer_bytes', 'cached_sessions'))
    names = dict(fp16='全FP16', w8='校准W8', mixed='混合50%', selected='逐层37.5%')
    lines = ['# 校准W8与混合精度：端到端效率验收', '',
        '2026-10-16，Windows / Core Ultra 7 258V / Lunar Lake NPU4000 `8086:643e`。',
        '使用现有本地SmolLM2-135M-Instruct、Torch 2.14.1+cpu，CPU 8线程、interop 1线程；'
        '完整30层decoder与KV在NPU，embedding/final norm/lm_head显式在CPU。', '',
        '本轮只新增研究脚本与文档，未修改生产适配器、量化器、内核、C++、DLL/PYD或默认配置，未推送GitHub。', '',
        '## 固定工作量与计时范围', '',
        '- 复用上一轮的同一份逐行GPTQ校准W8，权重NPZ SHA256为 `'+data['freeze']['weights_sha256']+'`。',
        '- 四种既定策略：全FP16；全部210个投影W8；所有层Up/Down保留FP16（50%投影元素）；'
        '之前选定45个Up/Down保留FP16（37.5%）。不根据此次结果重新选择保护项。',
        '- batch=1、24-token提示、32个预测位置、最终KV长度55，容量64；不遇EOS提前停止。'
        '每种配置都消费CPU FP16生成的同一条冻结token轨迹，同时实际计算自身argmax。'
        '这是相同计算量的teacher-forced性能比较，不是各精度沿自身路径的自由生成。',
        '- 每种配置单独进程，首次无磁盘图缓存运行后预热2次，正式7轮交替；只允许一个进程执行NPU任务。'
        '四个进程在正式轮次都驻留，前四轮每种配置在每个顺序位置恰好一次，七轮各位置出现1或2次。',
        '- 热首token包括CPU embedding、上传、完整prefill、下载、CPU norm/head、argmax和token bookkeeping；'
        '沿用当前参考路径，在prefill的全部24个位置计算CPU norm/head，然后取最后位置。',
        '- 解码统计后31步；tokens/s=31/解码秒数。32-token整体速度另按32/总秒数计算。'
        '计时包含同步及显式CPU两端，不含加载、首次编译和区间后的hash/数值验收。',
        '- 编译图常量与实际进程内存分开列出；不把W8A16当作全部INT8运算，不以本测试推算芯片TOPS。',
        '- 桌面其他负载、系统调度与频率未锁定；7轮中位数和范围只描述本次本机条件。', '',
        'CPU数值参考为官方FP16模型对完整55-token序列做一次causal前向，取位置23..54；'
        '冻结token轨迹本身通过CPU DynamicCache逐步生成。两种CPU执行调度可能有舍入差异。'
        '提示与历史性能测试相同，只作为性能回归，不是新增独立质量集。', '',
        '## 生产运行时：热请求', '',
        '| 配置 | 热首token ms | 首token范围 ms | 解码 ms/token | 解码 token/s | 整体32-token token/s | 解码总耗时范围 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label in bench.LABELS:
        rows = data['samples'][label]; decode = median(rows, 'decode_ms')
        lines.append(f"| {names[label]} | {f(median(rows,'first_token_ms'))} | "
            f"{f(min(r['first_token_ms'] for r in rows))}–{f(max(r['first_token_ms'] for r in rows))} | {f(decode/31)} | {f(31000/decode)} | "
            f"{f(32000/median(rows,'elapsed_ms'))} | {f(min(r['decode_ms'] for r in rows))}–{f(max(r['decode_ms'] for r in rows))} |")
    lines += ['', '逐轮解码ms/token，列顺序固定为FP16/W8/混合50%/逐层37.5%；实际执行顺序另列：', '',
              '| 轮次 | FP16 | W8 | 混合50% | 逐层37.5% | 实际顺序 |', '|---|---:|---:|---:|---:|---|']
    for i, order in enumerate(data['orders']):
        lines.append('| '+str(i+1)+' | '+' | '.join(f(data['samples'][label][i]['decode_ms']/31) for label in bench.LABELS)+
                     ' | '+' → '.join(names[label] for label in order)+' |')
    lines += ['', '相对FP16的同轮配对解码耗时降低（正数为更快）：', '',
              '| 配置 | 配对中位降低 | 更快轮数/7 | 配对范围 |', '|---|---:|---:|---:|']
    for label in bench.LABELS[1:]:
        changes = [100*(1-a['decode_ms']/b['decode_ms']) for a,b in zip(data['samples'][label],data['samples']['fp16'])]
        lines.append(f'| {names[label]} | {st.median(changes):.2f}% | {sum(v>0 for v in changes)}/7 | {min(changes):.2f}%–{max(changes):.2f}% |')
    lines += ['', '## 首次编译与已有磁盘缓存的新进程', '',
        '| 配置 | 无缓存首token s，1次 | 无缓存脚本入口到首token s | 已有缓存首token s，2次中位 | 缓存首token范围 s | 已有缓存入口到首token s | 临时缓存 MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label in bench.LABELS:
        ready = data['ready'][label]; rows = startup['variants'][label]['rows'][1:]
        ttft = [r['ready']['cold']['first_token_ms']/1000 for r in rows]
        entry = [r['ready']['entry_to_first_token_ms']/1000 for r in rows]
        lines.append(f"| {names[label]} | {f(ready['cold']['first_token_ms']/1000)} | {f(ready['entry_to_first_token_ms']/1000)} | "
            f"{f(st.median(ttft))} | {f(min(ttft))}–{f(max(ttft))} | {f(st.median(entry))} | {f(mib(startup['variants'][label]['temporary_cache_bytes']))} |")
    lines += ['', '无缓存初始化拆分（完整首次请求包含30份prefill与30份decode图）：', '',
        '| 配置 | 后端初始化 ms | CPU模型与适配器准备 ms | 60份图的compile API总计 s | session创建 ms |',
        '|---|---:|---:|---:|---:|']
    for label in bench.LABELS:
        ready = data['ready'][label]; hooks = ready['compile_hooks']
        assert hooks['compile_inclusive']['calls'] == 60
        lines.append(f"| {names[label]} | {f(ready['backend_ms'])} | {f(ready['setup_ms'])} | "
            f"{f(hooks['compile_inclusive']['ms']/1000)} | {f(hooks['session_create']['ms'])} |")
    lines += ['', 'compile API为inclusive host计时，包含编译与SHAVE patch，不含调用前的IR序列化；'
        '60份图总计还包含首token之后才创建的decode图，不能加到首token耗时上。', '',
        '无缓存首token的1次数据取主测速进程；缓存数据来自每种配置另外两个全新进程。'
        '每种配置先在一个临时目录建立60份prefill/decode图，再验证两个新进程均60次命中、0次编译。'
        '四种策略的30份prefill命中均跳过IR重建；decode仍使用完整建图的缓存键。',
        '缓存目录位于系统临时目录，每种策略完成后删除，最多同时保留一份、不超过512MiB，磁盘图缓存默认仍关闭。'
        '无缓存编译与缓存启动样本数少，不作稳定速度排名。脚本入口计时包括导入、加载、权重准备和后端初始化，'
        '不含PowerShell或Python进程创建之前的时间。', '',
        '## 实际进程内存与图常量', '',
        '| 配置 | 热请求后私有提交 MiB，中位 | 生命周期峰值提交 MiB | 热工作集 MiB，中位 | 峰值工作集 MiB | 30份decode图常量 MiB | prefill+decode图常量 MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label in bench.LABELS:
        rows = data['samples'][label]; ready = data['ready'][label]; mem = data['closed'][label]['memory']
        graphs = ready['graphs']; decode = sum(g['evidence']['constant_bytes'] for g in graphs if g['key'][-1] == 'decode')
        all_bytes = sum(g['evidence']['constant_bytes'] for g in graphs)
        lines.append(f"| {names[label]} | {f(mib(st.median(r['memory']['private_bytes'] for r in rows)))} | {f(mib(mem['peak_commit_bytes']))} | "
            f"{f(mib(st.median(r['memory']['working_set_bytes'] for r in rows)))} | {f(mib(mem['peak_working_set_bytes']))} | {f(mib(decode))} | {f(mib(all_bytes))} |")
    lines += ['', '通过Windows GetProcessMemoryInfo读取：PrivateUsage为私有提交量；PeakPagefileUsage为该进程生命周期的峰值提交，'
        '不是已写入磁盘的pagefile字节；WorkingSet/PeakWorkingSet为驻留页，可能包含共享页，不能直接相加。'
        '生命周期峰值包含导入、模型/校准权重加载、编译、运行和区间外验收，不是单个内核的显存峰值。',
        '现有测试路径保留完整CPU模型，NPU适配器又保留原FP16权重；W8配置还保留packed权重、图blob、session常量与运行缓冲。'
        '因此图常量压缩不等于整进程同比压缩。四进程驻留可能影响工作集与缓存，私有提交量更适合此处比较，'
        '本轮不声称测到了全部驱动/固件内存。', '',
        '## 热运行分段：生产DLL', '',
        '每个数字先取一次请求后31步的平均ms/token，再取7轮中位数。分项中位数不必恰好相加为总中位数。', '',
        '| 配置 | '+ ' | '.join(bench.STAGES)+' |', '|---|'+'---:|'*len(bench.STAGES)]
    for label in bench.LABELS:
        values = [st.median(st.mean(r['steps'][i][key] for i in range(1,32)) for r in data['samples'][label]) for key in bench.STAGES]
        lines.append('| '+names[label]+' | '+' | '.join(f(v) for v in values)+' |')
    lines += ['', '热首token分段ms（7轮中位数）：', '',
        '| 配置 | '+ ' | '.join(bench.STAGES)+' |', '|---|'+'---:|'*len(bench.STAGES)]
    for label in bench.LABELS:
        values = [st.median(r['steps'][0][key] for r in data['samples'][label]) for key in bench.STAGES]
        lines.append('| '+names[label]+' | '+' | '.join(f(v) for v in values)+' |')
    lines += ['', 'decoder分项同时包含Python/ATen准备、图绑定、驱动提交、等待与设备执行；'
        'upload/download仅为显式Tensor复制，不能从它们推断内部DMA不存在。', '',
        '## 提交与等待：隔离的诊断DLL', '',
        '复用既有QPC host profiler构建临时DLL，不替换生产二进制。每种配置单独进程、2次预热、3次采样；'
        '其延迟不混入上面的生产数据。wait包含NPU执行、排队及完成通知等待，不等于纯DPU计算时间。', '',
        '| 配置 | 诊断decoder ms/token | fence wait ms/token | append_graph ms/token | set_argument ms/token | 其他计数API ms/token | decoder减计数API ms/token |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label in bench.LABELS:
        rows = profile['samples'][label]
        dec = st.median(st.mean(row['steps'][i]['decoder'] for i in range(1,32)) for row in rows)
        stages = {key:st.median(st.mean(row['host'][i]['ms'][key] for i in range(1,32)) for row in rows)
                  for key in rows[0]['host'][0]['ms']}
        other = sum(value for key,value in stages.items() if key not in ('wait','append_graph','set_argument'))
        residual = st.median(st.mean(row['steps'][i]['decoder']-sum(row['host'][i]['ms'].values()) for i in range(1,32)) for row in rows)
        lines.append('| '+names[label]+' | '+' | '.join(f(v) for v in (dec, stages['wait'], stages['append_graph'], stages['set_argument'], other, residual))+' |')
    lines += ['', '残差包含未计数的Python/ATen、校验、内存准备与计时开销，不单独认定为Python时间；'
        '计数API逐项相加的微小计时误差与各项中位数顺序也会影响分解。'
        '每次prefill为30次等待；每次30层decode按4层分组，共8次等待。', '',
        '## 数值、执行路线与资源验收', '',
        '| 配置 | 固定32个位置对CPU FP16平均logits相对L2 | 最大相对L2 | 旧3%门槛内位置/32 | 同配置所有正式重复logits逐位一致 |',
        '|---|---:|---:|---:|---|']
    for label in bench.LABELS:
        quality = data['ready'][label]['cold']['quality']
        lines.append(f"| {names[label]} | {quality['mean_relative_l2']*100:.4f}% | {quality['max_relative_l2']*100:.4f}% | {quality['within_old_3_percent']}/32 | 是 |")
    total = sum(record['closed'][label]['stats']['npu_calls'] for record in (data,profile) for label in bench.LABELS)
    total += sum(entry['closed']['stats']['npu_calls'] for value in startup['variants'].values() for entry in value['rows'])
    lines += ['', '- 全部正式请求每次960次层图调用；31次decode各30图，248个chain groups；'
        '热区间无新session或驱动allocation，恰好32次显式上传、32次下载。',
        '- 每种配置的60份图均检查DPU模式：全FP16没有整数权重矩阵任务；其余各层prefill/decode均包含真实W8A16矩阵任务（amode=0/wmode=2）。',
        '- 生产、诊断DLL及所有缓存新进程对同配置的32个logits全部逐位一致；全部关闭后live/cache/quarantine/session归零，无context failure。',
        f'- 本轮累计 **{total:,}次真实NPU调用**。计时用参考token轨迹，不把自身argmax与参考一致率当成自由生成正确率。',
        '- 旧logits门槛保留，原PPL/自由生成报告仍为质量依据；32个固定位置不新增通用质量认证。', '',
        '## 复现', '', '```powershell', "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:PYTHONIOENCODING='utf-8'", "$env:NPUNLOCK_GRAPH_CACHE=$null",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_efficiency.py -ScriptArguments '--case','selfcheck'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_efficiency.py -ScriptArguments '--case','prepare'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_efficiency.py -ScriptArguments '--case','benchmark'",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_efficiency.py -ScriptArguments '--case','profile'",
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_efficiency_startup.py',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\w8_efficiency_report.py', '```', '',
        '## 指纹', '', '| 对象 | SHA256 |', '|---|---|']
    sources = dict(data['sources'])
    for path in ('npunlock-probe/research/w8_efficiency_startup.py', 'npunlock-probe/research/w8_efficiency_report.py'):
        sources[path] = bench.sha(bench.ROOT/path)
    sources['models/SmolLM2-135M-Instruct/model.safetensors'] = data['freeze']['checkpoint_sha256']
    for path, digest in sources.items(): lines.append(f'| `{path}` | `{digest}` |')
    lines += ['', '临时验收材料清理前的指纹（复现需重新prepare，清理后不能直接离线重算）：', '',
              '| 材料 | SHA256 |', '|---|---|']
    for name in ('prepare.json', 'weights.npz', 'selection.json', 'freeze.json', 'reference.npz',
                 'benchmark.json', 'profile.json', 'startup.json'):
        lines.append(f'| `{name}` | `{bench.sha(bench.OUT/name)}` |')
    lines += ['', '| 配置 | 32个固定位置的NPU logits SHA256 |', '|---|---|']
    for label in bench.LABELS:
        lines.append(f"| {names[label]} | `{data['ready'][label]['cold']['logits_sha256']}` |")
    temporary = [p for p in bench.OUT.rglob('*') if p.is_file() and p.name != 'REPORT.md']
    size = sum(p.stat().st_size for p in temporary)
    cache_size = sum(value['temporary_cache_bytes'] for value in startup['variants'].values())
    lines += ['', '## 交付与清理', '',
        f'验收后按用户约定删除本轮{len(temporary)}份临时文件，共{size:,}字节（{mib(size):.2f}MiB）：'
        '校准权重、参考logits、token轨迹、原始计时JSON和生成的诊断C/DLL/构建清单。'
        '保留3份新增研究脚本、既有profiler构建源码及本报告。',
        f'4份系统临时图缓存共{mib(cache_size):.2f}MiB、240个文件，已逐种策略删除；'
        '它们没有同时驻留磁盘，单份最多438.36MiB。没有下载模型、安装依赖或产生图片。']
    base_rate = 31000/median(data['samples']['fp16'], 'decode_ms')
    changes = {label:100*((31000/median(data['samples'][label], 'decode_ms'))/base_rate-1) for label in bench.LABELS}
    intro = ['## 本轮判断与下一步', '',
        f"全W8相对FP16的解码吞吐变化为{changes['w8']:+.2f}%，混合50%为{changes['mixed']:+.2f}%，"
        f"逐层37.5%为{changes['selected']:+.2f}%。本轮低位的主要收益是减少内存，"
        '没有显示数量级的单请求加速；混合50%的速度差尤其小，应结合逐轮波动阅读。',
        '热首token的中位数均约22–23ms；首轮若干配置升至43–51ms，已经保留全部范围，'
        '不据此承诺低位精度改善热首token。原有独立质量门槛仍未全面通过，FP16默认不变。',
        'FP16热decode的显式上传与下载合计约0.29ms/token，占总耗时约2%；decoder约11.54ms/token，'
        '占约84%。诊断DLL显示每token约7.4–7.9ms在fence等待、约1ms在提交，'
        '另有约2.2–2.3ms未计数的准备开销。等待包含设备执行与排队，尚不能把它归因于某种硬件引擎或内部DMA。',
        'prefill提前命中已覆盖FP16、校准W8与混合策略。后续启动优化应测量CPU模型/权重准备及ELF/DPU重复解析；'
        '稳态解码另需比较更长的层链式提交，并以设备任务profile分解等待时间。', '',
        '本轮没有扩展INT4/NF4、上下文长度、batch范围或通用PyTorch算子，也未重新认证训练路径。', '']
    lines[lines.index('## 固定工作量与计时范围'):lines.index('## 固定工作量与计时范围')] = intro
    (bench.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Audit passed; report written. NPU calls:', total)
    for label in bench.LABELS:
        rows = data['samples'][label]
        print(label, 'TTFT ms', median(rows,'first_token_ms'), 'decode tok/s', 31000/median(rows,'decode_ms'))


if __name__ == '__main__': main()
