"""Serial, warm FP16 SmolLM generation; NPU decoder with explicit CPU ends."""
import argparse
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import statistics as st
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/SmolLM2-135M-Instruct'
OUT = ROOT / 'npunlock-probe/research/results/model-cross-device'
TOKENS, ROUNDS = 32, 7
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'


def sha(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def distribution(values):
    quartiles = st.quantiles(values, n=4, method='inclusive')
    return dict(median=st.median(values), q1=quartiles[0], q3=quartiles[2], samples=values)


def metric(actual, reference):
    import numpy as np
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    # NumPy BLAS loads a second OpenMP runtime in this conda/PyTorch process.
    return dict(relative_l2=float(np.sqrt(np.sum((a-b)**2))/max(np.sqrt(np.sum(b**2)), 1e-8)),
                max_abs=float(np.max(np.abs(a-b))), argmax_equal=bool(a.argmax() == b.argmax()))


def execute(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    npu = None
    sources = [Path(__file__), ROOT/'torch_npu_smollm.py', ROOT/'torch_intel_npu.py',
               ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd',
               ROOT/'npunlock-probe/session-runtime/npunlock_session.dll']
    report = dict(status='running', label=args.label, device=args.device, threads=args.threads,
        started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), platform=platform.platform(),
        python=platform.python_version(), packages={n:metadata.version(n) for n in
            ('torch','transformers','safetensors','tokenizers','huggingface-hub','numpy')},
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        checkpoint_sha256=sha(MODEL/'model.safetensors'), dtype='float16', decoder_layers=30,
        batch=1, new_tokens=TOKENS, rounds=ROUNDS, samples=[])
    path = OUT / (args.label+'.json')
    def save(stage):
        report['stage'] = stage
        write(path, report)
        print(args.label, stage, flush=True)
    try:
        save('initialization')
        start = time.perf_counter_ns()
        if args.device == 'npu':
            import torch_intel_npu as npu
            from torch_npu_smollm import FrozenSmolStack
            from decode_chain_api import delta
        if args.device == 'xpu':
            assert torch.xpu.is_available(), 'XPU is unavailable'
            report['device_name'] = torch.xpu.get_device_name(0)
            report['device_properties'] = str(torch.xpu.get_device_properties(0))
        report['backend_initialization_ms'] = (time.perf_counter_ns()-start)/1e6
        device = 'xpu' if args.device == 'xpu' else 'cpu'
        sync = torch.xpu.synchronize if device == 'xpu' else lambda:None

        def work():
            started = time.perf_counter_ns()
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                attn_implementation='eager', local_files_only=True).eval().to(device)
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            text = tokenizer.apply_chat_template([dict(role='system', content='Tell stories.'),
                dict(role='user', content='Tell a story about a bird.')],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, return_tensors='pt').input_ids
            assert ids.shape == (1,24) and ids.shape[1]+TOKENS-1 <= 64
            stack = FrozenSmolStack.from_checkpoint(MODEL, num_layers=30, mode='fp16') if npu else None
            sync()
            report.update(model_setup_ms=(time.perf_counter_ns()-started)/1e6, input_ids=ids.tolist(),
                          input_tokens=ids.shape[1], attention='eager HF; fixed-capacity 64-token graph for NPU',
                          placement='30 NPU decoder layers; CPU embedding/norm/head/argmax' if npu else
                          'full model and argmax on '+device)
            if device == 'xpu':
                report['xpu_model_memory_allocated_bytes'] = torch.xpu.memory_allocated()
                torch.xpu.reset_peak_memory_stats()
            fixture = None
            if args.device != 'prepare':
                with np.load(OUT/'fixture.npz', allow_pickle=False) as saved:
                    fixture = {k:saved[k].copy() for k in saved.files}
                assert np.array_equal(fixture['tokens'][:,:ids.shape[1]], ids.numpy())
                report['fixture_sha256'] = sha(OUT/'fixture.npz')

            def run(replay=None):
                before = npu.stats() if npu else None
                sync()
                begin = time.perf_counter_ns()
                generated = ids.clone().to(device)
                state = None
                logits_rows, emitted, step_ms = [], [], []
                stage_ms = []
                for step in range(TOKENS):
                    mark = begin if step == 0 else time.perf_counter_ns()
                    source = generated if step == 0 else generated[:,-1:]
                    if npu:
                        host = model.model.embed_tokens(source)
                        x = host.to('intel_npu')
                        decoder_start = time.perf_counter_ns()
                        hidden,state = stack.prefill(x) if state is None else stack.decode(x,state)
                        decoder_end = time.perf_counter_ns()
                        hidden = hidden.cpu()
                        logits = model.lm_head(model.model.norm(hidden))[:,-1].float()
                        stage_ms.append((decoder_end-decoder_start)/1e6)
                    else:
                        result = model.model(source, past_key_values=state, use_cache=True)
                        state = result.past_key_values
                        logits = model.lm_head(result.last_hidden_state)[:,-1].float()
                    token = logits.argmax(-1,keepdim=True)
                    emitted.append(token.cpu())  # Each token must be observable on the host.
                    chosen = token if replay is None else replay[:,ids.shape[1]+step:ids.shape[1]+step+1]
                    generated = torch.cat((generated, chosen), dim=1)
                    logits_rows.append(logits)
                    sync()
                    step_ms.append((time.perf_counter_ns()-mark)/1e6)
                elapsed = (time.perf_counter_ns()-begin)/1e6
                change = delta(before,npu.stats()) if npu else None
                row = dict(elapsed_ms=elapsed, first_token_ms=step_ms[0], step_ms=step_ms,
                    decode_ms=sum(step_ms[1:]), overall_tokens_per_second=32000/elapsed,
                    decode_tokens_per_second=31000/sum(step_ms[1:]), npu_delta=change)
                if npu:
                    row['decoder_ms'] = stage_ms
                    assert len(state)==30 and all(cache.length==55 for cache in state)
                # Validation/download and final KV destruction are outside the timer.
                logits_cpu = torch.stack(logits_rows).cpu().numpy().copy()
                predicted = torch.cat(emitted, dim=1).numpy().copy()
                generated_cpu = generated.cpu().numpy().copy()
                assert np.isfinite(logits_cpu).all() and generated_cpu.shape == (1,56)
                return row, logits_cpu, predicted, generated_cpu

            save('cold_generation')
            cold,anchor_logits,anchor_pred,anchor_tokens = run()
            report['cold'] = cold
            if args.device == 'prepare':
                np.savez_compressed(OUT/'fixture.npz', tokens=anchor_tokens, logits=anchor_logits)
                report['fixture_sha256'] = sha(OUT/'fixture.npz')
                report['continuation'] = tokenizer.decode(anchor_pred[0], skip_special_tokens=True)
                return
            save('warm_generation')
            for index in range(2+ROUNDS):
                row,actual,predicted,tokens = run()
                np.testing.assert_array_equal(predicted, anchor_pred)
                np.testing.assert_array_equal(tokens, anchor_tokens)
                np.testing.assert_array_equal(actual.view(np.uint8), anchor_logits.view(np.uint8))
                if npu:
                    d = row['npu_delta']
                    assert d['npu_calls']==960 and d['chain_graphs']==930 and d['chain_groups']==248
                    assert d['host_upload_copies']==d['host_download_copies']==32
                    assert all(d[k]==0 for k in ('driver_allocations','session_creations','host_device_copies','host_pack_copies'))
                if index >= 2:
                    report['samples'].append(row)
            report['summary'] = {k:distribution([row[k] for row in report['samples']]) for k in
                ('elapsed_ms','first_token_ms','decode_ms','overall_tokens_per_second','decode_tokens_per_second')}
            if npu:
                report['summary']['decoder_total_ms'] = distribution([sum(r['decoder_ms']) for r in report['samples']])
            save('fixed_token_accuracy_replay')
            replay = torch.from_numpy(fixture['tokens']).to(device)
            _,actual,predicted,tokens = run(replay)
            np.testing.assert_array_equal(tokens, fixture['tokens'])
            report['replay_errors'] = [metric(a,b) for a,b in zip(actual,fixture['logits'])]
            report['max_replay_relative_l2'] = max(r['relative_l2'] for r in report['replay_errors'])
            report['replay_argmax_matches'] = sum(r['argmax_equal'] for r in report['replay_errors'])
            report['quality_gate_passed'] = report['max_replay_relative_l2'] < .03
            report['greedy_tokens_equal_reference'] = bool(np.array_equal(anchor_tokens,fixture['tokens']))
            report['continuation'] = tokenizer.decode(anchor_pred[0], skip_special_tokens=True)
            report['repeated_logits_and_tokens_bit_equal'] = True
            evidence = OUT/(args.label+'-outputs.npz')
            np.savez_compressed(evidence, replay_logits=actual, replay_predictions=predicted,
                greedy_logits=anchor_logits, greedy_tokens=anchor_tokens)
            report['outputs_sha256'] = sha(evidence)
            if device == 'xpu':
                report['xpu_peak_memory_allocated_bytes'] = torch.xpu.max_memory_allocated()
            if npu:
                report['stats_before_shutdown'] = npu.stats()
            print(args.label, json.dumps({k:v['median'] for k,v in report['summary'].items()}), flush=True)

        with torch.no_grad():
            work()
        if npu:
            from shave_fp32_pipeline import final
            final(npu, report)
        gc.collect()
        report['status'] = 'passed'
        save('complete')
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if npu:
            report['failure_stats'] = npu.stats()
        save('failed')
        print(report['traceback'], flush=True)
        raise


def audit():
    import numpy as np
    labels = ('cpu-native-1','cpu-native-8','cpu-xpu-1','cpu-xpu-8','npu','xpu')
    rows = []
    with np.load(OUT/'fixture.npz',allow_pickle=False) as saved:
        reference = saved['logits'].copy()
        tokens = saved['tokens'].copy()
    for label in labels:
        report = json.loads((OUT/(label+'.json')).read_text())
        assert report['status']=='passed' and len(report['samples'])==ROUNDS
        assert report['fixture_sha256']==sha(OUT/'fixture.npz')
        for name,digest in report['sources'].items():assert sha(ROOT/name)==digest, name
        assert report['checkpoint_sha256']==sha(MODEL/'model.safetensors')
        path = OUT/(label+'-outputs.npz')
        assert sha(path)==report['outputs_sha256']
        with np.load(path,allow_pickle=False) as saved:
            actual = [metric(a,b) for a,b in zip(saved['replay_logits'],reference)]
            assert actual == report['replay_errors']
            np.testing.assert_array_equal(saved['greedy_tokens'][:,:24],tokens[:,:24])
        for key,value in report['summary'].items():
            numbers = [sum(r['decoder_ms']) if key=='decoder_total_ms' else r[key] for r in report['samples']]
            assert distribution(numbers)==value
        rows.append(dict(label=label,torch=report['packages']['torch'],threads=report['threads'],
            ttft_ms=report['summary']['first_token_ms']['median'],
            total_ms=report['summary']['elapsed_ms']['median'],
            tokens_per_second=32000/report['summary']['elapsed_ms']['median'],
            decode_tokens_per_second=31000/report['summary']['decode_ms']['median'],
            max_replay_relative_l2=report['max_replay_relative_l2'],
            replay_argmax_matches=report['replay_argmax_matches'],
            greedy_tokens_equal_reference=report['greedy_tokens_equal_reference']))
    write(OUT/'summary.json',dict(status='passed',rows=rows,quality_gate_passed=all(
        row['max_replay_relative_l2'] < .03 for row in rows)))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('prepare','cpu','npu','xpu','audit'), required=True)
    parser.add_argument('--threads', type=int, choices=(1,8), default=8)
    parser.add_argument('--label', default='fixture')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.device=='audit':audit()
    else:execute(args)
