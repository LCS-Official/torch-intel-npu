"""Research-only W8 calibration on quantized prefixes and projection stages.

Reuses calibrate_int8 (GPTQ adaptation, LICENSES/GPTQ.txt); layerwise replay
follows the idea in IST-DASLab/gptq opt.py at
2d65066eeb06a5c9ff5184d8cebdf33662c67faf. No upstream code is copied here.
Only token IDs, packed weights and scalar metrics are written to disk.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from torch_npu_lowbit import PackedWeight, calibrate_int8, quantize_weight
from torch_npu_smollm import FrozenSmolStack
from torch_npu_smollm_stack_demo import errors
from w8_calibration import input_name
from w8_calibration_probe import FRESH, HOLDOUT, metric
from w8_continuation_calibration import PROMPTS
from w8_full_calibration import chat
from w8_quality_probe import MODEL, ROOT, sha

OUT = ROOT / 'npunlock-probe/research/results/w8-sequential-calibration'
STAGES = (
    ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'),
    ('self_attn.o_proj.weight',),
    ('mlp.gate_proj.weight', 'mlp.up_proj.weight'),
    ('mlp.down_proj.weight',),
)
SOURCES = ['torch_npu_lowbit.py', 'torch_npu_smollm.py', 'torch_npu_transformer.py',
           'npunlock-probe/research/w8_sequential_calibration.py',
           'npunlock-probe/research/w8_continuation_calibration.py',
           'npunlock-probe/research/w8_calibration_probe.py',
           'models/SmolLM2-135M-Instruct/config.json',
           'models/SmolLM2-135M-Instruct/tokenizer.json',
           'models/SmolLM2-135M-Instruct/tokenizer_config.json']


def write(name, report):
    (OUT / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def read(name):
    return json.loads((OUT / (name + '.json')).read_text(encoding='utf-8'))


def fingerprints():
    return {name: sha(ROOT / name) for name in SOURCES}


def model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cpu = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation='eager', local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert len(cpu.model.layers) == 30
    return cpu, tokenizer


def trajectory(cpu, ids, total):
    from transformers import DynamicCache
    assert ids.shape[0] == 1 and ids.shape[1] <= total <= 64
    cache = DynamicCache(config=cpu.config)
    generated = ids.clone()
    output = cpu(ids, past_key_values=cache, use_cache=True)
    while generated.shape[1] < total:
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated = torch.cat((generated, token), 1)
        output = cpu(token, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == total
    return generated


def save_weights(label, packed):
    arrays = {}
    for i, weights in packed.items():
        for name, q in weights.items():
            arrays[f'{i}/{name}/codes'] = np.frombuffer(q.data, dtype=np.int8).reshape(q.shape)
            arrays[f'{i}/{name}/scales'] = np.frombuffer(q.scales, dtype=np.float16)
    assert len(arrays) == 420
    path = OUT / (label + '-weights.npz')
    np.savez_compressed(path, **arrays)
    return sha(path)


def load_weights(label):
    metadata = read('fit')
    assert metadata['sources'] == fingerprints()
    assert metadata['checkpoint_sha256'] == sha(MODEL / 'model.safetensors')
    path = OUT / (label + '-weights.npz')
    assert sha(path) == metadata['weight_sha256'][label]
    result = {i: {} for i in range(30)}
    with np.load(path, allow_pickle=False) as arrays:
        assert len(arrays.files) == 420
        for key in arrays.files:
            if not key.endswith('/codes'):
                continue
            index, name, _ = key.split('/')
            codes, scales = arrays[key], arrays[key[:-5] + 'scales']
            assert codes.dtype == np.int8 and scales.dtype == np.float16 and codes.ndim == 2
            result[int(index)][name] = PackedWeight(
                codes.shape, 'int8', codes.shape[1], codes.tobytes(), scales.tobytes())
    assert all(len(weights) == 7 for weights in result.values())
    return result


def reconstruct(q, x, weight):
    codes = torch.from_numpy(np.frombuffer(q.data, dtype=np.int8).reshape(q.shape).astype(np.float64))
    scales = torch.from_numpy(np.frombuffer(q.scales, dtype=np.float16).astype(np.float64))
    return ((x.double() @ codes.t()) * scales[None] - x.double() @ weight.double().t()).square().sum(0)


def fit():
    assert 'torch_intel_npu' not in sys.modules
    assert len(PROMPTS) == len(set(PROMPTS)) == 40
    assert set(PROMPTS).isdisjoint((system, prompt) for _, system, prompt in HOLDOUT + FRESH)
    started = time.perf_counter()
    cpu, tokenizer = model()
    report = dict(status='running', sources=fingerprints(), checkpoint_sha256=sha(MODEL / 'model.safetensors'),
                  criteria=dict(logits_relative_l2=.03, next_token_kl=.02), trajectories=[], matrices=[],
                  policy='Fit trajectories i%5!=4; select on i%5==4, before independent holdout. '
                         'Original FP16 row scales; public GPTQ solver unchanged. '
                         'Sequential layer/substage inputs use the existing staged CPU W8 reference.')
    for i, (system, prompt) in enumerate(PROMPTS):
        ids = chat(tokenizer, system, prompt)
        tokens = trajectory(cpu, ids, 64)
        report['trajectories'].append(dict(index=i, system=system, prompt=prompt,
                                          prefix=ids.shape[1], token_ids=tokens.tolist()))
        if (i + 1) % 8 == 0:
            print('generated calibration trajectories', i + 1, '/40', flush=True)
    fit_rows = [row for row in report['trajectories'] if row['index'] % 5 != 4]
    ids = torch.cat([torch.tensor(row['token_ids']) for row in fit_rows])
    official = cpu.model.embed_tokens(ids)
    sequential = official.clone()
    positions = torch.arange(64)[None].expand(8, -1)
    mask = torch.triu(torch.full((64, 64), -float('inf'), dtype=torch.float16), diagonal=1)[None, None]
    stack = FrozenSmolStack.from_checkpoint(MODEL, num_layers=30, mode='w8a16')
    prepared = {'simultaneous': {}, 'sequential': {}}
    for i, (original_layer, layer) in enumerate(zip(cpu.model.layers, stack.layers)):
        captured = {}
        hooks = []
        for name in ('self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj'):
            key = input_name(name + '.weight')
            captured[key] = []
            def capture(module, args, key=key):
                captured[key].append(args[0].detach().reshape(-1, args[0].shape[-1]).clone())
            hooks.append(original_layer.get_submodule(name).register_forward_pre_hook(capture))
        try:
            outputs = []
            for chunk in official.split(8):
                outputs.append(original_layer(chunk, position_ids=positions, attention_mask=mask,
                                              position_embeddings=cpu.model.rotary_emb(chunk, positions), use_cache=False))
            official = torch.cat(outputs)
        finally:
            for hook in hooks:
                hook.remove()
        observed = {key: torch.cat(values) for key, values in captured.items()}
        prepared['simultaneous'][i] = {}
        prepared['sequential'][i] = {}
        for names in STAGES:
            key = input_name(names[0])
            chunks = [layer.reference_cpu(chunk)[1][key].reshape(-1, 1536 if key == 'gated' else 576)
                      for chunk in sequential.split(8)]
            actual = torch.cat(chunks)
            assert actual.shape == observed[key].shape and actual.shape[0] == 2048
            drift = errors(actual, observed[key])['relative_l2']
            for name in names:
                weight = original_layer.get_parameter(name).detach().half().contiguous()
                rtn = quantize_weight(weight, format='int8')
                for label, x in (('simultaneous', observed[key]), ('sequential', actual)):
                    q = calibrate_int8(weight, x)
                    before, after = reconstruct(rtn, x, weight), reconstruct(q, x, weight)
                    assert torch.all(after <= before + 1e-18) and q.scales == rtn.scales
                    prepared[label][i][name] = q
                    report['matrices'].append(dict(label=label, layer=i, name=name, rows=x.shape[0],
                                                   input_relative_l2=drift, rtn_sse=float(before.sum()),
                                                   calibrated_sse=float(after.sum())))
                layer._packed[name] = prepared['sequential'][i][name]
        sequential = torch.cat([layer.reference_cpu(chunk)[0] for chunk in sequential.split(8)])
        print('calibrated layer', i + 1, '/30, hidden drift', errors(sequential, official)['relative_l2'], flush=True)
    assert len(report['matrices']) == 420
    report['weight_sha256'] = {label: save_weights(label, packed) for label, packed in prepared.items()}
    report.update(status='passed', elapsed_seconds=time.perf_counter() - started, npu_calls=0)
    assert 'torch_intel_npu' not in sys.modules
    write('fit', report)


def summary(rows):
    return dict(positions=len(rows), quality_passed=sum(row['quality_gate_passed'] for row in rows),
                mean_relative_l2=float(np.mean([row['relative_l2'] for row in rows])),
                max_relative_l2=max(row['relative_l2'] for row in rows),
                max_kl=max(row['next_token_kl'] for row in rows),
                next_token_equal=sum(row['next_token_equal'] for row in rows))


def ranking(record):
    return (record['positions'] - record['quality_passed'], record['max_relative_l2'], record['mean_relative_l2'])


def evaluate_cpu(independent=False):
    metadata = read('fit')
    cpu, tokenizer = model()
    stacks = {label: FrozenSmolStack.from_checkpoint(MODEL, num_layers=30, mode='w8a16', packed_weights=load_weights(label))
              for label in ('simultaneous', 'sequential')}
    stacks['fp16'] = FrozenSmolStack.from_checkpoint(MODEL, num_layers=30, mode='fp16')
    cases = []
    if independent:
        selection = read('selection')  # Freeze selection before looking at independent metrics.
        for tag, system, prompt in HOLDOUT + FRESH:
            ids = chat(tokenizer, system, prompt)
            count = 32 if tag in ('bird', 'garden', 'cat') else 1
            cases.append((tag, ids.shape[1] - 1, trajectory(cpu, ids, ids.shape[1] + count - 1)))
        ids = tokenizer('Once upon a time, a curious child asked how birds fly.', return_tensors='pt').input_ids
        ids = ids.repeat(1, (32 + ids.shape[1]) // ids.shape[1])[:, :32]
        cases.append(('repeat32', 31, trajectory(cpu, ids, 64)))
    else:
        for row in metadata['trajectories']:
            if row['index'] % 5 == 4:
                cases.append((str(row['index']), row['prefix'] - 1, torch.tensor(row['token_ids'])))
    report = dict(status='running', sources=fingerprints(), weight_sha256=metadata['weight_sha256'],
                  criteria=metadata['criteria'], cases=[], positions={label: [] for label in stacks}, npu_calls=0)
    for tag, offset, ids in cases:
        baseline = cpu(ids, use_cache=False).logits[:, offset:].float()
        report['cases'].append(dict(tag=tag, offset=offset, token_ids=ids.tolist()))
        for label, stack in stacks.items():
            hidden = cpu.model.embed_tokens(ids)
            for layer in stack.layers:
                hidden, _ = layer.reference_cpu(hidden)
            logits = cpu.lm_head(cpu.model.norm(hidden))[:, offset:].float()
            assert logits.shape == baseline.shape and torch.isfinite(logits).all()
            for j in range(logits.shape[1]):
                record = metric(logits[:, j], baseline[:, j])
                # Independent FP64 checks guard metric bookkeeping without storing vocabulary-sized arrays.
                a, b = logits[:, j].double(), baseline[:, j].double()
                l2 = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-5))
                kl = float((b.log_softmax(-1).exp() * (b.log_softmax(-1) - a.log_softmax(-1))).sum())
                assert abs(record['relative_l2'] - l2) < 1e-6 and abs(record['next_token_kl'] - kl) < 1e-5, (tag, j, record, l2, kl)
                report['positions'][label].append(dict(tag=tag, position=offset + j, **record))
        print('CPU quality', tag, {label: round(report['positions'][label][-1]['relative_l2'], 5) for label in stacks}, flush=True)
    report['summary'] = {label: summary(rows) for label, rows in report['positions'].items()}
    if independent:
        report['selected'] = selection['selected']
    else:
        report['selected'] = min(('simultaneous', 'sequential'), key=lambda label: ranking(report['summary'][label]))
    assert 'torch_intel_npu' not in sys.modules
    report.update(status='passed', npu_backend_imported=False)
    write('independent-cpu' if independent else 'selection', report)
    print(json.dumps(dict(selected=report['selected'], summary=report['summary']), indent=2), flush=True)


def selfcheck():
    rows = [dict(quality_gate_passed=True, relative_l2=.01, next_token_kl=.001, next_token_equal=True),
            dict(quality_gate_passed=False, relative_l2=.04, next_token_kl=.003, next_token_equal=False)]
    record = summary(rows)
    assert record['positions'] == 2 and record['quality_passed'] == record['next_token_equal'] == 1
    assert ranking(record) == (1, .04, .025)
    assert len({name for names in STAGES for name in names}) == 7
    assert [input_name(names[0]) for names in STAGES] == ['norm1', 'attention', 'norm2', 'gated']
    print('PASS: stage ordering, quality summary and candidate ranking self-checks')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('fit', 'selection', 'independent-cpu', 'selfcheck'), required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        selfcheck()
        if args.case == 'fit':
            fit()
        elif args.case != 'selfcheck':
            evaluate_cpu(args.case == 'independent-cpu')


if __name__ == '__main__':
    main()
