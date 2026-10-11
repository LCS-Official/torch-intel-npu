"""Compare the first late-state convolution against CPU FP16 and FP64 rounding."""
import copy
import gc
import json
import torch
import torch_intel_npu as npu
from diagnose_fp16_training import OUT, metrics
from torch_npu_training_demo import make_small_resnet


def compare():
    saved = torch.load(OUT / 'fp16-residual-late.pt', weights_only=True)
    network = make_small_resnet(torch.float16)
    network.load_state_dict(saved['state'])
    cpu = network.layers[0]
    model = copy.deepcopy(cpu).to('intel_npu')
    data = saved['input'].to('intel_npu')
    with torch.no_grad():
        reference = cpu(saved['input'])
        exact = copy.deepcopy(cpu).double()(saved['input'].double()).half()
        actual = model(data).cpu()
    mask = actual != reference
    return {'npu_vs_cpu': metrics(actual, reference), 'npu_vs_fp64_rounded': metrics(actual, exact),
            'cpu_vs_fp64_rounded': metrics(reference, exact),
            'cpu_differing_values': reference[mask].tolist(), 'npu_differing_values': actual[mask].tolist(),
            'fp64_rounded_differing_values': exact[mask].tolist()}


if __name__ == '__main__':
    torch.backends.mkldnn.enabled = False
    report = compare()
    gc.collect()
    stats = npu.stats()
    assert not stats['context_failed']
    assert stats['live_buffers'] == stats['cached_scalar_buffers'], stats
    npu.shutdown()
    report['final_stats'] = npu.stats()
    (OUT / 'fp16-late-stem-rounding.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
