"""Isolated OpenVINO baseline against the exact saved npunlock IR and inputs.

Run with .conda/openvino-bench/Scripts/python.exe; no torch dependency.
Input/output access and model compilation are outside the warm infer timing.
"""
import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov

OUT = Path(__file__).resolve().parent/'results/dpu-diagnostic'
SHAPES = ((1,1024,1024),(128,1024,1024),(1024,1024,1024),(2048,512,256))


def properties(target, names):
    result = {}
    for name in names:
        try:
            result[name] = str(target.get_property(name))
        except RuntimeError as error:
            result[name] = {'unavailable': str(error)}
    return result


def execute(tiles_test=False):
    core = ov.Core()
    assert 'NPU' in core.available_devices, core.available_devices
    keys = ('FULL_DEVICE_NAME','DEVICE_ARCHITECTURE','NPU_DRIVER_VERSION',
            'NPU_COMPILER_VERSION','NPU_COMPILER_TYPE','NPU_MAX_TILES','NPU_TILES',
            'NPU_TURBO','PERFORMANCE_HINT','EXECUTION_MODE_HINT','EXECUTION_DEVICES')
    report = dict(status='running',openvino_version=ov.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  device_properties={},
                  supported_properties=str(core.get_property('NPU','SUPPORTED_PROPERTIES')),
                  scope='resident InferRequest buffers; synchronous infer; explicit NPU, no fallback; '
                        'same saved IR, weights and fixture as native replay; share_outputs=True; '
                        '5 warmups, 31 rotating rounds',
                  tiles_test=tiles_test,rows=[],successful_graph_calls=0)
    # Core has a two-argument get_property, while CompiledModel has one.
    for key in keys:
        try:
            report['device_properties'][key] = str(core.get_property('NPU',key))
        except RuntimeError as error:
            report['device_properties'][key] = {'unavailable':str(error)}
    assert report['device_properties']['NPU_DRIVER_VERSION'] == '1005540'
    calls = 0
    try:
        for shape in (((128,1024,1024),(1024,1024,1024)) if tiles_test else SHAPES):
            m,k,n = shape
            tag = '-'.join(map(str,shape))
            with np.load(OUT/f'fixture-{tag}.npz',allow_pickle=False) as fixture:
                expected = fixture['expected'].copy()
                hosts = {'fp16-fq':fixture['fp16'].copy(),'prequant-u8':fixture['u8'].copy()}
            prepared = []
            for path,host in hosts.items():
                xml = OUT/f'{tag}-{path}.xml'
                weights = xml.with_suffix('.bin')
                model = core.read_model(str(xml))
                if tiles_test and path != 'prequant-u8': continue
                configurations = ((c,t) for c in ('DRIVER','PLUGIN') for t in (4,6)) if tiles_test else ((c,None) for c in ('DRIVER','PLUGIN'))
                for compiler,tiles in configurations:
                    config = {'PERFORMANCE_HINT':'LATENCY','EXECUTION_MODE_HINT':'ACCURACY',
                              'NPU_COMPILER_TYPE':compiler}
                    if tiles is not None: config['NPU_TILES'] = tiles
                    start = time.perf_counter_ns()
                    compiled = core.compile_model(model,'NPU',config)
                    compile_ms = (time.perf_counter_ns()-start)/1e6
                    request = compiled.create_infer_request()
                    inp = request.get_input_tensor().data
                    output = request.get_output_tensor().data
                    assert inp.shape == host.shape and inp.dtype == host.dtype,(inp.shape,inp.dtype)
                    assert output.shape == expected.shape and output.dtype == expected.dtype
                    np.copyto(inp,host)
                    request.infer(share_outputs=True); calls += 1
                    np.testing.assert_array_equal(output,expected)
                    inp.fill(128 if path == 'prequant-u8' else 0)
                    request.infer(share_outputs=True); calls += 1
                    np.testing.assert_array_equal(output,np.zeros_like(expected))
                    np.copyto(inp,host)
                    request.infer(share_outputs=True); calls += 1
                    np.testing.assert_array_equal(output,expected)
                    exported = compiled.export_model()
                    if hasattr(exported,'getvalue'): exported = exported.getvalue()
                    assert isinstance(exported,bytes)
                    blob = OUT/(f'tiles-{tag}-{path}-{compiler.lower()}-{tiles}.blob' if tiles_test else
                                f'openvino-{tag}-{path}-{compiler.lower()}.blob')
                    blob.write_bytes(exported)
                    row = dict(shape_mkn=list(shape),path=path,compiler=compiler,tiles=tiles,config=config,
                               compile_ms=compile_ms,compiled_properties=properties(compiled,keys),
                               input_dtype=str(inp.dtype),output_dtype=str(output.dtype),
                               bit_exact_grid=True,changed_input_and_restore_passed=True,
                               ir_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                               weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                               fixture_sha256=hashlib.sha256((OUT/f'fixture-{tag}.npz').read_bytes()).hexdigest(),
                               blob_file=blob.name,export_sha256=hashlib.sha256(exported).hexdigest(),
                               samples_ms=[])
                    prepared.append((request,compiled,row,output))
            for request,compiled,row,output in prepared:
                for _ in range(5): request.infer(share_outputs=True); calls += 1
            for round_index in range(31):
                offset = round_index%len(prepared)
                order = prepared[offset:]+prepared[:offset]
                if round_index%2: order = list(reversed(order))
                for request,compiled,row,output in order:
                    start = time.perf_counter_ns()
                    request.infer(share_outputs=True)
                    row['samples_ms'].append((time.perf_counter_ns()-start)/1e6)
                    calls += 1
            for request,compiled,row,output in prepared:
                np.testing.assert_array_equal(output,expected)
                median = statistics.median(row['samples_ms'])
                row.update(median_ms=median,min_ms=min(row['samples_ms']),max_ms=max(row['samples_ms']),
                           effective_tops=2*m*k*n/(median*1e9))
                report['rows'].append(row)
                print(json.dumps({key:row[key] for key in ('shape_mkn','path','compiler','tiles','median_ms','effective_tops')}),flush=True)
            del request,compiled,output,inp,prepared,order
            gc.collect()
        report['status'] = 'passed'
    finally:
        report['successful_graph_calls'] = calls
        (OUT/('openvino-tiles.json' if tiles_test else 'openvino.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-test',action='store_true')
    raise SystemExit(execute(parser.parse_args().tiles_test))
