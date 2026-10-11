"""One-shot, checked promotion of the measured chain prototype."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'npunlock-probe/research/results/decode-chain-api'


def replace(path, old, new):
    text = path.read_text(encoding='utf-8-sig')
    assert text.count(old) == 1, (path, old[:80])
    path.write_text(text.replace(old, new), encoding='utf-8')


def main():
    before = OUT / 'before'
    names = ['npunlock/include/npunlock/graphinfer.h','npunlock/src/graphinfer/session_win32.c',
             'npunlock/src/workers/level_zero_min.h','npunlock/tests/unit/test_session_failure.c',
             'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/session-runtime/build.json',
             'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd',
             'npunlock-probe/torch-device/build.json','torch_intel_npu.py','torch_npu_smollm.py',
             'build_npu_session.ps1','TORCH_NPU_README.md']
    assert not before.exists(), 'Promotion is deliberately one-shot'
    hashes = {}
    for name in names:
        source = ROOT / name; target = before / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (before/'sources.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    header = ROOT/'npunlock/include/npunlock/graphinfer.h'
    replace(header, 'typedef struct graphinfer_session_infer_result {', '''/* Bindings and sessions are borrowed until the synchronous chain returns. */
typedef struct graphinfer_chain_entry {
  uint32_t struct_size;
  graphinfer_session *session;
  const graphinfer_shared_tensor *inputs;
  size_t input_count;
  const graphinfer_shared_tensor *outputs;
  size_t output_count;
} graphinfer_chain_entry;

typedef struct graphinfer_session_infer_result {''')
    replace(header, 'NPUNLOCK_GRAPHINFER_API void\ngraphinfer_session_infer_result_release', '''/* Execute 1..4 distinct graphs in order in one context. Validate all bindings
 * before execution; outputs cannot overwrite an earlier input/output. A later
 * graph may consume an earlier output. Cold/recreated sessions run sequentially
 * for initialization; warmed sessions share one submission and final fence.
 * All borrowed objects must remain alive throughout the call, including while
 * acquiring the context lock. Uncertain completion quarantines the context. */
NPUNLOCK_GRAPHINFER_API npunlock_status graphinfer_session_infer_chain(
    const graphinfer_chain_entry *entries, size_t count, graphinfer_session_infer_result *result);

NPUNLOCK_GRAPHINFER_API void
graphinfer_session_infer_result_release''')
    minimal = ROOT/'npunlock/src/workers/level_zero_min.h'
    replace(minimal, 'typedef uint32_t(__cdecl *npunlock_ze_command_list_reset_fn)', '''typedef uint32_t(__cdecl *npunlock_ze_command_list_append_barrier_fn)(
    npunlock_ze_command_list_handle, npunlock_ze_event_handle, uint32_t, npunlock_ze_event_handle *);
typedef uint32_t(__cdecl *npunlock_ze_command_list_reset_fn)''')
    native = ROOT/'npunlock/src/graphinfer/session_win32.c'
    replace(native, '  npunlock_ze_command_list_close_fn command_list_close;',
            '  npunlock_ze_command_list_close_fn command_list_close;\n  npunlock_ze_command_list_append_barrier_fn command_list_append_barrier;')
    text = (ROOT/'npunlock-probe/research/decode_chain_extra.c').read_text(encoding='utf-8')
    text = text[text.index('__declspec(dllexport)'):]
    text = text.replace('__declspec(dllexport) npunlock_status diagnostic_infer_chain', 'npunlock_status graphinfer_session_infer_chain')
    text = text.replace('diagnostic_chain_entry', 'graphinfer_chain_entry').replace('diagnostic.chain', 'graphinfer.chain')
    text = text.replace('  typedef uint32_t(__cdecl *barrier_fn)(npunlock_ze_command_list_handle, void *, uint32_t, void *);\n  barrier_fn barrier = NULL;\n', '')
    text = text.replace('  bool recorded = false;', '  bool recorded = false, cold = false;')
    start = text.index('  if (count > 1 && !load_export('); end = text.index('  /* Validate', start)
    text = text[:start] + text[end:]
    text = text.replace('    if (!s || s->context_owner', '    if (entries[i].struct_size < sizeof(entries[i]) || !s || s->context_owner')
    text = text.replace(' || !s->initialized ||\n        !s->list || !s->fence ||', ' ||')
    marker = '    for (j = 0; j < i; ++j) if (entries[j].session == s) goto invalid;'
    text = text.replace(marker, marker+'\n    cold |= !s->initialized || !s->list || !s->fence;')
    marker = '  for (i = 0; i < count; ++i) {\n    graphinfer_session *session = entries[i].session;'
    inserted = '''  /* Reject forward references and overwrites, allowing producer -> consumer. */
  for (i = 0; i < count; ++i)
    for (j = 0; j < entries[i].session->argument_count; ++j)
      if (entries[i].session->arguments[j].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT)
        for (size_t earlier = 0; earlier < i; ++earlier)
          for (k = 0; k < entries[earlier].session->argument_count; ++k)
            if (allocations[i][j] == allocations[earlier][k]) goto invalid;
  if (cold || count == 1) {
    for (i = 0; i < count; ++i) {
      status = execute_allocations_locked(entries[i].session, allocations[i], &result->diagnostic);
      if (status != NPUNLOCK_STATUS_OK) break;
    }
    goto done;
  }
  if (!first->context_owner->ze.command_list_append_barrier &&
      !load_export(first->context_owner->loader, "zeCommandListAppendBarrier",
          &first->context_owner->ze.command_list_append_barrier,
          sizeof(first->context_owner->ze.command_list_append_barrier))) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
        "graphinfer.chain", "barrier export unavailable");
    goto done;
  }
'''
    assert text.count(marker) == 1
    text = text.replace(marker, inserted+marker)
    text = text.replace('code = barrier(', 'code = first->context_owner->ze.command_list_append_barrier(')
    text = text.replace('cold/closed/duplicate session', 'closed/duplicate session')
    import re
    text = re.sub(r'CHAIN_TIME\(\d, (code = [^\n]+)\);', r'\1;', text)
    assert 'CHAIN_TIME' not in text and 'diagnostic_' not in text
    with native.open('a', encoding='utf-8') as file: file.write('\n'+text)
    backend = ROOT/'npunlock-probe/torch-device/backend.cpp'
    replace(backend, '  decltype(&graphinfer_session_infer) infer = nullptr;',
            '  decltype(&graphinfer_session_infer) infer = nullptr;\n  decltype(&graphinfer_session_infer_chain) infer_chain = nullptr;')
    extra = (ROOT/'npunlock-probe/research/decode_chain_extra.cpp').read_text(encoding='utf-8')
    extra = extra[extra.index('void diagnostic_chain'):].replace('diagnostic_chain(', 'run_chain(')
    extra = extra.replace('diagnostic', 'NPU')
    start = extra.index('  using Fn ='); end = extra.index('  std::vector<std::vector', start)
    extra = extra[:start] + extra[end:]
    extra = extra.replace('DiagnosticChainEntry', 'graphinfer_chain_entry')
    extra = extra.replace('entries.push_back({reinterpret_cast', 'entries.push_back({sizeof(graphinfer_chain_entry), reinterpret_cast')
    extra = extra.replace('auto status=infer(', 'auto status=s.infer_chain(')
    # diagnostic() was an existing helper, not a diagnostic-only entry.
    extra = extra.replace('auto message=NPU(result.NPU);', 'auto message=diagnostic(result.diagnostic);')
    extra = extra.replace('    s.calls+=count;', '    s.calls+=count; s.chain_groups++; s.chain_graphs+=count;')
    replace(backend, 'void run(uintptr_t session,', extra+'\nvoid run(uintptr_t session,')
    replace(backend, '  LOAD(infer, graphinfer_session_infer);',
            '  LOAD(infer, graphinfer_session_infer);\n  LOAD(infer_chain, graphinfer_session_infer_chain);')
    replace(backend, '  int64_t allocations = 0, live = 0, bytes = 0, peak = 0, calls = 0;',
            '  int64_t allocations = 0, live = 0, bytes = 0, peak = 0, calls = 0;\n  int64_t chain_groups = 0, chain_graphs = 0;')
    replace(backend, 'std::array<int64_t, 14> values;', 'std::array<int64_t, 16> values;')
    replace(backend, 'static_cast<int64_t>(s.cached.size()), s.cached_bytes};',
            'static_cast<int64_t>(s.cached.size()), s.cached_bytes, s.chain_groups, s.chain_graphs};')
    replace(backend, '"cached_buffer_blocks", "cached_buffer_bytes"};',
            '"cached_buffer_blocks", "cached_buffer_bytes", "chain_groups", "chain_graphs"};')
    replace(backend, '  m.def("run_many", run_many, pybind11::call_guard<pybind11::gil_scoped_release>());',
            '  m.def("run_many", run_many, pybind11::call_guard<pybind11::gil_scoped_release>());\n  m.def("run_chain", run_chain, pybind11::call_guard<pybind11::gil_scoped_release>());')
    print('Native API and C++ entry promoted; before snapshot saved', flush=True)


if __name__ == '__main__': main()
