"""Build an isolated, host-instrumented session DLL; no production source edits."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'npunlock-probe/research/results/dpu-diagnostic'
source = ROOT / 'npunlock/src/graphinfer/session_win32.c'
extra = ROOT / 'npunlock-probe/research/dpu_diagnostic_extra.c'
OUT.mkdir(parents=True, exist_ok=True)
text = source.read_text(encoding='utf-8')

def change(old, new, count=1):
    global text
    assert text.count(old) == count, (old, text.count(old))
    text = text.replace(old, new)

change('struct graphinfer_session {', '''static uint64_t diag_clock(void) {
  LARGE_INTEGER now;
  QueryPerformanceCounter(&now);
  return (uint64_t)now.QuadPart;
}
struct graphinfer_session {''')
change('  uint32_t device_id;\n};', '''  uint32_t device_id;
  uint64_t diag_ticks[9];
  void *diag_pool;
  npunlock_ze_graph_profiling_query_handle diag_query;
  uint32_t(__cdecl *diag_pool_destroy)(void *);
  uint32_t(__cdecl *diag_query_destroy)(void *);
  uint32_t(__cdecl *diag_get_data)(void *, uint32_t, uint32_t *, uint8_t *);
};''')
change('  free(session->arguments);', '''  if (!owner->failed && session->diag_query) session->diag_query_destroy(session->diag_query);
  if (!owner->failed && session->diag_pool) session->diag_pool_destroy(session->diag_pool);
  free(session->arguments);''')
change('  queue_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;',
       '  uint64_t diag_t;\n  queue_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;')
change('  uint32_t code;\n  for (index = 0; index < session->argument_count; ++index) {',
       '  uint32_t code;\n  uint64_t diag_t, diag_begin = diag_clock();\n  memset(session->diag_ticks, 0, sizeof(session->diag_ticks));\n  for (index = 0; index < session->argument_count; ++index) {')

def time_call(call, slot, count=1):
    change(call, 'diag_t = diag_clock();\n  ' + call + f'\n  session->diag_ticks[{slot}] += diag_clock() - diag_t;', count)

time_call('code = session->graph.set_argument(session->handle, (uint32_t)index, allocations[index]);', 0, 2)
time_call('code = session->ze.command_list_create(session->context, session->device, &list_desc, &list);', 1)
time_call('code = session->ze.fence_create(queue, &fence_desc, &fence);', 2)
change('session->graph.append_execute(list, session->handle, NULL, NULL, 0, NULL)',
       'session->graph.append_execute(list, session->handle, session->diag_query, NULL, 0, NULL)')
time_call('code = initialize ? session->graph.append_initialize(list, session->handle, NULL, 0, NULL)\n                    : session->graph.append_execute(list, session->handle, session->diag_query, NULL, 0, NULL);', 3)
time_call('code = session->ze.command_list_close(list);', 4)
time_call('code = session->ze.command_queue_execute(queue, 1, &list, fence);', 5)
time_call('code = session->ze.fence_synchronize(fence, session->fence_timeout_ns);', 6)
change('done:\n  if (!session->context_owner->failed && fence != NULL)',
       'done:\n  diag_t = diag_clock();\n  if (!session->context_owner->failed && fence != NULL)')
change('  return status;\n}\n\nstatic npunlock_status session_create(',
       '  session->diag_ticks[7] += diag_clock() - diag_t;\n  return status;\n}\n\nstatic npunlock_status session_create(')
change('  return submit(session, false, diagnostic);',
       '  status = submit(session, false, diagnostic);\n  session->diag_ticks[8] = diag_clock() - diag_begin;\n  return status;')
text += '\n' + extra.read_text(encoding='utf-8')
generated = OUT/'session_diagnostic.c'
generated.write_text(text, encoding='utf-8')
compiler = Path('C:/Program Files/mingw64/bin/gcc.exe')
sources = [ROOT/'npunlock'/p for p in ('src/common/buffer.c','src/common/checked.c','src/common/diagnostic.c','src/common/status.c','src/graphinfer/graphinfer.c')]+[generated]
args = [str(compiler),'-shared','-O2','-std=c17','-Wall','-Wextra','-Werror','-Wpedantic','-static-libgcc','-DNPUNLOCK_BUILDING','-DNPUNLOCK_SHARED']
for path in ('include','src','src/common','src/graphinfer'):
    args += ['-I', str(ROOT/'npunlock'/path)]
args += [str(p) for p in sources]+['-o',str(OUT/'session_diagnostic.dll')]
env = dict(os.environ, PATH=str(compiler.parent)+';'+os.environ['PATH'])
result = subprocess.run(args,env=env,capture_output=True,text=True)
(OUT/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
print(result.stdout+result.stderr)
result.check_returncode()
(OUT/'build.json').write_text(json.dumps(dict(args=args,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (*sources,source,extra,Path(__file__))},dll_sha256=hashlib.sha256((OUT/'session_diagnostic.dll').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print('built',OUT/'session_diagnostic.dll')
