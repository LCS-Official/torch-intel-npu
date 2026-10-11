/* Exercise the exact experimental chain against the existing fake driver. */
#define main original_session_failure_main
#include "../../npunlock/tests/unit/test_session_failure.c"
#undef main

static unsigned barrier_calls, execute_calls, wait_calls, chain_count;
static uint32_t barrier_result, close_result, set_result;
static void *graph_bindings[2][2], *chain_bindings[4][2];

static uint32_t chain_barrier(npunlock_ze_command_list_handle list, void *signal,
                               uint32_t count, void *waits) {
  (void)list; (void)signal; (void)count; (void)waits;
  assert(recorded && !pending); ++barrier_calls; return barrier_result;
}

static bool chain_load_export(HMODULE loader, const char *name, void *destination, size_t size) {
  typedef uint32_t(__cdecl *fn)(npunlock_ze_command_list_handle, void *, uint32_t, void *);
  fn pointer = chain_barrier;
  (void)loader;
  assert(strcmp(name, "zeCommandListAppendBarrier") == 0 && size == sizeof(pointer));
  memcpy(destination, &pointer, size); return true;
}

#define load_export chain_load_export
#include "decode_chain_extra.c"
#undef load_export

static uint32_t chain_set(npunlock_ze_graph_handle graph, uint32_t index, const void *pointer) {
  size_t slot = (size_t)(uintptr_t)graph - 1;
  assert(slot < 2 && index < 2);
  graph_bindings[slot][index] = (void *)pointer; return set_result;
}

static uint32_t chain_append(npunlock_ze_command_list_handle list, npunlock_ze_graph_handle graph,
                           npunlock_ze_graph_profiling_query_handle profiling,
                           npunlock_ze_event_handle signal, uint32_t count,
                           npunlock_ze_event_handle *waits) {
  size_t slot = (size_t)(uintptr_t)graph - 1;
  (void)list; (void)profiling; (void)signal; (void)count; (void)waits;
  assert(!pending && slot < 2 && chain_count < 4); ++append_calls;
  if (!append_result) {
    memcpy(chain_bindings[chain_count++], graph_bindings[slot], sizeof(graph_bindings[slot]));
    recorded = true;
  }
  return append_result;
}

static uint32_t chain_close(npunlock_ze_command_list_handle list) {
  (void)list; return close_result;
}

static uint32_t chain_execute(npunlock_ze_command_queue_handle queue, uint32_t count,
                            npunlock_ze_command_list_handle *lists, npunlock_ze_fence_handle fence) {
  (void)queue; (void)count; (void)lists; (void)fence;
  assert(recorded && !signaled && !pending && chain_count == 2);
  ++execute_calls; pending = true;
  if (!execute_result) for (unsigned i = 0; i < chain_count; ++i)
    *(float *)chain_bindings[i][1] = *(float *)chain_bindings[i][0] + 1.0f;
  return execute_result;
}

static uint32_t chain_wait(npunlock_ze_fence_handle fence, uint64_t timeout) {
  ++wait_calls; return mock_wait(fence, timeout);
}

static void check_chain(unsigned stage) {
  graphinfer_session *first = mock_session(), *second = mock_session();
  graphinfer_session_result root = {0}, child = {0};
  graphinfer_session_infer_result run = {0};
  graphinfer_shared_buffer buffers[3] = {{0}};
  graphinfer_shared_tensor in[2] = {{0}}, out[2] = {{0}};
  diagnostic_chain_entry entries[2];
  void *pointers[3];
  bool uncertain = stage == 3 || stage == 4 || stage == 5;
  second->context_owner = first; InterlockedIncrement(&first->references);
  second->handle = (npunlock_ze_graph_handle)(uintptr_t)2;
  first->queue = (npunlock_ze_command_queue_handle)(uintptr_t)1;
  graphinfer_session *sessions[2] = {first, second};
  for (unsigned i = 0; i < 2; ++i) {
    graphinfer_session *s = sessions[i];
    s->initialized = true;
    s->list = (npunlock_ze_command_list_handle)(uintptr_t)(i + 1);
    s->fence = (npunlock_ze_fence_handle)(uintptr_t)(i + 1);
    s->argument_count = 2; s->arguments = calloc(2, sizeof(*s->arguments));
    assert(s->arguments);
    s->graph.set_argument = chain_set; s->graph.append_execute = chain_append;
    s->ze.command_list_close = chain_close;
    s->ze.command_queue_execute = chain_execute; s->ze.fence_synchronize = chain_wait;
    for (unsigned j = 0; j < 2; ++j) {
      s->arguments[j].index = j;
      s->arguments[j].type = j ? NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT : NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT;
      s->arguments[j].data_size = sizeof(float);
    }
  }
  for (unsigned i = 0; i < 3; ++i) {
    assert(graphinfer_shared_buffer_create(first, sizeof(float), &buffers[i]) == NPUNLOCK_STATUS_OK);
    pointers[i] = buffers[i].data; *(float *)pointers[i] = i ? 0 : 7.0f;
  }
  for (unsigned i = 0; i < 2; ++i) {
    in[i].struct_size = out[i].struct_size = sizeof(graphinfer_shared_tensor);
    in[i].argument_index = 0; out[i].argument_index = 1;
    in[i].buffer = &buffers[i]; out[i].buffer = &buffers[i+1];
    entries[i] = (diagnostic_chain_entry){sessions[i], &in[i], 1, &out[i], 1};
  }
  barrier_calls = execute_calls = wait_calls = chain_count = 0;
  barrier_result = close_result = set_result = 0;
  if (stage == 1) append_result = 0x78000004u;
  if (stage == 2) barrier_result = 0x78000004u;
  if (stage == 3) execute_result = 0x70000001u;
  if (stage == 4) fence_result = 1;
  if (stage == 5) fence_result = 0x70000001u;
  if (stage == 6) list_reset_result = 0x78000004u;
  if (stage == 7) fence_reset_result = 0x78000004u;
  if (stage == 8) close_result = 0x78000004u;
  if (stage == 9) set_result = 0x78000004u;
  assert(diagnostic_infer_chain(entries, 2, &run) ==
      (stage ? NPUNLOCK_STATUS_DRIVER_FAILED : NPUNLOCK_STATUS_OK));
  assert(first->failed == uncertain);
  if (uncertain) {
    assert(pending && destroy_calls == 0 && list_resets == 0 && fence_resets == 0);
    assert(execute_calls == 1 && wait_calls == (stage == 3 ? 0u : 1u));
  } else if (!stage) {
    assert(*(float *)buffers[2].data == 9.0f && !pending && !recorded && !signaled);
    assert(append_calls == 2 && barrier_calls == 1 && execute_calls == 1 && wait_calls == 1);
    assert(list_resets == 1 && fence_resets == 1 && destroy_calls == 0);
  } else if (stage != 9) {
    assert(!pending && !first->list && !first->fence && destroy_calls == 2);
  } else assert(!pending && first->list && first->fence && destroy_calls == 0);
  graphinfer_session_infer_result_release(&run);
  /* Failures known to be complete permit a fresh list/fence and new bindings. */
  if (stage && !uncertain) {
    append_result = barrier_result = list_reset_result = fence_reset_result = close_result = set_result = 0;
    chain_count = 0;
    first->list = (npunlock_ze_command_list_handle)(uintptr_t)1;
    first->fence = (npunlock_ze_fence_handle)(uintptr_t)1;
    assert(diagnostic_infer_chain(entries, 2, &run) == NPUNLOCK_STATUS_OK);
    assert(*(float *)buffers[2].data == 9.0f && !pending && !first->failed);
    graphinfer_session_infer_result_release(&run);
  }
  child.session = second; root.session = first;
  graphinfer_session_result_release(&child);
  for (unsigned i = 0; i < 3; ++i) graphinfer_shared_buffer_release(&buffers[i]);
  graphinfer_session_result_release(&root);
  if (uncertain) assert(destroy_calls == 0);
  for (unsigned i = 0; i < 3; ++i) free(pointers[i]); /* No actual device work in this fixture. */
}

int main(void) {
  assert(original_session_failure_main() == 0);
  for (unsigned stage = 0; stage < 10; ++stage) check_chain(stage);
  puts("chain success, dependency barrier, 9 failure stages and safe recovery: passed");
  return 0;
}
