/* Exercise the public synchronous chain against the existing fake driver. */
#define main original_session_failure_main
#include "test_session_failure.c"
#undef main

static unsigned barrier_calls, execute_calls, wait_calls, chain_count;
static uint32_t barrier_result, close_result, set_result;
static void *graph_bindings[2][2], *chain_bindings[4][2];

static uint32_t chain_barrier(npunlock_ze_command_list_handle list, npunlock_ze_event_handle signal,
                               uint32_t count, npunlock_ze_event_handle *waits) {
  (void)list; (void)signal; (void)count; (void)waits;
  assert(recorded && !pending); ++barrier_calls; return barrier_result;
}

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
  assert(recorded && !signaled && !pending && chain_count >= 1 && chain_count <= 2);
  ++execute_calls; pending = true;
  if (!execute_result) for (unsigned i = 0; i < chain_count; ++i)
    *(float *)chain_bindings[i][1] = *(float *)chain_bindings[i][0] + 1.0f;
  return execute_result;
}

static uint32_t chain_wait(npunlock_ze_fence_handle fence, uint64_t timeout) {
  ++wait_calls; return mock_wait(fence, timeout);
}

static uint32_t chain_reset(npunlock_ze_command_list_handle list) {
  uint32_t code = mock_list_reset(list);
  if (!code) chain_count = 0;
  return code;
}

static void check_chain(unsigned stage) {
  graphinfer_session *first = mock_session(), *second = mock_session();
  graphinfer_session_result root = {0}, child = {0};
  graphinfer_session_infer_result run = {0};
  graphinfer_shared_buffer buffers[3] = {{0}};
  graphinfer_shared_tensor in[2] = {{0}}, out[2] = {{0}};
  graphinfer_chain_entry entries[2];
  void *pointers[3];
  bool uncertain = stage == 3 || stage == 4 || stage == 5;
  bool failed_stage = stage >= 1 && stage <= 9;
  first->ze.command_list_append_barrier = chain_barrier; second->context_owner = first; InterlockedIncrement(&first->references);
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
    s->ze.command_list_reset = chain_reset;
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
    entries[i] = (graphinfer_chain_entry){sizeof(graphinfer_chain_entry), sessions[i], &in[i], 1, &out[i], 1};
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
  if (stage == 10) first->initialized = second->initialized = false;
  if (stage == 11) second->initialized = false;
  if (stage == 12) entries[1].struct_size = 0;
  if (stage == 13) second->active = false;
  if (stage == 14) second->context_owner = second;
  if (stage == 15) entries[1].input_count = 0;
  if (stage == 16) out[0].buffer = &buffers[0];
  if (stage == 17) out[1].buffer = &buffers[0];
  if (stage == 18) entries[1].session = first;
  if (stage == 19) second->arguments[0].data_size = 8;
  if (stage == 20) out[1].buffer = &buffers[1];
  assert(graphinfer_session_infer_chain(stage == 22 ? NULL : entries, stage == 21 ? 5 : 2, &run) ==
      (stage >= 12 ? NPUNLOCK_STATUS_INVALID_ARGUMENT : failed_stage ? NPUNLOCK_STATUS_DRIVER_FAILED : NPUNLOCK_STATUS_OK));
  second->context_owner = first;
  assert(first->failed == uncertain);
  if (stage >= 12) {
    assert(!pending && !recorded && !append_calls && !execute_calls && !wait_calls && !destroy_calls);
  } else if (uncertain) {
    assert(pending && destroy_calls == 0 && list_resets == 0 && fence_resets == 0);
    assert(execute_calls == 1 && wait_calls == (stage == 3 ? 0u : 1u));
  } else if (!failed_stage) {
    assert(*(float *)buffers[2].data == 9.0f && !pending && !recorded && !signaled);
    unsigned submissions = stage >= 10 ? 2 : 1;
    assert(append_calls == 2 && barrier_calls == (stage >= 10 ? 0u : 1u));
    assert(execute_calls == submissions && wait_calls == submissions);
    assert(list_resets == submissions && fence_resets == submissions && destroy_calls == 0);
  } else if (stage != 9) {
    assert(!pending && !first->list && !first->fence && destroy_calls == 2);
  } else assert(!pending && first->list && first->fence && destroy_calls == 0);
  graphinfer_session_infer_result_release(&run);
  /* Failures known to be complete permit a fresh list/fence and new bindings. */
  if (failed_stage && !uncertain) {
    append_result = barrier_result = list_reset_result = fence_reset_result = close_result = set_result = 0;
    chain_count = 0;
    first->list = (npunlock_ze_command_list_handle)(uintptr_t)1;
    first->fence = (npunlock_ze_fence_handle)(uintptr_t)1;
    assert(graphinfer_session_infer_chain(entries, 2, &run) == NPUNLOCK_STATUS_OK);
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
  for (unsigned stage = 0; stage < 23; ++stage) check_chain(stage);
  puts("chain: success, 2 cold/mixed paths, 11 rejections, 9 failures and safe recovery passed");
  return 0;
}

