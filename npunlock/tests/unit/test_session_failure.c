/* Exercise uncertain completion without deliberately crashing a real NPU.
 * Including the implementation permits replacing its private driver table. */
#include "../../src/graphinfer/session_win32.c"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

static unsigned destroy_calls;
static uint32_t execute_result;
static uint32_t fence_result;
static void *allocation;
static unsigned list_creations, fence_creations, append_calls, list_resets, fence_resets;
static uint32_t append_result, list_reset_result, fence_reset_result;
static bool pending, recorded, signaled;
static void *bound[2], *captured[2];

static uint32_t mock_queue_create(npunlock_ze_context_handle context,
                                 npunlock_ze_device_handle device,
                                 const npunlock_ze_command_queue_desc *desc,
                                 npunlock_ze_command_queue_handle *handle) {
  (void)context; (void)device; (void)desc;
  *handle = (npunlock_ze_command_queue_handle)(uintptr_t)1;
  return 0;
}

static uint32_t mock_list_create(npunlock_ze_context_handle context,
                                npunlock_ze_device_handle device,
                                const npunlock_ze_command_list_desc *desc,
                                npunlock_ze_command_list_handle *handle) {
  (void)context; (void)device; (void)desc;
  ++list_creations;
  *handle = (npunlock_ze_command_list_handle)(uintptr_t)1;
  return 0;
}

static uint32_t mock_fence_create(npunlock_ze_command_queue_handle queue,
                                 const npunlock_ze_fence_desc *desc,
                                 npunlock_ze_fence_handle *handle) {
  (void)queue; (void)desc;
  ++fence_creations;
  *handle = (npunlock_ze_fence_handle)(uintptr_t)1;
  return 0;
}

static uint32_t mock_append(npunlock_ze_command_list_handle list, npunlock_ze_graph_handle graph,
                           npunlock_ze_graph_profiling_query_handle profiling,
                           npunlock_ze_event_handle signal, uint32_t count,
                           npunlock_ze_event_handle *waits) {
  (void)list; (void)graph; (void)profiling; (void)signal; (void)count; (void)waits;
  ++append_calls;
  assert(!recorded);
  captured[0] = bound[0]; captured[1] = bound[1];
  recorded = append_result == 0;
  return append_result;
}

static uint32_t mock_close(npunlock_ze_command_list_handle list) { (void)list; return 0; }

static uint32_t mock_execute(npunlock_ze_command_queue_handle queue, uint32_t count,
                            npunlock_ze_command_list_handle *lists, npunlock_ze_fence_handle fence) {
  (void)queue; (void)count; (void)lists; (void)fence;
  assert(recorded && !signaled && !pending);
  pending = true;
  if (!execute_result && captured[0] && captured[1]) {
    *(float *)captured[1] = *(float *)captured[0] + 1.0f;
  }
  return execute_result;
}

static uint32_t mock_wait(npunlock_ze_fence_handle fence, uint64_t timeout) {
  (void)fence; (void)timeout;
  if (!fence_result) { pending = false; signaled = true; }
  return fence_result;
}

static uint32_t mock_list_reset(npunlock_ze_command_list_handle handle) {
  (void)handle; assert(!pending); ++list_resets;
  if (!list_reset_result) { recorded = false; captured[0] = captured[1] = NULL; }
  return list_reset_result;
}

static uint32_t mock_fence_reset(npunlock_ze_fence_handle handle) {
  (void)handle; assert(!pending); ++fence_resets;
  if (!fence_reset_result) signaled = false;
  return fence_reset_result;
}

static uint32_t mock_set_argument(npunlock_ze_graph_handle graph, uint32_t index,
                                  const void *pointer) {
  (void)graph; assert(index < 2); bound[index] = (void *)pointer; return 0;
}

static uint32_t mock_queue_destroy(npunlock_ze_command_queue_handle handle) {
  (void)handle; ++destroy_calls; return 0;
}
static uint32_t mock_list_destroy(npunlock_ze_command_list_handle handle) {
  (void)handle; assert(!pending); recorded = false; ++destroy_calls; return 0;
}
static uint32_t mock_fence_destroy(npunlock_ze_fence_handle handle) {
  (void)handle; assert(!pending); signaled = false; ++destroy_calls; return 0;
}
static uint32_t mock_graph_destroy(npunlock_ze_graph_handle handle) {
  (void)handle; ++destroy_calls; return 0;
}
static uint32_t mock_context_destroy(npunlock_ze_context_handle handle) {
  (void)handle; ++destroy_calls; return 0;
}

static uint32_t mock_allocate(npunlock_ze_context_handle context,
                              const npunlock_ze_device_mem_alloc_desc *device_desc,
                              const npunlock_ze_host_mem_alloc_desc *host_desc,
                              size_t size, size_t alignment, npunlock_ze_device_handle device,
                              void **pointer) {
  (void)context; (void)device_desc; (void)host_desc; (void)alignment; (void)device;
  *pointer = allocation = malloc(size);
  assert(*pointer != NULL);
  return 0;
}

static uint32_t mock_free(npunlock_ze_context_handle context, void *pointer) {
  (void)context; (void)pointer; ++destroy_calls; return 0;
}

static graphinfer_session *mock_session(void) {
  graphinfer_session *session = calloc(1, sizeof(*session));
  assert(session != NULL);
  InitializeCriticalSection(&session->lock);
  session->context_owner = session;
  session->references = 1;
  session->active = true;
  session->input_count = session->output_count = 1;
  session->context = (npunlock_ze_context_handle)(uintptr_t)1;
  session->handle = (npunlock_ze_graph_handle)(uintptr_t)1;
  session->ze.command_queue_create = mock_queue_create;
  session->ze.command_list_create = mock_list_create;
  session->ze.fence_create = mock_fence_create;
  session->ze.command_list_close = mock_close;
  session->ze.command_list_reset = mock_list_reset;
  session->ze.command_queue_execute = mock_execute;
  session->ze.fence_synchronize = mock_wait;
  session->ze.fence_reset = mock_fence_reset;
  session->ze.command_queue_destroy = mock_queue_destroy;
  session->ze.command_list_destroy = mock_list_destroy;
  session->ze.fence_destroy = mock_fence_destroy;
  session->ze.context_destroy = mock_context_destroy;
  session->ze.mem_alloc_shared = mock_allocate;
  session->ze.mem_free = mock_free;
  session->graph.append_execute = mock_append;
  session->graph.destroy = mock_graph_destroy;
  session->graph.set_argument = mock_set_argument;
  execute_result = fence_result = append_result = list_reset_result = fence_reset_result = 0;
  destroy_calls = list_creations = fence_creations = append_calls = list_resets = fence_resets = 0;
  pending = recorded = signaled = false;
  bound[0] = bound[1] = captured[0] = captured[1] = NULL;
  return session;
}

static void check_quarantine(uint32_t queue_status, uint32_t fence_status) {
  graphinfer_session *session = mock_session();
  graphinfer_session_result result = {0};
  graphinfer_shared_buffer buffer = {0};
  graphinfer_session_infer_result run = {0};
  graphinfer_shared_tensor binding = {0};
  npunlock_diagnostic diagnostic = {0};
  execute_result = queue_status;
  fence_result = fence_status;
  destroy_calls = 0;
  assert(graphinfer_shared_buffer_create(session, 68, &buffer) == NPUNLOCK_STATUS_OK);
  assert(submit(session, false, &diagnostic) == NPUNLOCK_STATUS_DRIVER_FAILED);
  assert(session->failed && destroy_calls == 0);
  assert(list_resets == 0 && fence_resets == 0);
  assert(graphinfer_session_infer(session, &binding, 1, &binding, 1, &run) ==
         NPUNLOCK_STATUS_INVALID_ARGUMENT);
  result.session = session;
  graphinfer_session_result_release(&result);
  graphinfer_shared_buffer_release(&buffer);
  assert(destroy_calls == 0); /* No destruction/free while completion is unknown. */
  free(allocation); /* This fixture has no actual device work in flight. */
  npunlock_diagnostic_release(&diagnostic);
  graphinfer_session_infer_result_release(&run);
}

static void check_reuse(void) {
  graphinfer_session *session = mock_session();
  graphinfer_session_result result = {0};
  npunlock_diagnostic diagnostic = {0};
  float input[2] = {2.0f, 9.0f}, output[2] = {0};
  void *addresses[2];
  session->argument_count = 2;
  for (unsigned i = 0; i < 100; ++i) {
    unsigned slot = i % 2;
    input[slot] += 1.0f;
    addresses[0] = &input[slot]; addresses[1] = &output[slot];
    assert(execute_allocations_locked(session, addresses, &diagnostic) == NPUNLOCK_STATUS_OK);
    assert(output[slot] == input[slot] + 1.0f);
    assert(!pending && !recorded && !signaled && !captured[0] && !captured[1]);
  }
  assert(list_creations == 1 && fence_creations == 1 && append_calls == 100);
  assert(list_resets == 100 && fence_resets == 100 && destroy_calls == 0);
  result.session = session;
  graphinfer_session_result_release(&result);
  assert(destroy_calls == 5); /* list, fence, graph, queue, context */
  npunlock_diagnostic_release(&diagnostic);
}

static void check_safe_failure(unsigned stage) {
  graphinfer_session *session = mock_session();
  graphinfer_session_result result = {0};
  npunlock_diagnostic diagnostic = {0};
  if (stage == 0) append_result = 0x78000004u;
  else if (stage == 1) list_reset_result = 0x78000004u;
  else fence_reset_result = 0x78000004u;
  assert(submit(session, false, &diagnostic) == NPUNLOCK_STATUS_DRIVER_FAILED);
  assert(!session->failed && !pending && !session->list && !session->fence);
  assert(destroy_calls == 2);
  append_result = list_reset_result = fence_reset_result = 0;
  npunlock_diagnostic_release(&diagnostic);
  assert(submit(session, false, &diagnostic) == NPUNLOCK_STATUS_OK);
  assert(list_creations == 2 && fence_creations == 2);
  result.session = session;
  graphinfer_session_result_release(&result);
  assert(destroy_calls == 7);
  npunlock_diagnostic_release(&diagnostic);
}

int main(void) {
  npunlock_ze_graph_argument_properties_3 properties = {0};
  size_t bytes = 0;
  properties.dims_count = 3;
  properties.dims[0] = properties.dims[1] = 256;
  properties.dims[2] = 4;
  properties.devicePrecision = NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
  assert(argument_data_size(&properties, &bytes) && bytes == 262144);
  properties.devicePrecision = NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP32;
  assert(argument_data_size(&properties, &bytes) && bytes == 1048576);
  properties.devicePrecision = 0;
  assert(!argument_data_size(&properties, &bytes));
  properties.devicePrecision = NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
  properties.dims[2] = 0;
  assert(!argument_data_size(&properties, &bytes));
  properties.dims[2] = UINT32_MAX;
  assert(!argument_data_size(&properties, &bytes));
  check_quarantine(0, 1);          /* ZE_RESULT_NOT_READY / fence timeout. */
  check_quarantine(0, 0x70000001); /* ZE_RESULT_ERROR_DEVICE_LOST. */
  check_quarantine(0x70000001, 0); /* Queue submission failure. */
  check_reuse();
  for (unsigned stage = 0; stage < 3; ++stage) check_safe_failure(stage);
  puts("session reuse, rebinding, safe failures and failure quarantine: passed");
  return 0;
}
