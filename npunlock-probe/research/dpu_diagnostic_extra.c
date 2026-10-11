/* Diagnostic-only additions to a copied session_win32.c. Production ABI and
 * DLL stay untouched. Profiling ABI follows Intel's MIT-licensed
 * level-zero-npu-extensions/ze_graph_profiling_ext.h (version 1.0). */
typedef uint32_t(__cdecl *diag_pool_create_fn)(npunlock_ze_graph_handle, uint32_t, void **);
typedef uint32_t(__cdecl *diag_destroy_fn)(void *);
typedef uint32_t(__cdecl *diag_query_create_fn)(void *, uint32_t, npunlock_ze_graph_profiling_query_handle *);
typedef uint32_t(__cdecl *diag_get_data_fn)(void *, uint32_t, uint32_t *, uint8_t *);

__declspec(dllexport) uint32_t diag_extensions(graphinfer_session *s, uint32_t *count,
                                              npunlock_ze_driver_extension_properties *values) {
  if (!s || !s->active || !count) return 0x78000004u;
  return s->ze.driver_get_extensions(s->driver, count, values);
}

__declspec(dllexport) uint32_t diag_argument(graphinfer_session *s, uint32_t index,
                                            session_argument *value) {
  if (!s || !s->active || !value || index >= s->argument_count) return 0x78000004u;
  memcpy(value, &s->arguments[index], sizeof(*value));
  return 0;
}

__declspec(dllexport) uint32_t diag_times(graphinfer_session *s, uint64_t *values,
                                         uint32_t count) {
  LARGE_INTEGER frequency;
  if (!s || !s->active || !values || count != 10) return 0x78000004u;
  QueryPerformanceFrequency(&frequency);
  for (uint32_t i = 0; i < 9; ++i) values[i] = s->diag_ticks[i] * 1000000000ull / (uint64_t)frequency.QuadPart;
  values[9] = s->context_owner->failed;
  return 0;
}

__declspec(dllexport) uint32_t diag_profile_enable(graphinfer_session *s) {
  void *driver_ddi = NULL;
  npunlock_ze_npu_get_extension_fn get = NULL;
  void **ddi = NULL;
  npunlock_ze_driver_extension_npu request = {NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT,
      NULL, "ZE_extension_profiling_data", 0x10000u, (void **)&ddi};
  uint32_t code;
  diag_pool_create_fn pool_create = NULL;
  diag_query_create_fn query_create = NULL;
  if (!s || !s->active || s->context_owner->failed || s->diag_pool) return 0x78000004u;
  code = s->ze.driver_get_extension_address(s->driver, "ZE_extension_driver_npu", &driver_ddi);
  if (code || !driver_ddi) return code ? code : 0x78000003u;
  memcpy(&get, driver_ddi, sizeof(get));
  code = get(s->driver, &request);
  if (code || !ddi) return code ? code : 0x78000003u;
  for (uint32_t i = 0; i < 5; ++i) if (!ddi[i]) return 0x78000003u;
  memcpy(&pool_create, &ddi[0], sizeof(pool_create));
  memcpy(&s->diag_pool_destroy, &ddi[1], sizeof(s->diag_pool_destroy));
  memcpy(&query_create, &ddi[2], sizeof(query_create));
  memcpy(&s->diag_query_destroy, &ddi[3], sizeof(s->diag_query_destroy));
  memcpy(&s->diag_get_data, &ddi[4], sizeof(s->diag_get_data));
  code = pool_create(s->handle, 1, &s->diag_pool);
  if (code) return code;
  code = query_create(s->diag_pool, 0, &s->diag_query);
  if (code) { s->diag_pool_destroy(s->diag_pool); s->diag_pool = NULL; }
  return code;
}

__declspec(dllexport) uint32_t diag_profile_data(graphinfer_session *s, uint32_t type,
                                                uint32_t *size, uint8_t *data) {
  if (!s || !s->active || s->context_owner->failed || !s->diag_query || !size ||
      (type != 1 && type != 2 && type != 4)) return 0x78000004u;
  return s->diag_get_data(s->diag_query, type, size, data);
}

/* ponytail: fixed addresses, synchronous replay only. General dynamic binding
 * and production caching need separate lifetime/invalidation integration. */
typedef struct diag_replay {
  graphinfer_session *session;
  npunlock_ze_command_list_handle list;
  npunlock_ze_fence_handle fence;
  uint32_t(__cdecl *reset)(npunlock_ze_fence_handle);
  graphinfer_shared_buffer *buffers[SESSION_MAX_ARGUMENTS];
  void *addresses[SESSION_MAX_ARGUMENTS];
  size_t sizes[SESSION_MAX_ARGUMENTS];
} diag_replay;

__declspec(dllexport) npunlock_status diag_replay_create(
    graphinfer_session *s, const graphinfer_shared_tensor *inputs, size_t input_count,
    const graphinfer_shared_tensor *outputs, size_t output_count, diag_replay **out,
    graphinfer_session_infer_result *result) {
  void *allocations[SESSION_MAX_ARGUMENTS] = {0};
  bool matched[SESSION_MAX_ARGUMENTS] = {false};
  npunlock_status status;
  diag_replay *r = NULL;
  npunlock_ze_command_list_desc ld = {NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, 0, 0};
  npunlock_ze_fence_desc fd = {NPUNLOCK_ZE_STRUCTURE_TYPE_FENCE_DESC, NULL, 0};
  uint32_t code;
  if (!s || !out || !result || !inputs || !outputs || input_count != s->input_count ||
      output_count != s->output_count) return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  *out = NULL;
  memset(result, 0, sizeof(*result)); result->struct_size = sizeof(*result);
  EnterCriticalSection(session_lock(s));
  if (!s->active || s->context_owner->failed || !s->initialized || !s->context_owner->queue) {
    status = NPUNLOCK_STATUS_INVALID_ARGUMENT; goto done;
  }
  status = collect_bindings(s, inputs, input_count, NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT,
                            allocations, matched, &result->diagnostic);
  if (!status) status = collect_bindings(s, outputs, output_count, NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT,
                                        allocations, matched, &result->diagnostic);
  if (status) goto done;
  for (size_t i = 0; i < s->argument_count; ++i) {
    if (!matched[i]) { status = NPUNLOCK_STATUS_INVALID_ARGUMENT; goto done; }
    for (size_t j = 0; j < i; ++j) if (allocations[i] == allocations[j] &&
        (s->arguments[i].type || s->arguments[j].type)) {
      status = NPUNLOCK_STATUS_INVALID_ARGUMENT; goto done;
    }
  }
  r = (diag_replay *)calloc(1, sizeof(*r));
  if (!r) { status = NPUNLOCK_STATUS_OUT_OF_MEMORY; goto done; }
  r->session = s;
  /* The Python owner retains every bound descriptor until this handle closes. */
  for (size_t i = 0; i < input_count + output_count; ++i) {
    graphinfer_shared_buffer *b = (i < input_count ? inputs[i].buffer : outputs[i-input_count].buffer);
    r->buffers[i] = b; r->addresses[i] = b->data; r->sizes[i] = b->size;
  }
  if (!load_export(s->context_owner->loader, "zeFenceReset", &r->reset, sizeof(r->reset))) {
    status = NPUNLOCK_STATUS_UNSUPPORTED; goto done;
  }
  for (size_t i = 0; i < s->argument_count; ++i) {
    code = s->graph.set_argument(s->handle, (uint32_t)i, allocations[i]);
    if (code) { status = session_driver_error(&result->diagnostic, "replay.set_argument", code); goto done; }
  }
  ld.commandQueueGroupOrdinal = s->queue_ordinal;
  code = s->ze.command_list_create(s->context, s->device, &ld, &r->list);
  if (!code) code = s->ze.fence_create(s->context_owner->queue, &fd, &r->fence);
  if (!code) code = s->graph.append_execute(r->list, s->handle, s->diag_query, NULL, 0, NULL);
  if (!code) code = s->ze.command_list_close(r->list);
  if (code) { status = session_driver_error(&result->diagnostic, "replay.record", code); goto done; }
  InterlockedIncrement(&s->references);
  *out = r; r = NULL; status = NPUNLOCK_STATUS_OK;
done:
  if (r) {
    if (r->fence) s->ze.fence_destroy(r->fence);
    if (r->list) s->ze.command_list_destroy(r->list);
    free(r);
  }
  LeaveCriticalSection(session_lock(s));
  return status;
}

__declspec(dllexport) npunlock_status diag_replay_run(diag_replay *r, uint32_t repetitions,
                                                    graphinfer_session_infer_result *result) {
  graphinfer_session *s;
  npunlock_status status = NPUNLOCK_STATUS_OK;
  uint32_t code;
  uint64_t begin, t;
  if (!r || !result || !repetitions || repetitions > 32) return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  memset(result, 0, sizeof(*result)); result->struct_size = sizeof(*result);
  s = r->session;
  EnterCriticalSection(session_lock(s));
  memset(s->diag_ticks, 0, sizeof(s->diag_ticks)); begin = diag_clock();
  if (!s->active || s->context_owner->failed) { status = NPUNLOCK_STATUS_INVALID_ARGUMENT; goto done; }
  for (size_t i = 0; i < s->argument_count; ++i) {
    graphinfer_shared_buffer *b = r->buffers[i];
    shared_buffer_impl *impl = b ? (shared_buffer_impl *)b->implementation : NULL;
    if (!impl || b->data != r->addresses[i] || b->size != r->sizes[i] ||
        impl->session->context_owner != s->context_owner || impl->allocation != b->data || impl->size != b->size) {
      status = NPUNLOCK_STATUS_INVALID_ARGUMENT; goto done;
    }
  }
  for (uint32_t i = 0; i < repetitions; ++i) {
    t = diag_clock(); code = r->reset(r->fence); s->diag_ticks[2] += diag_clock()-t;
    if (code) { status = session_driver_error(&result->diagnostic, "replay.fence_reset", code); goto done; }
    t = diag_clock(); code = s->ze.command_queue_execute(s->context_owner->queue, 1, &r->list, r->fence);
    s->diag_ticks[5] += diag_clock()-t;
    if (code) { s->context_owner->failed = true; status = session_driver_error(&result->diagnostic, "replay.submit", code); goto done; }
    t = diag_clock(); code = s->ze.fence_synchronize(r->fence, s->fence_timeout_ns);
    s->diag_ticks[6] += diag_clock()-t;
    if (code) { s->context_owner->failed = true; status = session_driver_error(&result->diagnostic, "replay.wait", code); goto done; }
  }
done:
  s->diag_ticks[8] = diag_clock()-begin;
  LeaveCriticalSection(session_lock(s));
  return status;
}

__declspec(dllexport) uint32_t diag_replay_destroy(diag_replay *r) {
  graphinfer_session *s;
  if (!r) return 0x78000004u;
  s = r->session;
  EnterCriticalSection(session_lock(s));
  if (!s->context_owner->failed) {
    s->ze.fence_destroy(r->fence); s->ze.command_list_destroy(r->list);
  }
  LeaveCriticalSection(session_lock(s));
  free(r); session_unref(s);
  return 0;
}
