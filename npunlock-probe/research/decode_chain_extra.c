/* Research-only synchronous chain; appended to the unchanged session source. */
typedef struct diagnostic_chain_entry {
  graphinfer_session *session;
  const graphinfer_shared_tensor *inputs;
  size_t input_count;
  const graphinfer_shared_tensor *outputs;
  size_t output_count;
} diagnostic_chain_entry;

#ifdef CHAIN_PROFILE
#define CHAIN_TIME(slot, statement) DIAG_TIME(slot, statement)
#else
#define CHAIN_TIME(slot, statement) do { statement; } while (0)
#endif

__declspec(dllexport) npunlock_status diagnostic_infer_chain(
    const diagnostic_chain_entry *entries, size_t count,
    graphinfer_session_infer_result *result) {
  void *allocations[4][SESSION_MAX_ARGUMENTS] = {{0}};
  bool matched[4][SESSION_MAX_ARGUMENTS] = {{false}};
  typedef uint32_t(__cdecl *barrier_fn)(npunlock_ze_command_list_handle, void *, uint32_t, void *);
  barrier_fn barrier = NULL;
  graphinfer_session *first;
  npunlock_status status = NPUNLOCK_STATUS_INVALID_ARGUMENT;
  uint32_t code;
  size_t i, j, k;
  bool recorded = false;
  if (!result) return status;
  memset(result, 0, sizeof(*result)); result->struct_size = sizeof(*result);
  if (!entries || count < 1 || count > 4 || !entries[0].session)
    return npunlock_set_diagnostic(&result->diagnostic, status, "diagnostic.chain", "requires 1..4 live sessions");
  first = entries[0].session;
  EnterCriticalSection(session_lock(first));
  if (first->context_owner->failed) goto invalid;
  if (count > 1 && !load_export(first->context_owner->loader, "zeCommandListAppendBarrier", &barrier, sizeof(barrier))) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED, "diagnostic.chain", "barrier export unavailable");
    goto done;
  }
  /* Validate the entire batch before any binding or command is recorded. */
  for (i = 0; i < count; ++i) {
    graphinfer_session *s = entries[i].session;
    if (!s || s->context_owner != first->context_owner || !s->active || !s->initialized ||
        !s->list || !s->fence || !entries[i].inputs || !entries[i].outputs ||
        entries[i].input_count != s->input_count || entries[i].output_count != s->output_count)
      goto invalid;
    for (j = 0; j < i; ++j) if (entries[j].session == s) goto invalid;
    status = collect_bindings(s, entries[i].inputs, entries[i].input_count,
        NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT, allocations[i], matched[i], &result->diagnostic);
    if (status == NPUNLOCK_STATUS_OK)
      status = collect_bindings(s, entries[i].outputs, entries[i].output_count,
          NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT, allocations[i], matched[i], &result->diagnostic);
    if (status != NPUNLOCK_STATUS_OK) goto done;
    for (j = 0; j < s->argument_count; ++j) {
      if (!matched[i][j]) goto invalid;
      for (k = 0; k < j; ++k)
        if (allocations[i][j] == allocations[i][k] &&
            (s->arguments[j].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT ||
             s->arguments[k].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT)) goto invalid;
      if (s->arguments[j].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
        for (size_t earlier = 0; earlier < i; ++earlier)
          for (k = 0; k < entries[earlier].session->argument_count; ++k)
            if (entries[earlier].session->arguments[k].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT &&
                allocations[i][j] == allocations[earlier][k]) goto invalid;
      }
    }
  }
  for (i = 0; i < count; ++i) {
    graphinfer_session *session = entries[i].session;
    for (j = 0; j < session->argument_count; ++j) {
      CHAIN_TIME(2, code = session->graph.set_argument(session->handle, (uint32_t)j, allocations[i][j]));
      if (code) { status = session_driver_error(&result->diagnostic, "chain.SetArgumentValue", code); goto done; }
    }
  }
  for (i = 0; i < count; ++i) {
    if (i) {
      code = barrier(first->list, NULL, 0, NULL);
      if (code) { status = session_driver_error(&result->diagnostic, "zeCommandListAppendBarrier(chain)", code); goto done; }
    }
    recorded = true;
    CHAIN_TIME(3, code = first->graph.append_execute(first->list, entries[i].session->handle, NULL, NULL, 0, NULL));
    if (code) { status = session_driver_error(&result->diagnostic, "chain.AppendGraphExecute", code); goto done; }
  }
  CHAIN_TIME(4, code = first->ze.command_list_close(first->list));
  if (code) { status = session_driver_error(&result->diagnostic, "zeCommandListClose(chain)", code); goto done; }
  CHAIN_TIME(5, code = first->ze.command_queue_execute(first->context_owner->queue, 1, &first->list, first->fence));
  if (code) { first->context_owner->failed = true; status = session_driver_error(&result->diagnostic, "zeCommandQueueExecuteCommandLists(chain)", code); goto done; }
  CHAIN_TIME(6, code = first->ze.fence_synchronize(first->fence, first->fence_timeout_ns));
  if (code) { first->context_owner->failed = true; status = session_driver_error(&result->diagnostic, "zeFenceHostSynchronize(chain)", code); goto done; }
  CHAIN_TIME(7, code = first->ze.command_list_reset(first->list));
  if (code) { status = session_driver_error(&result->diagnostic, "zeCommandListReset(chain)", code); goto done; }
  CHAIN_TIME(8, code = first->ze.fence_reset(first->fence));
  status = code ? session_driver_error(&result->diagnostic, "zeFenceReset(chain)", code) : NPUNLOCK_STATUS_OK;
  goto done;
invalid:
  status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
      "diagnostic.chain", "cold/closed/duplicate session, wrong context, or aliased/incomplete bindings");
done:
  if (status != NPUNLOCK_STATUS_OK && recorded) release_submission(first);
  LeaveCriticalSection(session_lock(first));
  return status;
}
