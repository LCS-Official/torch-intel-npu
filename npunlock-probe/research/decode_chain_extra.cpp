// Research-only entry; Tensor vectors retain every binding until synchronous completion.
struct DiagnosticChainEntry {
  graphinfer_session* session;
  const graphinfer_shared_tensor* inputs;
  size_t input_count;
  const graphinfer_shared_tensor* outputs;
  size_t output_count;
};
void diagnostic_chain(const std::vector<uintptr_t>& sessions,
    const std::vector<std::vector<at::Tensor>>& inputs,
    const std::vector<std::vector<std::string>>& names,
    const std::vector<std::vector<at::Tensor>>& outputs,
    const std::vector<std::vector<std::string>>& output_names, int64_t group) {
  auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex); ready();
  auto n = sessions.size();
  TORCH_CHECK(n && inputs.size()==n && names.size()==n && outputs.size()==n && output_names.size()==n,
      "Invalid diagnostic chain lengths");
  TORCH_CHECK(group==1 || group==2 || group==4, "Diagnostic group must be 1, 2 or 4");
  using Fn = npunlock_status(__cdecl *)(const DiagnosticChainEntry*, size_t, graphinfer_session_infer_result*);
  auto infer = reinterpret_cast<Fn>(GetProcAddress(s.library, "diagnostic_infer_chain"));
  TORCH_CHECK(infer, "Missing diagnostic chain export");
  std::vector<std::vector<graphinfer_shared_tensor>> in(n), out(n);
  std::vector<DiagnosticChainEntry> entries;
  for (size_t i=0;i<n;++i) {
    TORCH_CHECK(sessions[i] && inputs[i].size()==names[i].size() && !inputs[i].empty() &&
        outputs[i].size()==output_names[i].size() && !outputs[i].empty(), "Invalid diagnostic bindings");
    for (size_t j=0;j<inputs[i].size();++j)
      in[i].push_back({sizeof(graphinfer_shared_tensor), GRAPHINFER_AUTO_INDEX,
          {reinterpret_cast<const uint8_t*>(names[i][j].data()),names[i][j].size()},buffer_for(inputs[i][j])});
    for (size_t j=0;j<outputs[i].size();++j)
      out[i].push_back({sizeof(graphinfer_shared_tensor), GRAPHINFER_AUTO_INDEX,
          {reinterpret_cast<const uint8_t*>(output_names[i][j].data()),output_names[i][j].size()},buffer_for(outputs[i][j])});
    entries.push_back({reinterpret_cast<graphinfer_session*>(sessions[i]),in[i].data(),in[i].size(),out[i].data(),out[i].size()});
  }
  for (size_t i=0;i<n;i+=group) {
    auto count=std::min(n-i,static_cast<size_t>(group));
    graphinfer_session_infer_result result{}; result.struct_size=sizeof(result);
    auto status=infer(entries.data()+i,count,&result);
    auto message=diagnostic(result.diagnostic); s.release_result(&result);
    if (status==NPUNLOCK_STATUS_DRIVER_FAILED) {
      s.failed=true;
      if (message.find("zeFenceHostSynchronize")!=std::string::npos ||
          message.find("zeCommandQueueExecuteCommandLists")!=std::string::npos)
        s.quarantined=s.bytes+s.cached_bytes;
    }
    TORCH_CHECK(status==NPUNLOCK_STATUS_OK,"Diagnostic chain failed: ",message);
    s.calls+=count;
  }
}
