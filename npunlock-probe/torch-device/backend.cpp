// Synchronous, single-device PrivateUse1 storage backed by npunlock/Level Zero.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/t_native.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/library.h>
#include <npunlock/graphinfer.h>
#include <cstring>
#include <array>
#include <atomic>
#include <algorithm>
#include <mutex>

namespace {
constexpr auto device_type = c10::DeviceType::PrivateUse1;
constexpr int64_t max_tensor_elements = 1048576;
constexpr int64_t max_cached_buffer_bytes = 16 * 1024 * 1024;
constexpr size_t max_cached_buffers = 256;
struct State {
  // ponytail: synchronous global lock; add streams after asynchronous submission exists.
  std::recursive_mutex mutex;
  HMODULE library = nullptr;
  std::atomic<graphinfer_session*> root{nullptr};
  decltype(&graphinfer_shared_buffer_create) create = nullptr;
  decltype(&graphinfer_shared_buffer_release) release = nullptr;
  decltype(&graphinfer_session_infer) infer = nullptr;
  decltype(&graphinfer_session_infer_chain) infer_chain = nullptr;
  decltype(&graphinfer_session_infer_result_release) release_result = nullptr;
  std::atomic<bool> failed{false};
  int64_t allocations = 0, live = 0, bytes = 0, peak = 0, calls = 0;
  int64_t chain_groups = 0, chain_graphs = 0;
  int64_t uploads = 0, downloads = 0, copies = 0, packs = 0, quarantined = 0;
  // ponytail: exact-size blocks, bounded linear search; add size bins if measured.
  std::vector<graphinfer_shared_buffer*> cached;
  int64_t cached_bytes = 0, driver_allocations = 0, allocation_reuses = 0;
};
State& state() {
  // PyTorch's allocator/guard registries do not own their entries.
  static auto* value = new State;
  return *value;
}
void ready() {
  TORCH_CHECK(state().root, "intel_npu backend is not initialized or has been shut down");
  TORCH_CHECK(!state().failed, "NPU execution failed; restart process before accessing buffers");
}
std::string diagnostic(const npunlock_diagnostic& value) {
  if (!value.json.data) return "No native diagnostic";
  return std::string(reinterpret_cast<const char*>(value.json.data), value.json.size);
}
void free_allocation(void* ptr) {
  auto& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  auto* buffer = static_cast<graphinfer_shared_buffer*>(ptr);
  s.live--;
  s.bytes -= buffer->size;
  // The deleter runs only after the last Storage/view reference is gone.
  // Every graph call has completed and cleared its recorded bindings by now.
  if (s.root && !s.failed && buffer->size <= static_cast<size_t>(max_cached_buffer_bytes)) {
    // Retain recently returned sizes when workloads change shape. An idle
    // block at the front has the oldest release; live Storage is never here.
    while (s.cached.size() >= max_cached_buffers ||
           buffer->size > static_cast<size_t>(max_cached_buffer_bytes - s.cached_bytes)) {
      auto* oldest = s.cached.front();
      s.cached.erase(s.cached.begin());
      s.cached_bytes -= oldest->size;
      s.release(oldest);
      delete oldest;
    }
    s.cached.push_back(buffer); // Capacity reserved at configure: deleters cannot allocate.
    s.cached_bytes += buffer->size;
  } else {
    s.release(buffer); // Native runtime quarantines allocations after uncertain completion.
    delete buffer;
  }
}
void empty_cache() {
  auto& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  for (auto* buffer : s.cached) {
    s.release(buffer);
    delete buffer;
  }
  s.cached.clear();
  s.cached_bytes = 0;
}
struct NpuAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t bytes) override {
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    ready();
    TORCH_CHECK(bytes > 0 && bytes <= max_tensor_elements * 4, "NPU allocation requires 1-4194304 bytes");
    std::unique_ptr<graphinfer_shared_buffer> buffer;
    for (size_t i = s.cached.size(); i > 0; --i) {
      if (s.cached[i-1]->size == bytes) {
        buffer.reset(s.cached[i-1]);
        s.cached.erase(s.cached.begin() + (i-1)); // Preserve idle release order.
        s.cached_bytes -= bytes;
        s.allocation_reuses++;
        break;
      }
    }
    if (!buffer) {
      buffer = std::make_unique<graphinfer_shared_buffer>();
      *buffer = {};
      buffer->struct_size = sizeof(*buffer);
      auto status = s.create(s.root, bytes, buffer.get());
      if (status != NPUNLOCK_STATUS_OK) {
        auto message = diagnostic(buffer->diagnostic);
        s.release(buffer.get());
        TORCH_CHECK(false, "NPU allocation failed: ", message);
      }
      s.driver_allocations++;
    }
    auto* data = buffer->data;
    s.allocations++;
    s.live++;
    s.bytes += bytes;
    s.peak = std::max(s.peak, s.bytes);
    return {data, buffer.release(), free_allocation, c10::Device(device_type, 0)};
  }
  void copy_data(void* dst, const void* src, size_t bytes) const override {
    std::lock_guard<std::recursive_mutex> lock(state().mutex);
    ready();
    std::memmove(dst, src, bytes);
    state().copies++;
  }
};
graphinfer_shared_buffer* buffer_for(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().type() == device_type && tensor.device().index() == 0,
              "Expected intel_npu:0 tensor");
  TORCH_CHECK(!at::GradMode::is_enabled() || !tensor.requires_grad(),
              "intel_npu requires no_grad/inference_mode for tensors requiring gradients");
  TORCH_CHECK(tensor.is_contiguous() && tensor.storage_offset() == 0 &&
              !tensor.is_neg() && !tensor.is_conj(),
              "NPU kernels require contiguous, offset-zero tensors with resolved value bits");
  const auto& data = tensor.storage().data_ptr();
  TORCH_CHECK(data.get_deleter() == free_allocation, "Tensor does not own an NPU allocation");
  auto* buffer = static_cast<graphinfer_shared_buffer*>(data.get_context());
  TORCH_CHECK(buffer && buffer->size == tensor.nbytes(), "NPU binding must cover its entire allocation");
  return buffer;
}
void check_shape(at::IntArrayRef shape, at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat, "intel_npu requires FP16 or FP32");
  TORCH_CHECK(shape.size() <= 4, "intel_npu supports rank 0-4");
  int64_t elements = 1;
  for (auto dim : shape) {
    TORCH_CHECK(dim > 0 && dim <= max_tensor_elements / elements, "intel_npu requires 1-1048576 elements");
    elements *= dim;
  }
}
void check_options(std::optional<at::Layout> layout, std::optional<at::Device> device,
                   std::optional<bool> pinned) {
  TORCH_CHECK(!layout || *layout == at::kStrided, "intel_npu requires strided layout");
  TORCH_CHECK(!device || (device->type() == device_type && device->index() <= 0),
              "Only intel_npu:0 is supported");
  TORCH_CHECK(!pinned.value_or(false), "intel_npu factory does not support pin_memory");
}
at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pinned, std::optional<at::MemoryFormat> format) {
  check_options(layout, device, pinned);
  auto type = dtype.value_or(at::typeMetaToScalarType(c10::get_default_dtype()));
  check_shape(size, type);
  TORCH_CHECK(!format || *format == at::MemoryFormat::Contiguous,
              "intel_npu requires contiguous memory format");
  return at::Tensor(at::detail::empty_generic(size, c10::GetAllocator(device_type),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), type, format));
}
at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout> layout, std::optional<at::Device> device,
                         std::optional<bool> pinned) {
  check_options(layout, device, pinned);
  auto type = dtype.value_or(at::typeMetaToScalarType(c10::get_default_dtype()));
  check_shape(size, type);
  TORCH_CHECK(size.size() == stride.size(), "NPU size/stride ranks differ");
  std::vector<size_t> axes(size.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    TORCH_CHECK(stride[i] >= 0, "intel_npu empty_strided requires nonnegative strides");
    axes[i] = i;
  }
  std::sort(axes.begin(), axes.end(), [&](size_t a, size_t b) { return stride[a] < stride[b]; });
  int64_t expected = 1;
  for (auto axis : axes) {
    if (size[axis] == 1) continue;
    TORCH_CHECK(stride[axis] == expected,
                "intel_npu empty_strided requires a complete dense permutation without holes/overlap");
    expected *= size[axis];
  }
  return at::Tensor(at::detail::empty_strided_generic(size, stride,
      c10::GetAllocator(device_type), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), type));
}
at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  std::lock_guard<std::recursive_mutex> lock(state().mutex);
  ready();
  TORCH_CHECK(!non_blocking, "intel_npu copies are synchronous; non_blocking is unsupported");
  TORCH_CHECK(dst.sizes() == src.sizes() && dst.scalar_type() == src.scalar_type(),
              "intel_npu copy requires identical shape and dtype");
  TORCH_CHECK(!at::GradMode::is_enabled() || (!src.requires_grad() && !dst.requires_grad()),
              "intel_npu requires no_grad/inference_mode for tensors requiring gradients");
  TORCH_CHECK(!src.is_neg() && !src.is_conj(), "Copy requires resolved source value bits");
  bool src_npu = src.device().type() == device_type;
  bool dst_npu = dst.device().type() == device_type;
  TORCH_CHECK(src_npu || dst_npu, "Expected an NPU copy endpoint");
  TORCH_CHECK((src_npu || src.device().is_cpu()) && (dst_npu || dst.device().is_cpu()),
              "Only CPU/intel_npu copies are implemented");
  if (src_npu) buffer_for(src);
  if (dst_npu) buffer_for(dst);
  TORCH_CHECK(dst.is_contiguous(), "Copy destination must be contiguous");
  auto packed = src_npu ? src : src.contiguous();
  if (dst.data_ptr() != packed.data_ptr()) {
    std::memmove(dst.mutable_data_ptr(), packed.const_data_ptr(), src.nbytes());
    if (!src_npu) state().uploads++;
    else if (!dst_npu) state().downloads++;
    else state().copies++;
    if (!src.is_contiguous()) state().packs++;
  }
  return dst;
}
void run_many(uintptr_t session, const std::vector<at::Tensor>& inputs,
              const std::vector<std::string>& names, const std::vector<at::Tensor>& outputs,
              const std::vector<std::string>& output_names) {
  auto& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  ready();
  TORCH_CHECK(session && inputs.size() == names.size() && !inputs.empty(), "Invalid NPU invocation");
  TORCH_CHECK(!outputs.empty() && outputs.size() == output_names.size(), "Invalid NPU output bindings");
  std::vector<graphinfer_shared_tensor> bindings, out_bindings;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto* buffer = buffer_for(outputs[i]);
    for (const auto& existing : out_bindings)
      TORCH_CHECK(existing.buffer != buffer, "NPU outputs may not alias each other");
    out_bindings.push_back({sizeof(graphinfer_shared_tensor), GRAPHINFER_AUTO_INDEX,
        {reinterpret_cast<const uint8_t*>(output_names[i].data()), output_names[i].size()}, buffer});
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    auto* buffer = buffer_for(inputs[i]);
    for (const auto& out : out_bindings)
      TORCH_CHECK(buffer != out.buffer, "NPU output may not alias an input");
    bindings.push_back({sizeof(graphinfer_shared_tensor), GRAPHINFER_AUTO_INDEX,
        {reinterpret_cast<const uint8_t*>(names[i].data()), names[i].size()}, buffer});
  }
  graphinfer_session_infer_result result{};
  result.struct_size = sizeof(result);
  auto status = s.infer(reinterpret_cast<graphinfer_session*>(session), bindings.data(),
                        bindings.size(), out_bindings.data(), out_bindings.size(), &result);
  auto message = diagnostic(result.diagnostic);
  s.release_result(&result);
  if (status == NPUNLOCK_STATUS_DRIVER_FAILED) {
    s.failed = true;
    if (message.find("zeFenceHostSynchronize") != std::string::npos ||
        message.find("zeCommandQueueExecuteCommandLists") != std::string::npos)
      s.quarantined = s.bytes + s.cached_bytes;
  }
  TORCH_CHECK(status == NPUNLOCK_STATUS_OK, "NPU submission failed: ", message);
  s.calls++;
}
void run_chain(const std::vector<uintptr_t>& sessions,
    const std::vector<std::vector<at::Tensor>>& inputs,
    const std::vector<std::vector<std::string>>& names,
    const std::vector<std::vector<at::Tensor>>& outputs,
    const std::vector<std::vector<std::string>>& output_names, int64_t group) {
  auto& s = state(); std::lock_guard<std::recursive_mutex> lock(s.mutex); ready();
  auto n = sessions.size();
  TORCH_CHECK(n && inputs.size()==n && names.size()==n && outputs.size()==n && output_names.size()==n,
      "Invalid NPU chain lengths");
  TORCH_CHECK(group==1 || group==2 || group==4, "NPU group must be 1, 2 or 4");
  std::vector<std::vector<graphinfer_shared_tensor>> in(n), out(n);
  std::vector<graphinfer_chain_entry> entries;
  for (size_t i=0;i<n;++i) {
    TORCH_CHECK(sessions[i] && inputs[i].size()==names[i].size() && !inputs[i].empty() &&
        outputs[i].size()==output_names[i].size() && !outputs[i].empty(), "Invalid NPU bindings");
    for (size_t j=0;j<inputs[i].size();++j)
      in[i].push_back({sizeof(graphinfer_shared_tensor), GRAPHINFER_AUTO_INDEX,
          {reinterpret_cast<const uint8_t*>(names[i][j].data()),names[i][j].size()},buffer_for(inputs[i][j])});
    for (size_t j=0;j<outputs[i].size();++j)
      out[i].push_back({sizeof(graphinfer_shared_tensor), GRAPHINFER_AUTO_INDEX,
          {reinterpret_cast<const uint8_t*>(output_names[i][j].data()),output_names[i][j].size()},buffer_for(outputs[i][j])});
    entries.push_back({sizeof(graphinfer_chain_entry), reinterpret_cast<graphinfer_session*>(sessions[i]),in[i].data(),in[i].size(),out[i].data(),out[i].size()});
  }
  for (size_t i=0;i<n;i+=group) {
    auto count=std::min(n-i,static_cast<size_t>(group));
    graphinfer_session_infer_result result{}; result.struct_size=sizeof(result);
    auto status=s.infer_chain(entries.data()+i,count,&result);
    auto message=diagnostic(result.diagnostic); s.release_result(&result);
    if (status==NPUNLOCK_STATUS_DRIVER_FAILED) {
      s.failed=true;
      if (message.find("zeFenceHostSynchronize")!=std::string::npos ||
          message.find("zeCommandQueueExecuteCommandLists")!=std::string::npos)
        s.quarantined=s.bytes+s.cached_bytes;
    }
    TORCH_CHECK(status==NPUNLOCK_STATUS_OK,"NPU chain failed: ",message);
    s.calls+=count; s.chain_groups++; s.chain_graphs+=count;
  }
}

void run(uintptr_t session, const std::vector<at::Tensor>& inputs,
         const std::vector<std::string>& names, const at::Tensor& output,
         const std::string& output_name) {
  run_many(session, inputs, names, {output}, {output_name});
}
struct Guard final : c10::impl::NoOpDeviceGuardImpl<device_type> {
  c10::Device getDevice() const override { return c10::Device(device_type, 0); }
  c10::Device exchangeDevice(c10::Device device) const override { setDevice(device); return getDevice(); }
  void setDevice(c10::Device device) const override {
    TORCH_CHECK(device.type() == device_type && device.index() <= 0, "Only intel_npu:0 is supported");
  }
  c10::Stream getStream(c10::Device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(device_type, 0));
  }
  c10::Stream getDefaultStream(c10::Device device) const override { return getStream(device); }
  c10::Stream exchangeStream(c10::Stream stream) const noexcept override { return getStream(stream.device()); }
};
struct Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return state().root != nullptr && !state().failed; }
  bool hasPrimaryContext(c10::DeviceIndex index) const override { return index == 0 && state().root; }
  void init() const override { ready(); }
  c10::DeviceIndex deviceCount() const override { return 1; }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  void setCurrentDevice(c10::DeviceIndex index) const override { TORCH_CHECK(index <= 0, "Only NPU 0 exists"); }
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex index) const override { setCurrentDevice(index); return 0; }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex index) const override { return exchangeDevice(index); }
  c10::Device getDeviceFromPtr(void*) const override {
    TORCH_CHECK(false, "External pointers cannot be imported as intel_npu storage");
  }
};
void configure(const std::wstring& path, uintptr_t root) {
  auto& s = state();
  std::lock_guard<std::recursive_mutex> lock(s.mutex);
  TORCH_CHECK(root && !s.root && !s.library, "NPU backend can only initialize once per process");
  s.library = LoadLibraryW(path.c_str());
  TORCH_CHECK(s.library, "Cannot load NPU session DLL");
#define LOAD(field, symbol) \
  s.field = reinterpret_cast<decltype(s.field)>(GetProcAddress(s.library, #symbol)); \
  TORCH_CHECK(s.field, "Missing NPU runtime export: ", #symbol)
  LOAD(create, graphinfer_shared_buffer_create);
  LOAD(release, graphinfer_shared_buffer_release);
  LOAD(infer, graphinfer_session_infer);
  LOAD(infer_chain, graphinfer_session_infer_chain);
  LOAD(release_result, graphinfer_session_infer_result_release);
#undef LOAD
  s.cached.reserve(max_cached_buffers);
  s.root = reinterpret_cast<graphinfer_session*>(root);
}
pybind11::dict stats() {
  std::array<int64_t, 16> values;
  bool failed;
  {
    // ATen factory/copy calls may need the GIL while holding the native lock.
    // Never wait on that lock with the GIL held by a Python caller.
    pybind11::gil_scoped_release release;
    std::lock_guard<std::recursive_mutex> lock(state().mutex);
    auto& s = state();
    values = {s.allocations, s.live, s.bytes, s.peak, s.calls, s.uploads,
              s.downloads, s.copies, s.packs, s.quarantined,
              s.driver_allocations, s.allocation_reuses, static_cast<int64_t>(s.cached.size()), s.cached_bytes, s.chain_groups, s.chain_graphs};
    failed = s.failed;
  }
  pybind11::dict result;
  const char* keys[] = {"buffer_allocations", "live_buffers", "live_buffer_bytes", "peak_buffer_bytes",
      "npu_calls", "host_upload_copies", "host_download_copies", "host_device_copies",
      "host_pack_copies", "quarantined_buffer_bytes", "driver_allocations", "allocation_reuses",
      "cached_buffer_blocks", "cached_buffer_bytes", "chain_groups", "chain_graphs"};
  for (size_t i = 0; i < values.size(); i++) result[keys[i]] = values[i];
  result["context_failed"] = failed;
  return result;
}
} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty);
  m.impl("empty_strided", empty_strided);
  m.impl("view", at::native::view);
  m.impl("view.dtype", at::native::view_dtype);
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("t", at::native::t);
  m.impl("transpose.int", at::native::transpose);
  m.impl("permute", at::native::permute);
  m.impl("detach", at::native::detach);
}

PYBIND11_MODULE(_torch_intel_npu, m) {
  m.attr("max_tensor_elements") = max_tensor_elements;
  TORCH_CHECK(!at::isPrivateUse1HooksRegistered(), "PrivateUse1 hooks already belong to another backend");
  c10::impl::registerDeviceGuard(device_type, new Guard);
  at::RegisterPrivateUse1HooksInterface(new Hooks);
  c10::SetAllocator(device_type, new NpuAllocator);
  m.def("configure", configure, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("run", run, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("run_many", run_many, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("run_chain", run_chain, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("copy", copy, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("stats", stats);
  m.def("empty_cache", empty_cache, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("shutdown", [] { std::lock_guard<std::recursive_mutex> lock(state().mutex); state().root = nullptr; empty_cache(); },
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
