#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/graphinfer.h"

#include "internal.h"
#include "session_internal.h"
#include "workers/level_zero_min.h"

#define SESSION_MAX_ARGUMENTS 64u
#define SESSION_MAX_TENSOR_SIZE (64u * 1024u * 1024u)
#define SESSION_PAGE_SIZE 4096u

typedef struct session_argument {
  uint32_t index;
  uint32_t type;
  uint32_t precision;
  uint32_t dims_count;
  uint32_t dims[GRAPHINFER_MAX_DIMS];
  uint8_t name[256];
  size_t name_size;
  size_t data_size;
} session_argument;

typedef struct session_ze_functions {
  npunlock_ze_init_drivers_fn init_drivers;
  npunlock_ze_driver_get_properties_fn driver_get_properties;
  npunlock_ze_driver_get_extensions_fn driver_get_extensions;
  npunlock_ze_driver_get_extension_address_fn driver_get_extension_address;
  npunlock_ze_device_get_fn device_get;
  npunlock_ze_device_get_properties_fn device_get_properties;
  npunlock_ze_context_create_fn context_create;
  npunlock_ze_context_destroy_fn context_destroy;
  npunlock_ze_device_get_queue_groups_fn device_get_queue_groups;
  npunlock_ze_mem_alloc_host_fn mem_alloc_host;
  npunlock_ze_mem_alloc_shared_fn mem_alloc_shared;
  npunlock_ze_mem_free_fn mem_free;
  npunlock_ze_command_queue_create_fn command_queue_create;
  npunlock_ze_command_queue_destroy_fn command_queue_destroy;
  npunlock_ze_command_queue_execute_fn command_queue_execute;
  npunlock_ze_command_list_create_fn command_list_create;
  npunlock_ze_command_list_close_fn command_list_close;
  npunlock_ze_command_list_append_barrier_fn command_list_append_barrier;
  npunlock_ze_command_list_reset_fn command_list_reset;
  npunlock_ze_command_list_destroy_fn command_list_destroy;
  npunlock_ze_fence_create_fn fence_create;
  npunlock_ze_fence_destroy_fn fence_destroy;
  npunlock_ze_fence_synchronize_fn fence_synchronize;
  npunlock_ze_fence_reset_fn fence_reset;
} session_ze_functions;

typedef struct session_graph_functions {
  npunlock_ze_graph_create2_fn create2;
  npunlock_ze_graph_destroy_fn destroy;
  npunlock_ze_graph_get_properties3_fn get_properties3;
  npunlock_ze_graph_get_argument_properties3_fn get_argument_properties3;
  npunlock_ze_graph_set_argument_fn set_argument;
  npunlock_ze_graph_initialize_fn initialize;
  npunlock_ze_graph_append_initialize_fn append_initialize;
  npunlock_ze_graph_append_execute_fn append_execute;
} session_graph_functions;

struct graphinfer_session {
  CRITICAL_SECTION lock;
  LONG references;
  graphinfer_session *context_owner;
  bool active;
  bool failed;
  bool initialized;
  HMODULE loader;
  session_ze_functions ze;
  session_graph_functions graph;
  npunlock_ze_driver_handle driver;
  npunlock_ze_device_handle device;
  npunlock_ze_context_handle context;
  npunlock_ze_command_queue_handle queue;
  npunlock_ze_command_list_handle list;
  npunlock_ze_fence_handle fence;
  npunlock_ze_graph_handle handle;
  npunlock_buffer graph_blob;
  session_argument *arguments;
  size_t argument_count;
  size_t input_count;
  size_t output_count;
  uint32_t queue_ordinal;
  uint32_t init_stages;
  uint64_t fence_timeout_ns;
  uint32_t driver_index;
  uint32_t device_index;
  uint32_t driver_version;
  uint32_t vendor_id;
  uint32_t device_id;
};

typedef struct shared_buffer_impl {
  graphinfer_session *session;
  void *allocation;
  size_t size;
} shared_buffer_impl;

static CRITICAL_SECTION *session_lock(graphinfer_session *session) {
  /* ponytail: one lock per context; add parallel queues only after profiling. */
  return &session->context_owner->lock;
}

static void session_unref(graphinfer_session *session);

static bool session_view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static npunlock_status session_driver_error(npunlock_diagnostic *diagnostic, const char *stage,
                                            uint32_t code) {
  char message[160];
  snprintf(message, sizeof(message), "%s returned Level Zero result 0x%08x", stage, code);
  return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_DRIVER_FAILED, "graphinfer.session",
                                 message);
}

static bool load_export(HMODULE library, const char *name, void *destination, size_t size) {
  FARPROC function = GetProcAddress(library, name);
  if (function == NULL || size != sizeof(function)) {
    return false;
  }
  memcpy(destination, &function, size);
  return true;
}

static bool load_ze_functions(HMODULE library, session_ze_functions *ze) {
  memset(ze, 0, sizeof(*ze));
#define LOAD(name, field) load_export(library, name, &ze->field, sizeof(ze->field))
  return LOAD("zeInitDrivers", init_drivers) &&
         LOAD("zeDriverGetProperties", driver_get_properties) &&
         LOAD("zeDriverGetExtensionProperties", driver_get_extensions) &&
         LOAD("zeDriverGetExtensionFunctionAddress", driver_get_extension_address) &&
         LOAD("zeDeviceGet", device_get) && LOAD("zeDeviceGetProperties", device_get_properties) &&
         LOAD("zeContextCreate", context_create) && LOAD("zeContextDestroy", context_destroy) &&
         LOAD("zeDeviceGetCommandQueueGroupProperties", device_get_queue_groups) &&
         LOAD("zeMemAllocHost", mem_alloc_host) && LOAD("zeMemAllocShared", mem_alloc_shared) &&
         LOAD("zeMemFree", mem_free) && LOAD("zeCommandQueueCreate", command_queue_create) &&
         LOAD("zeCommandQueueDestroy", command_queue_destroy) &&
         LOAD("zeCommandQueueExecuteCommandLists", command_queue_execute) &&
         LOAD("zeCommandListCreate", command_list_create) &&
         LOAD("zeCommandListClose", command_list_close) &&
         LOAD("zeCommandListReset", command_list_reset) &&
         LOAD("zeCommandListDestroy", command_list_destroy) &&
         LOAD("zeFenceCreate", fence_create) && LOAD("zeFenceDestroy", fence_destroy) &&
         LOAD("zeFenceHostSynchronize", fence_synchronize) && LOAD("zeFenceReset", fence_reset);
#undef LOAD
}

static npunlock_status select_npu(graphinfer_session *session, const graphinfer_options *options,
                                  graphinfer_session_result *result) {
  npunlock_ze_init_driver_type_desc init = {NPUNLOCK_ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, NULL,
                                            NPUNLOCK_ZE_INIT_DRIVER_TYPE_FLAG_NPU};
  npunlock_ze_driver_handle *drivers = NULL;
  uint32_t driver_count = 0;
  uint32_t driver_index;
  uint32_t code = session->ze.init_drivers(&driver_count, NULL, &init);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    return session_driver_error(&result->diagnostic, "zeInitDrivers(count)", code);
  }
  if (driver_count == 0 ||
      (options->driver_index != GRAPHINFER_AUTO_INDEX && options->driver_index >= driver_count)) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_NOT_FOUND,
                                   "graphinfer.session", "requested NPU driver was not found");
  }
  drivers = (npunlock_ze_driver_handle *)calloc(driver_count, sizeof(*drivers));
  if (drivers == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  code = session->ze.init_drivers(&driver_count, drivers, &init);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    free(drivers);
    return session_driver_error(&result->diagnostic, "zeInitDrivers(list)", code);
  }
  for (driver_index = 0; driver_index < driver_count; ++driver_index) {
    npunlock_ze_device_handle *devices = NULL;
    uint32_t device_count = 0;
    uint32_t device_index;
    if (options->driver_index != GRAPHINFER_AUTO_INDEX && options->driver_index != driver_index) {
      continue;
    }
    code = session->ze.device_get(drivers[driver_index], &device_count, NULL);
    if (code != NPUNLOCK_ZE_SUCCESS || device_count == 0) {
      continue;
    }
    devices = (npunlock_ze_device_handle *)calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
      free(drivers);
      return NPUNLOCK_STATUS_OUT_OF_MEMORY;
    }
    code = session->ze.device_get(drivers[driver_index], &device_count, devices);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      free(devices);
      continue;
    }
    for (device_index = 0; device_index < device_count; ++device_index) {
      npunlock_ze_device_properties properties = {0};
      if (options->device_index != GRAPHINFER_AUTO_INDEX && options->device_index != device_index) {
        continue;
      }
      properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      code = session->ze.device_get_properties(devices[device_index], &properties);
      if (code == NPUNLOCK_ZE_SUCCESS && properties.type == NPUNLOCK_ZE_DEVICE_TYPE_VPU) {
        session->driver = drivers[driver_index];
        session->device = devices[device_index];
        result->selected_driver_index = driver_index;
        result->selected_device_index = device_index;
        result->device_vendor_id = properties.vendorId;
        result->device_id = properties.deviceId;
        free(devices);
        free(drivers);
        return NPUNLOCK_STATUS_OK;
      }
    }
    free(devices);
  }
  free(drivers);
  return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_NOT_FOUND,
                                 "graphinfer.session", "requested Level Zero VPU was not found");
}

static uint32_t find_graph_version(graphinfer_session *session, graphinfer_session_result *result) {
  npunlock_ze_driver_extension_properties *extensions = NULL;
  uint32_t count = 0;
  uint32_t index;
  uint32_t version = 0;
  uint32_t code = session->ze.driver_get_extensions(session->driver, &count, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    session_driver_error(&result->diagnostic, "zeDriverGetExtensionProperties(count)", code);
    return 0;
  }
  extensions = (npunlock_ze_driver_extension_properties *)calloc(count, sizeof(*extensions));
  if (extensions == NULL && count != 0) {
    npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                            "graphinfer.session", "failed to allocate extension list");
    return 0;
  }
  code = session->ze.driver_get_extensions(session->driver, &count, extensions);
  if (code == NPUNLOCK_ZE_SUCCESS) {
    for (index = 0; index < count; ++index) {
      if (memchr(extensions[index].name, 0, sizeof(extensions[index].name)) != NULL &&
          strcmp(extensions[index].name, "ZE_extension_graph") == 0) {
        version = extensions[index].version;
        break;
      }
    }
  }
  free(extensions);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    session_driver_error(&result->diagnostic, "zeDriverGetExtensionProperties(list)", code);
  } else if (version == 0) {
    npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED, "graphinfer.session",
                            "NPU graph extension is unavailable");
  }
  return version;
}

static npunlock_status load_graph_functions(graphinfer_session *session, uint32_t version,
                                            graphinfer_session_result *result) {
  void *raw_driver_ddi = NULL;
  npunlock_ze_npu_get_extension_fn get_extension = NULL;
  npunlock_ze_graph_ddi_prefix *ddi = NULL;
  npunlock_ze_driver_extension_npu request = {0};
  uint32_t code = session->ze.driver_get_extension_address(
      session->driver, "ZE_extension_driver_npu", &raw_driver_ddi);
  if (code != NPUNLOCK_ZE_SUCCESS || raw_driver_ddi == NULL) {
    return session_driver_error(&result->diagnostic, "zeDriverGetExtensionFunctionAddress", code);
  }
  memcpy(&get_extension, raw_driver_ddi, sizeof(get_extension));
  if (get_extension == NULL) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                   "graphinfer.session", "NPU extension table is invalid");
  }
  request.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT;
  request.name = "ZE_extension_graph";
  request.version = version;
  request.ppFunctionAddress = (void **)&ddi;
  code = get_extension(session->driver, &request);
  if (code != NPUNLOCK_ZE_SUCCESS || ddi == NULL) {
    return session_driver_error(&result->diagnostic, "pfnGetExtension", code);
  }
#define GET(slot, field)                                                                           \
  memcpy(&session->graph.field, &ddi->slots[slot], sizeof(session->graph.field))
  GET(NPUNLOCK_ZE_GRAPH_DDI_CREATE2, create2);
  GET(NPUNLOCK_ZE_GRAPH_DDI_DESTROY, destroy);
  GET(NPUNLOCK_ZE_GRAPH_DDI_GET_PROPERTIES3, get_properties3);
  GET(NPUNLOCK_ZE_GRAPH_DDI_GET_ARGUMENT_PROPERTIES3, get_argument_properties3);
  GET(NPUNLOCK_ZE_GRAPH_DDI_SET_ARGUMENT, set_argument);
  GET(NPUNLOCK_ZE_GRAPH_DDI_INITIALIZE, initialize);
  GET(NPUNLOCK_ZE_GRAPH_DDI_APPEND_INITIALIZE, append_initialize);
  GET(NPUNLOCK_ZE_GRAPH_DDI_APPEND_EXECUTE, append_execute);
#undef GET
  if (session->graph.create2 == NULL || session->graph.destroy == NULL ||
      session->graph.get_properties3 == NULL || session->graph.get_argument_properties3 == NULL ||
      session->graph.set_argument == NULL || session->graph.append_execute == NULL) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                   "graphinfer.session",
                                   "required graph execution entry point is unavailable");
  }
  return NPUNLOCK_STATUS_OK;
}

static bool argument_data_size(const npunlock_ze_graph_argument_properties_3 *properties,
                               size_t *data_size) {
  size_t elements = 1;
  size_t element_size;
  uint32_t index;
  if (properties->dims_count == 0 || properties->dims_count > GRAPHINFER_MAX_DIMS) {
    return false;
  }
  if (properties->devicePrecision == NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
    element_size = sizeof(uint16_t);
  } else if (properties->devicePrecision == NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
    element_size = sizeof(uint32_t);
  } else if (properties->devicePrecision == NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_UINT8) {
    element_size = sizeof(uint8_t); /* Exact DMA transport, not a Torch arithmetic dtype. */
  } else {
    return false;
  }
  for (index = 0; index < properties->dims_count; ++index) {
    if (properties->dims[index] == 0 ||
        !npunlock_checked_mul_size(elements, properties->dims[index], &elements)) {
      return false;
    }
  }
  return npunlock_checked_mul_size(elements, element_size, data_size) && *data_size != 0 &&
         *data_size <= SESSION_MAX_TENSOR_SIZE;
}

static npunlock_status query_arguments(graphinfer_session *session,
                                       graphinfer_session_result *result) {
  npunlock_ze_graph_properties_3 properties = {0};
  uint32_t index;
  uint32_t code;
  properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
  code = session->graph.get_properties3(session->handle, &properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    return session_driver_error(&result->diagnostic, "pfnGetProperties3", code);
  }
  if (properties.numGraphArgs == 0 || properties.numGraphArgs > SESSION_MAX_ARGUMENTS) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                   "graphinfer.session", "unsupported graph argument count");
  }
  session->arguments =
      (session_argument *)calloc(properties.numGraphArgs, sizeof(*session->arguments));
  if (session->arguments == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  session->argument_count = properties.numGraphArgs;
  session->init_stages = properties.initStageRequired;
  for (index = 0; index < properties.numGraphArgs; ++index) {
    npunlock_ze_graph_argument_properties_3 metadata = {0};
    session_argument *argument = &session->arguments[index];
    const char *name_end;
    metadata.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;
    code = session->graph.get_argument_properties3(session->handle, index, &metadata);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      return session_driver_error(&result->diagnostic, "pfnGetArgumentProperties3", code);
    }
    if ((metadata.type != NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT &&
         metadata.type != NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) ||
        !argument_data_size(&metadata, &argument->data_size)) {
      return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                     "graphinfer.session",
                                     "only bounded static FP16/FP32/U8 tensors are supported");
    }
    name_end = (const char *)memchr(metadata.name, 0, sizeof(metadata.name));
    if (name_end == NULL) {
      return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_DRIVER_FAILED,
                                     "graphinfer.session", "graph argument name is invalid");
    }
    argument->index = index;
    argument->type = metadata.type;
    argument->precision = metadata.devicePrecision;
    argument->dims_count = metadata.dims_count;
    memcpy(argument->dims, metadata.dims, sizeof(argument->dims));
    argument->name_size = (size_t)(name_end - metadata.name);
    memcpy(argument->name, metadata.name, argument->name_size);
    if (argument->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
      ++session->input_count;
    } else {
      ++session->output_count;
    }
  }
  return session->input_count != 0 && session->output_count != 0
             ? NPUNLOCK_STATUS_OK
             : npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                       "graphinfer.session",
                                       "graph requires at least one input and output");
}

static npunlock_status find_compute_queue(graphinfer_session *session,
                                          npunlock_diagnostic *diagnostic) {
  npunlock_ze_command_queue_group_properties *groups = NULL;
  uint32_t count = 0;
  uint32_t index;
  uint32_t code = session->ze.device_get_queue_groups(session->device, &count, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    return session_driver_error(diagnostic, "zeDeviceGetCommandQueueGroupProperties(count)", code);
  }
  groups = (npunlock_ze_command_queue_group_properties *)calloc(count, sizeof(*groups));
  if (groups == NULL && count != 0) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  for (index = 0; index < count; ++index) {
    groups[index].stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  code = session->ze.device_get_queue_groups(session->device, &count, groups);
  if (code == NPUNLOCK_ZE_SUCCESS) {
    for (index = 0; index < count; ++index) {
      if (groups[index].numQueues != 0 &&
          (groups[index].flags & NPUNLOCK_ZE_COMMAND_QUEUE_GROUP_COMPUTE) != 0) {
        session->queue_ordinal = index;
        free(groups);
        return NPUNLOCK_STATUS_OK;
      }
    }
  }
  free(groups);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    return session_driver_error(diagnostic, "zeDeviceGetCommandQueueGroupProperties(list)", code);
  }
  return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED, "graphinfer.session",
                                 "NPU exposes no compute queue");
}

static void release_submission(graphinfer_session *session) {
  if (!session->context_owner->failed) {
    if (session->fence != NULL) {
      session->ze.fence_destroy(session->fence);
      session->fence = NULL;
    }
    if (session->list != NULL) {
      session->ze.command_list_destroy(session->list);
      session->list = NULL;
    }
  }
}

static void session_destroy(graphinfer_session *session) {
  graphinfer_session *owner;
  if (session == NULL) {
    return;
  }
  owner = session->context_owner;
  release_submission(session);
  free(session->arguments);
  if (owner == session && !session->failed && session->queue != NULL) {
    session->ze.command_queue_destroy(session->queue);
  }
  if (!owner->failed && session->handle != NULL && session->graph.destroy != NULL) {
    session->graph.destroy(session->handle);
  }
  if (!owner->failed) {
    npunlock_buffer_release(&session->graph_blob);
  }
  if (owner == session && !session->failed && session->context != NULL &&
      session->ze.context_destroy != NULL) {
    session->ze.context_destroy(session->context);
  }
  if (!owner->failed && session->loader != NULL) {
    FreeLibrary(session->loader);
  }
  DeleteCriticalSection(&session->lock);
  free(session);
  if (owner != session) {
    session_unref(owner);
  }
}

static void session_unref(graphinfer_session *session) {
  if (session != NULL && InterlockedDecrement(&session->references) == 0) {
    session_destroy(session);
  }
}

static npunlock_status submit(graphinfer_session *session, bool initialize,
                              npunlock_diagnostic *diagnostic) {
  npunlock_ze_command_queue_desc queue_desc = {0};
  npunlock_ze_command_list_desc list_desc = {0};
  npunlock_ze_fence_desc fence_desc = {0};
  npunlock_ze_command_queue_handle queue = session->context_owner->queue;
  npunlock_status status = NPUNLOCK_STATUS_DRIVER_FAILED;
  uint32_t code;
  queue_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue_desc.ordinal = session->queue_ordinal;
  list_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  list_desc.commandQueueGroupOrdinal = session->queue_ordinal;
  fence_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_FENCE_DESC;
  if (queue == NULL) {
    code = session->ze.command_queue_create(session->context, session->device, &queue_desc, &queue);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      status = session_driver_error(diagnostic, "zeCommandQueueCreate", code);
      goto done;
    }
    session->context_owner->queue = queue;
  }
  if (session->list == NULL) {
    code = session->ze.command_list_create(session->context, session->device, &list_desc,
                                            &session->list);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      status = session_driver_error(diagnostic, "zeCommandListCreate", code);
      goto done;
    }
  }
  if (session->fence == NULL) {
    code = session->ze.fence_create(queue, &fence_desc, &session->fence);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      status = session_driver_error(diagnostic, "zeFenceCreate", code);
      goto done;
    }
  }
  code = initialize ? session->graph.append_initialize(session->list, session->handle, NULL, 0, NULL)
                    : session->graph.append_execute(session->list, session->handle, NULL, NULL, 0, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    status = session_driver_error(
        diagnostic, initialize ? "pfnAppendGraphInitialize" : "pfnAppendGraphExecute", code);
    goto done;
  }
  code = session->ze.command_list_close(session->list);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    status = session_driver_error(diagnostic, "zeCommandListClose", code);
    goto done;
  }
  code = session->ze.command_queue_execute(queue, 1, &session->list, session->fence);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    session->context_owner->failed = true;
    status = session_driver_error(diagnostic, "zeCommandQueueExecuteCommandLists", code);
    goto done;
  }
  code = session->ze.fence_synchronize(session->fence, session->fence_timeout_ns);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    /* Completion is uncertain (including timeouts). Quarantine this context:
     * never free allocations or unload its driver while work may be in flight.
     * The caller must restart its process after such an execution failure. */
    session->context_owner->failed = true;
    status = session_driver_error(diagnostic, "zeFenceHostSynchronize", code);
    goto done;
  }
  /* ponytail: reuse objects, re-record each call. Reset only after completion,
   * before callers can release buffers. No cached Tensor addresses or retained
   * allocations; fixed-address/mutable-command replay can be added if measured. */
  code = session->ze.command_list_reset(session->list);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    status = session_driver_error(diagnostic, "zeCommandListReset", code);
    goto done;
  }
  code = session->ze.fence_reset(session->fence);
  status = code == NPUNLOCK_ZE_SUCCESS ? NPUNLOCK_STATUS_OK
                                     : session_driver_error(diagnostic, "zeFenceReset", code);

done:
  if (status != NPUNLOCK_STATUS_OK) {
    release_submission(session);
  }
  return status;
}

static npunlock_status session_create(graphinfer_session *parent, const graphinfer_options *options,
                                      npunlock_view graph_blob,
                                      graphinfer_session_result *result) {
  graphinfer_session *session = NULL;
  npunlock_ze_driver_properties driver_properties = {0};
  npunlock_ze_context_desc context_desc = {0};
  npunlock_ze_graph_desc_2 graph_desc = {0};
  npunlock_status status;
  uint32_t graph_version;
  uint32_t code;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) || options->timeout_ms == 0 ||
      !npunlock_view_is_valid(graph_blob) || graph_blob.size == 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.session", "invalid options or graph blob");
  }
  session = (graphinfer_session *)calloc(1, sizeof(*session));
  if (session == NULL) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "graphinfer.session", "failed to allocate session");
  }
  InitializeCriticalSection(&session->lock);
  session->context_owner = session;
  session->references = 1;
  session->active = true;
  session->fence_timeout_ns = (uint64_t)options->timeout_ms * 1000000u;
  /* Native import/initialization can retain ELF-backed code/constant views.
   * Keep the image alive through graph destruction, including deferred init. */
  status = npunlock_buffer_copy(graph_blob, &session->graph_blob);
  if (status != NPUNLOCK_STATUS_OK) {
    npunlock_set_diagnostic(&result->diagnostic, status, "graphinfer.session",
                            "failed to retain native graph image");
    goto fail;
  }
  if (parent != NULL) {
    graphinfer_session *owner = parent->context_owner;
    EnterCriticalSection(session_lock(parent));
    if (!parent->active || owner->failed ||
        (options->driver_index != GRAPHINFER_AUTO_INDEX &&
         options->driver_index != owner->driver_index) ||
        (options->device_index != GRAPHINFER_AUTO_INDEX &&
         options->device_index != owner->device_index)) {
      LeaveCriticalSection(session_lock(parent));
      status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                       "graphinfer.session",
                                       "parent is closed, failed, or selects another device");
      goto fail;
    }
    InterlockedIncrement(&owner->references);
    session->context_owner = owner;
    session->ze = owner->ze;
    session->graph = owner->graph;
    session->driver = owner->driver;
    session->device = owner->device;
    session->context = owner->context;
    result->selected_driver_index = owner->driver_index;
    result->selected_device_index = owner->device_index;
    result->driver_version = owner->driver_version;
    result->device_vendor_id = owner->vendor_id;
    result->device_id = owner->device_id;
    LeaveCriticalSection(session_lock(parent));
    goto create_graph;
  }
  session->loader = LoadLibraryExW(L"ze_loader.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (session->loader == NULL) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_NOT_FOUND,
                                     "graphinfer.session", "ze_loader.dll was not found");
    goto fail;
  }
  if (!load_ze_functions(session->loader, &session->ze)) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                     "graphinfer.session",
                                     "Level Zero loader lacks a required function");
    goto fail;
  }
  status = select_npu(session, options, result);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  driver_properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  code = session->ze.driver_get_properties(session->driver, &driver_properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    status = session_driver_error(&result->diagnostic, "zeDriverGetProperties", code);
    goto fail;
  }
  result->driver_version = driver_properties.driverVersion;
  graph_version = find_graph_version(session, result);
  if (graph_version == 0) {
    status = NPUNLOCK_STATUS_UNSUPPORTED;
    goto fail;
  }
  status = load_graph_functions(session, graph_version, result);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  context_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  code = session->ze.context_create(session->driver, &context_desc, &session->context);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    status = session_driver_error(&result->diagnostic, "zeContextCreate", code);
    goto fail;
  }
  session->driver_index = result->selected_driver_index;
  session->device_index = result->selected_device_index;
  session->driver_version = result->driver_version;
  session->vendor_id = result->device_vendor_id;
  session->device_id = result->device_id;
create_graph:
  graph_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
  graph_desc.format = NPUNLOCK_ZE_GRAPH_FORMAT_NATIVE;
  graph_desc.inputSize = session->graph_blob.size;
  graph_desc.pInput = session->graph_blob.data;
  graph_desc.pBuildFlags = NULL;
  code = session->graph.create2(session->context, session->device, &graph_desc, &session->handle);
  if (code != NPUNLOCK_ZE_SUCCESS || session->handle == NULL) {
    status = session_driver_error(&result->diagnostic, "pfnCreate2(native)", code);
    goto fail;
  }
  status = query_arguments(session, result);
  if (status == NPUNLOCK_STATUS_OK) {
    status = find_compute_queue(session, &result->diagnostic);
  }
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  result->session = session;
  return NPUNLOCK_STATUS_OK;

fail:
  session_destroy(session);
  return status;
}

npunlock_status graphinfer_session_create(const graphinfer_options *options,
                                          npunlock_view graph_blob,
                                          graphinfer_session_result *result) {
  return session_create(NULL, options, graph_blob, result);
}

npunlock_status graphinfer_session_create_child(graphinfer_session *parent,
                                                const graphinfer_options *options,
                                                npunlock_view graph_blob,
                                                graphinfer_session_result *result) {
  if (parent == NULL) {
    if (result == NULL) {
      return NPUNLOCK_STATUS_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    result->struct_size = sizeof(*result);
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.session", "child requires a parent session");
  }
  return session_create(parent, options, graph_blob, result);
}

void graphinfer_session_result_release(graphinfer_session_result *result) {
  graphinfer_session *session;
  if (result == NULL) {
    return;
  }
  session = result->session;
  if (session != NULL) {
    EnterCriticalSection(session_lock(session));
    session->active = false;
    release_submission(session);
    if (!session->context_owner->failed && session->handle != NULL) {
      session->graph.destroy(session->handle);
      session->handle = NULL;
    }
    LeaveCriticalSection(session_lock(session));
    session_unref(session);
  }
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}

npunlock_status graphinfer_shared_buffer_create(graphinfer_session *session, size_t size,
                                                graphinfer_shared_buffer *buffer) {
  shared_buffer_impl *implementation = NULL;
  npunlock_ze_device_mem_alloc_desc device_desc = {0};
  npunlock_ze_host_mem_alloc_desc host_desc = {0};
  uint32_t code;
  if (buffer == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(buffer, 0, sizeof(*buffer));
  buffer->struct_size = sizeof(*buffer);
  if (session == NULL || size == 0 || size > SESSION_MAX_TENSOR_SIZE) {
    return npunlock_set_diagnostic(&buffer->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.shared_buffer", "invalid session or buffer size");
  }
  implementation = (shared_buffer_impl *)calloc(1, sizeof(*implementation));
  if (implementation == NULL) {
    return npunlock_set_diagnostic(&buffer->diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "graphinfer.shared_buffer", "failed to allocate buffer state");
  }
  EnterCriticalSection(session_lock(session));
  if (!session->active || session->context_owner->failed) {
    LeaveCriticalSection(session_lock(session));
    free(implementation);
    return npunlock_set_diagnostic(&buffer->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.shared_buffer", "session is closed");
  }
  device_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  host_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  code =
      session->ze.mem_alloc_shared(session->context, &device_desc, &host_desc, size,
                                   SESSION_PAGE_SIZE, session->device, &implementation->allocation);
  if (code == NPUNLOCK_ZE_SUCCESS) {
    implementation->session = session;
    implementation->size = size;
    InterlockedIncrement(&session->references);
  }
  LeaveCriticalSection(session_lock(session));
  if (code != NPUNLOCK_ZE_SUCCESS) {
    free(implementation);
    return session_driver_error(&buffer->diagnostic, "zeMemAllocShared", code);
  }
  memset(implementation->allocation, 0, size);
  buffer->data = (uint8_t *)implementation->allocation;
  buffer->size = size;
  buffer->implementation = implementation;
  return NPUNLOCK_STATUS_OK;
}

void graphinfer_shared_buffer_release(graphinfer_shared_buffer *buffer) {
  shared_buffer_impl *implementation;
  graphinfer_session *session;
  if (buffer == NULL) {
    return;
  }
  implementation = (shared_buffer_impl *)buffer->implementation;
  if (implementation != NULL) {
    session = implementation->session;
    EnterCriticalSection(session_lock(session));
    if (!session->context_owner->failed) {
      session->ze.mem_free(session->context, implementation->allocation);
    }
    LeaveCriticalSection(session_lock(session));
    free(implementation);
    session_unref(session);
  }
  npunlock_diagnostic_release(&buffer->diagnostic);
  memset(buffer, 0, sizeof(*buffer));
}

static bool binding_name_matches(const graphinfer_shared_tensor *binding,
                                 const session_argument *argument) {
  return binding->argument_name_utf8.size == argument->name_size &&
         memcmp(binding->argument_name_utf8.data, argument->name, argument->name_size) == 0;
}

static npunlock_status collect_bindings(graphinfer_session *session,
                                        const graphinfer_shared_tensor *bindings,
                                        size_t binding_count, uint32_t required_type,
                                        void **allocations, bool *matched,
                                        npunlock_diagnostic *diagnostic) {
  size_t binding_index;
  for (binding_index = 0; binding_index < binding_count; ++binding_index) {
    const graphinfer_shared_tensor *binding = &bindings[binding_index];
    shared_buffer_impl *implementation;
    session_argument *argument = NULL;
    bool by_index;
    bool by_name;
    size_t argument_index;
    if (binding->struct_size < sizeof(*binding) || binding->buffer == NULL ||
        !npunlock_view_is_valid(binding->argument_name_utf8) ||
        session_view_has_nul(binding->argument_name_utf8)) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.session", "invalid shared tensor binding");
    }
    by_index = binding->argument_index != GRAPHINFER_AUTO_INDEX;
    by_name = binding->argument_name_utf8.size != 0;
    if (by_index == by_name) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.session", "binding needs one selector");
    }
    for (argument_index = 0; argument_index < session->argument_count; ++argument_index) {
      session_argument *candidate = &session->arguments[argument_index];
      if ((by_index && binding->argument_index == candidate->index) ||
          (by_name && binding_name_matches(binding, candidate))) {
        if (argument != NULL) {
          return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                         "graphinfer.session", "binding selector is ambiguous");
        }
        argument = candidate;
      }
    }
    if (argument == NULL || argument->type != required_type || matched[argument->index]) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.session",
                                     "binding does not uniquely select the required argument");
    }
    implementation = (shared_buffer_impl *)binding->buffer->implementation;
    if (implementation == NULL || implementation->session->context_owner != session->context_owner ||
        binding->buffer->data != implementation->allocation ||
        binding->buffer->size != implementation->size ||
        implementation->size != argument->data_size) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.session",
                                     "shared buffer has the wrong owner or byte size");
    }
    allocations[argument->index] = implementation->allocation;
    matched[argument->index] = true;
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status execute_allocations_locked(graphinfer_session *session,
                                                  void *const *allocations,
                                                  npunlock_diagnostic *diagnostic) {
  npunlock_status status;
  size_t index;
  uint32_t code;
  if (!session->initialized) {
    for (index = 0; index < session->argument_count; ++index) {
      code = session->graph.set_argument(session->handle, (uint32_t)index, allocations[index]);
      if (code != NPUNLOCK_ZE_SUCCESS) {
        return session_driver_error(diagnostic, "pfnSetArgumentValue(pre-init)", code);
      }
    }
    if ((session->init_stages & NPUNLOCK_ZE_GRAPH_STAGE_INITIALIZE) != 0) {
      if (session->graph.initialize == NULL) {
        return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                       "graphinfer.session",
                                       "direct graph initialization is unavailable");
      }
      code = session->graph.initialize(session->handle);
      if (code != NPUNLOCK_ZE_SUCCESS) {
        return session_driver_error(diagnostic, "pfnGraphInitialize", code);
      }
    }
    if ((session->init_stages & NPUNLOCK_ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) != 0) {
      if (session->graph.append_initialize == NULL) {
        return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                       "graphinfer.session",
                                       "command-list initialization is unavailable");
      }
      status = submit(session, true, diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        return status;
      }
    }
    session->initialized = true;
  }
  for (index = 0; index < session->argument_count; ++index) {
    code = session->graph.set_argument(session->handle, (uint32_t)index, allocations[index]);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      return session_driver_error(diagnostic, "pfnSetArgumentValue(post-init)", code);
    }
  }
  return submit(session, false, diagnostic);
}

npunlock_status graphinfer_session_infer(graphinfer_session *session,
                                         const graphinfer_shared_tensor *inputs, size_t input_count,
                                         const graphinfer_shared_tensor *outputs,
                                         size_t output_count,
                                         graphinfer_session_infer_result *result) {
  void *allocations[SESSION_MAX_ARGUMENTS] = {0};
  bool matched[SESSION_MAX_ARGUMENTS] = {false};
  npunlock_status status;
  size_t index;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = sizeof(*result);
  if (session == NULL || inputs == NULL || outputs == NULL || input_count != session->input_count ||
      output_count != session->output_count) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.session", "bindings do not cover graph tensors");
  }
  EnterCriticalSection(session_lock(session));
  if (!session->active || session->context_owner->failed) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.session",
                                     "session is closed or context failed; restart process");
    goto done;
  }
  status = collect_bindings(session, inputs, input_count, NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT,
                            allocations, matched, &result->diagnostic);
  if (status == NPUNLOCK_STATUS_OK) {
    status =
        collect_bindings(session, outputs, output_count, NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT,
                         allocations, matched, &result->diagnostic);
  }
  if (status != NPUNLOCK_STATUS_OK) {
    goto done;
  }
  for (index = 0; index < session->argument_count; ++index) {
    size_t previous;
    if (!matched[index]) {
      status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                       "graphinfer.session", "graph argument is not bound");
      goto done;
    }
    for (previous = 0; previous < index; ++previous) {
      if (allocations[previous] == allocations[index] &&
          (session->arguments[previous].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT ||
           session->arguments[index].type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT)) {
        status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                         "graphinfer.session", "output aliases another graph argument");
        goto done;
      }
    }
  }
  status = execute_allocations_locked(session, allocations, &result->diagnostic);

done:
  LeaveCriticalSection(session_lock(session));
  return status;
}

static bool copied_input_matches(const graphinfer_input *input, const session_argument *argument) {
  if (input->argument_index != GRAPHINFER_AUTO_INDEX) {
    return input->argument_index == argument->index;
  }
  return input->argument_name_utf8.size == argument->name_size &&
         memcmp(input->argument_name_utf8.data, argument->name, argument->name_size) == 0;
}

npunlock_status npunlock_graphinfer_infer_copied(graphinfer_session *session,
                                                 const graphinfer_input *inputs, size_t input_count,
                                                 graphinfer_result *result) {
  void *allocations[SESSION_MAX_ARGUMENTS] = {0};
  bool matched_inputs[SESSION_MAX_ARGUMENTS] = {false};
  npunlock_ze_host_mem_alloc_desc host_desc = {0};
  npunlock_status status = NPUNLOCK_STATUS_OK;
  size_t index;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  if (session == NULL || inputs == NULL || input_count != session->input_count) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.infer", "inputs do not cover graph tensors");
  }
  host_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  EnterCriticalSection(session_lock(session));
  if (!session->active || session->context_owner->failed) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.infer", "session is closed");
    goto done;
  }
  for (index = 0; index < input_count; ++index) {
    size_t argument_index;
    session_argument *argument = NULL;
    for (argument_index = 0; argument_index < session->argument_count; ++argument_index) {
      session_argument *candidate = &session->arguments[argument_index];
      if (candidate->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT &&
          copied_input_matches(&inputs[index], candidate)) {
        if (argument != NULL) {
          status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                           "graphinfer.infer", "input selector is ambiguous");
          goto done;
        }
        argument = candidate;
      }
    }
    if (argument == NULL || matched_inputs[argument->index] ||
        inputs[index].data.size != argument->data_size) {
      status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                       "graphinfer.infer",
                                       "input selector or byte size does not match the graph");
      goto done;
    }
    matched_inputs[argument->index] = true;
  }
  result->outputs = (graphinfer_output *)calloc(session->output_count, sizeof(*result->outputs));
  if (result->outputs == NULL) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                     "graphinfer.infer", "failed to allocate output descriptors");
    goto done;
  }
  for (index = 0; index < session->argument_count; ++index) {
    session_argument *argument = &session->arguments[index];
    uint32_t code;
    host_desc.flags = argument->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT
                          ? NPUNLOCK_ZE_HOST_MEM_ALLOC_WRITE_COMBINED
                          : 0;
    code = session->ze.mem_alloc_host(session->context, &host_desc, argument->data_size,
                                      SESSION_PAGE_SIZE, &allocations[index]);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      status = session_driver_error(&result->diagnostic, "zeMemAllocHost(copied tensor)", code);
      goto done;
    }
    if (argument->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
      size_t input_index;
      for (input_index = 0; input_index < input_count; ++input_index) {
        if (copied_input_matches(&inputs[input_index], argument)) {
          memcpy(allocations[index], inputs[input_index].data.data, argument->data_size);
          break;
        }
      }
    } else {
      memset(allocations[index], 0, argument->data_size);
    }
  }
  status = execute_allocations_locked(session, allocations, &result->diagnostic);
  if (status == NPUNLOCK_STATUS_OK) {
    size_t output_index = 0;
    for (index = 0; index < session->argument_count; ++index) {
      session_argument *argument = &session->arguments[index];
      graphinfer_output *output;
      if (argument->type != NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
        continue;
      }
      output = &result->outputs[output_index++];
      result->output_count = output_index;
      output->struct_size = sizeof(*output);
      output->argument_index = argument->index;
      output->precision = argument->precision;
      output->dims_count = argument->dims_count;
      memcpy(output->dims, argument->dims, sizeof(output->dims));
      status = npunlock_buffer_copy((npunlock_view){argument->name, argument->name_size},
                                    &output->argument_name_utf8);
      if (status == NPUNLOCK_STATUS_OK) {
        status = npunlock_buffer_copy((npunlock_view){allocations[index], argument->data_size},
                                      &output->data);
      }
      if (status != NPUNLOCK_STATUS_OK) {
        npunlock_set_diagnostic(&result->diagnostic, status, "graphinfer.infer",
                                "failed to retain graph output");
        goto done;
      }
    }
  }

done:
  for (index = 0; index < session->argument_count; ++index) {
    if (!session->context_owner->failed && allocations[index] != NULL) {
      session->ze.mem_free(session->context, allocations[index]);
    }
  }
  LeaveCriticalSection(session_lock(session));
  return status;
}

void graphinfer_session_infer_result_release(graphinfer_session_infer_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}

npunlock_status graphinfer_session_infer_chain(
    const graphinfer_chain_entry *entries, size_t count,
    graphinfer_session_infer_result *result) {
  void *allocations[4][SESSION_MAX_ARGUMENTS] = {{0}};
  bool matched[4][SESSION_MAX_ARGUMENTS] = {{false}};
  graphinfer_session *first;
  npunlock_status status = NPUNLOCK_STATUS_INVALID_ARGUMENT;
  uint32_t code;
  size_t i, j, k;
  bool recorded = false, cold = false;
  if (!result) return status;
  memset(result, 0, sizeof(*result)); result->struct_size = sizeof(*result);
  if (!entries || count < 1 || count > 4 || entries[0].struct_size < sizeof(entries[0]) || !entries[0].session)
    return npunlock_set_diagnostic(&result->diagnostic, status, "graphinfer.chain", "requires 1..4 live sessions");
  first = entries[0].session;
  EnterCriticalSection(session_lock(first));
  if (first->context_owner->failed) goto invalid;
  /* Validate the entire batch before any binding or command is recorded. */
  for (i = 0; i < count; ++i) {
    graphinfer_session *s;
    if (entries[i].struct_size < sizeof(entries[i])) goto invalid;
    s = entries[i].session;
    if (!s || s->context_owner != first->context_owner || !s->active || !entries[i].inputs || !entries[i].outputs ||
        entries[i].input_count != s->input_count || entries[i].output_count != s->output_count)
      goto invalid;
    for (j = 0; j < i; ++j) if (entries[j].session == s) goto invalid;
    cold |= !s->initialized || !s->list || !s->fence;
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
    }
  }
  /* Reject forward references and overwrites, allowing producer -> consumer. */
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
  for (i = 0; i < count; ++i) {
    graphinfer_session *session = entries[i].session;
    for (j = 0; j < session->argument_count; ++j) {
      code = session->graph.set_argument(session->handle, (uint32_t)j, allocations[i][j]);
      if (code) { status = session_driver_error(&result->diagnostic, "chain.SetArgumentValue", code); goto done; }
    }
  }
  for (i = 0; i < count; ++i) {
    if (i) {
      code = first->context_owner->ze.command_list_append_barrier(first->list, NULL, 0, NULL);
      if (code) { status = session_driver_error(&result->diagnostic, "zeCommandListAppendBarrier(chain)", code); goto done; }
    }
    recorded = true;
    code = first->graph.append_execute(first->list, entries[i].session->handle, NULL, NULL, 0, NULL);
    if (code) { status = session_driver_error(&result->diagnostic, "chain.AppendGraphExecute", code); goto done; }
  }
  code = first->ze.command_list_close(first->list);
  if (code) { status = session_driver_error(&result->diagnostic, "zeCommandListClose(chain)", code); goto done; }
  code = first->ze.command_queue_execute(first->context_owner->queue, 1, &first->list, first->fence);
  if (code) { first->context_owner->failed = true; status = session_driver_error(&result->diagnostic, "zeCommandQueueExecuteCommandLists(chain)", code); goto done; }
  code = first->ze.fence_synchronize(first->fence, first->fence_timeout_ns);
  if (code) { first->context_owner->failed = true; status = session_driver_error(&result->diagnostic, "zeFenceHostSynchronize(chain)", code); goto done; }
  code = first->ze.command_list_reset(first->list);
  if (code) { status = session_driver_error(&result->diagnostic, "zeCommandListReset(chain)", code); goto done; }
  code = first->ze.fence_reset(first->fence);
  status = code ? session_driver_error(&result->diagnostic, "zeFenceReset(chain)", code) : NPUNLOCK_STATUS_OK;
  goto done;
invalid:
  status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
      "graphinfer.chain", "closed/duplicate session, wrong context, or aliased/incomplete bindings");
done:
  if (status != NPUNLOCK_STATUS_OK && recorded) release_submission(first);
  LeaveCriticalSection(session_lock(first));
  return status;
}
