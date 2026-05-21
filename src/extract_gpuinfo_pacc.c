/*
 *
 * PACC support through jobd shared DDR.
 *
 * This file is part of Nvtop.
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 */

#include "nvtop/extract_gpuinfo_common.h"
#include "nvtop/pacc_monitor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct gpu_info_pacc {
  struct gpu_info base;
  unsigned monitor_index;
  struct pacc_monitor_device_desc desc;
  struct pacc_monitor_sample last_sample;
};

static struct pacc_monitor_backend *pacc_backend;
static struct gpu_info_pacc *pacc_infos;

static bool gpuinfo_pacc_init(void);
static void gpuinfo_pacc_shutdown(void);
static const char *gpuinfo_pacc_last_error_string(void);
static bool gpuinfo_pacc_get_device_handles(struct list_head *devices, unsigned *count);
static void gpuinfo_pacc_populate_static_info(struct gpu_info *_gpu_info);
static void gpuinfo_pacc_refresh_dynamic_info(struct gpu_info *_gpu_info);
static void gpuinfo_pacc_get_running_processes(struct gpu_info *_gpu_info);

struct gpu_vendor gpu_vendor_pacc = {
    .init = gpuinfo_pacc_init,
    .shutdown = gpuinfo_pacc_shutdown,
    .last_error_string = gpuinfo_pacc_last_error_string,
    .get_device_handles = gpuinfo_pacc_get_device_handles,
    .populate_static_info = gpuinfo_pacc_populate_static_info,
    .refresh_dynamic_info = gpuinfo_pacc_refresh_dynamic_info,
    .refresh_running_processes = gpuinfo_pacc_get_running_processes,
    .name = "PACC",
};

__attribute__((constructor)) static void init_extract_gpuinfo_pacc(void) {
  register_gpu_vendor(&gpu_vendor_pacc);
}

static bool gpuinfo_pacc_init(void) {
  struct pacc_monitor_options options = {
      .abi_version = PACC_MONITOR_ABI_VERSION,
      .client = pacc_monitor_client_nvtop,
      .device_paths = NULL,
      .busy_window_ms = 1000U,
  };

  if (pacc_backend) {
    return pacc_monitor_device_count(pacc_backend) > 0;
  }

  return pacc_monitor_open(&pacc_backend, &options) == 0 && pacc_monitor_device_count(pacc_backend) > 0;
}

static void gpuinfo_pacc_shutdown(void) {
  free(pacc_infos);
  pacc_infos = NULL;
  pacc_monitor_close(pacc_backend);
  pacc_backend = NULL;
}

static const char *gpuinfo_pacc_last_error_string(void) {
  return "PACC shared-DDR monitor error";
}

static bool gpuinfo_pacc_get_device_handles(struct list_head *devices, unsigned *count) {
  unsigned device_count;

  if (!pacc_backend || !count) {
    return false;
  }

  *count = 0;
  device_count = pacc_monitor_device_count(pacc_backend);
  if (device_count == 0) {
    return false;
  }

  pacc_infos = calloc(device_count, sizeof(*pacc_infos));
  if (!pacc_infos) {
    return false;
  }

  for (unsigned i = 0; i < device_count; i++) {
    struct gpu_info_pacc *pacc_info = &pacc_infos[i];
    if (pacc_monitor_describe_device(pacc_backend, i, &pacc_info->desc) != 0) {
      continue;
    }
    pacc_info->monitor_index = i;
    pacc_info->base.vendor = &gpu_vendor_pacc;
    pacc_info->base.processes_count = 0;
    pacc_info->base.processes = NULL;
    pacc_info->base.processes_array_size = 0;
    snprintf(pacc_info->base.pdev, PDEV_LEN, "PACC%u", pacc_info->desc.pacc_id);
    list_add_tail(&pacc_info->base.list, devices);
    (*count)++;
  }

  return *count > 0;
}

static void gpuinfo_pacc_populate_static_info(struct gpu_info *_gpu_info) {
  struct gpu_info_pacc *pacc_info = container_of(_gpu_info, struct gpu_info_pacc, base);
  struct gpuinfo_static_info *static_info = &pacc_info->base.static_info;

  RESET_ALL(static_info->valid);
  static_info->integrated_graphics = false;
  static_info->encode_decode_shared = false;

  snprintf(static_info->device_name, sizeof(static_info->device_name), "PACC%u shared-DDR",
           pacc_info->desc.pacc_id);
  SET_VALID(gpuinfo_device_name_valid, static_info->valid);
  SET_GPUINFO_STATIC(static_info, n_exec_engines, 1);
  SET_GPUINFO_STATIC(static_info, engine_count, 1);
}

static void gpuinfo_pacc_refresh_dynamic_info(struct gpu_info *_gpu_info) {
  struct gpu_info_pacc *pacc_info = container_of(_gpu_info, struct gpu_info_pacc, base);
  struct gpuinfo_dynamic_info *dynamic_info = &pacc_info->base.dynamic_info;
  struct pacc_monitor_sample sample;

  RESET_ALL(dynamic_info->valid);
  if (!pacc_backend ||
      pacc_monitor_sample_device(pacc_backend, pacc_info->monitor_index, &sample) != 0) {
    return;
  }

  pacc_info->last_sample = sample;
  SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, sample.gpu_util_percent);
  SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, sample.mem_util_percent);
  if (sample.total_memory > 0) {
    SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, sample.total_memory);
    SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, sample.used_memory);
    SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, sample.free_memory);
  }
}

static void gpuinfo_pacc_get_running_processes(struct gpu_info *_gpu_info) {
  struct gpu_info_pacc *pacc_info = container_of(_gpu_info, struct gpu_info_pacc, base);
  struct pacc_monitor_process_sample samples[PACC_MONITOR_MAX_PIDS];
  size_t process_count = 0;

  _gpu_info->processes_count = 0;
  if (!pacc_backend ||
      pacc_monitor_sample_processes(pacc_backend, pacc_info->monitor_index, samples,
                                    sizeof(samples) / sizeof(samples[0]), &process_count) != 0 ||
      process_count == 0) {
    return;
  }

  if (_gpu_info->processes_array_size < process_count) {
    struct gpu_process *new_processes = realloc(_gpu_info->processes, process_count * sizeof(*_gpu_info->processes));
    if (!new_processes) {
      _gpu_info->processes_array_size = 0;
      free(_gpu_info->processes);
      _gpu_info->processes = NULL;
      return;
    }
    _gpu_info->processes = new_processes;
    _gpu_info->processes_array_size = process_count;
  }

  _gpu_info->processes_count = (unsigned)process_count;
  for (size_t i = 0; i < process_count; i++) {
    memset(&_gpu_info->processes[i], 0, sizeof(_gpu_info->processes[i]));
    _gpu_info->processes[i].type = gpu_process_compute;
    _gpu_info->processes[i].pid = samples[i].pid;
    SET_GPUINFO_PROCESS(&_gpu_info->processes[i], gpu_usage, samples[i].gpu_util_percent);
    SET_GPUINFO_PROCESS(&_gpu_info->processes[i], gpu_memory_usage, samples[i].gpu_memory_usage);
  }
}
