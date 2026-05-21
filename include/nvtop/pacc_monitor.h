/*
 *
 * PACC shared-DDR monitor ABI for nvtop and future PACC tools.
 *
 * This file is part of Nvtop.
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 */

#ifndef NVTOP_PACC_MONITOR_H__
#define NVTOP_PACC_MONITOR_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define PACC_MONITOR_ABI_VERSION 1U
#define PACC_MONITOR_MAX_DEVICES 4U
#define PACC_MONITOR_PATH_MAX 256U
#define PACC_MONITOR_MAX_PIDS 256U

enum pacc_monitor_client {
  pacc_monitor_client_nvtop = 0,
  pacc_monitor_client_nvprof = 1,
  pacc_monitor_client_cuda_gdb = 2,
};

enum pacc_monitor_job_id {
  pacc_monitor_job_kernel = 0,
  pacc_monitor_job_gemm = 1,
  pacc_monitor_job_softmax = 2,
  pacc_monitor_job_rmsnorm = 3,
  pacc_monitor_job_allreduce = 4,
  pacc_monitor_job_mmvf = 5,
  pacc_monitor_job_count = 6,
};

struct pacc_monitor_options {
  uint32_t abi_version;
  enum pacc_monitor_client client;
  const char *device_paths;
  uint32_t busy_window_ms;
};

struct pacc_monitor_device_desc {
  uint32_t abi_version;
  uint32_t pacc_id;
  char name[PACC_MONITOR_PATH_MAX];
  char mbox_path[PACC_MONITOR_PATH_MAX];
  uint64_t shared_ddr_base;
  uint64_t shared_ddr_size;
  uint64_t control_offset;
  bool control_mapped;
};

struct pacc_monitor_job_counter {
  uint64_t submitted_seq;
  uint64_t completed_seq;
  uint64_t submit_count;
  uint64_t complete_count;
};

struct pacc_monitor_sample {
  uint32_t abi_version;
  uint32_t pacc_id;
  uint64_t timestamp_ns;
  bool online;
  bool active;

  uint32_t gpu_util_percent;
  uint32_t mem_util_percent;
  uint64_t total_memory;
  uint64_t used_memory;
  uint64_t free_memory;

  uint32_t active_job_id;
  uint32_t submitted_job_id;
  uint64_t submitted_seq;
  uint32_t completed_job_id;
  uint32_t completion_status;
  uint64_t completed_seq;
  uint64_t runtime_seq;

  uint32_t beacon_phase;
  uint32_t beacon_detail;
  uint64_t beacon_seq;
  uint64_t diag_count;

  uint64_t job_submit_count;
  uint64_t job_complete_count;
  struct pacc_monitor_job_counter jobs[pacc_monitor_job_count];
};

struct pacc_monitor_process_sample {
  pid_t pid;
  uint32_t gpu_util_percent;
  uint64_t gpu_active_ns;
  uint64_t sample_window_ns;
  uint64_t gpu_memory_usage;
};

struct pacc_monitor_backend;

const char *pacc_monitor_job_name(uint32_t job_id);

int pacc_monitor_open(struct pacc_monitor_backend **backend, const struct pacc_monitor_options *options);
void pacc_monitor_close(struct pacc_monitor_backend *backend);
unsigned pacc_monitor_device_count(const struct pacc_monitor_backend *backend);

int pacc_monitor_describe_device(const struct pacc_monitor_backend *backend, unsigned index,
                                 struct pacc_monitor_device_desc *desc);
int pacc_monitor_sample_device(struct pacc_monitor_backend *backend, unsigned index,
                               struct pacc_monitor_sample *sample);

int pacc_monitor_read_control(struct pacc_monitor_backend *backend, unsigned index, uint64_t offset, void *dst,
                              size_t len);
int pacc_monitor_write_control(struct pacc_monitor_backend *backend, unsigned index, uint64_t offset, const void *src,
                               size_t len);
int pacc_monitor_scan_processes(const struct pacc_monitor_backend *backend, unsigned index, pid_t *pids,
                                size_t max_pids, size_t *count);
int pacc_monitor_sample_processes(struct pacc_monitor_backend *backend, unsigned index,
                                  struct pacc_monitor_process_sample *processes, size_t max_processes,
                                  size_t *count);

#endif // NVTOP_PACC_MONITOR_H__
