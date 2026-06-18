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

#include "nvtop/pacc_monitor.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define HETGPU_PACC_JOB_MAGIC UINT64_C(0x4847505550414343)
#define HETGPU_PACC_BEACON_MAGIC UINT64_C(0x4847505542434e31)
#define HETGPU_PACC_JOB_VERSION 1U
#define PACC_JOB_MAGIC UINT64_C(0x504143434a4f4231)
#define HETGPU_PACC_RUNTIME_TABLE_MAGIC UINT64_C(0x4847505554424c31)
#define HETGPU_PACC_RUNTIME_TABLE_VERSION 1U
#define HETGPU_PACC_COUNT 4U
#define HETGPU_PACC_MAX_JOB_ID 16U
#define HETGPU_PACC_ARG_SLOT_BYTES UINT64_C(0x400)
#define HETGPU_PACC_CONTROL_BYTES UINT64_C(0x2000)
#define HETGPU_PACC_ARG_BASE_OFF UINT64_C(0x100)
#define HETGPU_PACC_RUNTIME_TABLE_OFF UINT64_C(0x1400)
#define HETGPU_PACC_SHARED_DDR_USER_OFF UINT64_C(0x00100000)
#define HETGPU_PACC_COMPLETION_OFF UINT64_C(0x1f20)
#define HETGPU_PACC_BEACON_OFF UINT64_C(0x1f40)
#define HETGPU_PACC_DIAG_RING_SLOT 8U
#define HETGPU_PACC_DIAG_RING_RECORDS 192U
#define HETGPU_PACC_DIAG_MAGIC UINT64_C(0x4847505544494147)
#define HETGPU_PACC_DEFAULT_SHARED_DDR_BYTES UINT64_C(0x100000000)
#define PACC_MONITOR_DEFAULT_SAMPLE_US UINT64_C(20000)
#define PACC_MONITOR_MAX_TRACKED_PROCESSES 256U
#define PACC_MONITOR_ALLOC_STATS_DEFAULT_DIR "/tmp/hetgpu_pacc_allocs"

#define PACC_IOC_MAGIC 'p'
#define PACC_IOC_ZLUDA_GET_DDR_BASE _IOR(PACC_IOC_MAGIC, 6, struct pacc_zluda_ddr_info)
#define PACC_IOC_GET_PACC_ID _IOR(PACC_IOC_MAGIC, 7, unsigned long)

struct pacc_zluda_ddr_info {
  uint64_t ddr_base;
  uint64_t ddr_size;
};

struct pacc_monitor_doorbell {
  uint64_t magic;
  uint32_t version;
  uint32_t job_id;
  uint32_t flags;
  uint32_t status;
  uint64_t seq;
};

struct pacc_monitor_host_status {
  uint64_t magic;
  uint32_t version;
  uint32_t job_id;
  uint32_t status;
  uint64_t seq;
};

struct pacc_monitor_jobd_beacon {
  uint64_t magic;
  uint32_t version;
  uint32_t job_id;
  uint32_t phase;
  uint32_t detail;
  uint64_t seq;
};

struct pacc_monitor_diag_event {
  uint64_t magic;
  uint32_t index;
  uint32_t status;
  uint32_t job_id;
  uint32_t aux;
  uint64_t seq;
};

struct pacc_monitor_arg_slot_header {
  uint64_t magic;
  uint32_t version;
  uint32_t job_id;
  uint64_t seq;
  uint64_t arg_len;
};

struct pacc_monitor_job_desc {
  uint64_t addr;
  uint64_t len;
  uint64_t seq;
  uint64_t buf_info;
};

struct pacc_monitor_runtime_table_prefix {
  uint64_t magic;
  uint32_t version;
  uint32_t flags;
  uint64_t seq;
  uint32_t have_gemm;
  uint32_t have_softmax;
  uint32_t have_rmsnorm;
  uint32_t have_allreduce;
  uint32_t have_mmvf;
  uint32_t reserved0;
};

struct pacc_monitor_process_accum {
  pid_t pid;
  uint64_t active_ns;
  uint64_t last_seen_ns;
  uint64_t gpu_memory_usage;
};

struct pacc_monitor_pid_stats {
  bool have_memory_usage;
  bool have_memory_bandwidth;
  uint64_t gpu_memory_usage;
  uint64_t memory_read_bytes;
  uint64_t memory_write_bytes;
};

struct pacc_monitor_device {
  int fd;
  uint32_t pacc_id;
  char path[PACC_MONITOR_PATH_MAX];
  uint64_t shared_ddr_base;
  uint64_t shared_ddr_size;
  uint64_t fd_user_off;
  bool control_mapped;
  void *control_map;
  size_t control_map_len;
  uint8_t *control_window;
  uint64_t control_mmap_offset;

  bool have_last;
  uint64_t last_activity_ns;
  uint64_t last_submitted_seq;
  uint64_t last_completed_seq;
  uint64_t last_beacon_seq;
  uint64_t last_diag_count;
  uint64_t job_submit_count;
  uint64_t job_complete_count;
  struct pacc_monitor_job_counter jobs[pacc_monitor_job_count];

  bool sampler_have_last;
  uint64_t sampler_last_ns;
  uint64_t sampler_last_submitted_seq;
  uint64_t sampler_last_completed_seq;
  uint64_t sampler_last_beacon_seq;
  uint64_t sample_window_elapsed_ns;
  uint64_t sample_window_active_ns;
  uint64_t process_window_elapsed_ns;
  bool have_memory_bandwidth_last;
  uint64_t last_memory_bandwidth_ns;
  uint64_t last_memory_read_bytes;
  uint64_t last_memory_write_bytes;
  size_t process_count;
  struct pacc_monitor_process_accum processes[PACC_MONITOR_MAX_TRACKED_PROCESSES];
};

struct pacc_monitor_backend {
  enum pacc_monitor_client client;
  uint32_t busy_window_ms;
  unsigned count;
  struct pacc_monitor_device devices[PACC_MONITOR_MAX_DEVICES];
  pthread_mutex_t lock;
  bool lock_initialized;
  pthread_t sampler_thread;
  bool sampler_running;
  bool stop_sampler;
  uint64_t sample_interval_us;
};

static uint64_t read_diag_count(struct pacc_monitor_device *dev);
static int scan_processes_for_device(const struct pacc_monitor_backend *backend, const struct pacc_monitor_device *dev,
                                     pid_t *pids, size_t max_pids, size_t *count);
static void read_device_snapshot(struct pacc_monitor_device *dev, struct pacc_monitor_sample *sample,
                                 bool include_diag);
static void start_sampler(struct pacc_monitor_backend *backend);
static void stop_sampler(struct pacc_monitor_backend *backend);

static uint64_t monotonic_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void sleep_us(uint64_t usec) {
  struct timespec ts;
  ts.tv_sec = (time_t)(usec / UINT64_C(1000000));
  ts.tv_nsec = (long)((usec % UINT64_C(1000000)) * UINT64_C(1000));
  while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
  }
}

static bool parse_u64_checked(const char *s, uint64_t *out) {
  char *end = NULL;
  unsigned long long value;

  if (!s || !*s || !out) {
    return false;
  }

  errno = 0;
  value = strtoull(s, &end, 0);
  if (errno || end == s) {
    return false;
  }
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
    end++;
  }
  if (*end) {
    return false;
  }
  *out = (uint64_t)value;
  return true;
}

static uint64_t env_u64_default(const char *name, uint64_t fallback) {
  uint64_t value;
  return parse_u64_checked(getenv(name), &value) ? value : fallback;
}

static bool read_file_to_buffer(const char *path, char *buf, size_t buf_size) {
  int fd;
  ssize_t n;

  if (!path || !buf || buf_size == 0) {
    return false;
  }

  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  n = read(fd, buf, buf_size - 1);
  close(fd);
  if (n <= 0) {
    return false;
  }
  buf[n] = '\0';
  return true;
}

static bool read_u64_file(const char *path, uint64_t *out) {
  char buf[64];

  if (!read_file_to_buffer(path, buf, sizeof(buf))) {
    return false;
  }
  return parse_u64_checked(buf, out);
}

static bool parse_key_value_u64(const char *buf, const char *key, uint64_t *out) {
  size_t key_len;
  const char *line;

  if (!buf || !key || !out) {
    return false;
  }

  key_len = strlen(key);
  line = buf;
  while (*line) {
    const char *line_end = strchr(line, '\n');
    size_t line_len = line_end ? (size_t)(line_end - line) : strlen(line);
    if (line_len > key_len && !strncmp(line, key, key_len) && line[key_len] == '=') {
      char value[64];
      size_t value_len = line_len - key_len - 1;
      if (value_len >= sizeof(value)) {
        return false;
      }
      memcpy(value, line + key_len + 1, value_len);
      value[value_len] = '\0';
      return parse_u64_checked(value, out);
    }
    if (!line_end) {
      break;
    }
    line = line_end + 1;
  }
  return false;
}

static void stats_saturating_add(uint64_t *total, uint64_t value) {
  if (!total) {
    return;
  }
  if (UINT64_MAX - *total < value) {
    *total = UINT64_MAX;
    return;
  }
  *total += value;
}

static bool read_pid_alloc_stats(pid_t pid, struct pacc_monitor_pid_stats *stats) {
  char path[PATH_MAX];
  char bw_path[PATH_MAX];
  char buf[1024];
  const char *dir;
  int len;
  uint64_t value;
  uint64_t shared_ddr = 0;
  uint64_t shared_ddr_ipc = 0;
  uint64_t driver = 0;
  bool have_device_memory_key = false;
  bool have_any = false;

  if (pid <= 0 || !stats) {
    return false;
  }

  memset(stats, 0, sizeof(*stats));
  dir = getenv("HETGPU_PACC_ALLOC_STATS_DIR");
  if (!dir || !*dir) {
    dir = PACC_MONITOR_ALLOC_STATS_DEFAULT_DIR;
  }
  len = snprintf(path, sizeof(path), "%s/%ld", dir, (long)pid);
  if (len < 0 || (size_t)len >= sizeof(path)) {
    return false;
  }

  if (read_file_to_buffer(path, buf, sizeof(buf))) {
    have_any = true;
    if (parse_key_value_u64(buf, "device_memory", &value)) {
      stats->gpu_memory_usage = value;
      stats->have_memory_usage = true;
    } else {
      have_device_memory_key = parse_key_value_u64(buf, "shared_ddr", &shared_ddr);
      have_device_memory_key = parse_key_value_u64(buf, "shared_ddr_ipc", &shared_ddr_ipc) || have_device_memory_key;
      have_device_memory_key = parse_key_value_u64(buf, "driver", &driver) || have_device_memory_key;
      if (have_device_memory_key) {
        stats->gpu_memory_usage = shared_ddr;
        stats_saturating_add(&stats->gpu_memory_usage, shared_ddr_ipc);
        stats_saturating_add(&stats->gpu_memory_usage, driver);
        stats->have_memory_usage = true;
      } else if (parse_key_value_u64(buf, "total", &value) || parse_u64_checked(buf, &value)) {
        stats->gpu_memory_usage = value;
        stats->have_memory_usage = true;
      }
    }
    if (parse_key_value_u64(buf, "memory_read_bytes", &value)) {
      stats->memory_read_bytes = value;
      stats->have_memory_bandwidth = true;
    }
    if (parse_key_value_u64(buf, "memory_write_bytes", &value)) {
      stats->memory_write_bytes = value;
      stats->have_memory_bandwidth = true;
    }
  }

  len = snprintf(bw_path, sizeof(bw_path), "%s/%ld.bw", dir, (long)pid);
  if (len >= 0 && (size_t)len < sizeof(bw_path) && read_file_to_buffer(bw_path, buf, sizeof(buf))) {
    have_any = true;
    if (parse_key_value_u64(buf, "memory_read_bytes", &value)) {
      stats->memory_read_bytes = value;
      stats->have_memory_bandwidth = true;
    }
    if (parse_key_value_u64(buf, "memory_write_bytes", &value)) {
      stats->memory_write_bytes = value;
      stats->have_memory_bandwidth = true;
    }
  }

  return have_any;
}

static bool sum_pid_alloc_stats(const pid_t *pids, size_t count, uint64_t *memory_usage,
                                uint64_t *memory_read_bytes, uint64_t *memory_write_bytes,
                                bool *memory_usage_valid, bool *memory_bandwidth_valid) {
  bool have_any = false;

  if (memory_usage) {
    *memory_usage = 0;
  }
  if (memory_read_bytes) {
    *memory_read_bytes = 0;
  }
  if (memory_write_bytes) {
    *memory_write_bytes = 0;
  }
  if (memory_usage_valid) {
    *memory_usage_valid = false;
  }
  if (memory_bandwidth_valid) {
    *memory_bandwidth_valid = false;
  }
  if (!pids) {
    return false;
  }

  for (size_t i = 0; i < count; i++) {
    struct pacc_monitor_pid_stats stats;
    if (!read_pid_alloc_stats(pids[i], &stats)) {
      continue;
    }
    if (stats.have_memory_usage && memory_usage) {
      stats_saturating_add(memory_usage, stats.gpu_memory_usage);
      if (memory_usage_valid) {
        *memory_usage_valid = true;
      }
      have_any = true;
    }
    if (stats.have_memory_bandwidth) {
      if (memory_read_bytes) {
        stats_saturating_add(memory_read_bytes, stats.memory_read_bytes);
      }
      if (memory_write_bytes) {
        stats_saturating_add(memory_write_bytes, stats.memory_write_bytes);
      }
      if (memory_bandwidth_valid) {
        *memory_bandwidth_valid = true;
      }
      have_any = true;
    }
  }
  return have_any;
}

static uint32_t pacc_kib_per_second(uint64_t byte_delta, uint64_t elapsed_ns) {
  long double rate;

  if (byte_delta == 0 || elapsed_ns == 0) {
    return 0;
  }
  rate = ((long double)byte_delta * 1000000000.0L) / ((long double)elapsed_ns * 1024.0L);
  if (rate >= (long double)UINT32_MAX) {
    return UINT32_MAX;
  }
  return (uint32_t)(rate + 0.5L);
}

static void update_memory_bandwidth(struct pacc_monitor_device *dev, struct pacc_monitor_sample *sample,
                                    uint64_t read_bytes, uint64_t write_bytes) {
  uint64_t elapsed_ns;

  if (!dev || !sample) {
    return;
  }

  sample->memory_bandwidth_valid = true;
  sample->memory_read_bytes = read_bytes;
  sample->memory_write_bytes = write_bytes;
  if (dev->have_memory_bandwidth_last && sample->timestamp_ns > dev->last_memory_bandwidth_ns &&
      read_bytes >= dev->last_memory_read_bytes && write_bytes >= dev->last_memory_write_bytes) {
    elapsed_ns = sample->timestamp_ns - dev->last_memory_bandwidth_ns;
    sample->memory_read_kb_s = pacc_kib_per_second(read_bytes - dev->last_memory_read_bytes, elapsed_ns);
    sample->memory_write_kb_s = pacc_kib_per_second(write_bytes - dev->last_memory_write_bytes, elapsed_ns);
  } else {
    sample->memory_read_kb_s = 0;
    sample->memory_write_kb_s = 0;
  }
  dev->have_memory_bandwidth_last = true;
  dev->last_memory_bandwidth_ns = sample->timestamp_ns;
  dev->last_memory_read_bytes = read_bytes;
  dev->last_memory_write_bytes = write_bytes;
}

static void set_sample_memory_usage(struct pacc_monitor_sample *sample, uint64_t used_memory) {
  uint64_t total_memory;
  uint64_t util;

  if (!sample) {
    return;
  }

  total_memory = sample->total_memory;
  if (total_memory > 0 && used_memory > total_memory) {
    used_memory = total_memory;
  }
  sample->used_memory = used_memory;
  sample->free_memory = total_memory > used_memory ? total_memory - used_memory : 0;
  if (total_memory == 0) {
    sample->mem_util_percent = 0;
    return;
  }
  util = (used_memory * UINT64_C(100) + total_memory / 2) / total_memory;
  sample->mem_util_percent = util > 100 ? 100 : (uint32_t)util;
}

static bool read_shared_ddr_info_from_env_or_debugfs(struct pacc_zluda_ddr_info *info) {
  uint64_t base = 0;
  uint64_t size = 0;

  if (!info) {
    return false;
  }

  parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_BASE"), &base);
  parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_BYTES"), &size);
  if (!size) {
    parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_SIZE"), &size);
  }
  if (!base) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_base", &base);
  }
  if (!base) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_base", &base);
  }
  if (!base) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_base", &base);
  }
  if (!base) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base", &base);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_bytes", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr_coh/shared_ddr_size", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_bytes", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_ddr/shared_ddr_size", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_bytes", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox_full/shared_ddr_size", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_bytes", &size);
  }
  if (!size) {
    read_u64_file("/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_size", &size);
  }
  if (!base) {
    base = UINT64_C(0x20110600000);
  }
  if (!size && base) {
    size = HETGPU_PACC_DEFAULT_SHARED_DDR_BYTES;
  }
  if (!base || size < HETGPU_PACC_CONTROL_BYTES) {
    return false;
  }

  info->ddr_base = base;
  info->ddr_size = size;
  return true;
}

static void add_u64_candidate(uint64_t *values, size_t *count, size_t max_count, uint64_t value) {
  if (!values || !count || *count >= max_count) {
    return;
  }
  for (size_t i = 0; i < *count; i++) {
    if (values[i] == value) {
      return;
    }
  }
  values[(*count)++] = value;
}

static bool pread_exact(int fd, uint64_t off, void *dst, size_t len) {
  uint8_t *buf = dst;
  size_t done = 0;

  if (fd < 0 || !dst) {
    return false;
  }

  while (done < len) {
    ssize_t got = pread(fd, buf + done, len - done, (off_t)(off + done));
    if (got <= 0) {
      return false;
    }
    done += (size_t)got;
  }
  __sync_synchronize();
  return true;
}

static bool pwrite_exact(int fd, uint64_t off, const void *src, size_t len) {
  const uint8_t *buf = src;
  size_t done = 0;

  if (fd < 0 || !src) {
    return false;
  }

  __sync_synchronize();
  while (done < len) {
    ssize_t put = pwrite(fd, buf + done, len - done, (off_t)(off + done));
    if (put <= 0) {
      return false;
    }
    done += (size_t)put;
  }
  __sync_synchronize();
  return true;
}

static bool read_ddr_relative(struct pacc_monitor_device *dev, uint64_t rel, void *dst, size_t len) {
  uint64_t candidates[4];
  size_t candidate_count = 0;

  if (!dev || !dst || len == 0) {
    return false;
  }
  if (dev->shared_ddr_size && (rel > dev->shared_ddr_size || (uint64_t)len > dev->shared_ddr_size - rel)) {
    return false;
  }

  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]), rel);
  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]), dev->fd_user_off + rel);
  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]),
                    HETGPU_PACC_SHARED_DDR_USER_OFF + rel);

  for (size_t i = 0; i < candidate_count; i++) {
    memset(dst, 0, len);
    if (pread_exact(dev->fd, candidates[i], dst, len)) {
      return true;
    }
  }
  return false;
}

static bool write_ddr_relative(struct pacc_monitor_device *dev, uint64_t rel, const void *src, size_t len) {
  uint64_t candidates[4];
  size_t candidate_count = 0;

  if (!dev || !src || len == 0) {
    return false;
  }
  if (dev->shared_ddr_size && (rel > dev->shared_ddr_size || (uint64_t)len > dev->shared_ddr_size - rel)) {
    return false;
  }

  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]), rel);
  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]), dev->fd_user_off + rel);
  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]),
                    HETGPU_PACC_SHARED_DDR_USER_OFF + rel);

  for (size_t i = 0; i < candidate_count; i++) {
    if (pwrite_exact(dev->fd, candidates[i], src, len)) {
      return true;
    }
  }
  return false;
}

static bool read_control(struct pacc_monitor_device *dev, uint64_t off, void *dst, size_t len) {
  uint64_t rel;

  if (!dev || !dst || len == 0 || off > HETGPU_PACC_CONTROL_BYTES ||
      (uint64_t)len > HETGPU_PACC_CONTROL_BYTES - off) {
    return false;
  }

  if (dev->control_mapped && dev->control_window) {
    __sync_synchronize();
    memcpy(dst, dev->control_window + off, len);
    __sync_synchronize();
    return true;
  }

  rel = (uint64_t)dev->pacc_id * HETGPU_PACC_CONTROL_BYTES + off;
  return read_ddr_relative(dev, rel, dst, len);
}

static bool write_control(struct pacc_monitor_device *dev, uint64_t off, const void *src, size_t len) {
  uint64_t rel;

  if (!dev || !src || len == 0 || off > HETGPU_PACC_CONTROL_BYTES ||
      (uint64_t)len > HETGPU_PACC_CONTROL_BYTES - off) {
    return false;
  }

  if (dev->control_mapped && dev->control_window) {
    memcpy(dev->control_window + off, src, len);
    __sync_synchronize();
    if (dev->control_map && dev->control_map_len) {
      (void)msync(dev->control_map, dev->control_map_len, MS_SYNC);
    }
    return true;
  }

  rel = (uint64_t)dev->pacc_id * HETGPU_PACC_CONTROL_BYTES + off;
  return write_ddr_relative(dev, rel, src, len);
}

static bool map_control_window(struct pacc_monitor_device *dev) {
  uint64_t configured_off;
  uint64_t candidates[4];
  uint64_t rel;
  size_t candidate_count = 0;

  if (!dev || dev->fd < 0 || dev->pacc_id >= HETGPU_PACC_COUNT) {
    return false;
  }

  rel = (uint64_t)dev->pacc_id * HETGPU_PACC_CONTROL_BYTES;
  if (dev->shared_ddr_size &&
      (rel > dev->shared_ddr_size || HETGPU_PACC_CONTROL_BYTES > dev->shared_ddr_size - rel)) {
    return false;
  }

  if (parse_u64_checked(getenv("HETGPU_PACC_SHARED_DDR_MMAP_USER_OFF"), &configured_off)) {
    add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]), configured_off);
  }
  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]),
                    HETGPU_PACC_SHARED_DDR_USER_OFF);
  add_u64_candidate(candidates, &candidate_count, sizeof(candidates) / sizeof(candidates[0]), 0);

  for (size_t i = 0; i < candidate_count; i++) {
    void *map = mmap(NULL, (size_t)HETGPU_PACC_CONTROL_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                     (off_t)(candidates[i] + rel));
    if (map != MAP_FAILED) {
      dev->control_mapped = true;
      dev->control_map = map;
      dev->control_map_len = (size_t)HETGPU_PACC_CONTROL_BYTES;
      dev->control_window = map;
      dev->control_mmap_offset = candidates[i] + rel;
      return true;
    }
  }

  return false;
}

static bool path_exists(const char *path) {
  return path && access(path, R_OK | W_OK) == 0;
}

static bool path_already_added(const struct pacc_monitor_backend *backend, const char *path) {
  for (unsigned i = 0; i < backend->count; i++) {
    if (!strcmp(backend->devices[i].path, path)) {
      return true;
    }
  }
  return false;
}

static bool parse_path_suffix_id(const char *path, uint32_t *pacc_id) {
  const char *p;
  unsigned long value;

  if (!path || !pacc_id) {
    return false;
  }

  p = path + strlen(path);
  while (p > path && isdigit((unsigned char)p[-1])) {
    p--;
  }
  if (!*p) {
    return false;
  }
  value = strtoul(p, NULL, 10);
  if (value >= HETGPU_PACC_COUNT) {
    return false;
  }
  *pacc_id = (uint32_t)value;
  return true;
}

static bool init_device(struct pacc_monitor_device *dev, const char *path) {
  struct pacc_zluda_ddr_info info;
  unsigned long ioctl_pacc_id = 0;
  int fd;

  if (!dev || !path) {
    return false;
  }

  fd = open(path, O_RDWR | O_SYNC | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }

  memset(dev, 0, sizeof(*dev));
  dev->fd = fd;
  dev->fd_user_off = env_u64_default("HETGPU_PACC_SHARED_DDR_FD_USER_OFF",
                                     env_u64_default("HETGPU_PACC_SHARED_DDR_USER_OFF",
                                                     HETGPU_PACC_SHARED_DDR_USER_OFF));
  snprintf(dev->path, sizeof(dev->path), "%s", path);

  if (ioctl(fd, PACC_IOC_GET_PACC_ID, &ioctl_pacc_id) == 0 && ioctl_pacc_id < HETGPU_PACC_COUNT) {
    dev->pacc_id = (uint32_t)ioctl_pacc_id;
  } else if (!parse_path_suffix_id(path, &dev->pacc_id)) {
    close(fd);
    dev->fd = -1;
    return false;
  }

  memset(&info, 0, sizeof(info));
  if (ioctl(fd, PACC_IOC_ZLUDA_GET_DDR_BASE, &info) != 0 || !info.ddr_base ||
      info.ddr_size < HETGPU_PACC_CONTROL_BYTES) {
    memset(&info, 0, sizeof(info));
    (void)read_shared_ddr_info_from_env_or_debugfs(&info);
  }
  dev->shared_ddr_base = info.ddr_base;
  dev->shared_ddr_size = info.ddr_size;

  (void)map_control_window(dev);
  return true;
}

static void close_device(struct pacc_monitor_device *dev) {
  if (!dev) {
    return;
  }
  if (dev->control_mapped && dev->control_map && dev->control_map_len) {
    munmap(dev->control_map, dev->control_map_len);
  }
  if (dev->fd >= 0) {
    close(dev->fd);
  }
  memset(dev, 0, sizeof(*dev));
  dev->fd = -1;
}

static void add_device_path(struct pacc_monitor_backend *backend, const char *path) {
  struct pacc_monitor_device candidate;

  if (!backend || !path || backend->count >= PACC_MONITOR_MAX_DEVICES || path_already_added(backend, path)) {
    return;
  }
  if (!path_exists(path)) {
    return;
  }
  memset(&candidate, 0, sizeof(candidate));
  candidate.fd = -1;
  if (!init_device(&candidate, path)) {
    return;
  }
  backend->devices[backend->count++] = candidate;
}

static void add_configured_paths(struct pacc_monitor_backend *backend, const char *paths) {
  char *copy;
  char *saveptr = NULL;
  char *token;

  if (!paths || !*paths) {
    return;
  }

  copy = strdup(paths);
  if (!copy) {
    return;
  }

  for (token = strtok_r(copy, ",:", &saveptr); token; token = strtok_r(NULL, ",:", &saveptr)) {
    while (*token == ' ' || *token == '\t') {
      token++;
    }
    if (*token) {
      add_device_path(backend, token);
    }
  }
  free(copy);
}

static void discover_default_devices(struct pacc_monitor_backend *backend) {
  char path[PACC_MONITOR_PATH_MAX];

  for (unsigned i = 0; i < HETGPU_PACC_COUNT; i++) {
    unsigned before = backend->count;

    snprintf(path, sizeof(path), "/dev/hetgpu_pacc_mbox_ddr_coh%u", i);
    add_device_path(backend, path);
    if (backend->count != before) {
      continue;
    }
    snprintf(path, sizeof(path), "/dev/hetgpu_pacc_mbox%u", i);
    add_device_path(backend, path);
    if (backend->count != before) {
      continue;
    }
    snprintf(path, sizeof(path), "/dev/pacc%u", i);
    add_device_path(backend, path);
  }
}

const char *pacc_monitor_job_name(uint32_t job_id) {
  switch (job_id) {
  case pacc_monitor_job_kernel:
    return "KERNEL_ELF";
  case pacc_monitor_job_gemm:
    return "GEMM";
  case pacc_monitor_job_softmax:
    return "SOFTMAX";
  case pacc_monitor_job_rmsnorm:
    return "RMSNORM";
  case pacc_monitor_job_allreduce:
    return "ALLREDUCE";
  case pacc_monitor_job_mmvf:
    return "MMVF";
  default:
    return "UNKNOWN";
  }
}

int pacc_monitor_open(struct pacc_monitor_backend **backend, const struct pacc_monitor_options *options) {
  struct pacc_monitor_backend *created;
  const char *configured_paths = NULL;

  if (!backend) {
    return -EINVAL;
  }
  *backend = NULL;

  created = calloc(1, sizeof(*created));
  if (!created) {
    return -ENOMEM;
  }
  for (unsigned i = 0; i < PACC_MONITOR_MAX_DEVICES; i++) {
    created->devices[i].fd = -1;
  }
  if (pthread_mutex_init(&created->lock, NULL) == 0) {
    created->lock_initialized = true;
  }

  created->client = options ? options->client : pacc_monitor_client_nvtop;
  created->busy_window_ms = options && options->busy_window_ms ? options->busy_window_ms : 1000U;

  configured_paths = options && options->device_paths ? options->device_paths : getenv("PACC_MONITOR_DEVICES");
  if (!configured_paths || !*configured_paths) {
    configured_paths = getenv("PACC_NVTOP_DEVICES");
  }

  add_configured_paths(created, configured_paths);
  if (created->count == 0) {
    discover_default_devices(created);
  }

  if (created->count == 0) {
    if (created->lock_initialized) {
      pthread_mutex_destroy(&created->lock);
    }
    free(created);
    return -ENODEV;
  }

  start_sampler(created);
  *backend = created;
  return 0;
}

void pacc_monitor_close(struct pacc_monitor_backend *backend) {
  if (!backend) {
    return;
  }
  stop_sampler(backend);
  for (unsigned i = 0; i < backend->count; i++) {
    close_device(&backend->devices[i]);
  }
  if (backend->lock_initialized) {
    pthread_mutex_destroy(&backend->lock);
  }
  free(backend);
}

unsigned pacc_monitor_device_count(const struct pacc_monitor_backend *backend) {
  return backend ? backend->count : 0;
}

int pacc_monitor_describe_device(const struct pacc_monitor_backend *backend, unsigned index,
                                 struct pacc_monitor_device_desc *desc) {
  const struct pacc_monitor_device *dev;

  if (!backend || !desc || index >= backend->count) {
    return -EINVAL;
  }

  dev = &backend->devices[index];
  memset(desc, 0, sizeof(*desc));
  desc->abi_version = PACC_MONITOR_ABI_VERSION;
  desc->pacc_id = dev->pacc_id;
  snprintf(desc->name, sizeof(desc->name), "PACC%u", dev->pacc_id);
  snprintf(desc->mbox_path, sizeof(desc->mbox_path), "%s", dev->path);
  desc->shared_ddr_base = dev->shared_ddr_base;
  desc->shared_ddr_size = dev->shared_ddr_size;
  desc->control_offset = (uint64_t)dev->pacc_id * HETGPU_PACC_CONTROL_BYTES;
  desc->control_mapped = dev->control_mapped;
  return 0;
}

static void update_submit(struct pacc_monitor_sample *sample, uint32_t job_id, uint64_t seq) {
  if (!sample || seq == 0) {
    return;
  }
  if (seq >= sample->submitted_seq) {
    sample->submitted_seq = seq;
    sample->submitted_job_id = job_id;
  }
}

static void read_device_snapshot(struct pacc_monitor_device *dev, struct pacc_monitor_sample *sample,
                                 bool include_diag) {
  struct pacc_monitor_doorbell doorbell;
  struct pacc_monitor_job_desc kernel_desc;
  struct pacc_monitor_host_status completion;
  struct pacc_monitor_jobd_beacon beacon;
  struct pacc_monitor_runtime_table_prefix runtime;
  uint64_t memory_total;

  memset(sample, 0, sizeof(*sample));
  sample->abi_version = PACC_MONITOR_ABI_VERSION;
  sample->pacc_id = dev->pacc_id;
  sample->timestamp_ns = monotonic_ns();
  sample->online = dev->fd >= 0;

  memset(&doorbell, 0, sizeof(doorbell));
  if (read_control(dev, 0, &doorbell, sizeof(doorbell)) && doorbell.magic == HETGPU_PACC_JOB_MAGIC &&
      doorbell.version == HETGPU_PACC_JOB_VERSION) {
    update_submit(sample, doorbell.job_id, doorbell.seq);
  }

  memset(&kernel_desc, 0, sizeof(kernel_desc));
  if (read_control(dev, 0, &kernel_desc, sizeof(kernel_desc)) && kernel_desc.buf_info == PACC_JOB_MAGIC) {
    update_submit(sample, pacc_monitor_job_kernel, kernel_desc.seq);
  }

  for (uint32_t slot = 0; slot < HETGPU_PACC_MAX_JOB_ID; slot++) {
    struct pacc_monitor_arg_slot_header header;
    uint64_t off = HETGPU_PACC_ARG_BASE_OFF + (uint64_t)slot * HETGPU_PACC_ARG_SLOT_BYTES;
    memset(&header, 0, sizeof(header));
    if (off + sizeof(header) <= HETGPU_PACC_CONTROL_BYTES &&
        read_control(dev, off, &header, sizeof(header)) && header.magic == HETGPU_PACC_JOB_MAGIC &&
        header.version == HETGPU_PACC_JOB_VERSION) {
      update_submit(sample, header.job_id, header.seq);
    }
  }

  memset(&runtime, 0, sizeof(runtime));
  if (read_control(dev, HETGPU_PACC_RUNTIME_TABLE_OFF, &runtime, sizeof(runtime)) &&
      runtime.magic == HETGPU_PACC_RUNTIME_TABLE_MAGIC && runtime.version == HETGPU_PACC_RUNTIME_TABLE_VERSION) {
    sample->runtime_seq = runtime.seq;
  }

  memset(&completion, 0, sizeof(completion));
  if (read_control(dev, HETGPU_PACC_COMPLETION_OFF, &completion, sizeof(completion)) &&
      completion.magic == HETGPU_PACC_JOB_MAGIC && completion.version == HETGPU_PACC_JOB_VERSION) {
    sample->completed_job_id = completion.job_id;
    sample->completion_status = completion.status;
    sample->completed_seq = completion.seq;
  }

  memset(&beacon, 0, sizeof(beacon));
  if (read_control(dev, HETGPU_PACC_BEACON_OFF, &beacon, sizeof(beacon)) &&
      beacon.magic == HETGPU_PACC_BEACON_MAGIC && beacon.version == HETGPU_PACC_JOB_VERSION) {
    sample->beacon_phase = beacon.phase;
    sample->beacon_detail = beacon.detail;
    sample->beacon_seq = beacon.seq;
  }

  sample->diag_count = include_diag ? read_diag_count(dev) : dev->last_diag_count;
  sample->active = sample->submitted_seq != 0 && sample->submitted_seq != sample->completed_seq;
  sample->active_job_id = sample->active ? sample->submitted_job_id : sample->completed_job_id;

  memory_total = dev->shared_ddr_size > HETGPU_PACC_SHARED_DDR_USER_OFF
                     ? dev->shared_ddr_size - HETGPU_PACC_SHARED_DDR_USER_OFF
                     : dev->shared_ddr_size;
  sample->total_memory = memory_total;
  sample->used_memory = 0;
  sample->free_memory = memory_total;
  sample->mem_util_percent = 0;
}

static struct pacc_monitor_process_accum *get_process_accum(struct pacc_monitor_device *dev, pid_t pid,
                                                            uint64_t now_ns) {
  size_t oldest = 0;

  for (size_t i = 0; i < dev->process_count; i++) {
    if (dev->processes[i].pid == pid) {
      return &dev->processes[i];
    }
    if (dev->processes[i].last_seen_ns < dev->processes[oldest].last_seen_ns) {
      oldest = i;
    }
  }

  if (dev->process_count < PACC_MONITOR_MAX_TRACKED_PROCESSES) {
    struct pacc_monitor_process_accum *entry = &dev->processes[dev->process_count++];
    memset(entry, 0, sizeof(*entry));
    entry->pid = pid;
    entry->last_seen_ns = now_ns;
    return entry;
  }

  memset(&dev->processes[oldest], 0, sizeof(dev->processes[oldest]));
  dev->processes[oldest].pid = pid;
  dev->processes[oldest].last_seen_ns = now_ns;
  return &dev->processes[oldest];
}

static void attribute_active_time(struct pacc_monitor_device *dev, const pid_t *pids, size_t pid_count,
                                  uint64_t active_ns, uint64_t now_ns) {
  if (!dev || !pids || pid_count == 0 || active_ns == 0) {
    return;
  }

  uint64_t share_ns = active_ns / pid_count;
  uint64_t remainder_ns = active_ns % pid_count;
  for (size_t i = 0; i < pid_count; i++) {
    struct pacc_monitor_process_accum *entry = get_process_accum(dev, pids[i], now_ns);
    entry->active_ns += share_ns + (i == 0 ? remainder_ns : 0);
    entry->last_seen_ns = now_ns;
  }
}

static void *sampler_main(void *opaque) {
  struct pacc_monitor_backend *backend = opaque;

  while (!backend->stop_sampler) {
    uint64_t now_ns = monotonic_ns();

    for (unsigned i = 0; i < backend->count; i++) {
      struct pacc_monitor_sample sample;
      pid_t pids[PACC_MONITOR_MAX_PIDS];
      size_t pid_count = 0;
      struct pacc_monitor_device *dev = &backend->devices[i];
      bool active_or_changed;

      read_device_snapshot(dev, &sample, false);
      active_or_changed =
          sample.active ||
          (dev->sampler_have_last &&
           (sample.submitted_seq != dev->sampler_last_submitted_seq ||
            sample.completed_seq != dev->sampler_last_completed_seq ||
            sample.beacon_seq != dev->sampler_last_beacon_seq));
      if (active_or_changed) {
        (void)scan_processes_for_device(backend, dev, pids, sizeof(pids) / sizeof(pids[0]), &pid_count);
      }

      pthread_mutex_lock(&backend->lock);
      if (dev->sampler_have_last && now_ns > dev->sampler_last_ns) {
        uint64_t elapsed_ns = now_ns - dev->sampler_last_ns;
        if (elapsed_ns <= backend->sample_interval_us * UINT64_C(1000) * 10) {
          dev->sample_window_elapsed_ns += elapsed_ns;
          dev->process_window_elapsed_ns += elapsed_ns;
          if (active_or_changed) {
            dev->sample_window_active_ns += elapsed_ns;
            attribute_active_time(dev, pids, pid_count, elapsed_ns, now_ns);
          }
        }
      }
      dev->sampler_last_ns = now_ns;
      dev->sampler_last_submitted_seq = sample.submitted_seq;
      dev->sampler_last_completed_seq = sample.completed_seq;
      dev->sampler_last_beacon_seq = sample.beacon_seq;
      dev->sampler_have_last = true;
      pthread_mutex_unlock(&backend->lock);
    }

    sleep_us(backend->sample_interval_us);
  }

  return NULL;
}

static void start_sampler(struct pacc_monitor_backend *backend) {
  uint64_t interval_us;

  if (!backend || !backend->lock_initialized || backend->sampler_running) {
    return;
  }

  interval_us = env_u64_default("PACC_MONITOR_SAMPLE_US", PACC_MONITOR_DEFAULT_SAMPLE_US);
  if (interval_us < 1000) {
    interval_us = 1000;
  } else if (interval_us > 1000000) {
    interval_us = 1000000;
  }
  backend->sample_interval_us = interval_us;
  backend->stop_sampler = false;
  if (pthread_create(&backend->sampler_thread, NULL, sampler_main, backend) == 0) {
    backend->sampler_running = true;
  }
}

static void stop_sampler(struct pacc_monitor_backend *backend) {
  if (!backend || !backend->sampler_running) {
    return;
  }
  backend->stop_sampler = true;
  pthread_join(backend->sampler_thread, NULL);
  backend->sampler_running = false;
}

static void update_job_counters(struct pacc_monitor_device *dev, struct pacc_monitor_sample *sample, bool activity) {
  uint32_t submitted_job = sample->submitted_job_id;
  uint32_t completed_job = sample->completed_job_id;

  if (!dev->have_last) {
    dev->last_submitted_seq = sample->submitted_seq;
    dev->last_completed_seq = sample->completed_seq;
    dev->last_beacon_seq = sample->beacon_seq;
    dev->last_diag_count = sample->diag_count;
    return;
  }

  if (sample->submitted_seq && sample->submitted_seq != dev->last_submitted_seq) {
    dev->job_submit_count++;
    if (submitted_job < pacc_monitor_job_count) {
      dev->jobs[submitted_job].submit_count++;
      dev->jobs[submitted_job].submitted_seq = sample->submitted_seq;
    }
  }
  if (sample->completed_seq && sample->completed_seq != dev->last_completed_seq) {
    dev->job_complete_count++;
    if (completed_job < pacc_monitor_job_count) {
      dev->jobs[completed_job].complete_count++;
      dev->jobs[completed_job].completed_seq = sample->completed_seq;
    }
  }
  if (activity) {
    dev->last_activity_ns = sample->timestamp_ns;
  }

  dev->last_submitted_seq = sample->submitted_seq;
  dev->last_completed_seq = sample->completed_seq;
  dev->last_beacon_seq = sample->beacon_seq;
  dev->last_diag_count = sample->diag_count;
}

static uint64_t read_diag_count(struct pacc_monitor_device *dev) {
  uint64_t rel = (uint64_t)HETGPU_PACC_DIAG_RING_SLOT * HETGPU_PACC_CONTROL_BYTES;
  uint64_t max_index = 0;
  bool have_event = false;

  for (unsigned i = 0; i < HETGPU_PACC_DIAG_RING_RECORDS; i++) {
    struct pacc_monitor_diag_event event;
    if (!read_ddr_relative(dev, rel + (uint64_t)i * sizeof(event), &event, sizeof(event))) {
      break;
    }
    if (event.magic == HETGPU_PACC_DIAG_MAGIC) {
      if (!have_event || event.index > max_index) {
        max_index = event.index;
      }
      have_event = true;
    }
  }

  return have_event ? max_index + 1 : 0;
}

int pacc_monitor_sample_device(struct pacc_monitor_backend *backend, unsigned index,
                               struct pacc_monitor_sample *sample) {
  struct pacc_monitor_device *dev;
  pid_t memory_pids[PACC_MONITOR_MAX_PIDS];
  size_t memory_pid_count = 0;
  bool have_memory_stats = false;
  bool have_memory_usage_stats = false;
  bool have_memory_bandwidth_stats = false;
  uint64_t memory_used = 0;
  uint64_t memory_read_bytes = 0;
  uint64_t memory_write_bytes = 0;
  bool activity = false;
  uint64_t active_ns = 0;
  uint64_t elapsed_ns = 0;

  if (!backend || !sample || index >= backend->count) {
    return -EINVAL;
  }

  dev = &backend->devices[index];
  read_device_snapshot(dev, sample, true);
  if (scan_processes_for_device(backend, dev, memory_pids,
                                sizeof(memory_pids) / sizeof(memory_pids[0]),
                                &memory_pid_count) == 0) {
    have_memory_stats = sum_pid_alloc_stats(memory_pids, memory_pid_count, &memory_used,
                                            &memory_read_bytes, &memory_write_bytes,
                                            &have_memory_usage_stats, &have_memory_bandwidth_stats);
    if (have_memory_usage_stats) {
      set_sample_memory_usage(sample, memory_used);
    }
  }

  activity = sample->active ||
             (dev->have_last && (sample->submitted_seq != dev->last_submitted_seq ||
                                 sample->completed_seq != dev->last_completed_seq ||
                                 sample->beacon_seq != dev->last_beacon_seq ||
                                 sample->diag_count != dev->last_diag_count));

  if (backend->lock_initialized) {
    pthread_mutex_lock(&backend->lock);
  }
  update_job_counters(dev, sample, activity);
  if (have_memory_stats && have_memory_bandwidth_stats) {
    update_memory_bandwidth(dev, sample, memory_read_bytes, memory_write_bytes);
  }
  if (backend->sampler_running) {
    active_ns = dev->sample_window_active_ns;
    elapsed_ns = dev->sample_window_elapsed_ns;
    dev->sample_window_active_ns = 0;
    dev->sample_window_elapsed_ns = 0;
  }
  sample->job_submit_count = dev->job_submit_count;
  sample->job_complete_count = dev->job_complete_count;
  memcpy(sample->jobs, dev->jobs, sizeof(sample->jobs));
  dev->have_last = true;
  if (backend->lock_initialized) {
    pthread_mutex_unlock(&backend->lock);
  }

  if (elapsed_ns > 0) {
    uint64_t util = (active_ns * UINT64_C(100) + elapsed_ns / 2) / elapsed_ns;
    sample->gpu_util_percent = util > 100 ? 100 : (uint32_t)util;
  } else if (activity || sample->active ||
             (dev->last_activity_ns && sample->timestamp_ns >= dev->last_activity_ns &&
              sample->timestamp_ns - dev->last_activity_ns <= (uint64_t)backend->busy_window_ms * UINT64_C(1000000))) {
    sample->gpu_util_percent = backend->sampler_running ? 0 : 100;
  } else {
    sample->gpu_util_percent = 0;
  }

  return 0;
}

int pacc_monitor_read_control(struct pacc_monitor_backend *backend, unsigned index, uint64_t offset, void *dst,
                              size_t len) {
  if (!backend || index >= backend->count || !dst) {
    return -EINVAL;
  }
  return read_control(&backend->devices[index], offset, dst, len) ? 0 : -EIO;
}

int pacc_monitor_write_control(struct pacc_monitor_backend *backend, unsigned index, uint64_t offset, const void *src,
                               size_t len) {
  if (!backend || index >= backend->count || !src) {
    return -EINVAL;
  }
  return write_control(&backend->devices[index], offset, src, len) ? 0 : -EIO;
}

static bool proc_entry_is_pid(const char *name) {
  if (!name || !*name) {
    return false;
  }
  for (const char *p = name; *p; p++) {
    if (!isdigit((unsigned char)*p)) {
      return false;
    }
  }
  return true;
}

static bool pid_already_seen(pid_t pid, const pid_t *pids, size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (pids[i] == pid) {
      return true;
    }
  }
  return false;
}

static bool pid_comm_equals(pid_t pid, const char *comm) {
  char path[64];
  char buf[64];
  int fd;
  ssize_t n;

  if (pid <= 0 || !comm) {
    return false;
  }

  snprintf(path, sizeof(path), "/proc/%ld/comm", (long)pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  n = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (n <= 0) {
    return false;
  }
  buf[n] = '\0';
  for (ssize_t i = 0; i < n; i++) {
    if (buf[i] == '\n' || buf[i] == '\r') {
      buf[i] = '\0';
      break;
    }
  }
  return !strcmp(buf, comm);
}

static bool fd_target_matches_pacc(const char *target, const struct pacc_monitor_device *dev) {
  char needle[64];

  if (!target || !dev) {
    return false;
  }
  if (!strcmp(target, dev->path)) {
    return true;
  }

  snprintf(needle, sizeof(needle), "/dev/pacc%u", dev->pacc_id);
  if (!strcmp(target, needle)) {
    return true;
  }
  snprintf(needle, sizeof(needle), "/dev/hetgpu_pacc_mbox%u", dev->pacc_id);
  if (!strcmp(target, needle)) {
    return true;
  }
  snprintf(needle, sizeof(needle), "/dev/hetgpu_pacc_mbox_ddr_coh%u", dev->pacc_id);
  return !strcmp(target, needle);
}

static int scan_processes_for_device(const struct pacc_monitor_backend *backend, const struct pacc_monitor_device *dev,
                                     pid_t *pids, size_t max_pids, size_t *count) {
  DIR *proc_dir;
  struct dirent *proc_entry;
  size_t found = 0;
  pid_t self_pid = getpid();

  if (!backend || !dev || !pids || !count) {
    return -EINVAL;
  }
  *count = 0;
  if (max_pids == 0) {
    return 0;
  }

  proc_dir = opendir("/proc");
  if (!proc_dir) {
    return -errno;
  }

  while ((proc_entry = readdir(proc_dir)) != NULL && found < max_pids) {
    char fd_dir_path[PATH_MAX];
    DIR *fd_dir;
    struct dirent *fd_entry;
    pid_t pid;

    if (!proc_entry_is_pid(proc_entry->d_name)) {
      continue;
    }
    pid = (pid_t)strtol(proc_entry->d_name, NULL, 10);
    if (pid <= 0 || pid_already_seen(pid, pids, found)) {
      continue;
    }
    if (backend->client == pacc_monitor_client_nvtop &&
        (pid == self_pid || pid_comm_equals(pid, "nvtop"))) {
      continue;
    }

    snprintf(fd_dir_path, sizeof(fd_dir_path), "/proc/%s/fd", proc_entry->d_name);
    fd_dir = opendir(fd_dir_path);
    if (!fd_dir) {
      continue;
    }

    while ((fd_entry = readdir(fd_dir)) != NULL) {
      char target[PATH_MAX];
      ssize_t n;

      if (fd_entry->d_name[0] == '.') {
        continue;
      }
      n = readlinkat(dirfd(fd_dir), fd_entry->d_name, target, sizeof(target) - 1);
      if (n <= 0) {
        continue;
      }
      target[n] = '\0';
      if (fd_target_matches_pacc(target, dev)) {
        pids[found++] = pid;
        break;
      }
    }
    closedir(fd_dir);
  }

  closedir(proc_dir);
  *count = found;
  return 0;
}

int pacc_monitor_scan_processes(const struct pacc_monitor_backend *backend, unsigned index, pid_t *pids,
                                size_t max_pids, size_t *count) {
  if (!backend || index >= backend->count) {
    return -EINVAL;
  }
  return scan_processes_for_device(backend, &backend->devices[index], pids, max_pids, count);
}

static bool pid_in_list(pid_t pid, const pid_t *pids, size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (pids[i] == pid) {
      return true;
    }
  }
  return false;
}

static bool process_sample_already_added(pid_t pid, const struct pacc_monitor_process_sample *processes,
                                         size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (processes[i].pid == pid) {
      return true;
    }
  }
  return false;
}

int pacc_monitor_sample_processes(struct pacc_monitor_backend *backend, unsigned index,
                                  struct pacc_monitor_process_sample *processes, size_t max_processes,
                                  size_t *count) {
  struct pacc_monitor_device *dev;
  pid_t current_pids[PACC_MONITOR_MAX_PIDS];
  struct pacc_monitor_pid_stats current_stats[PACC_MONITOR_MAX_PIDS];
  size_t current_count = 0;
  size_t out_count = 0;
  uint64_t window_ns = 0;
  uint64_t now_ns = monotonic_ns();

  if (!backend || index >= backend->count || !processes || !count) {
    return -EINVAL;
  }

  *count = 0;
  if (max_processes == 0) {
    return 0;
  }

  dev = &backend->devices[index];
  memset(current_stats, 0, sizeof(current_stats));
  (void)scan_processes_for_device(backend, dev, current_pids, sizeof(current_pids) / sizeof(current_pids[0]),
                                  &current_count);
  for (size_t i = 0; i < current_count; i++) {
    (void)read_pid_alloc_stats(current_pids[i], &current_stats[i]);
  }

  if (backend->lock_initialized) {
    pthread_mutex_lock(&backend->lock);
  }

  for (size_t i = 0; i < current_count; i++) {
    struct pacc_monitor_process_accum *entry = get_process_accum(dev, current_pids[i], now_ns);
    entry->gpu_memory_usage = current_stats[i].have_memory_usage ? current_stats[i].gpu_memory_usage : 0;
    entry->last_seen_ns = now_ns;
  }

  window_ns = dev->process_window_elapsed_ns;
  for (size_t i = 0; i < dev->process_count && out_count < max_processes; i++) {
    struct pacc_monitor_process_accum *entry = &dev->processes[i];
    bool current = pid_in_list(entry->pid, current_pids, current_count);
    if (entry->active_ns == 0 && !current) {
      continue;
    }

    processes[out_count].pid = entry->pid;
    processes[out_count].gpu_active_ns = entry->active_ns;
    processes[out_count].sample_window_ns = window_ns;
    processes[out_count].gpu_memory_usage = entry->gpu_memory_usage;
    if (window_ns > 0) {
      uint64_t util = (entry->active_ns * UINT64_C(100) + window_ns / 2) / window_ns;
      processes[out_count].gpu_util_percent = util > 100 ? 100 : (uint32_t)util;
    } else {
      processes[out_count].gpu_util_percent = 0;
    }
    out_count++;
  }

  for (size_t i = 0; i < current_count && out_count < max_processes; i++) {
    if (process_sample_already_added(current_pids[i], processes, out_count)) {
      continue;
    }
    memset(&processes[out_count], 0, sizeof(processes[out_count]));
    processes[out_count].pid = current_pids[i];
    processes[out_count].sample_window_ns = window_ns;
    processes[out_count].gpu_memory_usage = current_stats[i].have_memory_usage ? current_stats[i].gpu_memory_usage : 0;
    out_count++;
  }

  for (size_t i = 0; i < dev->process_count; i++) {
    dev->processes[i].active_ns = 0;
  }
  dev->process_window_elapsed_ns = 0;

  if (backend->lock_initialized) {
    pthread_mutex_unlock(&backend->lock);
  }

  *count = out_count;
  return 0;
}
