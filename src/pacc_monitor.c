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
};

struct pacc_monitor_backend {
  enum pacc_monitor_client client;
  uint32_t busy_window_ms;
  unsigned count;
  struct pacc_monitor_device devices[PACC_MONITOR_MAX_DEVICES];
};

static uint64_t monotonic_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
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

static bool read_u64_file(const char *path, uint64_t *out) {
  char buf[64];
  int fd;
  ssize_t n;

  if (!path || !out) {
    return false;
  }

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
  return parse_u64_checked(buf, out);
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
    free(created);
    return -ENODEV;
  }

  *backend = created;
  return 0;
}

void pacc_monitor_close(struct pacc_monitor_backend *backend) {
  if (!backend) {
    return;
  }
  for (unsigned i = 0; i < backend->count; i++) {
    close_device(&backend->devices[i]);
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
  struct pacc_monitor_doorbell doorbell;
  struct pacc_monitor_job_desc kernel_desc;
  struct pacc_monitor_host_status completion;
  struct pacc_monitor_jobd_beacon beacon;
  struct pacc_monitor_runtime_table_prefix runtime;
  bool activity = false;
  uint64_t memory_total;

  if (!backend || !sample || index >= backend->count) {
    return -EINVAL;
  }

  dev = &backend->devices[index];
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

  sample->diag_count = read_diag_count(dev);

  sample->active = sample->submitted_seq != 0 && sample->submitted_seq != sample->completed_seq;
  sample->active_job_id = sample->active ? sample->submitted_job_id : sample->completed_job_id;

  activity = sample->active ||
             (dev->have_last && (sample->submitted_seq != dev->last_submitted_seq ||
                                 sample->completed_seq != dev->last_completed_seq ||
                                 sample->beacon_seq != dev->last_beacon_seq ||
                                 sample->diag_count != dev->last_diag_count));
  update_job_counters(dev, sample, activity);

  memory_total = dev->shared_ddr_size > HETGPU_PACC_SHARED_DDR_USER_OFF
                     ? dev->shared_ddr_size - HETGPU_PACC_SHARED_DDR_USER_OFF
                     : dev->shared_ddr_size;
  sample->total_memory = memory_total;
  sample->used_memory = 0;
  sample->free_memory = memory_total;
  sample->mem_util_percent = 0;

  if (activity || sample->active ||
      (dev->last_activity_ns && sample->timestamp_ns >= dev->last_activity_ns &&
       sample->timestamp_ns - dev->last_activity_ns <= (uint64_t)backend->busy_window_ms * UINT64_C(1000000))) {
    sample->gpu_util_percent = 100;
  } else {
    sample->gpu_util_percent = 0;
  }

  sample->job_submit_count = dev->job_submit_count;
  sample->job_complete_count = dev->job_complete_count;
  memcpy(sample->jobs, dev->jobs, sizeof(sample->jobs));
  dev->have_last = true;
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

int pacc_monitor_scan_processes(const struct pacc_monitor_backend *backend, unsigned index, pid_t *pids,
                                size_t max_pids, size_t *count) {
  const struct pacc_monitor_device *dev;
  DIR *proc_dir;
  struct dirent *proc_entry;
  size_t found = 0;
  pid_t self_pid = getpid();

  if (!backend || index >= backend->count || !pids || !count) {
    return -EINVAL;
  }
  *count = 0;
  if (max_pids == 0) {
    return 0;
  }

  dev = &backend->devices[index];
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
    if (pid <= 0 || (backend->client == pacc_monitor_client_nvtop && pid == self_pid) ||
        pid_already_seen(pid, pids, found)) {
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
