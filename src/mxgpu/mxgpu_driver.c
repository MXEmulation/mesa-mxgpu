/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_driver.h"
#include "mxgpu_compiler.h"
#include "mxgpu_scene.h"

#include "mxgpu_drm_uapi.h"
#include "mxgpu_wire.h"
#include "mxsb.h"
#include "mx_le.h"

#include <errno.h>
#include <pthread.h>
#include <limits.h>
#include <math.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <glob.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


#define MXGPU_RESOURCE_SLOTS 128u
#define MXGPU_FRAMEBUFFER_CACHE_ENTRIES 32u
#define MXGPU_FRAMEBUFFER_CACHE_BYTES (128ull * 1024u * 1024u)
#define MXGPU_TEXTURE_CACHE_ENTRIES 32u
#define MXGPU_VERTEX_CACHE_ENTRIES 32u
#define MXGPU_VERTEX_CACHE_BYTES (16ull * 1024u * 1024u)
#define MXGPU_UNIFORM_CACHE_ENTRIES 8u
#define MXGPU_UNIFORM_CACHE_BYTES (8ull * 1024u * 1024u)
#define MXGPU_TEXTURE_CACHE_BYTES (128ull * 1024u * 1024u)

struct resource {
    int live;
    int host_live;
    int host_current;
    int upload_scheduled;
    int batch_pinned;
    unsigned format;
    unsigned kind;
    unsigned width;
    unsigned height;
    unsigned char *bytes;
    unsigned size;
    unsigned host_bytes;
    uint64_t host_usage;
    unsigned host_width;
    unsigned host_height;
    unsigned mip_count, host_mip_count;
    unsigned array_layers, host_array_layers, host_format;
    unsigned mip_offsets[32];
    unsigned dirty_begin;
    unsigned dirty_end;
};

struct shader_image {
    int live;
    unsigned len;
    unsigned char bytes[MXGPU_LINK_MODULE_CAPACITY];
};

struct pipeline_image {
    int live;
    unsigned shader;
    unsigned vs_entry;
    unsigned fs_entry;
};

struct native_state_cache_entry {
    uint8_t payload[MXGPU_BLEND_STATE_HEADER_SIZE + 8 * MXGPU_BLEND_TARGET_SIZE];
    uint32_t bytes, id;
    uint64_t used;
};

struct native_pipeline_cache_entry {
    uint8_t *module;
    uint32_t bytes, shader_id, pipeline_id;
    int shader_live, pipeline_live;
    uint64_t used;
};

struct native_texture_cache_entry {
    uint64_t identity, used;
    unsigned resource_id, legacy_slot;
    int blocked;
};

struct uniform_cache_entry {
    unsigned resource_id, logical_size;
    uint64_t used;
    int blocked, uncertain_create;
};

struct vertex_cache_entry {
    unsigned resource_id;
    uint32_t used_bytes, required_capacity, stride;
    uint64_t used;
    int blocked, uncertain_create;
};

struct scratch_buffer {
    uint8_t *bytes;
    uint32_t capacity;
};

struct mxgpu_framebuffer {
    unsigned char *pixels;
    uint32_t width, height, size;
    uint64_t cpu_revision;
    int valid, pending, error, unpublished;
    unsigned resource_id;
    int resource_blocked;
    uint64_t used;
};

struct device {
    struct mxgpu_framebuffer *framebuffer;
    struct mxgpu_framebuffer *framebuffer_cache[MXGPU_FRAMEBUFFER_CACHE_ENTRIES];
    uint64_t framebuffer_clock;
    unsigned immediate_color_id;
    int deferred_readback;
    struct scratch_buffer command_scratch, record_scratch, upload_scratch, readback_scratch;
    int open;
    int lost;
    int completion_valid;
    uint32_t completion_status;
    unsigned context;
    int context_owned;
    struct resource resources[MXGPU_RESOURCE_SLOTS];
    struct shader_image shaders[16];
    struct pipeline_image pipes[16];
    unsigned next_id;
    unsigned color_id;
    unsigned texture_id;
    unsigned vertex_id;
    struct vertex_cache_entry vertex_cache[MXGPU_VERTEX_CACHE_ENTRIES];
    uint64_t vertex_cache_clock;
    unsigned uniform_id[2];
    unsigned uniform_size[2];
    struct uniform_cache_entry uniform_cache[2][MXGPU_UNIFORM_CACHE_ENTRIES];
    uint64_t uniform_cache_clock;
    struct native_pipeline_cache_entry module_cache[8];
    uint32_t module_cache_next_id;
    uint64_t module_cache_clock;
    unsigned shader_id;
    unsigned pipeline_id;
    int shader_live;
    int pipeline_live;
    int pipeline_dirty;
    unsigned submits;
    unsigned frame_submits;
    unsigned long long sequence;
    int fd;
    int allow_executor;
    struct mxgpu_drm_caps caps;
    struct mxgpu_adapter_info adapter_info;
    int adapter_info_valid;
    struct mxgpu_drm_transfer_limits transfer_limits;
    struct mxgpu_drm_batch_limits batch_limits;
    struct scratch_buffer batch_commands, batch_record;
    struct mxgpu_drm_batch batch;
    uint32_t batch_bytes, batch_offsets[MXGPU_DRM_BATCH_MAX_COMMANDS];
    uint64_t batch_sequences[MXGPU_DRM_BATCH_MAX_COMMANDS];
    uint16_t batch_opcodes[MXGPU_DRM_BATCH_MAX_COMMANDS];
    int batch_collect;
    struct mxgpu_native_render_state native_state;
    int native_active;
    uint32_t native_state_id;
    unsigned depth_id;
    struct mxgpu_format_capabilities format_caps;
    int format_caps_valid;
    uint32_t active_depth_id;
    struct native_state_cache_entry depth_cache[4];
    uint32_t active_blend_id, active_rasterizer_id, active_sampler_ids[MXGPU_TEXTURE_INPUTS];
    uint64_t native_cache_clock;
    struct native_state_cache_entry blend_cache[4], rasterizer_cache[4], sampler_cache[MXGPU_TEXTURE_INPUTS];
    const struct mxgpu_texture_input *draw_textures;
    uint32_t draw_texture_count;
    unsigned texture_ids[MXGPU_TEXTURE_INPUTS];
    struct native_texture_cache_entry texture_cache[MXGPU_TEXTURE_CACHE_ENTRIES];
    uint64_t texture_clock;

};

static int apply_decoded(uint16_t opcode, const uint8_t *payload, uint32_t len);
static int mxgpu_debug_illegal_then_legal_unlocked(void);

static struct device g_dev = {.fd = -1};
static pthread_mutex_t g_device_mutex = PTHREAD_MUTEX_INITIALIZER;
static int batch_drain_unlocked(void);
static int vertex_cache_trim_unlocked(unsigned preserve, uint32_t replacement_size);
static int uniform_cache_trim_unlocked(unsigned preserve, uint32_t replacement_size);
static int framebuffer_trim_bytes_unlocked(struct mxgpu_framebuffer *preserve);
static int mxgpu_readback_ready_unlocked(void);
static void mxgpu_flush_frame_unlocked(void);
static int mxgpu_seed_color_unlocked(const unsigned char *pixels, unsigned width, unsigned height);
static int mxgpu_execute_module_uniforms_unlocked(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size, uint32_t vertex_stride_bytes);
static int mxgpu_execute_module_unlocked(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch);
static int mxgpu_execute_scene_unlocked(const float *vertices, int vertex_count,
                        const unsigned char *texels, int tw, int th,
                        unsigned char *color, int cw, int ch);
static int query_adapter_unlocked(void);
static void uniform_cache_record_create(unsigned id, int success);
static int framebuffer_sync_unlocked(struct mxgpu_framebuffer *framebuffer);
static int framebuffer_evict_unlocked(struct mxgpu_framebuffer *preserve);
static int texture_evict_unlocked(void);
static int resource_pressure_evict_unlocked(void);
static int native_color_sample_available_unlocked(void);
static int g_needs_clear = 1;

static int framebuffer_detach_unlocked(void);
static int mxgpu_device_open_fd_unlocked(int fd);
static int mxgpu_device_open_unlocked(void);
static void mxgpu_device_close_unlocked(void);
static unsigned mxgpu_last_submits_unlocked(void);
static int mxgpu_debug_illegal_then_legal_unlocked(void);

static unsigned long long take_sequence(void)
{
    struct timespec ts;
    unsigned long long now;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    now = (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
    if (now <= g_dev.sequence)
        now = g_dev.sequence + 1ull;
    g_dev.sequence = now;
    return now;
}

static void write_log(const char *leaf, const char *text)
{
    const char *dir = getenv("MXGPU_LOG_DIR");
    char path[768];
    size_t dir_len;
    size_t leaf_len;
    FILE *log;
    if (!dir || !dir[0])
        return;
    dir_len = strlen(dir);
    leaf_len = strlen(leaf);
    if (dir_len + 1 + leaf_len + 1 > sizeof path)
        return;
    memcpy(path, dir, dir_len);
    path[dir_len] = '/';
    memcpy(path + dir_len + 1, leaf, leaf_len + 1);
    log = fopen(path, "w");
    if (!log)
        return;
    fputs(text, log);
    fclose(log);
}

static int open_render_node(char *path, size_t path_size)
{
    glob_t nodes = {0};
    int fd = -1;
    int err = ENODEV;
    snprintf(path, path_size, "/dev/dri/renderD*");
    if (glob(path, 0, NULL, &nodes) == 0) {
        for (size_t i = 0; i < nodes.gl_pathc; i++) {
            char name[16] = {0};
            struct drm_version version = {.name = name, .name_len = sizeof name};
            int candidate = open(nodes.gl_pathv[i], O_RDWR | O_CLOEXEC);
            if (candidate < 0) {
                err = errno;
                continue;
            }
            if (mxgpu_ioctl(candidate, DRM_IOCTL_VERSION, &version) == 0 &&
                version.name_len == 5 && memcmp(name, "mxgpu", 5) == 0) {
                fd = candidate;
                snprintf(path, path_size, "%s", nodes.gl_pathv[i]);
                break;
            }
            close(candidate);
        }
    }
    globfree(&nodes);
    if (fd < 0)
        errno = err;
    return fd;
}

int mxgpu_device_available(void)
{
    char path[256];
    int fd = open_render_node(path, sizeof path);
    if (fd < 0)
        return 0;
    close(fd);
    return 1;
}

int mxgpu_device_flush(void)
{
    pthread_mutex_lock(&g_device_mutex);
    int had_pending = g_dev.batch.count != 0;
    int result = batch_drain_unlocked();
    int saved_errno = errno;
    if (!result && had_pending) {
        vertex_cache_trim_unlocked(0, 0);
        if (!g_dev.lost) uniform_cache_trim_unlocked(0, 0);
        if (!g_dev.lost) framebuffer_trim_bytes_unlocked(g_dev.framebuffer);
        if (g_dev.lost) result = -1;
    }
    errno = saved_errno;
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_device_lost(void)
{
    pthread_mutex_lock(&g_device_mutex);
    int lost = g_dev.lost;
    pthread_mutex_unlock(&g_device_mutex);
    return lost;
}

static void log_open_attempt(void)
{
    char path[256], line[512];
    int fd = open_render_node(path, sizeof path);
    int err = errno;
    if (fd < 0) {
        snprintf(line, sizeof line, "open %s failed errno %d %s\n", path, err, strerror(err));
        g_dev.fd = -1;
        g_dev.allow_executor = 0;
    } else {
        snprintf(line, sizeof line, "open %s succeeded fd %d\n", path, fd);
        g_dev.fd = fd;
        g_dev.allow_executor = 0;
    }
    write_log("device-access.log", line);
}

static int device_execute_submit(const struct mxgpu_drm_user *user)
{
    const uint8_t *record;
    struct mxgpu_command_header header;
    uint32_t context_id, queue, response_cap, command_bytes, payload_len;
    uint64_t fence_value;
    const uint8_t *command = NULL;
    const uint8_t *payload = NULL;
    if (!user || user->pointer == 0 || user->size == 0 ||
        user->size > 512u * 1024u + 512u || user->size > user->capacity)
        return -1;
    record = (const uint8_t *)(uintptr_t)user->pointer;
    if (mxgpu_drm_submit_decode(record, user->size, &context_id, &queue, &fence_value, &response_cap, &command, &command_bytes) != MXGPU_DRM_OK)
        return -1;
    if (mxgpu_command_decode(command, command_bytes, 512u * 1024u, &header, &payload, &payload_len) != MX_OK)
        return -1;
    if (apply_decoded(header.opcode, payload, payload_len) != 0)
        return -1;
    g_dev.submits++;
    if (header.opcode == MXGPU_OP_RENDER_SUBMIT)
        g_dev.frame_submits++;
    return MX_OK;
}

static uint8_t *scratch_reserve(struct scratch_buffer *scratch, uint32_t capacity)
{
    uint8_t *replacement;
    if (capacity <= scratch->capacity)
        return scratch->bytes;
    replacement = realloc(scratch->bytes, capacity);
    if (!replacement)
        return NULL;
    scratch->bytes = replacement;
    scratch->capacity = capacity;
    return replacement;
}

static void completed_submit_proof(void)
{
    const char *path = getenv("MXGPU_SUBMIT_FILE");
    if (path && g_dev.frame_submits) {
        FILE *out = fopen(path, "w");
        if (out) {
            fprintf(out, "%u\n", g_dev.frame_submits);
            fclose(out);
        }
    }
}

static int batch_drain_unlocked(void)
{
    if (g_dev.lost) return -1;
    if (!g_dev.batch.count) return 0;
    uint32_t capacity = g_dev.batch_limits.max_input_bytes;
    uint8_t *record = scratch_reserve(&g_dev.batch_record, capacity);
    uint32_t bytes;
    if (!record) return -1;
    for (unsigned i = 0; i < g_dev.batch.count; i++)
        g_dev.batch.commands[i].command = g_dev.batch_commands.bytes + g_dev.batch_offsets[i];
    if (mxgpu_drm_batch_encode(&g_dev.batch, record, capacity, &bytes) != MXGPU_DRM_OK) return -1;
    struct mxgpu_drm_user user = { .pointer = (uint64_t)(uintptr_t)record,
        .size = bytes, .capacity = capacity };
    int result = mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + MXGPU_DRM_IOCTL_SUBMIT_BATCH,
                       struct mxgpu_drm_user), &user);
    int saved_errno = errno;
    struct mxgpu_drm_batch_response response;
    int valid = user.size <= capacity &&
        mxgpu_drm_batch_response_decode(record, user.size, &response) == MXGPU_DRM_OK &&
        response.context_id == g_dev.context && response.count == g_dev.batch.count;
    if (valid) {
        for (unsigned i = 0; i < response.count; i++) {
            struct mxgpu_drm_batch_outcome *outcome = &response.outcomes[i];
            if (outcome->state != MXGPU_DRM_BATCH_NOT_POSTED &&
                (outcome->sequence != g_dev.batch_sequences[i] ||
                 outcome->fence_value != g_dev.batch.commands[i].fence_value)) valid = 0;
        }
    }
    if (valid) {
        for (unsigned i = 0; i < response.count; i++) {
            const struct mxgpu_drm_batch_outcome *outcome = &response.outcomes[i];
            if (outcome->state != MXGPU_DRM_BATCH_COMPLETED || outcome->status) continue;
            uint16_t opcode = g_dev.batch_opcodes[i];
            g_dev.submits++;
            if (opcode == MXGPU_OP_RENDER_SUBMIT_EXTENDED) g_dev.frame_submits++;
        }
        completed_submit_proof();
    }
    if (!valid || result || response.aggregate_result != MXGPU_DRM_BATCH_COMPLETE) {
        g_dev.lost = 1;
        for (unsigned i = 0; i < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; i++)
            if (g_dev.framebuffer_cache[i] && g_dev.framebuffer_cache[i]->pending)
                g_dev.framebuffer_cache[i]->error = 1;
        errno = result ? saved_errno : EIO;
        return -1;
    }
    for (unsigned i = 1; i < MXGPU_RESOURCE_SLOTS; i++) {
        struct resource *resource = &g_dev.resources[i];
        resource->batch_pinned = 0;
        if (resource->upload_scheduled) {
            resource->host_current = 1;
            resource->upload_scheduled = 0;
            resource->dirty_begin = resource->dirty_end = 0;
        }
    }
    g_dev.batch.count = g_dev.batch_bytes = 0;
    completed_submit_proof();
    return 0;
}

static int batch_enqueue_unlocked(uint16_t opcode, uint16_t queue, uint32_t context,
                                  const uint8_t *payload, uint32_t payload_bytes)
{
    uint8_t pins[MXGPU_RESOURCE_SLOTS] = {0};
    if (opcode == MXGPU_OP_TRANSFER_TO_HOST) {
        struct mxgpu_transfer transfer;
        const uint8_t *data;
        if (mxgpu_transfer_decode(payload, payload_bytes, &transfer, &data) != MX_OK ||
            !transfer.resource_id || transfer.resource_id >= MXGPU_RESOURCE_SLOTS) return -1;
        pins[transfer.resource_id] = 1;
    } else {
        struct mxgpu_render_extended render;
        struct mxgpu_execution_binding bindings[19];
        if (mxgpu_render_extended_decode(payload, payload_bytes, &render, bindings, 19) != MX_OK) return -1;
        for (unsigned i = 0; i < render.color_target_count; i++) {
            unsigned id = render.color_targets[i].resource_id;
            if (!id || id >= MXGPU_RESOURCE_SLOTS) return -1;
            pins[id] = 1;
        }
        for (unsigned i = 0; i < render.binding_count; i++) {
            if (bindings[i].kind == MXGPU_BIND_KIND_SAMPLER) continue;
            unsigned id = bindings[i].resource_id;
            if (!id || id >= MXGPU_RESOURCE_SLOTS) return -1;
            pins[id] = 1;
        }
    }
    uint32_t max = g_dev.batch_limits.max_command_bytes;
    if (payload_bytes > max || MXGPU_COMMAND_HEADER_SIZE > max - payload_bytes) return -1;
    uint32_t required = payload_bytes + MXGPU_COMMAND_HEADER_SIZE;
    if (g_dev.batch.count == g_dev.batch_limits.max_commands || g_dev.batch_bytes > max - required)
        if (batch_drain_unlocked()) return -1;
    uint8_t *arena = scratch_reserve(&g_dev.batch_commands, max);
    if (!arena) return -1;
    struct mxgpu_command_header header = {0};
    header.opcode = opcode;
    header.queue = queue;
    header.context_id = context;
    header.flags = MXGPU_CMD_SIGNAL_FENCE;
    header.sequence = take_sequence();
    header.fence_value = header.sequence;
    uint32_t bytes;
    if (mxgpu_command_encode(&header, payload, payload_bytes, g_dev.caps.max_command_bytes,
        arena + g_dev.batch_bytes, max - g_dev.batch_bytes, &bytes) != MX_OK) return -1;
    unsigned index = g_dev.batch.count;
    g_dev.batch.context_id = context;
    g_dev.batch_offsets[index] = g_dev.batch_bytes;
    g_dev.batch_sequences[index] = header.sequence;
    g_dev.batch_opcodes[index] = opcode;
    g_dev.batch.commands[index] = (struct mxgpu_drm_batch_command){
        .queue = queue, .command_bytes = bytes, .fence_value = header.fence_value };
    g_dev.batch_bytes += bytes;
    g_dev.batch.count++;
    for (unsigned i = 1; i < MXGPU_RESOURCE_SLOTS; i++)
        if (pins[i]) g_dev.resources[i].batch_pinned = 1;
    return 0;
}

static int winsys_submit_ioctl(uint16_t opcode, uint16_t queue, uint32_t context, const uint8_t *payload, uint32_t payload_len)
{
    if (g_dev.lost) return -1;
    if (g_dev.batch_collect && g_dev.batch_limits.max_commands &&
        context == g_dev.context && (opcode == MXGPU_OP_TRANSFER_TO_HOST ||
        opcode == MXGPU_OP_RENDER_SUBMIT_EXTENDED))
        return batch_enqueue_unlocked(opcode, queue, context, payload, payload_len);
    if (batch_drain_unlocked()) return -1;
    struct mxgpu_command_header header;
    struct mxgpu_drm_user user;
    uint8_t *command;
    uint8_t *record;
    uint32_t command_cap;
    uint32_t record_cap;
    uint32_t command_len = 0;
    uint32_t record_len = 0;
    int status;
    const uint32_t max_command = g_dev.caps.max_command_bytes ? g_dev.caps.max_command_bytes : 512u * 1024u;
    g_dev.completion_valid = 0;
    if (payload_len > UINT32_MAX - MXGPU_COMMAND_HEADER_SIZE ||
        payload_len + MXGPU_COMMAND_HEADER_SIZE > max_command || payload_len > UINT32_MAX - 768u)
        return -1;

    if (g_dev.lost)
        return -1;
    command_cap = payload_len + 256u;
    if (command_cap < 4096u)
        command_cap = 4096u;
    record_cap = command_cap + 512u;
    command = scratch_reserve(&g_dev.command_scratch, command_cap);
    record = scratch_reserve(&g_dev.record_scratch, record_cap);
    if (!command || !record) {
        return -1;
    }
    memset(&header, 0, sizeof header);
    header.opcode = opcode;
    header.flags = MXGPU_CMD_SIGNAL_FENCE;
    header.context_id = context;
    header.queue = queue;
    header.sequence = take_sequence();
    header.fence_value = header.sequence;
    status = mxgpu_command_encode(&header, payload, payload_len, max_command, command, command_cap, &command_len);
    if (status != MX_OK) {
        fprintf(stderr, "mxgpu command encode failed opcode %u payload %u status %d\n", opcode, payload_len, status);
        return status;
    }
    status = mxgpu_drm_submit_encode(context, queue, header.fence_value, 0, command, command_len, record, record_cap, &record_len);
    if (status != MXGPU_DRM_OK) {
        return status;
    }
    memset(&user, 0, sizeof user);
    user.pointer = (uint64_t)(uintptr_t)record;
    user.size = record_len;
    user.capacity = record_cap;
    if (g_dev.fd >= 0) {
        int posted = mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + 5, struct mxgpu_drm_user), &user);
        int submit_errno = errno;
        if (posted < 0) {
            if (submit_errno == EIO) {
                g_dev.completion_valid = 1;
                g_dev.completion_status = user.size;
            }
            if (getenv("MXGPU_TRACE"))
                fprintf(stderr, "mxgpu submit failed opcode %u queue %u user_size %u errno %d\n",
                        opcode, queue, user.size, submit_errno);
            if (submit_errno == EPIPE || submit_errno == ENODEV || submit_errno == ETIMEDOUT)
                g_dev.lost = 1;
            return -1;
        }
        g_dev.submits++;
        if (opcode == MXGPU_OP_RENDER_SUBMIT || opcode == MXGPU_OP_RENDER_SUBMIT_EXTENDED)
            g_dev.frame_submits++;
        return 0;
    }
    status = device_execute_submit(&user);
    return status;
}

static int native_cache_pinned(uint32_t id)
{
    if (g_dev.batch.count) return 1;
    unsigned i;
    if (id == g_dev.active_blend_id || id == g_dev.active_rasterizer_id || id == g_dev.active_depth_id)
        return 1;
    for (i = 0; i < MXGPU_TEXTURE_INPUTS; i++)
        if (id == g_dev.active_sampler_ids[i])
            return 1;
    return 0;
}

static int native_cache_destroy(struct native_state_cache_entry *entry, uint16_t opcode)
{
    uint8_t payload[MXGPU_RESOURCE_ID_SIZE];
    uint32_t bytes;
    if (!entry->id)
        return 0;
    if (mxgpu_resource_id_encode(entry->id, payload, sizeof payload, &bytes) != MX_OK ||
        winsys_submit_ioctl(opcode, MXGPU_QUEUE_CONTROL, g_dev.context, payload, bytes) != MX_OK) {
        g_dev.lost = 1;
        return -1;
    }
    memset(entry, 0, sizeof *entry);
    return 0;
}

static int native_cache_acquire(struct native_state_cache_entry *cache, unsigned count,
                                uint16_t create_opcode, uint16_t destroy_opcode,
                                const uint8_t *payload, uint32_t bytes, uint32_t *id)
{
    struct native_state_cache_entry *victim = NULL;
    uint8_t command[MXGPU_BLEND_STATE_HEADER_SIZE + 8 * MXGPU_BLEND_TARGET_SIZE];
    unsigned i;
    if (bytes < 4 || bytes > sizeof command || g_dev.lost)
        return -1;
    for (i = 0; i < count; i++) {
        if (cache[i].id && cache[i].bytes == bytes &&
            memcmp(cache[i].payload + 4, payload + 4, bytes - 4) == 0) {
            cache[i].used = ++g_dev.native_cache_clock;
            *id = cache[i].id;
            return 0;
        }
        if (!cache[i].id)
            victim = &cache[i];
    }
    if (!victim && g_dev.batch.count && batch_drain_unlocked()) return -1;
    if (!victim) {
        for (i = 0; i < count; i++)
            if (!native_cache_pinned(cache[i].id) && (!victim || cache[i].used < victim->used))
                victim = &cache[i];
    }
    if (!victim || g_dev.native_state_id == UINT32_MAX)
        return -1;
    if (native_cache_destroy(victim, destroy_opcode))
        return -1;
    memcpy(command, payload, bytes);
    uint32_t assigned = ++g_dev.native_state_id;
    mx_w32(command, 0, assigned);
    if (winsys_submit_ioctl(create_opcode, MXGPU_QUEUE_CONTROL, g_dev.context, command, bytes) != MX_OK)
        return -1;
    memcpy(victim->payload, payload, bytes);
    victim->bytes = bytes;
    victim->id = assigned;
    victim->used = ++g_dev.native_cache_clock;
    *id = assigned;
    return 0;
}

static void native_cache_release_all(void)
{
    unsigned i;
    for (i = 0; i < 4 && !g_dev.lost; i++)
        native_cache_destroy(&g_dev.depth_cache[i], MXGPU_OP_DEPTH_STENCIL_STATE_DESTROY);
    for (i = 0; i < 4 && !g_dev.lost; i++)
        native_cache_destroy(&g_dev.blend_cache[i], MXGPU_OP_BLEND_STATE_DESTROY);
    for (i = 0; i < 4 && !g_dev.lost; i++)
        native_cache_destroy(&g_dev.rasterizer_cache[i], MXGPU_OP_RASTERIZER_STATE_DESTROY);
    for (i = 0; i < MXGPU_TEXTURE_INPUTS && !g_dev.lost; i++)
        native_cache_destroy(&g_dev.sampler_cache[i], MXGPU_OP_SAMPLER_DESTROY);
}

static struct resource *res_slot(unsigned id)
{
    if (id == 0 || id >= MXGPU_RESOURCE_SLOTS)
        return NULL;
    if (!g_dev.resources[id].live)
        return NULL;
    return &g_dev.resources[id];
}

static unsigned new_resource(unsigned kind, unsigned width, unsigned height, unsigned size)
{
    unsigned id;
    struct resource *res;
    unsigned char *bytes;
    for (id = 1; id < MXGPU_RESOURCE_SLOTS && g_dev.resources[id].live; id++) {}
    if (id == MXGPU_RESOURCE_SLOTS) {
        if (resource_pressure_evict_unlocked()) return 0;
        for (id = 1; id < MXGPU_RESOURCE_SLOTS && g_dev.resources[id].live; id++) {}
        if (id == MXGPU_RESOURCE_SLOTS) return 0;
    }
    bytes = calloc(1, size ? size : 1);
    if (!bytes)
        return 0;
    res = &g_dev.resources[id];
    memset(res, 0, sizeof *res);
    res->live = 1;
    res->kind = kind;
    res->width = width;
    res->height = height;
    res->size = size;
    res->bytes = bytes;
    if (g_dev.next_id <= id) g_dev.next_id = id + 1;
    return id;
}

static int destroy_resource(unsigned id)
{
    uint8_t payload[16];
    uint32_t n = 0;
    if (mxgpu_resource_id_encode(id, payload, sizeof payload, &n) != MX_OK)
        return -1;
    if (winsys_submit_ioctl(MXGPU_OP_RESOURCE_DESTROY, MXGPU_QUEUE_CONTROL, g_dev.context, payload, n) != MX_OK)
        return -1;
    g_dev.resources[id].host_live = 0;
    g_dev.resources[id].host_bytes = 0;
    g_dev.resources[id].host_usage = 0;
    g_dev.resources[id].host_width = 0;
    g_dev.resources[id].host_height = 0;
    g_dev.resources[id].host_current = 0;
    g_dev.resources[id].dirty_begin = 0;
    g_dev.resources[id].dirty_end = g_dev.resources[id].size;
    return 0;
}

static int resource_create_submit(const uint8_t *payload, uint32_t size)
{
    for (unsigned attempt = 0; attempt <= MXGPU_TEXTURE_CACHE_ENTRIES + MXGPU_FRAMEBUFFER_CACHE_ENTRIES; attempt++) {
        int result = winsys_submit_ioctl(MXGPU_OP_RESOURCE_CREATE, MXGPU_QUEUE_CONTROL,
                                        g_dev.context, payload, size);
        if (result == MX_OK) return 0;
        int out_of_memory = g_dev.completion_valid &&
            g_dev.completion_status == MXGPU_COMPLETION_OUT_OF_MEMORY;
        if (!out_of_memory || attempt == MXGPU_TEXTURE_CACHE_ENTRIES + MXGPU_FRAMEBUFFER_CACHE_ENTRIES) return -1;
        if (resource_pressure_evict_unlocked()) return -1;
    }
    return -1;
}

static int create_buffer_resource(unsigned id)
{
    struct mxgpu_resource_create create;
    uint8_t payload[64];
    uint32_t n = 0;
    if (g_dev.resources[id].host_live && g_dev.resources[id].host_bytes == g_dev.resources[id].size)
        return 0;
    if (g_dev.resources[id].host_live && destroy_resource(id) != 0)
        return -1;
    memset(&create, 0, sizeof create);
    create.resource_id = id;
    create.kind = MXGPU_KIND_BUFFER;
    create.usage = MXGPU_USAGE_VERTEX | MXGPU_USAGE_UNIFORM | MXGPU_USAGE_TRANSFER_DESTINATION;
    create.width = g_dev.resources[id].size;
    create.height = 1;
    create.depth = 1;
    create.array_layers = 1;
    create.mip_levels = 1;
    create.sample_count = 1;
    create.byte_size = g_dev.resources[id].size;
    if (mxgpu_resource_create_encode(&create, g_dev.adapter_info_valid ? g_dev.adapter_info.max_buffer_bytes : 64ull << 20, payload, sizeof payload, &n) != MX_OK)
        return -1;
    int created = resource_create_submit(payload, n);
    uniform_cache_record_create(id, !created);
    if (created) return -1;
    g_dev.resources[id].host_live = 1;
    g_dev.resources[id].host_bytes = g_dev.resources[id].size;
    g_dev.resources[id].host_usage = create.usage;
    g_dev.resources[id].dirty_begin = 0;
    g_dev.resources[id].dirty_end = g_dev.resources[id].size;
    return 0;
}

static int create_texture_resource(unsigned id, unsigned usage)
{
    struct mxgpu_resource_create create;
    uint8_t payload[64];
    uint32_t n = 0;
    if (g_dev.resources[id].host_live && (g_dev.resources[id].host_usage & usage) == usage &&
        g_dev.resources[id].host_bytes == g_dev.resources[id].size &&
        g_dev.resources[id].host_width == g_dev.resources[id].width &&
        g_dev.resources[id].host_height == g_dev.resources[id].height &&
        g_dev.resources[id].host_mip_count == (g_dev.resources[id].mip_count ? g_dev.resources[id].mip_count : 1) &&
        g_dev.resources[id].host_array_layers == (g_dev.resources[id].array_layers ? g_dev.resources[id].array_layers : 1) &&
        g_dev.resources[id].host_format == (g_dev.resources[id].format ? g_dev.resources[id].format : MXGPU_FMT_RGBA8_UNORM))
        return 0;
    if (g_dev.resources[id].host_live && destroy_resource(id) != 0)
        return -1;
    memset(&create, 0, sizeof create);
    create.resource_id = id;
    create.kind = MXGPU_KIND_TEXTURE_2D;
    create.format = g_dev.resources[id].format ? g_dev.resources[id].format : MXGPU_FMT_RGBA8_UNORM;
    create.usage = usage;
    create.width = g_dev.resources[id].width;
    create.height = g_dev.resources[id].height;
    create.depth = 1;
    create.array_layers = g_dev.resources[id].array_layers ? g_dev.resources[id].array_layers : 1;
    create.mip_levels = g_dev.resources[id].mip_count ? g_dev.resources[id].mip_count : 1;
    create.sample_count = 1;
    create.byte_size = g_dev.resources[id].size;
    if (mxgpu_resource_create_encode(&create, g_dev.adapter_info_valid ? g_dev.adapter_info.max_buffer_bytes : 64ull << 20, payload, sizeof payload, &n) != MX_OK)
        return -1;
    if (resource_create_submit(payload, n))
        return -1;
    g_dev.resources[id].host_live = 1;
    g_dev.resources[id].host_bytes = g_dev.resources[id].size;
    g_dev.resources[id].host_usage = create.usage;
    g_dev.resources[id].host_width = g_dev.resources[id].width;
    g_dev.resources[id].host_height = g_dev.resources[id].height;
    g_dev.resources[id].host_mip_count = create.mip_levels;
    g_dev.resources[id].host_array_layers = create.array_layers;
    g_dev.resources[id].host_format = create.format;
    g_dev.resources[id].dirty_begin = 0;
    g_dev.resources[id].dirty_end = g_dev.resources[id].size;
    return 0;
}

#define MXGPU_UPLOAD_LIMIT (48u * 1024u)

static unsigned upload_limit(void)
{
    unsigned limit = g_dev.transfer_limits.max_transfer_to_host_bytes ?
                     g_dev.transfer_limits.max_transfer_to_host_bytes : MXGPU_UPLOAD_LIMIT;
    unsigned command_limit = g_dev.caps.max_command_bytes ? g_dev.caps.max_command_bytes : 512u * 1024u;
    unsigned overhead = MXGPU_COMMAND_HEADER_SIZE + MXGPU_TRANSFER_REQUEST_SIZE;
    if (command_limit <= overhead) return 0;
    if (limit > command_limit - overhead) limit = command_limit - overhead;
    return limit;
}

static unsigned resource_pixel_bytes(const struct resource *res)
{
    return res->format == MXGPU_FMT_DEPTH32_FLOAT_STENCIL8 ? 8 : 4;
}

static int transfer_span(struct resource *res, unsigned id, unsigned offset, unsigned length, unsigned y, unsigned height, unsigned level, unsigned layer)
{
    struct mxgpu_transfer transfer;
    uint8_t *payload;
    uint32_t n = 0;
    uint32_t cap = length + 256u;
    payload = scratch_reserve(&g_dev.upload_scratch, cap);
    if (!payload)
        return -1;
    memset(&transfer, 0, sizeof transfer);
    transfer.resource_id = id;
    transfer.data_bytes = length;
    transfer.mip_level = level;
    transfer.array_layer = layer;
    if (res->kind == MXGPU_KIND_BUFFER) {
        transfer.resource_offset = offset;
    } else {
        transfer.y = y;
        transfer.width = res->width >> level;
        if (!transfer.width) transfer.width = 1;
        transfer.height = height;
        transfer.depth = 1;
        transfer.row_bytes = transfer.width * resource_pixel_bytes(res);
    }
    if (mxgpu_transfer_encode(&transfer, res->bytes + offset, payload, cap, &n) != MX_OK) {
        return -1;
    }
    if (winsys_submit_ioctl(MXGPU_OP_TRANSFER_TO_HOST, MXGPU_QUEUE_TRANSFER, g_dev.context, payload, n) != MX_OK) {
        return -1;
    }
    return 0;
}

static int transfer_bytes(unsigned id)
{
    struct resource *res = res_slot(id);
    unsigned off;
    unsigned limit = upload_limit();
    if (!limit || !res || !res->bytes || res->size == 0)
        return -1;
    if (res->host_current || res->upload_scheduled)
        return 0;
    unsigned begin = res->dirty_begin;
    unsigned end = res->dirty_end;
    if (begin >= end || end > res->size) {
        begin = 0;
        end = res->size;
    }
    if (res->kind == MXGPU_KIND_BUFFER) {
        for (off = begin; off < end; ) {
            unsigned n = end - off;
            if (n > limit)
                n = limit;
            if (transfer_span(res, id, off, n, 0, 0, 0, 0) != 0)
                return -1;
            off += n;
        }
    } else {
        unsigned levels = res->mip_count ? res->mip_count : 1;
        for (unsigned level = 0; level < levels; level++) {
            unsigned width = res->width >> level, height = res->height >> level;
            unsigned base = res->mip_offsets[level];
            if (!width) width = 1;
            if (!height) height = 1;
            unsigned row = width * resource_pixel_bytes(res);
            if (!row || row > limit) return -1;
            unsigned rows = limit / row;
            unsigned layers = res->array_layers ? res->array_layers : 1;
            for (unsigned layer = 0; layer < layers; layer++) {
                unsigned plane = base + layer * row * height;
                unsigned plane_end = plane + row * height;
                if (begin >= plane_end || end <= plane) continue;
                unsigned first = begin > plane ? (begin - plane) / row : 0;
                unsigned last = end < plane_end ? (end - plane - 1u) / row + 1u : height;
                for (unsigned y = first; y < last;) {
                    unsigned count = last - y;
                    if (count > rows) count = rows;
                    if (transfer_span(res, id, plane + y * row, count * row, y, count, level, layer)) return -1;
                    y += count;
                }
            }
        }
    }

    if (g_dev.batch_collect && g_dev.batch.count) res->upload_scheduled = 1;
    else {
        res->host_current = 1;
        res->dirty_begin = 0;
        res->dirty_end = 0;
    }
    return 0;
}

struct sval {
    int live;
    uint32_t n;
    uint32_t type;
    uint32_t bits[4];
};

struct run_in {
    const uint8_t *mod;
    uint32_t mod_len;
    struct resource *bind_res[16];
    uint32_t vertex_id;
    float vary[8][4];
    int vary_n[8];
    int vary_live[8];
    float ret[4];
    int ret_n;
    int ret_ok;
    float outv[8][4];
    int out_n[8];
    int out_live[8];
};

static uint32_t word_at(const uint8_t *mod, uint32_t index)
{
    return mx_r32(mod, index * 4u);
}

static float u32_f(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static uint32_t f_u32(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static unsigned unorm8(float value)
{
    int c;
    if (value <= 0.f)
        return 0;
    if (value >= 1.f)
        return 255;
    c = (int)(value * 255.f + 0.5f);
    if (c < 0)
        return 0;
    if (c > 255)
        return 255;
    return (unsigned)c;
}

static float edge(float ax, float ay, float bx, float by, float px, float py)
{
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static int covered_edge(float ax, float ay, float bx, float by, float px, float py, float den)
{
    float value = edge(ax, ay, bx, by, px, py);
    float dx = bx - ax;
    float dy = by - ay;
    if (den > 0) {
        value = -value;
    } else {
        dx = -dx;
        dy = -dy;
    }
    return value > 0 || (value == 0 && (dy < 0 || (dy == 0 && dx < 0)));
}

static int coverage(float ax, float ay, float bx, float by, float cx, float cy, float px, float py, float *w0, float *w1, float *w2)
{
    float den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (den > -0.000001f && den < 0.000001f)
        return 0;
    if (!covered_edge(ax, ay, bx, by, px, py, den) ||
        !covered_edge(bx, by, cx, cy, px, py, den) ||
        !covered_edge(cx, cy, ax, ay, px, py, den))
        return 0;
    *w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den;
    *w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den;
    *w2 = 1.0f - *w0 - *w1;
    if (*w0 < -0.001f || *w1 < -0.001f || *w2 < -0.001f)
        return 0;
    return 1;
}

static int sample_nearest(struct resource *tex, float u, float v, uint32_t *bits)
{
    int tx, ty;
    const unsigned char *src;
    unsigned i;
    if (!tex || !tex->bytes || tex->width == 0 || tex->height == 0)
        return -1;
    if (u < 0.f)
        u = 0.f;
    if (v < 0.f)
        v = 0.f;
    if (u > 1.f)
        u = 1.f;
    if (v > 1.f)
        v = 1.f;
    tx = (int)(u * (float)tex->width);
    ty = (int)(v * (float)tex->height);
    if (tx >= (int)tex->width)
        tx = (int)tex->width - 1;
    if (ty >= (int)tex->height)
        ty = (int)tex->height - 1;
    if (tx < 0)
        tx = 0;
    if (ty < 0)
        ty = 0;
    if ((unsigned)(ty * (int)tex->width + tx) * 4u + 4u > tex->size)
        return -1;
    src = tex->bytes + ((unsigned)ty * tex->width + (unsigned)tx) * 4u;
    for (i = 0; i < 4; i++)
        bits[i] = f_u32((float)src[i] / 255.f);
    return 0;
}

static int exec_block(struct run_in *in, uint32_t entry)
{
    uint32_t words, bindings, entries, blocks, cursor, block;
    struct sval vals[4096];
    if (!in->mod || (in->mod_len & 3u) || in->mod_len < MXSB_HEADER_WORDS * 4u)
        return -1;
    words = in->mod_len / 4u;
    bindings = word_at(in->mod, 3);
    entries = word_at(in->mod, 4);
    blocks = word_at(in->mod, 5);
    cursor = MXSB_HEADER_WORDS + bindings * MXSB_BINDING_WORDS + entries * MXSB_ENTRY_WORDS;
    if (cursor > words)
        return -1;
    for (block = 0; block < blocks; block++) {
        uint32_t block_words, record_count, ent, rec, at;
        if (cursor + 4 > words)
            return -1;
        ent = word_at(in->mod, cursor + 1);
        block_words = word_at(in->mod, cursor + 2);
        record_count = word_at(in->mod, cursor + 3);
        if (block_words < 4 || cursor + block_words > words)
            return -1;
        if (ent != entry) {
            cursor += block_words;
            continue;
        }
        memset(vals, 0, sizeof vals);
        at = cursor + 4;
        for (rec = 0; rec < record_count; rec++) {
            uint32_t head, count, opcode, result, i;
            if (at >= cursor + block_words)
                return -1;
            head = word_at(in->mod, at);
            count = head >> 16;
            opcode = head & 0xffffu;
            if (count == 0 || at + count > cursor + block_words)
                return -1;
            if (opcode == MXSB_OP_CONSTANT || opcode == MXSB_OP_BUILTIN) {
                if (count < 4)
                    return -1;
                result = word_at(in->mod, at + 1);
                if (result == 0 || result >= 4096)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = 1;
                if (opcode == MXSB_OP_CONSTANT)
                    vals[result].bits[0] = word_at(in->mod, at + 3);
                else if (word_at(in->mod, at + 3) == MXSB_BUILTIN_VERTEX_ID)
                    vals[result].bits[0] = in->vertex_id;
                else
                    return -1;
            } else if (opcode == MXSB_OP_ADD || opcode == MXSB_OP_SUB ||
                       opcode == MXSB_OP_MUL || opcode == MXSB_OP_DIV) {
                uint32_t left, right, type;
                if (count != 5)
                    return -1;
                result = word_at(in->mod, at + 1);
                type = word_at(in->mod, at + 2);
                left = word_at(in->mod, at + 3);
                right = word_at(in->mod, at + 4);
                if (result == 0 || result >= 4096 || left >= 4096 || right >= 4096 ||
                    !vals[left].live || !vals[right].live || vals[left].n == 0 ||
                    vals[left].n > 4 || vals[left].n != vals[right].n)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = vals[left].n;
                for (i = 0; i < vals[left].n; i++) {
                    if (type == MXSB_TYPE_U32 && opcode != MXSB_OP_DIV) {
                        uint32_t x = vals[left].bits[i], y = vals[right].bits[i];
                        vals[result].bits[i] = opcode == MXSB_OP_ADD ? x + y :
                                              opcode == MXSB_OP_SUB ? x - y : x * y;
                    } else if (type == MXSB_TYPE_F32 || type == MXSB_TYPE_F32X2 ||
                               type == MXSB_TYPE_F32X3 || type == MXSB_TYPE_F32X4) {
                        float x = u32_f(vals[left].bits[i]);
                        float y = u32_f(vals[right].bits[i]);
                        float value = opcode == MXSB_OP_ADD ? x + y :
                                      opcode == MXSB_OP_SUB ? x - y :
                                      opcode == MXSB_OP_MUL ? x * y : x / y;
                        vals[result].bits[i] = f_u32(value);
                    } else {
                        return -1;
                    }
                }
            } else if (opcode == MXSB_OP_BIT_AND || opcode == MXSB_OP_BIT_OR ||
                       opcode == MXSB_OP_BIT_XOR || opcode == MXSB_OP_SHL || opcode == MXSB_OP_SHR) {
                uint32_t left, right;
                if (count != 5)
                    return -1;
                result = word_at(in->mod, at + 1);
                left = word_at(in->mod, at + 3);
                right = word_at(in->mod, at + 4);
                if (!result || result >= 4096 || left >= 4096 || right >= 4096 ||
                    !vals[left].live || !vals[right].live || !vals[left].n ||
                    vals[left].n > 4 || vals[left].n != vals[right].n)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = vals[left].n;
                for (i = 0; i < vals[left].n; i++) {
                    uint32_t x = vals[left].bits[i], y = vals[right].bits[i];
                    switch (opcode) {
                    case MXSB_OP_BIT_AND: vals[result].bits[i] = x & y; break;
                    case MXSB_OP_BIT_OR: vals[result].bits[i] = x | y; break;
                    case MXSB_OP_BIT_XOR: vals[result].bits[i] = x ^ y; break;
                    case MXSB_OP_SHL:
                        if (y >= 32) return -1;
                        vals[result].bits[i] = x << y;
                        break;
                    case MXSB_OP_SHR:
                        if (y >= 32) return -1;
                        vals[result].bits[i] = x >> y;
                        break;
                    default: return -1;
                    }
                }
            } else if (opcode == MXSB_OP_BIT_NOT || opcode == MXSB_OP_I2F || opcode == MXSB_OP_U2F ||
                       opcode == MXSB_OP_F2I || opcode == MXSB_OP_F2U) {
                uint32_t src;
                if (count != 4)
                    return -1;
                result = word_at(in->mod, at + 1);
                src = word_at(in->mod, at + 3);
                if (!result || result >= 4096 || src >= 4096 || !vals[src].live || !vals[src].n || vals[src].n > 4)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = vals[src].n;
                for (i = 0; i < vals[src].n; i++) {
                    uint32_t bits = vals[src].bits[i];
                    if (opcode == MXSB_OP_BIT_NOT)
                        vals[result].bits[i] = ~bits;
                    else if (opcode == MXSB_OP_I2F) {
                        int64_t value = (int64_t)bits - ((bits & 0x80000000u) ? 0x100000000ll : 0);
                        vals[result].bits[i] = f_u32((float)value);
                    } else if (opcode == MXSB_OP_U2F)
                        vals[result].bits[i] = f_u32((float)bits);
                    else {
                        double value = trunc((double)u32_f(bits));
                        if (!isfinite(value) ||
                            (opcode == MXSB_OP_F2I ? value < -2147483648.0 || value >= 2147483648.0 :
                                                   value < 0.0 || value >= 4294967296.0))
                            return -1;
                        vals[result].bits[i] = opcode == MXSB_OP_F2I ? (uint32_t)(int64_t)value : (uint32_t)value;
                    }
                }
            } else if (opcode == MXSB_OP_EQ || opcode == MXSB_OP_LT || opcode == MXSB_OP_LE) {
                uint32_t left, right;
                if (count != 5)
                    return -1;
                result = word_at(in->mod, at + 1);
                left = word_at(in->mod, at + 3);
                right = word_at(in->mod, at + 4);
                if (!result || result >= 4096 || left >= 4096 || right >= 4096 ||
                    !vals[left].live || !vals[right].live || vals[left].n != 1 || vals[right].n != 1)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = 1;
                if (vals[left].type == MXSB_TYPE_U32 && vals[right].type == MXSB_TYPE_U32) {
                    uint32_t x = vals[left].bits[0], y = vals[right].bits[0];
                    vals[result].bits[0] = opcode == MXSB_OP_EQ ? x == y : opcode == MXSB_OP_LT ? x < y : x <= y;
                } else {
                    float x = u32_f(vals[left].bits[0]), y = u32_f(vals[right].bits[0]);
                    vals[result].bits[0] = opcode == MXSB_OP_EQ ? x == y : opcode == MXSB_OP_LT ? x < y : x <= y;
                }
            } else if (opcode == MXSB_OP_SELECT) {
                uint32_t cond, yes, no, selected;
                if (count != 6)
                    return -1;
                result = word_at(in->mod, at + 1);
                cond = word_at(in->mod, at + 3);
                yes = word_at(in->mod, at + 4);
                no = word_at(in->mod, at + 5);
                if (!result || result >= 4096 || cond >= 4096 || yes >= 4096 || no >= 4096 ||
                    !vals[cond].live || vals[cond].n != 1 || !vals[yes].live || !vals[no].live ||
                    vals[yes].n != vals[no].n || !vals[yes].n || vals[yes].n > 4)
                    return -1;
                selected = vals[cond].bits[0] ? yes : no;
                vals[result] = vals[selected];
                vals[result].type = word_at(in->mod, at + 2);
            } else if (opcode == MXSB_OP_NOT || opcode == MXSB_OP_FABS || opcode == MXSB_OP_FLOOR ||
                       opcode == MXSB_OP_CEIL || opcode == MXSB_OP_FRACT || opcode == MXSB_OP_SQRT ||
                       opcode == MXSB_OP_RSQ || opcode == MXSB_OP_RCP || opcode == MXSB_OP_EXP2 ||
                       opcode == MXSB_OP_LOG2 || opcode == MXSB_OP_SIN || opcode == MXSB_OP_COS) {
                uint32_t src;
                if (count != 4)
                    return -1;
                result = word_at(in->mod, at + 1);
                src = word_at(in->mod, at + 3);
                if (!result || result >= 4096 || src >= 4096 || !vals[src].live || !vals[src].n || vals[src].n > 4)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = vals[src].n;
                for (i = 0; i < vals[src].n; i++) {
                    float x = u32_f(vals[src].bits[i]), value;
                    if (opcode == MXSB_OP_NOT) {
                        vals[result].bits[i] = !vals[src].bits[i];
                        continue;
                    }
                    switch (opcode) {
                    case MXSB_OP_FABS: value = fabsf(x); break;
                    case MXSB_OP_FLOOR: value = floorf(x); break;
                    case MXSB_OP_CEIL: value = ceilf(x); break;
                    case MXSB_OP_FRACT: value = x - floorf(x); break;
                    case MXSB_OP_SQRT: value = sqrtf(x); break;
                    case MXSB_OP_RSQ: value = 1.f / sqrtf(x); break;
                    case MXSB_OP_RCP: value = 1.f / x; break;
                    case MXSB_OP_EXP2: value = exp2f(x); break;
                    case MXSB_OP_LOG2: value = log2f(x); break;
                    case MXSB_OP_SIN: value = sinf(x); break;
                    case MXSB_OP_COS: value = cosf(x); break;
                    default: return -1;
                    }
                    vals[result].bits[i] = f_u32(value);
                }
            } else if (opcode == MXSB_OP_NEG || opcode == MXSB_OP_BITCAST) {
                uint32_t src, type;
                if (count != 4)
                    return -1;
                result = word_at(in->mod, at + 1);
                type = word_at(in->mod, at + 2);
                src = word_at(in->mod, at + 3);
                if (result == 0 || result >= 4096 || src >= 4096 || !vals[src].live ||
                    vals[src].n == 0 || vals[src].n > 4)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = vals[src].n;
                for (i = 0; i < vals[src].n; i++) {
                    if (opcode == MXSB_OP_BITCAST)
                        vals[result].bits[i] = vals[src].bits[i];
                    else if (type == MXSB_TYPE_U32)
                        vals[result].bits[i] = 0u - vals[src].bits[i];
                    else if (type == MXSB_TYPE_F32 || type == MXSB_TYPE_F32X2 ||
                             type == MXSB_TYPE_F32X3 || type == MXSB_TYPE_F32X4)
                        vals[result].bits[i] = f_u32(-u32_f(vals[src].bits[i]));
                    else
                        return -1;
                }
            } else if (opcode == MXSB_OP_EXTRACT) {
                uint32_t src, lane;
                if (count < 5)
                    return -1;
                result = word_at(in->mod, at + 1);
                src = word_at(in->mod, at + 3);
                lane = word_at(in->mod, at + 4);
                if (result == 0 || result >= 4096 || src >= 4096 || !vals[src].live || lane >= vals[src].n)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = 1;
                vals[result].bits[0] = vals[src].bits[lane];
            } else if (opcode == MXSB_OP_BUFFER_LOAD) {
                uint32_t binding, index_id, index;
                struct resource *res;
                if (count < 5)
                    return -1;
                result = word_at(in->mod, at + 1);
                binding = word_at(in->mod, at + 3);
                index_id = word_at(in->mod, at + 4);
                if (result == 0 || result >= 4096 || binding >= 16 || index_id >= 4096 || !vals[index_id].live)
                    return -1;
                res = in->bind_res[binding];
                index = vals[index_id].bits[0];
                if (!res || !res->bytes || (uint64_t)index * 16ull + 16ull > res->size)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = 4;
                for (i = 0; i < 4; i++)
                    vals[result].bits[i] = mx_r32(res->bytes, index * 16u + i * 4u);
            } else if (opcode == MXSB_OP_CONSTRUCT) {
                uint32_t lanes;
                if (count < 4)
                    return -1;
                result = word_at(in->mod, at + 1);
                lanes = word_at(in->mod, at + 3);
                if (result == 0 || result >= 4096 || lanes == 0 || lanes > 4 || count != 4u + lanes)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = lanes;
                for (i = 0; i < lanes; i++) {
                    uint32_t src = word_at(in->mod, at + 4u + i);
                    if (src >= 4096 || !vals[src].live)
                        return -1;
                    vals[result].bits[i] = vals[src].bits[0];
                }
            } else if (opcode == MXSB_OP_STAGE_OUTPUT) {
                uint32_t loc, src;
                if (count < 3)
                    return -1;
                loc = word_at(in->mod, at + 1);
                src = word_at(in->mod, at + 2);
                if (loc >= 8 || src >= 4096 || !vals[src].live || vals[src].n > 4)
                    return -1;
                in->out_live[loc] = 1;
                in->out_n[loc] = (int)vals[src].n;
                for (i = 0; i < vals[src].n; i++)
                    in->outv[loc][i] = u32_f(vals[src].bits[i]);
            } else if (opcode == MXSB_OP_STAGE_INPUT) {
                uint32_t loc;
                if (count < 5)
                    return -1;
                result = word_at(in->mod, at + 1);
                loc = word_at(in->mod, at + 3);
                if (result == 0 || result >= 4096 || loc >= 8 || !in->vary_live[loc] || in->vary_n[loc] < 1 || in->vary_n[loc] > 4)
                    return -1;
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = (uint32_t)in->vary_n[loc];
                for (i = 0; i < vals[result].n; i++)
                    vals[result].bits[i] = f_u32(in->vary[loc][i]);
            } else if (opcode == MXSB_OP_TEXTURE_SAMPLE) {
                uint32_t binding, coord, control;
                struct resource *tex;
                if (count < 6)
                    return -1;
                result = word_at(in->mod, at + 1);
                binding = word_at(in->mod, at + 3);
                coord = word_at(in->mod, at + 4);
                control = word_at(in->mod, at + 5);
                if (result == 0 || result >= 4096 || binding >= 16 || coord >= 4096 || !vals[coord].live || vals[coord].n < 2)
                    return -1;
                if (control != 0x1111u)
                    return -1;
                tex = in->bind_res[binding];
                vals[result].live = 1;
                vals[result].type = word_at(in->mod, at + 2);
                vals[result].n = 4;
                if (sample_nearest(tex, u32_f(vals[coord].bits[0]), u32_f(vals[coord].bits[1]), vals[result].bits) != 0)
                    return -1;
            } else if (opcode == MXSB_OP_RETURN_VALUE) {
                uint32_t src;
                if (count < 2)
                    return -1;
                src = word_at(in->mod, at + 1);
                if (src >= 4096 || !vals[src].live || vals[src].n == 0 || vals[src].n > 4)
                    return -1;
                in->ret_ok = 1;
                in->ret_n = (int)vals[src].n;
                for (i = 0; i < vals[src].n; i++)
                    in->ret[i] = u32_f(vals[src].bits[i]);
            } else {
                return -1;
            }
            at += count;
        }
        if (at != cursor + block_words)
            return -1;
        return 0;
    }
    return -1;
}

static int map_shader_bindings(const uint8_t *mod, const struct mxgpu_execution_binding *eb, uint32_t eb_n, struct resource **out)
{
    uint32_t bindings, i, j;
    bindings = word_at(mod, 3);
    for (i = 0; i < 16; i++)
        out[i] = NULL;
    for (i = 0; i < bindings; i++) {
        uint32_t base = MXSB_HEADER_WORDS + i * MXSB_BINDING_WORDS;
        uint32_t id = word_at(mod, base);
        uint32_t slot = word_at(mod, base + 1) & 0xffffu;
        if (id == 0 || id >= 16)
            return -1;
        for (j = 0; j < eb_n; j++) {
            if (eb[j].slot == slot)
                out[id] = res_slot(eb[j].resource_id);
        }
        if (!out[id])
            return -1;
    }
    return 0;
}

static int shade_frame(const uint8_t *mod, uint32_t mod_len, uint32_t vs_entry, uint32_t fs_entry, const struct mxgpu_execution_binding *eb, uint32_t eb_n, const struct mxgpu_render_submit *sub, struct resource *color)
{
    struct resource *mapped[16];
    struct vs_out {
        float x, y, w;
        float var[8][4];
        int var_n[8];
        int var_live[8];
    } vs[3];
    uint32_t tri, count;
    if (map_shader_bindings(mod, eb, eb_n, mapped) != 0)
        return -1;
    if (!color || !color->bytes || color->width == 0 || color->height == 0 || color->width > 64 || color->height > 64)
        return -1;
    if ((uint64_t)color->width * color->height * 4ull > color->size)
        return -1;
    if (sub->load_action == MXGPU_LOAD_CLEAR) {
        unsigned y, x, c;
        for (y = 0; y < color->height; y++) {
            for (x = 0; x < color->width; x++) {
                unsigned char *dst = color->bytes + ((size_t)y * color->width + x) * 4u;
                for (c = 0; c < 4; c++)
                    dst[c] = (unsigned char)unorm8(u32_f(sub->clear_rgba[c]));
            }
        }
    }
    count = sub->element_count;
    if (count < 3 || (count % 3u) != 0 || count > 96)
        return -1;
    for (tri = 0; tri < count / 3u; tri++) {
        int v;
        unsigned y, x;
        for (v = 0; v < 3; v++) {
            struct run_in in;
            int32_t vid = (int32_t)sub->element_start + (int32_t)(tri * 3u + (uint32_t)v) + sub->base_vertex;
            memset(&in, 0, sizeof in);
            if (vid < 0)
                return -1;
            in.mod = mod;
            in.mod_len = mod_len;
            memcpy(in.bind_res, mapped, sizeof mapped);
            in.vertex_id = (uint32_t)vid;
            if (exec_block(&in, vs_entry) != 0 || !in.ret_ok || in.ret_n < 4 || in.ret[3] == 0.f)
                return -1;
            vs[v].x = in.ret[0] / in.ret[3];
            vs[v].y = in.ret[1] / in.ret[3];
            vs[v].w = in.ret[3];
            memcpy(vs[v].var, in.outv, sizeof vs[v].var);
            memcpy(vs[v].var_n, in.out_n, sizeof vs[v].var_n);
            memcpy(vs[v].var_live, in.out_live, sizeof vs[v].var_live);
        }
        for (y = 0; y < color->height; y++) {
            for (x = 0; x < color->width; x++) {
                float px = ((x + 0.5f) / (float)color->width) * 2.f - 1.f;
                float py = ((y + 0.5f) / (float)color->height) * 2.f - 1.f;
                float w0, w1, w2, iw0, iw1, iw2, isum;
                struct run_in fin;
                unsigned char *dst;
                int loc, k;
                if (!coverage(vs[0].x, vs[0].y, vs[1].x, vs[1].y, vs[2].x, vs[2].y, px, py, &w0, &w1, &w2))
                    continue;
                iw0 = w0 / vs[0].w;
                iw1 = w1 / vs[1].w;
                iw2 = w2 / vs[2].w;
                isum = iw0 + iw1 + iw2;
                if (isum > -0.00000001f && isum < 0.00000001f)
                    continue;
                memset(&fin, 0, sizeof fin);
                fin.mod = mod;
                fin.mod_len = mod_len;
                memcpy(fin.bind_res, mapped, sizeof mapped);
                for (loc = 0; loc < 8; loc++) {
                    int n;
                    if (!vs[0].var_live[loc] || !vs[1].var_live[loc] || !vs[2].var_live[loc])
                        continue;
                    n = vs[0].var_n[loc];
                    if (n != vs[1].var_n[loc] || n != vs[2].var_n[loc] || n < 1 || n > 4)
                        continue;
                    fin.vary_live[loc] = 1;
                    fin.vary_n[loc] = n;
                    for (k = 0; k < n; k++)
                        fin.vary[loc][k] = (vs[0].var[loc][k] * iw0 + vs[1].var[loc][k] * iw1 + vs[2].var[loc][k] * iw2) / isum;
                }
                if (exec_block(&fin, fs_entry) != 0 || !fin.ret_ok || fin.ret_n < 4)
                    return -1;
                dst = color->bytes + ((size_t)y * color->width + x) * 4u;
                for (k = 0; k < 4; k++)
                    dst[k] = (unsigned char)unorm8(fin.ret[k]);
            }
        }
    }
    return 0;
}

static int apply_decoded(uint16_t opcode, const uint8_t *payload, uint32_t len)
{
    if (opcode == MXGPU_OP_CONTEXT_CREATE)
        return 0;
    if (opcode == MXGPU_OP_RESOURCE_CREATE) {
        struct mxgpu_resource_create rec;
        if (mxgpu_resource_create_decode(payload, len, 64ull << 20, &rec) != MX_OK)
            return -1;
        return res_slot(rec.resource_id) ? 0 : -1;
    }
    if (opcode == MXGPU_OP_RESOURCE_DESTROY) {
        uint32_t id;
        struct resource *res;
        if (mxgpu_resource_id_decode(payload, len, &id) != MX_OK || !(res = res_slot(id)) || !res->host_live)
            return -1;
        res->host_live = 0;
        res->host_bytes = 0;
        res->host_width = 0;
        res->host_height = 0;
        res->host_current = 0;
        return 0;
    }
    if (opcode == MXGPU_OP_TRANSFER_TO_HOST) {
        struct mxgpu_transfer transfer;
        const uint8_t *data = NULL;
        struct resource *res;
        if (mxgpu_transfer_decode(payload, len, &transfer, &data) != MX_OK || !data)
            return -1;
        res = res_slot(transfer.resource_id);
        if (!res || !res->bytes)
            return -1;
        if (res->kind == MXGPU_KIND_BUFFER) {
            if (transfer.resource_offset > res->size ||
                transfer.data_bytes > res->size - transfer.resource_offset)
                return -1;
            memmove(res->bytes + (size_t)transfer.resource_offset, data, transfer.data_bytes);
        } else if (res->kind == MXGPU_KIND_TEXTURE_2D) {
            uint64_t row, stride, offset, source_span, target_span;
            unsigned y;
            if (transfer.mip_level || transfer.array_layer || transfer.z ||
                transfer.resource_offset || transfer.depth != 1 ||
                !transfer.width || !transfer.height ||
                transfer.x > res->width || transfer.width > res->width - transfer.x ||
                transfer.y > res->height || transfer.height > res->height - transfer.y)
                return -1;
            row = (uint64_t)transfer.width * 4u;
            stride = (uint64_t)res->width * 4u;
            offset = (uint64_t)transfer.y * stride + (uint64_t)transfer.x * 4u;
            source_span = (uint64_t)(transfer.height - 1u) * transfer.row_bytes + row;
            target_span = (uint64_t)(transfer.height - 1u) * stride + row;
            if (transfer.row_bytes < row || source_span > transfer.data_bytes ||
                offset > res->size || target_span > res->size - offset)
                return -1;
            for (y = 0; y < transfer.height; y++)
                memmove(res->bytes + (size_t)(offset + y * stride),
                        data + (size_t)y * transfer.row_bytes, (size_t)row);
        } else {
            return -1;
        }
        return 0;
    }
    if (opcode == MXGPU_OP_PIPELINE_DESTROY || opcode == MXGPU_OP_SHADER_DESTROY) {
        uint32_t id;
        if (mxgpu_resource_id_decode(payload, len, &id) != MX_OK || id >= 16)
            return -1;
        if (opcode == MXGPU_OP_PIPELINE_DESTROY)
            memset(&g_dev.pipes[id], 0, sizeof g_dev.pipes[id]);
        else
            memset(&g_dev.shaders[id], 0, sizeof g_dev.shaders[id]);
        return 0;
    }
    if (opcode == MXGPU_OP_SHADER_CREATE) {
        uint32_t id, blen;
        struct mxsb_limits limits;
        if (!payload || len < MXGPU_SHADER_CREATE_HEADER_SIZE)
            return -1;
        id = mx_r32(payload, 0);
        blen = mx_r32(payload, 4);
        if (id == 0 || id >= 16 || blen == 0 || (blen & 3u) || MXGPU_SHADER_CREATE_HEADER_SIZE + blen != len || blen > sizeof g_dev.shaders[id].bytes)
            return -1;
        mxsb_limits_default(&limits);
        if (mxsb_verify(payload + MXGPU_SHADER_CREATE_HEADER_SIZE, blen, &limits) != MXSB_OK)
            return -1;
        memcpy(g_dev.shaders[id].bytes, payload + MXGPU_SHADER_CREATE_HEADER_SIZE, blen);
        g_dev.shaders[id].len = blen;
        g_dev.shaders[id].live = 1;
        return 0;
    }
    if (opcode == MXGPU_OP_PIPELINE_CREATE) {
        uint32_t id, shader, first, second;
        if (!payload || len < MXGPU_PIPELINE_CREATE_SIZE)
            return -1;
        id = mx_r32(payload, 0);
        shader = mx_r32(payload, 8);
        first = mx_r32(payload, 12);
        second = mx_r32(payload, 16);
        if (id == 0 || id >= 16 || shader == 0 || shader >= 16 || !g_dev.shaders[shader].live || first == 0 || second == 0)
            return -1;
        g_dev.pipes[id].live = 1;
        g_dev.pipes[id].shader = shader;
        g_dev.pipes[id].vs_entry = first;
        g_dev.pipes[id].fs_entry = second;
        return 0;
    }
    if (opcode == MXGPU_OP_RENDER_SUBMIT) {
        struct mxgpu_render_submit sub;
        if (g_dev.fd >= 0 || !g_dev.allow_executor)
            return g_dev.fd >= 0 ? 0 : -1;
        struct mxgpu_execution_binding binds[8];
        struct pipeline_image *pipe;
        struct shader_image *shader;
        struct resource *color;
        if (mxgpu_render_submit_decode(payload, len, &sub, binds, 8) != MX_OK)
            return -1;
        if (sub.pipeline_id >= 16 || sub.store_action != MXGPU_STORE_STORE)
            return -1;
        pipe = &g_dev.pipes[sub.pipeline_id];
        if (!pipe->live || pipe->shader == 0 || pipe->shader >= 16)
            return -1;
        shader = &g_dev.shaders[pipe->shader];
        if (!shader->live)
            return -1;
        color = res_slot(sub.color_target_id);
        return shade_frame(shader->bytes, shader->len, pipe->vs_entry, pipe->fs_entry, binds, sub.binding_count, &sub, color);
    }
    return -1;
}

static uint8_t g_user_mod[MXGPU_LINK_MODULE_CAPACITY];
static uint32_t g_user_len;
static unsigned g_draw_verts;

static int native_pipeline_entry_destroy(struct native_pipeline_cache_entry *entry)
{
    uint8_t payload[MXGPU_RESOURCE_ID_SIZE];
    uint32_t bytes;
    if (entry->pipeline_live) {
        if (mxgpu_resource_id_encode(entry->pipeline_id, payload, sizeof payload, &bytes) != MX_OK ||
            winsys_submit_ioctl(MXGPU_OP_PIPELINE_DESTROY, MXGPU_QUEUE_CONTROL, g_dev.context, payload, bytes)) {
            g_dev.lost = 1;
            return -1;
        }
        entry->pipeline_live = 0;
    }
    if (entry->shader_live) {
        if (mxgpu_resource_id_encode(entry->shader_id, payload, sizeof payload, &bytes) != MX_OK ||
            winsys_submit_ioctl(MXGPU_OP_SHADER_DESTROY, MXGPU_QUEUE_CONTROL, g_dev.context, payload, bytes)) {
            g_dev.lost = 1;
            return -1;
        }
        entry->shader_live = 0;
    }
    free(entry->module);
    memset(entry, 0, sizeof *entry);
    return 0;
}

static int ensure_native_pipeline(void)
{
    struct native_pipeline_cache_entry *entry = NULL, *victim = NULL;
    uint8_t payload[64];
    uint32_t bytes;
    unsigned i;
    if (g_dev.lost) return -1;
    for (i = 0; i < 8; i++) {
        struct native_pipeline_cache_entry *candidate = &g_dev.module_cache[i];
        if (candidate->module && candidate->bytes == g_user_len &&
            memcmp(candidate->module, g_user_mod, g_user_len) == 0) {
            entry = candidate;
            break;
        }
        if (!candidate->module || !victim || (victim->module && candidate->used < victim->used))
            victim = candidate;
    }
    if (!entry) {
        uint8_t *module;
        if (!victim || g_dev.module_cache_next_id == UINT32_MAX) return -1;
        module = malloc(g_user_len);
        if (!module) return -1;
        memcpy(module, g_user_mod, g_user_len);
        if (native_pipeline_entry_destroy(victim)) {
            free(module);
            return -1;
        }
        entry = victim;
        entry->module = module;
        entry->bytes = g_user_len;
        entry->shader_id = entry->pipeline_id = ++g_dev.module_cache_next_id;
    }
    if (!entry->shader_live) {
        uint32_t capacity = entry->bytes + MXGPU_SHADER_CREATE_HEADER_SIZE;
        uint8_t *shader_payload = malloc(capacity);
        int result;
        if (!shader_payload) return -1;
        result = mxgpu_shader_create_encode(entry->shader_id, entry->module, entry->bytes,
                                            shader_payload, capacity, &bytes);
        if (!result) result = winsys_submit_ioctl(MXGPU_OP_SHADER_CREATE, MXGPU_QUEUE_CONTROL,
                                                 g_dev.context, shader_payload, bytes);
        free(shader_payload);
        if (result) return -1;
        entry->shader_live = 1;
    }
    if (!entry->pipeline_live) {
        if (mxgpu_pipeline_create_encode(entry->pipeline_id, MXGPU_PIPELINE_RENDER,
                                         MXGPU_FMT_RGBA8_UNORM, entry->shader_id, 1, 2,
                                         payload, sizeof payload, &bytes) != MX_OK ||
            winsys_submit_ioctl(MXGPU_OP_PIPELINE_CREATE, MXGPU_QUEUE_CONTROL, g_dev.context, payload, bytes))
            return -1;
        entry->pipeline_live = 1;
    }
    entry->used = ++g_dev.module_cache_clock;
    g_dev.shader_id = entry->shader_id;
    g_dev.pipeline_id = entry->pipeline_id;
    g_dev.shader_live = g_dev.pipeline_live = 0;
    g_dev.pipeline_dirty = 0;
    return 0;
}

static int ensure_pipeline(void)
{
    uint8_t payload[64];
    uint32_t n = 0;
    struct mxsb_limits limits;
    if (g_dev.fd < 0 && g_dev.pipeline_live && !g_dev.pipeline_dirty)
        return 0;
    if (!g_user_len)
        return -1;
    mxsb_limits_default(&limits);
    if (mxsb_verify(g_user_mod, g_user_len, &limits) != MXSB_OK)
        return -1;
    if (g_dev.fd >= 0)
        return ensure_native_pipeline();
    if (g_dev.pipeline_dirty) {
        if (g_dev.pipeline_live) {
            if (mxgpu_resource_id_encode(g_dev.pipeline_id, payload, sizeof payload, &n) != MX_OK ||
                winsys_submit_ioctl(MXGPU_OP_PIPELINE_DESTROY, MXGPU_QUEUE_CONTROL,
                                    g_dev.context, payload, n) != MX_OK)
                return -1;
            g_dev.pipeline_live = 0;
        }
        if (g_dev.shader_live) {
            if (mxgpu_resource_id_encode(g_dev.shader_id, payload, sizeof payload, &n) != MX_OK ||
                winsys_submit_ioctl(MXGPU_OP_SHADER_DESTROY, MXGPU_QUEUE_CONTROL,
                                    g_dev.context, payload, n) != MX_OK)
                return -1;
            g_dev.shader_live = 0;
        }
        g_dev.pipeline_dirty = 0;
    }
    if (!g_dev.shader_id) {
        if (!g_dev.next_id || g_dev.next_id >= 15)
            return -1;
        g_dev.shader_id = g_dev.next_id++;
        g_dev.pipeline_id = g_dev.next_id++;
    }
    if (!g_dev.shader_live) {
        uint32_t capacity = g_user_len + MXGPU_SHADER_CREATE_HEADER_SIZE;
        uint8_t *shader_payload = malloc(capacity);
        int result;
        if (!shader_payload)
            return -1;
        result = mxgpu_shader_create_encode(g_dev.shader_id, g_user_mod, g_user_len,
                                            shader_payload, capacity, &n);
        if (!result)
            result = winsys_submit_ioctl(MXGPU_OP_SHADER_CREATE, MXGPU_QUEUE_CONTROL,
                                         g_dev.context, shader_payload, n);
        free(shader_payload);
        if (result)
            return -1;
        g_dev.shader_live = 1;
    }
    if (mxgpu_pipeline_create_encode(g_dev.pipeline_id, MXGPU_PIPELINE_RENDER,
                                     MXGPU_FMT_RGBA8_UNORM, g_dev.shader_id, 1, 2,
                                     payload, sizeof payload, &n) != MX_OK ||
        winsys_submit_ioctl(MXGPU_OP_PIPELINE_CREATE, MXGPU_QUEUE_CONTROL,
                            g_dev.context, payload, n) != MX_OK)
        return -1;
    g_dev.pipeline_live = 1;
    return 0;
}

#define MXGPU_READBACK_LIMIT (60u * 1024u)

static unsigned readback_limit(void)
{
    return g_dev.transfer_limits.max_transfer_from_host_bytes ?
           g_dev.transfer_limits.max_transfer_from_host_bytes : MXGPU_READBACK_LIMIT;
}

static int readback_rows(struct resource *color, unsigned y0, unsigned rows)
{
    struct mxgpu_command_header header;
    struct mxgpu_drm_user user;
    uint8_t request[MXGPU_TRANSFER_REQUEST_SIZE];
    uint8_t command[128];
    uint8_t *buf;
    unsigned row;
    unsigned bytes;
    unsigned capacity;
    unsigned i;
    uint32_t command_len = 0;
    uint32_t record_len = 0;
    int posted;

    if (g_dev.lost || !color || !color->bytes || !color->width || !rows ||
        y0 >= color->height || rows > color->height - y0 ||
        color->width > readback_limit() / resource_pixel_bytes(color))
        return -1;
    row = color->width * resource_pixel_bytes(color);
    if (rows > readback_limit() / row)
        return -1;
    bytes = rows * row;
    capacity = bytes > sizeof command + 512u ? bytes : sizeof command + 512u;
    buf = scratch_reserve(&g_dev.readback_scratch, capacity);
    if (!buf)
        return -1;
    memset(request, 0, sizeof request);
    mx_w32(request, 0, (uint32_t)(color - g_dev.resources));
    mx_w32(request, 12, y0);
    mx_w32(request, 20, color->width);
    mx_w32(request, 24, rows);
    mx_w32(request, 28, 1);
    mx_w32(request, 40, row);
    mx_w32(request, 44, bytes);
    memset(&header, 0, sizeof header);
    header.opcode = MXGPU_OP_TRANSFER_FROM_HOST;
    header.flags = MXGPU_CMD_SIGNAL_FENCE | MXGPU_CMD_RESPONSE_REQUIRED;
    header.context_id = g_dev.context;
    header.queue = MXGPU_QUEUE_TRANSFER;
    header.sequence = take_sequence();
    header.fence_value = header.sequence;
    if (mxgpu_command_encode(&header, request, MXGPU_TRANSFER_REQUEST_SIZE,
                              g_dev.caps.max_command_bytes ? g_dev.caps.max_command_bytes : 65536, command, sizeof command, &command_len) != MX_OK) {
        return -1;
    }
    if (mxgpu_drm_submit_encode(g_dev.context, MXGPU_QUEUE_TRANSFER, header.fence_value, bytes, command, command_len, buf, capacity, &record_len) != MXGPU_DRM_OK) {
        return -1;
    }
    memset(&user, 0, sizeof user);
    user.pointer = (uint64_t)(uintptr_t)buf;
    user.size = record_len;
    user.capacity = capacity;
    posted = mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + 5, struct mxgpu_drm_user), &user);
    if (posted < 0 && (errno == EPIPE || errno == ENODEV || errno == ETIMEDOUT))
        g_dev.lost = 1;
    if (posted < 0 || user.size != bytes) {
        return -1;
    }
    for (i = 0; i < rows; i++)
        memcpy(color->bytes + (size_t)(g_dev.native_active ? y0 + i : color->height - 1u - (y0 + i)) * row,
               buf + (size_t)i * row, row);
    return 0;
}

static int readback_attachment(struct resource *resource)
{
    unsigned row, rows_per, first = 0, end;

    if (!resource || !resource->bytes || !resource->size || !resource->width || !resource->height || g_dev.fd < 0)
        return -1;
    end = resource->height;
    row = resource->width * resource_pixel_bytes(resource);
    if (!row || row > readback_limit())
        return -1;
    rows_per = readback_limit() / row;
    if (g_dev.native_active && g_dev.native_state.rasterizer.scissor_enable) {
        const struct mxgpu_scissor *scissor = &g_dev.native_state.scissor;
        first = scissor->top < end ? scissor->top : end;
        end = scissor->bottom < end ? scissor->bottom : end;
        if (scissor->left >= resource->width || scissor->right <= scissor->left || end < first)
            end = first;
    }
    for (unsigned y = first; y < end; y += rows_per) {
        unsigned rows = end - y;
        if (rows > rows_per)
            rows = rows_per;
        if (readback_rows(resource, y, rows) != 0) {
            if (g_dev.native_active)
                resource->host_current = 0;
            return -1;
        }
    }
    if (g_dev.native_active)
        resource->host_current = 1;
    return 0;
}

static int readback_color(void)
{
    return readback_attachment(res_slot(g_dev.color_id));
}

static int framebuffer_sync_unlocked(struct mxgpu_framebuffer *framebuffer)
{
    if (batch_drain_unlocked() || !framebuffer || framebuffer->error) return -1;
    if (!framebuffer->pending) return 0;
    if (!framebuffer->resource_id || g_dev.lost || g_dev.fd < 0) return -1;
    struct resource *color = res_slot(framebuffer->resource_id);
    if (!color || color->width != framebuffer->width || color->height != framebuffer->height ||
        color->size != framebuffer->size) return -1;
    int native = g_dev.native_active;
    uint8_t scissor = g_dev.native_state.rasterizer.scissor_enable;
    g_dev.native_active = 1;
    g_dev.native_state.rasterizer.scissor_enable = 0;
    int result = readback_attachment(color);
    g_dev.native_state.rasterizer.scissor_enable = scissor;
    g_dev.native_active = native;
    if (result) {
        framebuffer->error = 1;
        color->host_current = 0;
        return result;
    }
    memcpy(framebuffer->pixels, color->bytes, framebuffer->size);
    framebuffer->pending = 0;
    return 0;
}

static int framebuffer_detach_unlocked(void)
{
    g_dev.framebuffer = NULL;
    g_dev.color_id = g_dev.immediate_color_id;
    return 0;
}

static int framebuffer_release_resource_unlocked(struct mxgpu_framebuffer *framebuffer)
{
    if (!framebuffer || !framebuffer->resource_id) return 0;
    if (framebuffer_sync_unlocked(framebuffer)) return -1;
    unsigned id = framebuffer->resource_id;
    struct resource *resource = res_slot(id);
    if (!resource) return -1;
    if (resource->host_live && destroy_resource(id)) { framebuffer->resource_blocked = 1; return -1; }
    free(resource->bytes);
    memset(resource, 0, sizeof *resource);
    framebuffer->resource_id = 0;
    framebuffer->resource_blocked = 0;
    if (g_dev.framebuffer == framebuffer) {
        g_dev.framebuffer = NULL;
        g_dev.color_id = g_dev.immediate_color_id;
    }
    for (unsigned i = 0; i < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; i++)
        if (g_dev.framebuffer_cache[i] == framebuffer) g_dev.framebuffer_cache[i] = NULL;
    return 0;
}

static int framebuffer_sample_pinned(const struct mxgpu_framebuffer *framebuffer)
{
    if (framebuffer && framebuffer->resource_id && g_dev.resources[framebuffer->resource_id].batch_pinned) return 1;
    for (unsigned i = 0; i < g_dev.draw_texture_count; i++)
        if (g_dev.draw_textures[i].framebuffer == framebuffer) return 1;
    return 0;
}

static int framebuffer_evict_unlocked(struct mxgpu_framebuffer *preserve)
{
    if (batch_drain_unlocked()) return -1;
    struct mxgpu_framebuffer *victim = NULL;
    for (unsigned i = 0; i < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; i++) {
        struct mxgpu_framebuffer *candidate = g_dev.framebuffer_cache[i];
        if (candidate && candidate != preserve && !framebuffer_sample_pinned(candidate) && (!victim || candidate->used < victim->used))
            victim = candidate;
    }
    return victim ? framebuffer_release_resource_unlocked(victim) : -1;
}

static int framebuffer_trim_bytes_unlocked(struct mxgpu_framebuffer *preserve)
{
    for (unsigned pass = 0; pass < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; pass++) {
        uint64_t retained = preserve ? preserve->size : 0;
        struct mxgpu_framebuffer *victim = NULL;
        for (unsigned i = 0; i < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; i++) {
            struct mxgpu_framebuffer *candidate = g_dev.framebuffer_cache[i];
            if (!candidate || candidate == preserve) continue;
            retained += candidate->size;
            if (!framebuffer_sample_pinned(candidate) && (!victim || candidate->used < victim->used))
                victim = candidate;
        }
        if (retained <= MXGPU_FRAMEBUFFER_CACHE_BYTES || !victim) return 0;
        if (framebuffer_release_resource_unlocked(victim)) return -1;
    }
    return 0;
}

static int framebuffer_resource_acquire_unlocked(struct mxgpu_framebuffer *framebuffer)
{
    if (framebuffer->resource_blocked && framebuffer_release_resource_unlocked(framebuffer)) return -1;
    if (g_dev.fd >= 0 && (query_adapter_unlocked() ||
        framebuffer->size > g_dev.adapter_info.max_buffer_bytes ||
        framebuffer->width > g_dev.adapter_info.max_texture_dimension_2d ||
        framebuffer->height > g_dev.adapter_info.max_texture_dimension_2d)) return -1;
    if (framebuffer_trim_bytes_unlocked(framebuffer)) return -1;
    if (!framebuffer->resource_id) {
        unsigned slot;
        for (slot = 0; slot < MXGPU_FRAMEBUFFER_CACHE_ENTRIES && g_dev.framebuffer_cache[slot]; slot++) {}
        if (slot == MXGPU_FRAMEBUFFER_CACHE_ENTRIES) {
            if (framebuffer_evict_unlocked(framebuffer)) return -1;
            for (slot = 0; slot < MXGPU_FRAMEBUFFER_CACHE_ENTRIES && g_dev.framebuffer_cache[slot]; slot++) {}
        }
        unsigned id = new_resource(MXGPU_KIND_TEXTURE_2D, framebuffer->width,
                                   framebuffer->height, framebuffer->size);
        if (!id) return -1;
        framebuffer->resource_id = id;
        g_dev.framebuffer_cache[slot] = framebuffer;
    }
    framebuffer->used = ++g_dev.framebuffer_clock;
    g_dev.framebuffer = framebuffer;
    g_dev.color_id = framebuffer->resource_id;
    g_needs_clear = 0;
    return 0;
}

struct mxgpu_framebuffer *mxgpu_framebuffer_create(uint32_t width, uint32_t height)
{
    if (!width || !height || width > 2048 || height > 2048 ||
        (uint64_t)width * height > UINT_MAX / 4u) return NULL;
    struct mxgpu_framebuffer *framebuffer = calloc(1, sizeof *framebuffer);
    if (!framebuffer) return NULL;
    framebuffer->size = width * height * 4u;
    framebuffer->pixels = malloc(framebuffer->size);
    if (!framebuffer->pixels) { free(framebuffer); return NULL; }
    framebuffer->width = width;
    framebuffer->height = height;
    return framebuffer;
}

int mxgpu_framebuffer_sync(struct mxgpu_framebuffer *framebuffer, unsigned char *pixels, int *changed)
{
    if (changed) *changed = 0;
    pthread_mutex_lock(&g_device_mutex);
    int result = !framebuffer || !pixels || !framebuffer->valid ? -1 : framebuffer_sync_unlocked(framebuffer);
    if (!result && framebuffer->valid) {
        memcpy(pixels, framebuffer->pixels, framebuffer->size);
        if (changed) *changed = framebuffer->unpublished;
        framebuffer->unpublished = 0;
    }
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_framebuffer_discard(struct mxgpu_framebuffer *framebuffer)
{
    pthread_mutex_lock(&g_device_mutex);
    struct resource *resource = framebuffer && framebuffer->resource_id ?
        res_slot(framebuffer->resource_id) : NULL;
    int result = batch_drain_unlocked() || !framebuffer || g_dev.lost || framebuffer->error ||
        framebuffer->resource_blocked || (framebuffer->resource_id &&
        (!resource || resource->width != framebuffer->width ||
         resource->height != framebuffer->height || resource->size != framebuffer->size)) ? -1 : 0;
    if (!result) {
        framebuffer->valid = 0;
        framebuffer->pending = 0;
        framebuffer->unpublished = 0;
        if (resource) resource->host_current = 0;
    }
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_framebuffer_destroy(struct mxgpu_framebuffer *framebuffer)
{
    if (!framebuffer) return 0;
    pthread_mutex_lock(&g_device_mutex);
    int result = framebuffer_release_resource_unlocked(framebuffer);
    if (g_dev.framebuffer == framebuffer) {
        g_dev.framebuffer = NULL;
        g_dev.color_id = g_dev.immediate_color_id;
    }
    for (unsigned i = 0; i < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; i++)
        if (g_dev.framebuffer_cache[i] == framebuffer) g_dev.framebuffer_cache[i] = NULL;
    free(framebuffer->pixels);
    free(framebuffer);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

static unsigned g_rt_w = MXGPU_FB_SIZE;
static unsigned g_rt_h = MXGPU_FB_SIZE;
static int g_readback_done;


static int present_color(void)
{
    struct mxgpu_present present;
    struct resource *color = res_slot(g_dev.color_id);
    uint8_t payload[64];
    uint32_t n = 0;

    if (!color || !color->host_live || !color->width || !color->height || g_dev.fd < 0)
        return -1;
    memset(&present, 0, sizeof present);
    present.resource_id = g_dev.color_id;
    present.scanout_id = 0;
    present.source.width = color->width;
    present.source.height = color->height;
    if (mxgpu_present_encode(&present, payload, sizeof payload, &n) != MX_OK)
        return -1;
    return winsys_submit_ioctl(MXGPU_OP_PRESENT, MXGPU_QUEUE_RENDER, g_dev.context, payload, n);
}

static int mxgpu_readback_ready_unlocked(void)
{
    return g_readback_done;
}

static void mxgpu_flush_frame_unlocked(void)
{
    struct resource *color = res_slot(g_dev.color_id);
    unsigned bytes;

    g_needs_clear = 1;
    if (!color || g_dev.fd < 0)
        return;
    bytes = color->width * color->height * 4u;
    if (bytes == 0 || bytes <= readback_limit())
        return;
    present_color();
}

static int query_adapter_unlocked(void)
{
    uint8_t command[64], record[128];
    uint32_t command_len, record_len;
    struct mxgpu_command_header header = {0};
    struct mxgpu_drm_user user = {0};
    if (g_dev.adapter_info_valid)
        return 0;
    if (g_dev.fd < 0 || g_dev.lost)
        return -1;
    header.opcode = MXGPU_OP_QUERY_ADAPTER;
    header.queue = MXGPU_QUEUE_CONTROL;
    header.flags = MXGPU_CMD_SIGNAL_FENCE | MXGPU_CMD_RESPONSE_REQUIRED;
    header.sequence = take_sequence();
    header.fence_value = header.sequence;
    if (mxgpu_command_encode(&header, NULL, 0, g_dev.caps.max_command_bytes,
            command, sizeof command, &command_len) != MX_OK ||
        mxgpu_drm_submit_encode(0, MXGPU_QUEUE_CONTROL, header.fence_value,
            MXGPU_ADAPTER_INFO_SIZE, command, command_len, record, sizeof record, &record_len) != MXGPU_DRM_OK)
        return -1;
    user.pointer = (uint64_t)(uintptr_t)record;
    user.size = record_len;
    user.capacity = sizeof record;
    int result = mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + 5, struct mxgpu_drm_user), &user);
    if (result < 0 && (errno == EPIPE || errno == ENODEV || errno == ETIMEDOUT))
        g_dev.lost = 1;
    if (result < 0 || mxgpu_adapter_info_decode(record, user.size, &g_dev.adapter_info) != MX_OK)
        return -1;
    g_dev.adapter_info_valid = 1;
    return 0;
}

static int native_color_sample_available_unlocked(void)
{
    if (g_dev.lost || g_dev.fd < 0 ||
        mxgpu_format_capabilities_features(g_dev.caps.features) != MX_OK || query_adapter_unlocked()) return 0;
    if (!g_dev.format_caps_valid) {
        uint8_t command[64], record[128];
        uint32_t command_size, record_size;
        struct mxgpu_command_header header = {0};
        struct mxgpu_drm_user user = {0};
        header.opcode = MXGPU_OP_QUERY_FORMAT_CAPABILITIES;
        header.queue = MXGPU_QUEUE_CONTROL;
        header.flags = MXGPU_CMD_SIGNAL_FENCE | MXGPU_CMD_RESPONSE_REQUIRED;
        header.sequence = take_sequence();
        header.fence_value = header.sequence;
        if (mxgpu_command_encode(&header, NULL, 0, g_dev.caps.max_command_bytes,
                command, sizeof command, &command_size) != MX_OK ||
            mxgpu_drm_submit_encode(0, MXGPU_QUEUE_CONTROL, header.fence_value,
                MXGPU_FORMAT_CAPABILITIES_SIZE, command, command_size,
                record, sizeof record, &record_size) != MXGPU_DRM_OK) return 0;
        user.pointer = (uint64_t)(uintptr_t)record;
        user.size = record_size;
        user.capacity = sizeof record;
        int posted = mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + 5, struct mxgpu_drm_user), &user);
        if (posted < 0 && (errno == EPIPE || errno == ENODEV || errno == ETIMEDOUT)) g_dev.lost = 1;
        if (posted < 0 || mxgpu_format_capabilities_decode(record, user.size,
                g_dev.adapter_info.pixel_format_mask, &g_dev.format_caps) != MX_OK) return 0;
        g_dev.format_caps_valid = 1;
    }
    uint32_t encoded_size;
    if (mxgpu_format_capabilities_encode(&g_dev.format_caps, g_dev.adapter_info.pixel_format_mask,
            NULL, 0, &encoded_size) != MX_OK) return 0;
    uint32_t bit = 1u << (MXGPU_FMT_RGBA8_UNORM - 1u);
    return (g_dev.format_caps.sampled & g_dev.format_caps.color_target &
            g_dev.format_caps.transfer_source & g_dev.format_caps.transfer_destination & bit) != 0;
}

static int vertex_backing_bytes(const uint8_t *module, uint32_t actual_bytes,
                                uint32_t stride, uint32_t *backing_bytes)
{
    uint32_t count = word_at(module, 3);
    *backing_bytes = actual_bytes;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t base = MXSB_HEADER_WORDS + i * MXSB_BINDING_WORDS;
        if (word_at(module, base) != 1)
            continue;
        uint32_t packed = word_at(module, base + 1);
        uint32_t elements = word_at(module, base + 3);
        if ((packed & 0xffffu) || ((packed >> 16) & 15u) != MXSB_BINDING_UNIFORM ||
            word_at(module, base + 2) != MXSB_TYPE_F32X4 || elements > UINT_MAX / 16u)
            return -1;
        uint32_t declared_bytes = elements * 16u;
        if (declared_bytes > actual_bytes) {
            if (declared_bytes % stride)
                return -1;
            *backing_bytes = declared_bytes;
        }
        return 0;
    }
    return -1;
}

static int replace_vertex_bytes(unsigned id, const void *vertices, uint32_t used, uint32_t capacity)
{
    struct resource *resource = res_slot(id);
    const uint8_t *source = vertices;
    if (!resource || !source || !used || used > capacity || g_dev.lost)
        return -1;
    if (resource->size > capacity) capacity = resource->size;
    if (g_dev.fd >= 0 && (query_adapter_unlocked() || capacity > g_dev.adapter_info.max_buffer_bytes))
        return -1;
    if ((resource->size != capacity || !resource->bytes) && batch_drain_unlocked()) return -1;
    if (resource->size != capacity || !resource->bytes) {
        uint8_t *replacement = calloc(1, capacity);
        if (!replacement)
            return -1;
        memcpy(replacement, source, used);
        if (resource->host_live && destroy_resource(id)) {
            free(replacement);
            return -1;
        }
        free(resource->bytes);
        resource->bytes = replacement;
        resource->size = capacity;
        resource->width = resource->height = 0;
        resource->host_current = 0;
        resource->dirty_begin = 0;
        resource->dirty_end = capacity;
        return 0;
    }
    uint32_t begin = 0, end = capacity;
    while (begin < end && resource->bytes[begin] == (begin < used ? source[begin] : 0))
        begin++;
    while (end > begin && resource->bytes[end - 1u] == (end <= used ? source[end - 1u] : 0))
        end--;
    if (begin == end)
        return 0;
    if (resource->batch_pinned && batch_drain_unlocked()) return -1;
    if (resource->host_current) {
        resource->dirty_begin = begin;
        resource->dirty_end = end;
    } else if (resource->dirty_begin < resource->dirty_end) {
        if (begin < resource->dirty_begin) resource->dirty_begin = begin;
        if (end > resource->dirty_end) resource->dirty_end = end;
    } else {
        resource->dirty_begin = begin;
        resource->dirty_end = end;
    }
    if (begin < used) {
        uint32_t copy_end = end < used ? end : used;
        memmove(resource->bytes + begin, source + begin, copy_end - begin);
    }
    if (end > used) {
        uint32_t zero_begin = begin > used ? begin : used;
        memset(resource->bytes + zero_begin, 0, end - zero_begin);
    }
    resource->host_current = 0;
    return 0;
}

static void uniform_cache_record_create(unsigned id, int success)
{
    for (unsigned stage = 0; stage < 2; stage++)
        for (unsigned i = 0; i < MXGPU_UNIFORM_CACHE_ENTRIES; i++) {
            struct uniform_cache_entry *entry = &g_dev.uniform_cache[stage][i];
            if (entry->resource_id != id) continue;
            if (success) entry->uncertain_create = 0;
            else if (!g_dev.completion_valid) entry->uncertain_create = 1;
            else if (!entry->uncertain_create) entry->logical_size = 0;
        }
    for (unsigned i = 0; i < MXGPU_VERTEX_CACHE_ENTRIES; i++) {
        struct vertex_cache_entry *entry = &g_dev.vertex_cache[i];
        if (entry->resource_id != id) continue;
        if (success) entry->uncertain_create = 0;
        else if (!g_dev.completion_valid) entry->uncertain_create = 1;
        else if (!entry->uncertain_create)
            entry->used_bytes = entry->required_capacity = entry->stride = 0;
    }
}

static int uniform_cache_release_unlocked(struct uniform_cache_entry *entry)
{
    struct resource *resource = res_slot(entry->resource_id);
    if (!resource) return -1;
    if (resource->host_live && destroy_resource(entry->resource_id)) {
        entry->blocked = 1;
        return -1;
    }
    free(resource->bytes);
    memset(resource, 0, sizeof *resource);
    memset(entry, 0, sizeof *entry);
    return 0;
}

static int vertex_cache_release_unlocked(struct vertex_cache_entry *entry)
{
    struct resource *resource = res_slot(entry->resource_id);
    if (!resource || entry->uncertain_create) return -1;
    if (resource->host_live && destroy_resource(entry->resource_id)) {
        entry->blocked = 1;
        return -1;
    }
    free(resource->bytes);
    memset(resource, 0, sizeof *resource);
    memset(entry, 0, sizeof *entry);
    return 0;
}

static int vertex_cache_evictable(const struct vertex_cache_entry *entry, unsigned preserve)
{
    struct resource *resource = res_slot(entry->resource_id);
    return resource && !resource->batch_pinned && entry->resource_id != preserve && entry->resource_id != g_dev.vertex_id &&
        !entry->uncertain_create && (!resource->host_live || resource->host_current || entry->blocked);
}

static int vertex_cache_pressure_evict_unlocked(void)
{
    struct vertex_cache_entry *victim = NULL;
    for (unsigned i = 0; i < MXGPU_VERTEX_CACHE_ENTRIES; i++) {
        struct vertex_cache_entry *entry = &g_dev.vertex_cache[i];
        if (vertex_cache_evictable(entry, 0) && (!victim || entry->used < victim->used)) victim = entry;
    }
    return victim ? vertex_cache_release_unlocked(victim) : 1;
}

static int vertex_cache_trim_unlocked(unsigned preserve, uint32_t replacement_size)
{
    for (unsigned pass = 0; pass < MXGPU_VERTEX_CACHE_ENTRIES; pass++) {
        uint64_t bytes = replacement_size;
        struct vertex_cache_entry *victim = NULL;
        for (unsigned i = 0; i < MXGPU_VERTEX_CACHE_ENTRIES; i++) {
            struct vertex_cache_entry *entry = &g_dev.vertex_cache[i];
            struct resource *resource = res_slot(entry->resource_id);
            if (!resource || entry->resource_id == preserve) continue;
            bytes += resource->size;
            if (vertex_cache_evictable(entry, preserve) && (!victim || entry->used < victim->used)) victim = entry;
        }
        if (bytes <= MXGPU_VERTEX_CACHE_BYTES || !victim) return 0;
        if (vertex_cache_release_unlocked(victim)) return -1;
    }
    return 0;
}

static unsigned vertex_cache_acquire_unlocked(const void *data, uint32_t used_bytes,
                                              uint32_t required_capacity, uint32_t stride)
{
    struct vertex_cache_entry *entry = NULL, *retry = NULL, *empty = NULL, *compatible = NULL, *victim = NULL;
    if (g_dev.lost || !data || !used_bytes || used_bytes > required_capacity || !stride ||
        stride % 16u || stride > MXGPU_SHADER_VERTEX_SLOTS * 16u || used_bytes % stride || required_capacity % stride)
        return 0;
    if (g_dev.fd >= 0 && (query_adapter_unlocked() || required_capacity > g_dev.adapter_info.max_buffer_bytes)) return 0;
    for (unsigned i = 0; i < MXGPU_VERTEX_CACHE_ENTRIES; i++) {
        struct vertex_cache_entry *candidate = &g_dev.vertex_cache[i];
        struct resource *resource = res_slot(candidate->resource_id);
        if (resource && !candidate->blocked && candidate->used_bytes == used_bytes &&
            candidate->required_capacity == required_capacity && candidate->stride == stride &&
            resource->size >= required_capacity && !memcmp(resource->bytes, data, used_bytes)) {
            if (resource->host_live && resource->host_current) { entry = candidate; break; }
            retry = candidate;
        }
        if (!candidate->resource_id) { if (!empty) empty = candidate; continue; }
        if (!vertex_cache_evictable(candidate, 0)) continue;
        if (!victim || candidate->used < victim->used) victim = candidate;
        if (resource->size >= required_capacity && (!compatible || candidate->used < compatible->used)) compatible = candidate;
    }
    if (!entry) entry = retry ? retry : empty ? empty : compatible ? compatible : victim;
    if (!entry && g_dev.batch.count) {
        if (batch_drain_unlocked()) return 0;
        return vertex_cache_acquire_unlocked(data, used_bytes, required_capacity, stride);
    }
    if (!entry) return 0;
    if (entry->blocked && vertex_cache_release_unlocked(entry)) return 0;
    struct resource *resource = res_slot(entry->resource_id);
    uint32_t capacity = resource && resource->size > required_capacity ? resource->size : required_capacity;
    if (vertex_cache_trim_unlocked(entry->resource_id, capacity)) return 0;
    if (!entry->resource_id) entry->resource_id = new_resource(MXGPU_KIND_BUFFER, 0, 0, required_capacity);
    if (!entry->resource_id) return 0;
    if (!resource || !resource->host_live || !resource->host_current ||
        entry->used_bytes != used_bytes || entry->required_capacity != required_capacity || entry->stride != stride ||
        memcmp(resource->bytes, data, used_bytes)) {
        if (replace_vertex_bytes(entry->resource_id, data, used_bytes, required_capacity)) {
            resource = res_slot(entry->resource_id);
            if (resource && resource->host_live) entry->blocked = 1;
            return 0;
        }
    }
    entry->used_bytes = used_bytes;
    entry->required_capacity = required_capacity;
    entry->stride = stride;
    entry->used = ++g_dev.vertex_cache_clock;
    return entry->resource_id;
}

static int uniform_cache_active(unsigned id)
{
    if (id && g_dev.resources[id].batch_pinned) return 1;
    return (g_dev.uniform_size[0] && id == g_dev.uniform_id[0]) ||
           (g_dev.uniform_size[1] && id == g_dev.uniform_id[1]);
}

static int uniform_cache_pressure_evict_unlocked(void)
{
    struct uniform_cache_entry *victim = NULL;
    for (unsigned stage = 0; stage < 2; stage++)
        for (unsigned i = 0; i < MXGPU_UNIFORM_CACHE_ENTRIES; i++) {
            struct uniform_cache_entry *entry = &g_dev.uniform_cache[stage][i];
            if (!entry->resource_id || entry->uncertain_create || uniform_cache_active(entry->resource_id)) continue;
            if (!victim || entry->used < victim->used) victim = entry;
        }
    return victim ? uniform_cache_release_unlocked(victim) : 1;
}

static int uniform_cache_trim_unlocked(unsigned preserve, uint32_t replacement_size)
{
    for (unsigned pass = 0; pass < 2 * MXGPU_UNIFORM_CACHE_ENTRIES; pass++) {
        uint64_t bytes = replacement_size;
        struct uniform_cache_entry *victim = NULL;
        for (unsigned stage = 0; stage < 2; stage++)
            for (unsigned i = 0; i < MXGPU_UNIFORM_CACHE_ENTRIES; i++) {
                struct uniform_cache_entry *entry = &g_dev.uniform_cache[stage][i];
                struct resource *resource = res_slot(entry->resource_id);
                if (!resource || entry->resource_id == preserve) continue;
                bytes += resource->size;
                if (entry->uncertain_create) continue;
                if (uniform_cache_active(entry->resource_id)) continue;
                if (!victim || entry->used < victim->used) victim = entry;
            }
        if (bytes <= MXGPU_UNIFORM_CACHE_BYTES || !victim) return 0;
        if (uniform_cache_release_unlocked(victim)) return -1;
    }
    return 0;
}

static unsigned uniform_cache_acquire_unlocked(unsigned stage, const void *data, uint32_t size)
{
    struct uniform_cache_entry *entry = NULL, *victim = NULL, *retry = NULL, *compatible = NULL;
    if (g_dev.lost || stage >= 2 || !data || !size) return 0;
    if (g_dev.fd >= 0 && (query_adapter_unlocked() || size > g_dev.adapter_info.max_buffer_bytes)) return 0;
    for (unsigned i = 0; i < MXGPU_UNIFORM_CACHE_ENTRIES; i++) {
        struct uniform_cache_entry *candidate = &g_dev.uniform_cache[stage][i];
        struct resource *resource = res_slot(candidate->resource_id);
        if (resource && !candidate->blocked && candidate->logical_size == size &&
            resource->size >= size && !memcmp(resource->bytes, data, size)) {
            if (resource->host_live && resource->host_current) { entry = candidate; break; }
            retry = candidate;
        }
        if (candidate->uncertain_create)
            continue;
        if (resource && resource->size >= size &&
            (!compatible || candidate->used < compatible->used)) compatible = candidate;
        if (!candidate->resource_id || !victim ||
            (victim->resource_id && candidate->used < victim->used)) victim = candidate;
    }
    if (!entry) {
        if (retry) entry = retry;
        else if (victim && !victim->resource_id) entry = victim;
        else entry = compatible ? compatible : victim;
    }
    if (!entry && g_dev.batch.count) {
        if (batch_drain_unlocked()) return 0;
        return uniform_cache_acquire_unlocked(stage, data, size);
    }
    if (!entry) return 0;
    if (entry->blocked && uniform_cache_release_unlocked(entry)) return 0;
    struct resource *resource = res_slot(entry->resource_id);
    uint32_t capacity = resource && resource->size > size ? resource->size : size;
    if (uniform_cache_trim_unlocked(entry->resource_id, capacity)) return 0;
    if (!entry->resource_id) entry->resource_id = new_resource(MXGPU_KIND_BUFFER, 0, 0, size);
    if (!entry->resource_id) return 0;
    if (replace_vertex_bytes(entry->resource_id, data, size, size)) {
        resource = res_slot(entry->resource_id);
        if (resource && resource->host_live) entry->blocked = 1;
        return 0;
    }
    entry->logical_size = size;
    entry->used = ++g_dev.uniform_cache_clock;
    return entry->resource_id;
}

static int ensure_color_resource(unsigned width, unsigned height);

static int texture_is_framebuffer_alias(unsigned index, unsigned id)
{
    return index < g_dev.draw_texture_count && g_dev.draw_textures[index].framebuffer &&
        g_dev.draw_textures[index].framebuffer->resource_id == id &&
        !g_dev.draw_textures[index].framebuffer->resource_blocked;
}

static int draw_scene(void)
{
    struct mxgpu_render_submit submit;
    struct mxgpu_execution_binding bindings[19];
    uint8_t payload[1024];
    uint32_t n = 0;
    unsigned stage;
    if (ensure_pipeline() != 0)
        return -1;
    if (create_buffer_resource(g_dev.vertex_id) != 0)
        return -1;
    if (!texture_is_framebuffer_alias(0, g_dev.texture_id) &&
        create_texture_resource(g_dev.texture_id, MXGPU_USAGE_SAMPLED | MXGPU_USAGE_TRANSFER_DESTINATION) != 0)
        return -1;
    if (ensure_color_resource(g_rt_w, g_rt_h) != 0)
        return -1;
    if (transfer_bytes(g_dev.vertex_id) != 0 ||
        (!texture_is_framebuffer_alias(0, g_dev.texture_id) && transfer_bytes(g_dev.texture_id) != 0))
        return -1;
    memset(&submit, 0, sizeof submit);
    memset(bindings, 0, sizeof bindings);
    submit.pipeline_id = g_dev.pipeline_id;
    submit.color_target_id = g_dev.color_id;
    submit.binding_count = 2;
    submit.load_action = g_needs_clear ? MXGPU_LOAD_CLEAR : MXGPU_LOAD_LOAD;
    submit.store_action = MXGPU_STORE_STORE;
    submit.draw_kind = MXGPU_DRAW_NON_INDEXED;
    submit.primitive = MXGPU_PRIM_TRIANGLE;
    submit.element_count = g_draw_verts ? g_draw_verts : 6;
    submit.instance_count = 1;
    bindings[0].access = MXGPU_BIND_ACCESS_READ;
    bindings[0].kind = MXGPU_BIND_KIND_BUFFER;
    bindings[0].resource_id = g_dev.vertex_id;
    bindings[0].size = g_dev.resources[g_dev.vertex_id].size;
    bindings[1].slot = 1;
    bindings[1].access = MXGPU_BIND_ACCESS_READ;
    bindings[1].kind = MXGPU_BIND_KIND_TEXTURE_2D;
    bindings[1].resource_id = g_dev.texture_id;
    for (stage = 0; stage < 2; stage++) {
        struct mxgpu_execution_binding *binding;
        if (!g_dev.uniform_size[stage])
            continue;
        if (create_buffer_resource(g_dev.uniform_id[stage]) != 0 || transfer_bytes(g_dev.uniform_id[stage]) != 0)
            return -1;
        binding = &bindings[submit.binding_count++];
        binding->slot = stage + 2;
        binding->access = MXGPU_BIND_ACCESS_READ;
        binding->kind = MXGPU_BIND_KIND_BUFFER;
        binding->resource_id = g_dev.uniform_id[stage];
        binding->size = g_dev.uniform_size[stage];
    }
    if (g_dev.draw_texture_count) {
        unsigned texture;
        bindings[1].slot = g_dev.draw_textures[0].texture_slot;
        bindings[1].kind = g_dev.draw_textures[0].binding_kind ?
                           g_dev.draw_textures[0].binding_kind : MXGPU_BIND_KIND_TEXTURE_2D;
        for (texture = 1; texture < g_dev.draw_texture_count; texture++) {
            struct mxgpu_execution_binding *binding = &bindings[submit.binding_count++];
            unsigned id = g_dev.texture_ids[texture];
            if (!texture_is_framebuffer_alias(texture, id) &&
                (create_texture_resource(id, MXGPU_USAGE_SAMPLED | MXGPU_USAGE_TRANSFER_DESTINATION) ||
                 transfer_bytes(id))) return -1;
            binding->slot = g_dev.draw_textures[texture].texture_slot;
            binding->access = MXGPU_BIND_ACCESS_READ;
            binding->kind = g_dev.draw_textures[texture].binding_kind ?
                            g_dev.draw_textures[texture].binding_kind : MXGPU_BIND_KIND_TEXTURE_2D;
            binding->resource_id = id;
        }
    }
    if (g_dev.native_active && g_dev.native_state.sampler_enabled) {
        unsigned texture, count = g_dev.draw_texture_count ? g_dev.draw_texture_count : 1;
        for (texture = 0; texture < count; texture++) {
            struct mxgpu_execution_binding *binding = &bindings[submit.binding_count++];
            binding->slot = g_dev.draw_texture_count ? g_dev.draw_textures[texture].sampler_slot : 4;
            binding->access = MXGPU_BIND_ACCESS_READ;
            binding->kind = MXGPU_BIND_KIND_SAMPLER;
            binding->resource_id = g_dev.active_sampler_ids[texture];
        }
    }
    if (g_dev.native_active) {
        struct mxgpu_render_extended extended = {0};
        extended.pipeline_id = submit.pipeline_id;
        extended.rasterizer_state_id = g_dev.active_rasterizer_id;
        extended.blend_state_id = g_dev.active_blend_id;
        if (g_dev.native_state.depth_enabled) {
            extended.depth_stencil_state_id = g_dev.active_depth_id;
            extended.depth_stencil_target_id = g_dev.depth_id;
            extended.depth_load_action = MXGPU_LOAD_LOAD;
            extended.depth_store_action = MXGPU_STORE_STORE;
            extended.stencil_load_action = MXGPU_LOAD_LOAD;
            extended.stencil_store_action = MXGPU_STORE_STORE;
            extended.stencil_reference = g_dev.native_state.stencil_reference;
        }
        extended.color_target_count = 1;
        extended.color_targets[0].resource_id = submit.color_target_id;
        extended.color_targets[0].format = MXGPU_FMT_RGBA8_UNORM;
        extended.color_targets[0].load_action = submit.load_action;
        extended.color_targets[0].store_action = submit.store_action;
        extended.viewport_count = 1;
        extended.viewports[0] = g_dev.native_state.viewport;
        extended.scissor_count = g_dev.native_state.rasterizer.scissor_enable ? 1 : 0;
        extended.scissors[0] = g_dev.native_state.scissor;
        memcpy(extended.blend_factor, g_dev.native_state.blend_factor, sizeof extended.blend_factor);
        extended.binding_count = submit.binding_count;
        extended.draw_kind = submit.draw_kind;
        extended.primitive = submit.primitive;
        extended.element_count = submit.element_count;
        extended.instance_count = submit.instance_count;
        if (mxgpu_render_extended_features(&extended, g_dev.caps.features) != MX_OK ||
            mxgpu_render_extended_encode(&extended, bindings, payload, sizeof payload, &n) != MX_OK)
            return -1;
    } else if (mxgpu_render_submit_encode(&submit, bindings, payload, sizeof payload, &n) != MX_OK)
        return -1;
    if (g_dev.native_active && g_dev.native_state.depth_enabled)
        g_dev.resources[g_dev.depth_id].host_current = 0;
    if (g_dev.native_active && g_dev.fd >= 0) {
        g_dev.resources[g_dev.color_id].host_current = 0;
        g_dev.resources[g_dev.color_id].upload_scheduled = 0;
    }
    if (winsys_submit_ioctl(g_dev.native_active ? MXGPU_OP_RENDER_SUBMIT_EXTENDED : MXGPU_OP_RENDER_SUBMIT,
                            MXGPU_QUEUE_RENDER, g_dev.context, payload, n) != MX_OK) {
        if (g_dev.deferred_readback && g_dev.framebuffer) g_dev.framebuffer->error = 1;
        return -1;
    }
    g_needs_clear = 0;
    g_readback_done = 0;
    if (!g_dev.deferred_readback && g_dev.fd >= 0 && (unsigned long)g_rt_w * g_rt_h * 4u <= readback_limit()) {
        if (readback_color() != 0)
            return -1;
        g_readback_done = 1;
    }
    return 0;
}

static unsigned g_scene_vbo;
static unsigned g_scene_tex;

static int replace_bytes(unsigned id, const void *data, unsigned size, unsigned width, unsigned height)
{
    struct resource *res = res_slot(id);
    unsigned char *replacement;
    if (!res || size == 0 || g_dev.lost)
        return -1;
    if (res->bytes && res->size == size && res->width == width && res->height == height && data &&
        memcmp(res->bytes, data, size) == 0)
        return 0;
    if (res->bytes && res->size == size && res->width == width && res->height == height && data) {
        if (res->batch_pinned && batch_drain_unlocked()) return -1;
        const unsigned char *source = data;
        unsigned begin = 0, end = size;
        while (begin < end && res->bytes[begin] == source[begin])
            begin++;
        while (end > begin && res->bytes[end - 1u] == source[end - 1u])
            end--;
        if (res->host_current) {
            res->dirty_begin = begin;
            res->dirty_end = end;
        } else if (res->dirty_begin < res->dirty_end && res->dirty_end <= size) {
            if (begin < res->dirty_begin)
                res->dirty_begin = begin;
            if (end > res->dirty_end)
                res->dirty_end = end;
        }
        memmove(res->bytes + begin, source + begin, end - begin);
        res->host_current = 0;
        return 0;
    }
    if (batch_drain_unlocked()) return -1;
    replacement = calloc(1, size);
    if (!replacement)
        return -1;
    if (data)
        memcpy(replacement, data, size);
    free(res->bytes);
    res->bytes = replacement;
    res->host_current = 0;
    res->dirty_begin = 0;
    res->dirty_end = size;
    res->width = width;
    res->height = height;
    res->size = size;
    return 0;
}

static int ensure_color_resource(unsigned width, unsigned height)
{
    struct resource *color;
    unsigned bytes;
    if (!width || !height || (uint64_t)width * height > UINT_MAX / 4u)
        return -1;
    bytes = width * height * 4u;
    if (!g_dev.color_id) {
        unsigned id = new_resource(MXGPU_KIND_TEXTURE_2D, width, height, bytes);
        if (!id)
            return -1;
        g_dev.color_id = id;
        if (!g_dev.deferred_readback) g_dev.immediate_color_id = id;
        g_needs_clear = 1;
    }
    color = res_slot(g_dev.color_id);
    if (!color)
        return -1;
    if (color->width != width || color->height != height) {
        if (replace_bytes(g_dev.color_id, NULL, bytes, width, height))
            return -1;
        g_needs_clear = 1;
    }
    unsigned usage = MXGPU_USAGE_COLOR_TARGET | MXGPU_USAGE_TRANSFER_SOURCE |
                     MXGPU_USAGE_SCANOUT | MXGPU_USAGE_TRANSFER_DESTINATION;
    if (g_dev.framebuffer && g_dev.framebuffer->resource_id == g_dev.color_id &&
        ((color->host_live && (color->host_usage & MXGPU_USAGE_SAMPLED)) ||
         (!color->host_live && native_color_sample_available_unlocked()))) usage |= MXGPU_USAGE_SAMPLED;
    return create_texture_resource(g_dev.color_id, usage);
}

static int mxgpu_seed_color_unlocked(const unsigned char *pixels, unsigned width, unsigned height)
{
    struct resource *color;
    if (!g_dev.deferred_readback && framebuffer_detach_unlocked()) return -1;
    if (!pixels || !width || !height || width > 2048 || height > 2048)
        return -1;
    if (mxgpu_device_open_unlocked() != 0 || ensure_color_resource(width, height) != 0)
        return -1;
    color = res_slot(g_dev.color_id);
    if ((!g_dev.native_active && g_needs_clear) || memcmp(color->bytes, pixels, color->size) != 0) {
        memcpy(color->bytes, pixels, color->size);
        color->host_current = 0;
        color->dirty_begin = 0;
        color->dirty_end = color->size;
    }
    if (transfer_bytes(g_dev.color_id) != 0)
        return -1;
    g_needs_clear = 0;
    return 0;
}

static int texture_cache_pinned(const struct native_texture_cache_entry *entry)
{
    if (entry->resource_id && g_dev.resources[entry->resource_id].batch_pinned) return 1;
    for (unsigned i = 0; i < g_dev.draw_texture_count; i++) {
        uint64_t identity = g_dev.draw_textures[i].identity;
        if (identity ? entry->identity == identity :
            !entry->identity && entry->legacy_slot == i)
            return 1;
    }
    return 0;
}

static int texture_cache_release_unlocked(struct native_texture_cache_entry *entry)
{
    struct resource *resource = res_slot(entry->resource_id);
    if (!resource) return -1;
    if (resource->host_live && destroy_resource(entry->resource_id)) { entry->blocked = 1; return -1; }
    free(resource->bytes);
    memset(resource, 0, sizeof *resource);
    memset(entry, 0, sizeof *entry);
    return 0;
}

static int texture_evict_except_unlocked(unsigned preserve_id)
{
    if (batch_drain_unlocked()) return -1;
    struct native_texture_cache_entry *victim = NULL;
    for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++) {
        struct native_texture_cache_entry *entry = &g_dev.texture_cache[i];
        if (entry->resource_id && entry->resource_id != preserve_id && !texture_cache_pinned(entry) &&
            (!victim || entry->used < victim->used)) victim = entry;
    }
    return victim ? texture_cache_release_unlocked(victim) : -1;
}

static int texture_evict_unlocked(void)
{
    return texture_evict_except_unlocked(0);
}

static int resource_pressure_evict_unlocked(void)
{
    if (batch_drain_unlocked()) return -1;
    int uniform = uniform_cache_pressure_evict_unlocked();
    if (uniform != 1) return uniform;
    int vertex = vertex_cache_pressure_evict_unlocked();
    if (vertex != 1) return vertex;
    for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++)
        if (g_dev.texture_cache[i].resource_id && !texture_cache_pinned(&g_dev.texture_cache[i]))
            return texture_evict_unlocked();
    return framebuffer_evict_unlocked(g_dev.framebuffer);
}

static unsigned texture_resource_acquire_unlocked(const struct mxgpu_texture_input *input, unsigned slot)
{
    struct native_texture_cache_entry *entry = NULL;
    for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++) {
        struct native_texture_cache_entry *candidate = &g_dev.texture_cache[i];
        if (candidate->resource_id && (input->identity ? candidate->identity == input->identity :
            !candidate->identity && candidate->legacy_slot == slot)) {
            if (candidate->blocked) {
                if (texture_cache_release_unlocked(candidate)) return 0;
            } else entry = candidate;
            break;
        }
    }
    if (!entry) {
        for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++)
            if (!g_dev.texture_cache[i].resource_id) { entry = &g_dev.texture_cache[i]; break; }
        if (!entry) {
            if (texture_evict_unlocked()) return 0;
            for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++)
                if (!g_dev.texture_cache[i].resource_id) { entry = &g_dev.texture_cache[i]; break; }
        }
        unsigned id = new_resource(MXGPU_KIND_TEXTURE_2D, 0, 0, 0);
        if (!id) return 0;
        entry->resource_id = id;
        entry->identity = input->identity;
        entry->legacy_slot = slot;
    }
    entry->used = ++g_dev.texture_clock;
    return entry->resource_id;
}

static unsigned texture_input_bytes(const struct mxgpu_texture_input *input)
{
    uint64_t total = 0;
    unsigned layers = input->array_layers ? input->array_layers : 1;
    unsigned kind = input->binding_kind ? input->binding_kind : MXGPU_BIND_KIND_TEXTURE_2D;
    unsigned format = input->format ? input->format : MXGPU_FMT_RGBA8_UNORM;
    if ((format != MXGPU_FMT_RGBA8_UNORM && format != MXGPU_FMT_DEPTH32_FLOAT) ||
        (kind != MXGPU_BIND_KIND_TEXTURE_2D && kind != MXGPU_BIND_KIND_TEXTURE_CUBE) ||
        (kind == MXGPU_BIND_KIND_TEXTURE_CUBE ? (layers != 6 || input->width != input->height) : layers != 1)) return 0;
    unsigned count = input->mip_count ? input->mip_count : 1;
    unsigned maximum = 0, extent = input->width > input->height ? input->width : input->height;
    while (extent) { maximum++; extent >>= 1; }
    if ((!input->pixels && !input->framebuffer) || !input->width || !input->height || count > maximum || count > 32 ||
        (input->mip_count && !input->levels)) return 0;
    for (unsigned level = 0; level < count; level++) {
        unsigned width = input->width >> level, height = input->height >> level;
        if (!width) width = 1;
        if (!height) height = 1;
        if (input->mip_count && (!input->levels[level].pixels ||
            input->levels[level].width != width || input->levels[level].height != height)) return 0;
        total += (uint64_t)width * height * 4u * layers;
        if (total > UINT_MAX) return 0;
    }
    return (unsigned)total;
}

static int texture_inputs_equal(const struct mxgpu_texture_input *a, const struct mxgpu_texture_input *b)
{
    if (a->framebuffer || b->framebuffer)
        return a->framebuffer && a->framebuffer == b->framebuffer &&
            a->framebuffer_revision == b->framebuffer_revision;
    unsigned levels = a->mip_count ? a->mip_count : 1;
    unsigned layers = a->array_layers ? a->array_layers : 1;
    if (a->width != b->width || a->height != b->height ||
        levels != (b->mip_count ? b->mip_count : 1) ||
        layers != (b->array_layers ? b->array_layers : 1) ||
        (a->binding_kind ? a->binding_kind : MXGPU_BIND_KIND_TEXTURE_2D) !=
        (b->binding_kind ? b->binding_kind : MXGPU_BIND_KIND_TEXTURE_2D) ||
        (a->format ? a->format : MXGPU_FMT_RGBA8_UNORM) !=
        (b->format ? b->format : MXGPU_FMT_RGBA8_UNORM)) return 0;
    for (unsigned i = 0; i < levels; i++) {
        const uint8_t *left = a->mip_count ? a->levels[i].pixels : a->pixels;
        const uint8_t *right = b->mip_count ? b->levels[i].pixels : b->pixels;
        unsigned width = a->mip_count ? a->levels[i].width : a->width;
        unsigned height = a->mip_count ? a->levels[i].height : a->height;
        if (memcmp(left, right, (size_t)width * height * layers * 4u)) return 0;
    }
    return 1;
}

static int texture_cache_trim_bytes(unsigned preserve_id, unsigned replacement_bytes)
{
    for (unsigned pass = 0; pass < MXGPU_TEXTURE_CACHE_ENTRIES; pass++) {
        uint64_t retained = replacement_bytes;
        for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++) {
            unsigned id = g_dev.texture_cache[i].resource_id;
            struct resource *resource = id != preserve_id ? res_slot(id) : NULL;
            if (resource) retained += resource->size;
        }
        if (retained <= MXGPU_TEXTURE_CACHE_BYTES) return 0;
        int eligible = 0;
        for (unsigned i = 0; i < MXGPU_TEXTURE_CACHE_ENTRIES; i++)
            eligible |= g_dev.texture_cache[i].resource_id && g_dev.texture_cache[i].resource_id != preserve_id &&
                !texture_cache_pinned(&g_dev.texture_cache[i]);
        if (!eligible) return 0;
        if (texture_evict_except_unlocked(preserve_id)) return -1;
    }
    return 0;
}

static int replace_texture_input(unsigned id, const struct mxgpu_texture_input *input)
{
    unsigned bytes = texture_input_bytes(input), count = input->mip_count ? input->mip_count : 1;
    unsigned layers = input->array_layers ? input->array_layers : 1;
    unsigned format = input->format ? input->format : MXGPU_FMT_RGBA8_UNORM;
    struct resource *resource = res_slot(id);
    if (!bytes || !resource) return -1;
    if (input->identity && bytes > resource->size && g_dev.fd >= 0 &&
        (query_adapter_unlocked() || bytes > g_dev.adapter_info.max_buffer_bytes)) return -1;
    if (texture_cache_trim_bytes(id, bytes)) return -1;
    if (resource->host_live && (resource->width != input->width || resource->height != input->height ||
        resource->size != bytes || resource->host_mip_count != count ||
        resource->host_array_layers != layers || resource->host_format != format) && destroy_resource(id)) return -1;
    const unsigned char *pixels = input->pixels;
    unsigned char *packed = NULL;
    if (input->mip_count) {
        packed = malloc(bytes);
        if (!packed) return -1;
        unsigned offset = 0;
        for (unsigned level = 0; level < count; level++) {
            unsigned size = input->levels[level].width * input->levels[level].height * 4u * layers;
            memcpy(packed + offset, input->levels[level].pixels, size);
            offset += size;
        }
        pixels = packed;
    }
    int result = replace_bytes(id, pixels, bytes, input->width, input->height);
    free(packed);
    if (result) return result;
    resource->mip_count = count;
    resource->array_layers = layers;
    resource->format = format;
    unsigned offset = 0;
    for (unsigned level = 0; level < count; level++) {
        unsigned width = input->width >> level, height = input->height >> level;
        if (!width) width = 1;
        if (!height) height = 1;
        resource->mip_offsets[level] = offset;
        offset += width * height * 4u * layers;
    }
    return 0;
}

static int mxgpu_execute_module_uniforms_unlocked(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size, uint32_t vertex_stride_bytes)
{
    struct resource *dst;
    const void *uniforms[2] = {vertex_uniforms, fragment_uniforms};
    uint32_t uniform_sizes[2] = {vertex_uniform_size, fragment_uniform_size};
    unsigned stage;
    if (!g_dev.deferred_readback && framebuffer_detach_unlocked()) return -1;
    struct mxsb_limits limits;
    const char *inject;
    if (!module || module_len == 0 || module_len > sizeof g_user_mod || !vertices || vertex_count < 3 || (!texels && !g_dev.draw_texture_count) || tw < 1 || th < 1 || !color || cw < 1 || ch < 1 || cw > 2048 || ch > 2048)
        return -1;
    if (!vertex_stride_bytes || vertex_stride_bytes % 16u || vertex_stride_bytes > MXGPU_SHADER_VERTEX_SLOTS * 16u ||
        (unsigned)vertex_count > UINT_MAX / vertex_stride_bytes ||
        (uint64_t)(unsigned)tw * (unsigned)th > UINT_MAX / 4u)
        return -1;
    for (stage = 0; stage < 2; stage++) {
        if ((uniform_sizes[stage] && !uniforms[stage]) || uniform_sizes[stage] > UINT_MAX - 15u)
            return -1;
    }
    mxsb_limits_default(&limits);
    if (mxsb_verify(module, module_len, &limits) != MXSB_OK)
        return -1;
    if (g_user_len != module_len || memcmp(g_user_mod, module, module_len) != 0) {
        memcpy(g_user_mod, module, module_len);
        g_user_len = module_len;
        g_dev.pipeline_dirty = 1;
    }
    g_draw_verts = (unsigned)vertex_count;
    g_rt_w = (unsigned)cw;
    g_rt_h = (unsigned)ch;
    g_readback_done = 0;
    if (mxgpu_device_open_unlocked() != 0)
        return -1;
    uint32_t vertex_used = (unsigned)vertex_count * vertex_stride_bytes;
    uint32_t vertex_capacity;
    if (vertex_backing_bytes(module, vertex_used, vertex_stride_bytes, &vertex_capacity))
        return -1;
    if (vertex_capacity > vertex_used && g_dev.fd >= 0 &&
        (query_adapter_unlocked() || vertex_capacity > g_dev.adapter_info.max_buffer_bytes))
        return -1;
    for (stage = 0; stage < 2; stage++) {
        uint32_t padded;
        uint8_t *data;
        int result;
        g_dev.uniform_size[stage] = 0;
        if (!uniform_sizes[stage])
            continue;
        padded = (uniform_sizes[stage] + 15u) & ~15u;
        data = NULL;
        if (padded != uniform_sizes[stage]) {
            data = calloc(1, padded);
            if (!data)
                return -1;
            memcpy(data, uniforms[stage], uniform_sizes[stage]);
        }
        unsigned uniform_id = uniform_cache_acquire_unlocked(stage, data ? data : uniforms[stage], padded);
        result = uniform_id ? 0 : -1;
        if (!result) g_dev.uniform_id[stage] = uniform_id;
        free(data);
        if (result)
            return -1;
        g_dev.uniform_size[stage] = padded;
    }
    g_dev.vertex_id = 0;
    g_scene_vbo = vertex_cache_acquire_unlocked(vertices, vertex_used, vertex_capacity, vertex_stride_bytes);
    if (!g_scene_vbo) return -1;
    g_dev.vertex_id = g_scene_vbo;
    if (g_dev.draw_texture_count) {
        for (unsigned texture = 0; texture < g_dev.draw_texture_count; texture++) {
            const struct mxgpu_texture_input *input = &g_dev.draw_textures[texture];
            struct mxgpu_texture_input fallback;
            if (input->framebuffer) {
                struct mxgpu_framebuffer *source = input->framebuffer;
                struct resource *resident = res_slot(source->resource_id);
                if (resident && !source->resource_blocked && resident->host_live && (resident->host_usage & MXGPU_USAGE_SAMPLED)) {
                    g_dev.texture_ids[texture] = source->resource_id;
                    source->used = ++g_dev.framebuffer_clock;
                    continue;
                }
                if (framebuffer_sync_unlocked(source)) return -1;
                fallback = *input;
                fallback.framebuffer = NULL;
                fallback.pixels = source->pixels;
                input = &fallback;
            }
            g_dev.texture_ids[texture] = texture_resource_acquire_unlocked(input, texture);
            if (!g_dev.texture_ids[texture] || replace_texture_input(g_dev.texture_ids[texture], input)) return -1;
        }
        g_dev.texture_id = g_dev.texture_ids[0];
    } else {
        if (!g_scene_tex)
            g_scene_tex = new_resource(MXGPU_KIND_TEXTURE_2D, (unsigned)tw, (unsigned)th, (unsigned)tw * (unsigned)th * 4u);
        if (!g_scene_tex || replace_bytes(g_scene_tex, texels, (unsigned)tw * (unsigned)th * 4u,
                                           (unsigned)tw, (unsigned)th)) return -1;
        g_dev.texture_id = g_scene_tex;
    }
    g_dev.vertex_id = g_scene_vbo;
    if (!g_dev.batch.count) g_dev.frame_submits = 0;
    inject = getenv("MXGPU_INJECT_ILLEGAL");
    if (inject && inject[0] == '1') {
        if (mxgpu_debug_illegal_then_legal_unlocked() != 0)
            return -1;
        fprintf(stderr, "legal_after_illegal ok\n");
    } else if (draw_scene() != 0) {
        return -1;
    }
    dst = res_slot(g_dev.color_id);
    if (!dst || !dst->bytes || dst->width != (unsigned)cw || dst->height != (unsigned)ch)
        return -1;
    if (!g_dev.deferred_readback && !g_readback_done && g_dev.fd >= 0) {
        if (readback_color() != 0)
            return -1;
        g_readback_done = 1;
    }
    if (!g_dev.deferred_readback && !(g_dev.native_active && g_dev.native_state.depth_enabled))
        memcpy(color, dst->bytes, (size_t)cw * (size_t)ch * 4u);
    if (getenv("MXGPU_TRACE"))
        fprintf(stderr, "submits %u\n", g_dev.frame_submits);
    if (!g_dev.batch.count) completed_submit_proof();
    return 0;
}

static int mxgpu_execute_module_unlocked(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch)
{
    return mxgpu_execute_module_uniforms_unlocked(module, module_len, vertices, vertex_count,
                                         texels, tw, th, color, cw, ch, NULL, 0, NULL, 0, 16);
}

static int mxgpu_execute_scene_unlocked(const float *vertices, int vertex_count,
                        const unsigned char *texels, int tw, int th,
                        unsigned char *color, int cw, int ch)
{
    uint8_t module[512];
    uint32_t module_len = 0;
    if (mxgpu_scene_encode(module, sizeof module, &module_len, (uint32_t)vertex_count) != MXSB_OK)
        return -1;
    return mxgpu_execute_module_unlocked(module, module_len, vertices, vertex_count, texels, tw, th, color, cw, ch);
}

static void private_context_release(void)
{
    uint8_t record[MXGPU_DRM_HEADER_SIZE + 8];
    uint32_t size;
    struct mxgpu_drm_user user = {0};
    if (g_dev.fd < 0 || !g_dev.context_owned)
        return;
    if (mxgpu_drm_ctx_destroy_encode(g_dev.context, record, sizeof record, &size) == MXGPU_DRM_OK) {
        user.pointer = (uint64_t)(uintptr_t)record;
        user.size = size;
        user.capacity = sizeof record;
        mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + 2, struct mxgpu_drm_user), &user);
    }
    g_dev.context_owned = 0;
}

static int device_initialize(int fd, int allow_executor)
{
    memset(&g_dev, 0, sizeof g_dev);
    g_dev.fd = fd;
    g_dev.allow_executor = allow_executor;
    if (fd < 0 && !allow_executor)
        return -1;
    g_dev.next_id = 1;
    if (fd >= 0) {
        uint8_t caps_record[48];
        uint32_t caps_size;
        struct mxgpu_drm_user caps_user = {0};
        if (mxgpu_drm_get_caps_encode(caps_record, sizeof caps_record, &caps_size) == MXGPU_DRM_OK) {
            caps_user.pointer = (uint64_t)(uintptr_t)caps_record;
            caps_user.size = caps_size;
            caps_user.capacity = sizeof caps_record;
            if (mxgpu_ioctl(fd, DRM_IOWR(DRM_COMMAND_BASE + 9, struct mxgpu_drm_user), &caps_user) == 0 &&
                caps_user.size <= sizeof caps_record)
                mxgpu_drm_get_caps_response_decode(caps_record, caps_user.size, &g_dev.caps);
        }
        uint8_t transfer_record[MXGPU_DRM_HEADER_SIZE + 8];
        uint32_t transfer_size;
        struct mxgpu_drm_user transfer_user = {0};
        if (mxgpu_drm_get_transfer_limits_encode(transfer_record, sizeof transfer_record, &transfer_size) == MXGPU_DRM_OK) {
            transfer_user.pointer = (uint64_t)(uintptr_t)transfer_record;
            transfer_user.size = transfer_size;
            transfer_user.capacity = sizeof transfer_record;
            if (mxgpu_ioctl(fd, DRM_IOWR(DRM_COMMAND_BASE + 10, struct mxgpu_drm_user), &transfer_user) == 0 &&
                transfer_user.size <= sizeof transfer_record)
                mxgpu_drm_get_transfer_limits_response_decode(transfer_record, transfer_user.size, &g_dev.transfer_limits);
        }
        uint8_t batch_record[MXGPU_DRM_HEADER_SIZE + 24];
        uint32_t batch_size;
        struct mxgpu_drm_user batch_user = {0};
        if (mxgpu_drm_get_batch_limits_encode(batch_record, sizeof batch_record, &batch_size) == MXGPU_DRM_OK) {
            batch_user.pointer = (uint64_t)(uintptr_t)batch_record;
            batch_user.size = batch_size;
            batch_user.capacity = sizeof batch_record;
            if (!mxgpu_ioctl(fd, DRM_IOWR(DRM_COMMAND_BASE + MXGPU_DRM_IOCTL_GET_BATCH_LIMITS,
                                   struct mxgpu_drm_user), &batch_user) && batch_user.size <= sizeof batch_record)
                mxgpu_drm_get_batch_limits_response_decode(batch_record, batch_user.size, &g_dev.batch_limits);
        }
        uint8_t record[MXGPU_DRM_HEADER_SIZE + 8];
        uint32_t size;
        struct mxgpu_drm_user user = {0};
        if (mxgpu_drm_ctx_create_encode(record, sizeof record, &size) != MXGPU_DRM_OK)
            goto failed;
        user.pointer = (uint64_t)(uintptr_t)record;
        user.size = size;
        user.capacity = sizeof record;
        if (mxgpu_ioctl(fd, DRM_IOWR(DRM_COMMAND_BASE + 1, struct mxgpu_drm_user), &user) < 0 ||
            user.size > sizeof record ||
            mxgpu_drm_ctx_create_response_decode(record, user.size, &g_dev.context) != MXGPU_DRM_OK)
            goto failed;
        g_dev.context_owned = 1;
        if (winsys_submit_ioctl(MXGPU_OP_CONTEXT_CREATE, MXGPU_QUEUE_CONTROL,
                                g_dev.context, NULL, 0) != 0)
            goto failed;
    } else {
        g_dev.context = 1;
    }
    g_dev.open = 1;
    return 0;
failed:
    private_context_release();
    close(fd);
    free(g_dev.command_scratch.bytes);
    free(g_dev.record_scratch.bytes);
    free(g_dev.upload_scratch.bytes);
    free(g_dev.readback_scratch.bytes);
    free(g_dev.batch_commands.bytes);
    free(g_dev.batch_record.bytes);
    memset(&g_dev, 0, sizeof g_dev);
    g_dev.fd = -1;
    return -1;
}

static int mxgpu_device_open_fd_unlocked(int fd)
{
    struct stat requested, current;
    int owned;
    if (fd < 0 || g_dev.lost || fstat(fd, &requested))
        return -1;
    if (g_dev.open) {
        if (g_dev.fd < 0 || fstat(g_dev.fd, &current))
            return -1;
        return requested.st_dev == current.st_dev && requested.st_ino == current.st_ino &&
               requested.st_rdev == current.st_rdev ? 0 : -1;
    }
    owned = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (owned < 0)
        return -1;
    return device_initialize(owned, 0);
}

static int mxgpu_device_open_unlocked(void)
{
    if (g_dev.lost)
        return -1;
    if (g_dev.open)
        return 0;
    log_open_attempt();
    return device_initialize(g_dev.fd, g_dev.allow_executor);
}

static void mxgpu_device_close_unlocked(void)
{
    batch_drain_unlocked();
    unsigned i;
    for (i = 0; i < MXGPU_FRAMEBUFFER_CACHE_ENTRIES; i++) {
        struct mxgpu_framebuffer *framebuffer = g_dev.framebuffer_cache[i];
        if (!framebuffer) continue;
        if (framebuffer_sync_unlocked(framebuffer)) framebuffer->error = 1;
        framebuffer->resource_id = 0;
        framebuffer->resource_blocked = 0;
    }
    g_dev.framebuffer = NULL;
    if (g_dev.fd >= 0 && g_dev.context_owned && g_dev.open && !g_dev.lost) {
        for (i = 0; i < 8 && !g_dev.lost; i++)
            native_pipeline_entry_destroy(&g_dev.module_cache[i]);
        native_cache_release_all();
        uint8_t payload[MXGPU_RESOURCE_ID_SIZE];
        uint32_t size;
        if (g_dev.pipeline_live &&
            mxgpu_resource_id_encode(g_dev.pipeline_id, payload, sizeof payload, &size) == MX_OK)
            winsys_submit_ioctl(MXGPU_OP_PIPELINE_DESTROY, MXGPU_QUEUE_CONTROL,
                                g_dev.context, payload, size);
        if (g_dev.shader_live &&
            mxgpu_resource_id_encode(g_dev.shader_id, payload, sizeof payload, &size) == MX_OK)
            winsys_submit_ioctl(MXGPU_OP_SHADER_DESTROY, MXGPU_QUEUE_CONTROL,
                                g_dev.context, payload, size);
        for (i = 1; i < MXGPU_RESOURCE_SLOTS && !g_dev.lost; i++)
            if (g_dev.resources[i].host_live)
                destroy_resource(i);
        if (!g_dev.lost)
            winsys_submit_ioctl(MXGPU_OP_CONTEXT_DESTROY, MXGPU_QUEUE_CONTROL,
                                g_dev.context, NULL, 0);
    }
    for (i = 0; i < 8; i++) {
        free(g_dev.module_cache[i].module);
        g_dev.module_cache[i].module = NULL;
    }
    private_context_release();
    for (i = 0; i < MXGPU_RESOURCE_SLOTS; i++)
        free(g_dev.resources[i].bytes);
    if (g_dev.fd >= 0)
        close(g_dev.fd);
    free(g_dev.command_scratch.bytes);
    free(g_dev.record_scratch.bytes);
    free(g_dev.upload_scratch.bytes);
    free(g_dev.readback_scratch.bytes);
    free(g_dev.batch_commands.bytes);
    free(g_dev.batch_record.bytes);
    memset(&g_dev, 0, sizeof g_dev);
    g_dev.fd = -1;
    g_scene_vbo = 0;
    g_scene_tex = 0;
    g_user_len = 0;
    g_needs_clear = 1;
    g_readback_done = 0;
}



static unsigned mxgpu_last_submits_unlocked(void)
{
    return g_dev.frame_submits;
}

static int mxgpu_debug_illegal_then_legal_unlocked(void)
{
    struct mxgpu_transfer transfer;
    uint8_t payload[80];
    uint8_t byte = 0;
    uint32_t n = 0;
    mxgpu_device_open_unlocked();
    memset(&transfer, 0, sizeof transfer);
    transfer.resource_id = 0x00ffffffu;
    transfer.data_bytes = 1;
    if (mxgpu_transfer_encode(&transfer, &byte, payload, sizeof payload, &n) != MX_OK || n == 0)
        return -1;
    if (winsys_submit_ioctl(MXGPU_OP_TRANSFER_TO_HOST, MXGPU_QUEUE_TRANSFER, g_dev.context ? g_dev.context : 1, payload, n) == 0)
        return -1;
    g_dev.frame_submits = 0;
    return draw_scene();
}



int mxgpu_readback_ready(void)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_readback_ready_unlocked();
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

void mxgpu_flush_frame(void)
{
    pthread_mutex_lock(&g_device_mutex);
    mxgpu_flush_frame_unlocked();
    pthread_mutex_unlock(&g_device_mutex);
}

int mxgpu_seed_color(const unsigned char *pixels, unsigned width, unsigned height)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_seed_color_unlocked(pixels, width, height);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_execute_module_uniforms(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_execute_module_uniforms_unlocked(module, module_len, vertices, vertex_count, texels, tw, th, color, cw, ch, vertex_uniforms, vertex_uniform_size, fragment_uniforms, fragment_uniform_size, 16);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_execute_module(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_execute_module_unlocked(module, module_len, vertices, vertex_count, texels, tw, th, color, cw, ch);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_execute_scene(const float *vertices, int vertex_count,
                        const unsigned char *texels, int tw, int th,
                        unsigned char *color, int cw, int ch)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_execute_scene_unlocked(vertices, vertex_count, texels, tw, th, color, cw, ch);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_device_open_fd(int fd)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_device_open_fd_unlocked(fd);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_device_open(void)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_device_open_unlocked();
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

void mxgpu_device_close(void)
{
    pthread_mutex_lock(&g_device_mutex);
    mxgpu_device_close_unlocked();
    pthread_mutex_unlock(&g_device_mutex);
}

__attribute__((visibility("default")))
unsigned mxgpu_last_submits(void)
{
    pthread_mutex_lock(&g_device_mutex);
    unsigned result = mxgpu_last_submits_unlocked();
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_debug_illegal_then_legal(void)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = mxgpu_debug_illegal_then_legal_unlocked();
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

static int native_depth_available_unlocked(uint32_t format)
{
    uint8_t command[64], record[128];
    uint32_t command_len, record_len;
    struct mxgpu_drm_user user = {0};
    struct mxgpu_command_header header = {0};
    if (g_dev.lost || g_dev.fd < 0 || mxgpu_depth_stencil_state_features(g_dev.caps.features) != MX_OK ||
        (format != MXGPU_FMT_DEPTH32_FLOAT && format != MXGPU_FMT_DEPTH32_FLOAT_STENCIL8 &&
         format != MXGPU_FMT_DEPTH24_UNORM_STENCIL8)) return 0;
    if (!g_dev.format_caps_valid) {
        if (mxgpu_format_capabilities_features(g_dev.caps.features) != MX_OK) return 0;
        header.opcode = MXGPU_OP_QUERY_FORMAT_CAPABILITIES;
        header.queue = MXGPU_QUEUE_CONTROL;
        header.flags = MXGPU_CMD_SIGNAL_FENCE | MXGPU_CMD_RESPONSE_REQUIRED;
        header.sequence = take_sequence(); header.fence_value = header.sequence;
        if (mxgpu_command_encode(&header, NULL, 0, g_dev.caps.max_command_bytes,
                                 command, sizeof command, &command_len) != MX_OK ||
            mxgpu_drm_submit_encode(0, MXGPU_QUEUE_CONTROL, header.fence_value,
                MXGPU_FORMAT_CAPABILITIES_SIZE, command, command_len, record, sizeof record, &record_len) != MXGPU_DRM_OK) return 0;
        user.pointer = (uint64_t)(uintptr_t)record; user.size = record_len; user.capacity = sizeof record;
        int posted = mxgpu_ioctl(g_dev.fd, DRM_IOWR(DRM_COMMAND_BASE + 5, struct mxgpu_drm_user), &user);
        if (posted < 0 && (errno == EPIPE || errno == ENODEV || errno == ETIMEDOUT))
            g_dev.lost = 1;
        if (posted < 0 || user.size != MXGPU_FORMAT_CAPABILITIES_SIZE ||
            mxgpu_format_capabilities_decode(record, user.size, UINT32_MAX, &g_dev.format_caps) != MX_OK) return 0;
        g_dev.format_caps_valid = 1;
    }
    uint32_t bit = 1u << (format - 1u);
    return (g_dev.format_caps.depth_stencil_target & bit) &&
           (g_dev.format_caps.transfer_source & bit) && (g_dev.format_caps.transfer_destination & bit);
}

int mxgpu_native_depth_available(int fd, uint32_t format)
{
    pthread_mutex_lock(&g_device_mutex);
    int result = fd >= 0 ? mxgpu_device_open_fd_unlocked(fd) : mxgpu_device_open_unlocked();
    result = !result && native_depth_available_unlocked(format);
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

static int execute_module_transaction_resources(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes,
                         const struct mxgpu_texture_input *textures, uint32_t texture_count,
                         struct mxgpu_framebuffer *framebuffer, uint64_t cpu_revision)
{
    int result = -1;
    const char *operation = "device-open";
    unsigned sampler_count = texture_count ? texture_count : 1;
    if (readback_complete)
        *readback_complete = 0;
    pthread_mutex_lock(&g_device_mutex);
    g_readback_done = 0;
    if (framebuffer && (!state || state->depth_enabled || !initial_color ||
        cw != (int)framebuffer->width || ch != (int)framebuffer->height)) goto invalid_arguments;
    if (texture_count > MXGPU_TEXTURE_INPUTS || (texture_count && !textures)) goto invalid_arguments;
    for (unsigned i = 0; i < texture_count; i++) {
        const struct mxgpu_texture_input *input = &textures[i];
        if (input->framebuffer && (!state || input->framebuffer == framebuffer ||
            !input->framebuffer->valid || input->framebuffer->error ||
            input->framebuffer_revision != input->framebuffer->cpu_revision ||
            input->width != input->framebuffer->width || input->height != input->framebuffer->height ||
            input->mip_count || (input->array_layers && input->array_layers != 1) ||
            (input->format && input->format != MXGPU_FMT_RGBA8_UNORM) ||
            (input->binding_kind && input->binding_kind != MXGPU_BIND_KIND_TEXTURE_2D))) goto invalid_arguments;
        if (!texture_input_bytes(input) ||
            input->texture_slot > UINT16_MAX || input->sampler_slot > UINT16_MAX ||
            input->texture_slot == 0 || input->texture_slot == 2 || input->texture_slot == 3 ||
            (state && state->sampler_enabled && (input->sampler_slot == 0 ||
             input->sampler_slot == 2 || input->sampler_slot == 3 || input->sampler_slot == input->texture_slot))) goto invalid_arguments;
        for (unsigned j = 0; j < i; j++)
            if ((input->identity && input->identity == textures[j].identity &&
                 !texture_inputs_equal(input, &textures[j])) ||
                input->texture_slot == textures[j].texture_slot ||
                (state && state->sampler_enabled && (input->sampler_slot == textures[j].sampler_slot ||
                 input->sampler_slot == textures[j].texture_slot || input->texture_slot == textures[j].sampler_slot))) goto invalid_arguments;
    }
    result = fd >= 0 ? mxgpu_device_open_fd_unlocked(fd) : mxgpu_device_open_unlocked();
    for (unsigned texture = 0; texture < texture_count && !result; texture++) {
        const struct mxgpu_texture_input *input = &textures[texture];
        if (input->binding_kind == MXGPU_BIND_KIND_TEXTURE_CUBE &&
            (g_dev.fd < 0 || g_dev.caps.major != 1 || g_dev.caps.minor < 28 ||
             !(g_dev.caps.features & MXGPU_FEAT_TEXTURE_ARRAY))) result = -1;
        if (input->format == MXGPU_FMT_DEPTH32_FLOAT &&
            (g_dev.fd < 0 || !(g_dev.caps.features & MXGPU_FEAT_EXTENDED_PIXEL_FORMATS))) result = -1;
        if (state && state->sampler_enabled && input->sampler.compare &&
            (!(g_dev.caps.features & MXGPU_FEAT_TEXTURE_COMPARE) ||
             input->format != MXGPU_FMT_DEPTH32_FLOAT)) result = -1;
    }
    g_dev.active_blend_id = g_dev.active_rasterizer_id = g_dev.active_depth_id = 0;
    memset(g_dev.active_sampler_ids, 0, sizeof g_dev.active_sampler_ids);
    if (!result && state) {
        uint8_t payloads[3 + MXGPU_TEXTURE_INPUTS][MXGPU_BLEND_STATE_HEADER_SIZE + 8 * MXGPU_BLEND_TARGET_SIZE];
        uint32_t sizes[3 + MXGPU_TEXTURE_INPUTS];
        uint64_t required = MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE |
                            MXGPU_FEAT_RASTERIZER_STATE | MXGPU_FEAT_VIEWPORT_SCISSOR;
        if (g_dev.fd < 0 || (g_dev.caps.features & required) != required ||
            (state->sampler_enabled && !(g_dev.caps.features & MXGPU_FEAT_SAMPLER_OBJECTS))) {
            result = -1;
        } else {
            g_dev.native_state = *state;
            g_dev.native_state.blend.state_id = 1;
            g_dev.native_state.rasterizer.state_id = 1;
            operation = "blend-encode";
            result = mxgpu_blend_state_encode(&g_dev.native_state.blend, payloads[0], sizeof payloads[0], &sizes[0]);
            if (!result) {
                operation = "rasterizer-encode";
                result = mxgpu_rasterizer_state_encode(&g_dev.native_state.rasterizer, payloads[1], sizeof payloads[1], &sizes[1]);
            }
            if (state->sampler_enabled) {
                for (unsigned texture = 0; texture < sampler_count && !result; texture++) {
                    struct mxgpu_sampler_state sampler = texture_count ? textures[texture].sampler : state->sampler;
                    sampler.sampler_id = 1;
                    operation = "sampler-encode";
                    result = mxgpu_sampler_state_encode(&sampler, payloads[2 + texture], sizeof payloads[0], &sizes[2 + texture]);
                }
            }
            if (!result && state->depth_enabled) {
                struct mxgpu_depth_stencil_state depth = state->depth_stencil;
                unsigned pixel_bytes = state->depth.format == MXGPU_FMT_DEPTH32_FLOAT_STENCIL8 ? 8 : 4;
                depth.state_id = 1;
                operation = "depth-validate";
                if (!state->depth.pixels || state->depth.width != (unsigned)cw || state->depth.height != (unsigned)ch ||
                    !cw || !ch || (uint64_t)(unsigned)cw * (unsigned)ch > UINT32_MAX / pixel_bytes ||
                    state->stencil_reference > 255 ||
                    (state->depth.format == MXGPU_FMT_DEPTH32_FLOAT && depth.stencil_enable) ||
                    !native_depth_available_unlocked(state->depth.format)) result = -1;
                if (!result) result = mxgpu_depth_stencil_state_encode(&depth,
                    payloads[2 + MXGPU_TEXTURE_INPUTS], sizeof payloads[0], &sizes[2 + MXGPU_TEXTURE_INPUTS]);
            }
            if (!result) {
                operation = "blend-cache";
                result = native_cache_acquire(g_dev.blend_cache, 4, MXGPU_OP_BLEND_STATE_CREATE,
                    MXGPU_OP_BLEND_STATE_DESTROY, payloads[0], sizes[0], &g_dev.active_blend_id);
            }
            if (!result) {
                operation = "rasterizer-cache";
                result = native_cache_acquire(g_dev.rasterizer_cache, 4, MXGPU_OP_RASTERIZER_STATE_CREATE,
                    MXGPU_OP_RASTERIZER_STATE_DESTROY, payloads[1], sizes[1], &g_dev.active_rasterizer_id);
            }
            if (state->sampler_enabled) {
                for (unsigned texture = 0; texture < sampler_count && !result; texture++) {
                    operation = "sampler-cache";
                    result = native_cache_acquire(g_dev.sampler_cache, MXGPU_TEXTURE_INPUTS, MXGPU_OP_SAMPLER_CREATE,
                        MXGPU_OP_SAMPLER_DESTROY, payloads[2 + texture], sizes[2 + texture], &g_dev.active_sampler_ids[texture]);
                }
            }
            if (!result && state->depth_enabled)
                result = native_cache_acquire(g_dev.depth_cache, 4, MXGPU_OP_DEPTH_STENCIL_STATE_CREATE,
                    MXGPU_OP_DEPTH_STENCIL_STATE_DESTROY, payloads[2 + MXGPU_TEXTURE_INPUTS],
                    sizes[2 + MXGPU_TEXTURE_INPUTS], &g_dev.active_depth_id);
            g_dev.native_active = !result;
        }
    }
    g_dev.draw_textures = textures;
    g_dev.draw_texture_count = texture_count;
    struct resource *framebuffer_resource = framebuffer ? res_slot(framebuffer->resource_id) : NULL;
    int framebuffer_needs_seed = framebuffer && (!framebuffer->resource_id || framebuffer->resource_blocked ||
        !framebuffer->valid || framebuffer->cpu_revision != cpu_revision || !framebuffer_resource ||
        !framebuffer_resource->host_live || (!framebuffer->pending && !framebuffer_resource->host_current));
    if (!result && framebuffer) {
        operation = "framebuffer-acquire";
        if (g_dev.framebuffer != framebuffer) result = framebuffer_detach_unlocked();
        if (!result && framebuffer->error && framebuffer->valid && framebuffer->cpu_revision == cpu_revision) result = -1;
        if (!result && (!framebuffer->valid || framebuffer->cpu_revision != cpu_revision)) {
            if (framebuffer->pending && !framebuffer->error) result = -1;
            else {
                memcpy(framebuffer->pixels, initial_color, framebuffer->size);
                framebuffer->cpu_revision = cpu_revision;
                framebuffer->valid = 1;
                framebuffer->pending = framebuffer->error = framebuffer->unpublished = 0;
            }
        }
        if (!result) {
            g_dev.deferred_readback = 1;
            result = framebuffer_resource_acquire_unlocked(framebuffer);
        }
    }
    if (!result) {
        operation = "color-seed";
        if (!framebuffer || framebuffer_needs_seed)
            result = mxgpu_seed_color_unlocked(framebuffer ? framebuffer->pixels : initial_color, cw, ch);
        if (!result && framebuffer) g_dev.framebuffer = framebuffer;
    }
    if (!result && state && state->depth_enabled) {
        unsigned pixel_bytes = state->depth.format == MXGPU_FMT_DEPTH32_FLOAT_STENCIL8 ? 8 : 4;
        unsigned bytes = state->depth.width * state->depth.height * pixel_bytes;
        if (!g_dev.depth_id) g_dev.depth_id = new_resource(MXGPU_KIND_TEXTURE_2D,
            state->depth.width, state->depth.height, bytes);
        struct resource *depth = res_slot(g_dev.depth_id);
        operation = "depth-seed";
        if (!depth) result = -1;
        if (!result && depth->host_live && (depth->format != state->depth.format ||
            depth->width != state->depth.width || depth->height != state->depth.height || depth->size != bytes))
            result = destroy_resource(g_dev.depth_id);
        if (!result) {
            depth->format = state->depth.format;
            result = replace_bytes(g_dev.depth_id, state->depth.pixels, bytes,
                                   state->depth.width, state->depth.height);
        }
        if (!result) result = create_texture_resource(g_dev.depth_id,
            MXGPU_USAGE_DEPTH_STENCIL | MXGPU_USAGE_TRANSFER_SOURCE | MXGPU_USAGE_TRANSFER_DESTINATION);
        if (!result) result = transfer_bytes(g_dev.depth_id);
    }
    if (!result) {
        operation = "module-execute";
        g_dev.batch_collect = framebuffer && state && !state->depth_enabled &&
                              g_dev.fd >= 0 && g_dev.batch_limits.max_commands;
        result = mxgpu_execute_module_uniforms_unlocked(module, module_len, vertices, vertex_count,
                                                       texels, tw, th, color, cw, ch,
                                                       vertex_uniforms, vertex_uniform_size,
                                                       fragment_uniforms, fragment_uniform_size, vertex_stride_bytes);
    }
    if (!result && state && state->depth_enabled) {
        struct resource *depth = res_slot(g_dev.depth_id);
        operation = "depth-readback";
        result = readback_attachment(depth);
        if (!result) {
            depth->host_current = 1;
            memcpy(state->depth.pixels, depth->bytes, depth->size);
            struct resource *color_resource = res_slot(g_dev.color_id);
            memcpy(color, color_resource->bytes, color_resource->size);
        }
    }
    if (result && state && state->depth_enabled) {
        struct resource *depth = res_slot(g_dev.depth_id);
        struct resource *color_resource = res_slot(g_dev.color_id);
        if (depth) depth->host_current = 0;
        if (color_resource) color_resource->host_current = 0;
        g_readback_done = 0;
    }
    if (result && getenv("MXGPU_TRACE"))
        fprintf(stderr, "mxgpu transaction failed operation=%s result=%d errno=%d native=%d features=%llx\n",
                operation, result, errno, state != NULL, (unsigned long long)g_dev.caps.features);

    if (!result && framebuffer) {
        framebuffer->pending = framebuffer->unpublished = 1;
    }
    if (!result && readback_complete)
        *readback_complete = framebuffer == NULL;
    g_dev.batch_collect = 0;
    if (result && g_dev.batch.count) batch_drain_unlocked();
    g_dev.deferred_readback = 0;
    g_dev.native_active = 0;
    g_dev.draw_textures = NULL;
    g_dev.draw_texture_count = 0;
    int transaction_errno = errno;
    if (!g_dev.lost && vertex_cache_trim_unlocked(0, 0) && getenv("MXGPU_TRACE"))
        fprintf(stderr, "mxgpu vertex retention cleanup deferred errno=%d\n", errno);
    if (!g_dev.lost && uniform_cache_trim_unlocked(0, 0) && getenv("MXGPU_TRACE"))
        fprintf(stderr, "mxgpu uniform retention cleanup deferred errno=%d\n", errno);
    if (!g_dev.lost && framebuffer_trim_bytes_unlocked(framebuffer) && getenv("MXGPU_TRACE"))
        fprintf(stderr, "mxgpu framebuffer retention cleanup deferred errno=%d\n", errno);
    errno = transaction_errno;
invalid_arguments:
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_execute_module_transaction_native_resources_deferred(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes,
                         const struct mxgpu_texture_input *textures, uint32_t texture_count,
                         struct mxgpu_framebuffer *framebuffer, uint64_t cpu_revision)
{
    if (!framebuffer) {
        if (readback_complete) *readback_complete = 0;
        pthread_mutex_lock(&g_device_mutex);
        g_readback_done = 0;
        pthread_mutex_unlock(&g_device_mutex);
        return -1;
    }
    return execute_module_transaction_resources(fd, module, module_len, vertices, vertex_count,
        texels, tw, th, initial_color, color, cw, ch, vertex_uniforms, vertex_uniform_size,
        fragment_uniforms, fragment_uniform_size, readback_complete, state, vertex_stride_bytes,
        textures, texture_count, framebuffer, cpu_revision);
}

int mxgpu_execute_module_transaction_native_resources(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes,
                         const struct mxgpu_texture_input *textures, uint32_t texture_count)
{
    return execute_module_transaction_resources(fd, module, module_len, vertices, vertex_count,
        texels, tw, th, initial_color, color, cw, ch, vertex_uniforms, vertex_uniform_size,
        fragment_uniforms, fragment_uniform_size, readback_complete, state, vertex_stride_bytes,
        textures, texture_count, NULL, 0);
}

int mxgpu_execute_module_transaction(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete)
{
    return mxgpu_execute_module_transaction_native(fd, module, module_len, vertices, vertex_count,
        texels, tw, th, initial_color, color, cw, ch, vertex_uniforms, vertex_uniform_size,
        fragment_uniforms, fragment_uniform_size, readback_complete, NULL);
}

static int native_caps_match(int fd, uint64_t required, unsigned minimum_minor)
{
    uint8_t record[48];
    uint32_t size;
    struct mxgpu_drm_caps caps = {0};
    struct mxgpu_drm_user user = {0};
    if (fd < 0 || mxgpu_drm_get_caps_encode(record, sizeof record, &size) != MXGPU_DRM_OK)
        return 0;
    user.pointer = (uint64_t)(uintptr_t)record;
    user.size = size;
    user.capacity = sizeof record;
    if (mxgpu_ioctl(fd, DRM_IOWR(DRM_COMMAND_BASE + 9, struct mxgpu_drm_user), &user) < 0 ||
        user.size > sizeof record ||
        mxgpu_drm_get_caps_response_decode(record, user.size, &caps) != MXGPU_DRM_OK)
        return 0;
    return (caps.features & required) == required &&
           (!minimum_minor || (caps.major == 1 && caps.minor >= minimum_minor));
}

int mxgpu_native_render_caps(int fd)
{
    return native_caps_match(fd, MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE |
                             MXGPU_FEAT_RASTERIZER_STATE | MXGPU_FEAT_VIEWPORT_SCISSOR, 0);
}

int mxgpu_native_mip_caps(int fd)
{
    return native_caps_match(fd, MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE |
        MXGPU_FEAT_RASTERIZER_STATE | MXGPU_FEAT_VIEWPORT_SCISSOR | MXGPU_FEAT_SAMPLER_OBJECTS, 0);
}

int mxgpu_native_cube_caps(int fd)
{
    return native_caps_match(fd, MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE |
        MXGPU_FEAT_RASTERIZER_STATE | MXGPU_FEAT_VIEWPORT_SCISSOR |
        MXGPU_FEAT_SAMPLER_OBJECTS | MXGPU_FEAT_TEXTURE_ARRAY, 28);
}

int mxgpu_native_shadow_caps(int fd)
{
    return native_caps_match(fd, MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE |
        MXGPU_FEAT_RASTERIZER_STATE | MXGPU_FEAT_VIEWPORT_SCISSOR | MXGPU_FEAT_SAMPLER_OBJECTS |
        MXGPU_FEAT_TEXTURE_COMPARE | MXGPU_FEAT_EXTENDED_PIXEL_FORMATS, 12);
}

int mxgpu_native_render_available(int fd)
{
    uint64_t required = MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE |
                        MXGPU_FEAT_RASTERIZER_STATE | MXGPU_FEAT_VIEWPORT_SCISSOR;
    pthread_mutex_lock(&g_device_mutex);
    int result = fd >= 0 ? mxgpu_device_open_fd_unlocked(fd) : mxgpu_device_open_unlocked();
    result = !result && g_dev.fd >= 0 && (g_dev.caps.features & required) == required;
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_native_sampler_available(int fd)
{
    uint64_t required = MXGPU_FEAT_RENDER | MXGPU_FEAT_BLEND_STATE | MXGPU_FEAT_RASTERIZER_STATE |
                        MXGPU_FEAT_VIEWPORT_SCISSOR | MXGPU_FEAT_SAMPLER_OBJECTS;
    pthread_mutex_lock(&g_device_mutex);
    int result = fd >= 0 ? mxgpu_device_open_fd_unlocked(fd) : mxgpu_device_open_unlocked();
    result = !result && g_dev.fd >= 0 && (g_dev.caps.features & required) == required;
    pthread_mutex_unlock(&g_device_mutex);
    return result;
}

int mxgpu_execute_module_transaction_native(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state)
{
    return mxgpu_execute_module_transaction_native_stride(fd, module, module_len, vertices, vertex_count,
        texels, tw, th, initial_color, color, cw, ch, vertex_uniforms, vertex_uniform_size,
        fragment_uniforms, fragment_uniform_size, readback_complete, state, 16);
}

int mxgpu_execute_module_transaction_native_stride(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes)
{
    return mxgpu_execute_module_transaction_native_resources(fd, module, module_len, vertices, vertex_count,
        texels, tw, th, initial_color, color, cw, ch, vertex_uniforms, vertex_uniform_size,
        fragment_uniforms, fragment_uniform_size, readback_complete, state, vertex_stride_bytes, NULL, 0);
}
