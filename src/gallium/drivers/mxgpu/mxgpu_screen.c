/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_public.h"
#include "mxgpu_driver.h"
#include "mxgpu_drm_uapi.h"
#include "mxgpu_compiler.h"
#include "mxsb.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "frontend/sw_winsys.h"
#include "frontend/winsys_handle.h"
#include "nir.h"
#include "nir_builder.h"
#include "tgsi/tgsi_exec.h"
#include "util/format/u_format.h"
#include "util/ralloc.h"
#include "util/u_debug_cb.h"
#include "util/u_inlines.h"
#include "util/u_framebuffer.h"
#include "util/u_surface.h"
#include "util/u_transfer.h"
#include "util/u_memory.h"
#include "util/u_endian.h"
#include "util/simple_mtx.h"
#include "util/u_screen.h"
#include "util/u_upload_mgr.h"

#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <drm.h>
#include <drm_mode.h>
#include <drm_fourcc.h>
#include <sys/stat.h>
#include <linux/dma-buf.h>
#include <errno.h>
#include <stdatomic.h>

struct mxgpu_view_identity {
    struct mxgpu_view_identity *next;
    uint32_t key[16];
    uint64_t identity;
};

struct mxgpu_resource {
    struct pipe_resource base;
    unsigned char *data;
    unsigned stride;
    unsigned size;
    unsigned level_offset[PIPE_MAX_TEXTURE_LEVELS];
    unsigned level_stride[PIPE_MAX_TEXTURE_LEVELS];
    unsigned gem;
    int dumb;
    void *import_map;
    size_t import_size;
    unsigned import_offset;
    int import_fd;
    atomic_bool sync_failed;
    int gem_dumb;
    struct mxgpu_framebuffer *framebuffer;
    unsigned char *framebuffer_pixels;
    uint64_t cpu_revision, framebuffer_revision;
    bool framebuffer_rendered, external, imported;
    bool framebuffer_dirty, framebuffer_publishing, framebuffer_discarding;
    struct mxgpu_view_identity *view_identities;
    struct mxgpu_resource *next;
};

struct mxgpu_screen {
    struct pipe_screen base;
    struct sw_winsys *winsys;
    int fd;
    struct mxgpu_resource *resources;
    simple_mtx_t resource_mutex;
};

struct mxgpu_sampler_view {
    struct pipe_sampler_view base;
    uint64_t identity;
};

static atomic_uint_fast64_t next_view_identity = 1;

struct mxgpu_velem_state {
    unsigned count;
    struct pipe_vertex_element el[PIPE_MAX_ATTRIBS];
};


struct mxgpu_gallium_shader {
    struct mxgpu_shader shader;
    struct mxgpu_shader *gl_position;
};

struct mxgpu_pending {
    struct pipe_resource *resource;
    struct mxgpu_pending *next;
};

struct mxgpu_context {
    struct pipe_context base;
    struct mxgpu_pending *pending;
    struct pipe_framebuffer_state fb;
    struct pipe_vertex_buffer vb[PIPE_MAX_ATTRIBS];
    unsigned vb_count;
    struct mxgpu_velem_state *velems;
    struct pipe_sampler_view *view;
    struct pipe_sampler_view *views[MXGPU_SHADER_TEXTURES];
    struct pipe_sampler_state samplers[MXGPU_SHADER_TEXTURES];
    bool sampler_bound[MXGPU_SHADER_TEXTURES];
    struct mxgpu_shader *vs;
    struct mxgpu_shader *fs;
    struct pipe_constant_buffer constants[2][MXGPU_UNIFORM_BUFFERS];
    void *constant_copies[2][MXGPU_UNIFORM_BUFFERS];
    bool constant_invalid[2][MXGPU_UNIFORM_BUFFERS];
    bool scissor_enabled;
    struct pipe_scissor_state scissor;
    struct pipe_blend_state blend;
    bool blend_bound;
    struct pipe_blend_color blend_color;
    struct pipe_rasterizer_state rasterizer;
    bool rasterizer_bound;
    struct pipe_viewport_state viewport;
    bool viewport_bound;
    struct pipe_depth_stencil_alpha_state dsa;
    bool dsa_bound;
    struct pipe_stencil_ref stencil_ref;
    unsigned draw_instance, draw_base_instance;
    int draw_base_vertex;
};

static struct mxgpu_resource *res_of(struct pipe_resource *resource)
{
    return (struct mxgpu_resource *)resource;
}

static struct mxgpu_context *ctx_of(struct pipe_context *pipe)
{
    return (struct mxgpu_context *)pipe;
}

static struct mxgpu_screen *screen_of(struct pipe_screen *screen)
{
    return (struct mxgpu_screen *)screen;
}

static void mxgpu_destroy_screen(struct pipe_screen *screen)
{
    struct mxgpu_screen *mx = screen_of(screen);
    mxgpu_device_flush();
    if (mx->fd >= 0)
        close(mx->fd);
    simple_mtx_destroy(&mx->resource_mutex);
    FREE(mx);
}

static const char *mxgpu_get_name(struct pipe_screen *screen)
{
    (void)screen;
    return "mxgpu";
}

static const char *mxgpu_get_vendor(struct pipe_screen *screen)
{
    (void)screen;
    return "MX";
}

static int mxgpu_get_fd(struct pipe_screen *screen)
{
    struct sw_winsys *winsys = screen_of(screen)->winsys;
    if (winsys && winsys->get_fd)
        return winsys->get_fd(winsys);
    return screen_of(screen)->fd;
}

static bool mxgpu_format_ok(struct pipe_screen *screen, enum pipe_format format, enum pipe_texture_target target, unsigned sample_count, unsigned storage_sample_count, unsigned bindings)
{
    (void)screen;
    (void)target;
    (void)bindings;
    if (sample_count > 1 || storage_sample_count > 1)
        return false;
    if (bindings == PIPE_BIND_VERTEX_BUFFER) {
        const struct util_format_description *desc = util_format_description(format);
        const struct util_format_unpack_description *unpack = util_format_unpack_description(format);
        if (desc && unpack && unpack->unpack_rgba && desc->layout == UTIL_FORMAT_LAYOUT_PLAIN &&
            desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB && desc->block.width == 1 &&
            desc->block.height == 1 && desc->block.depth == 1 && desc->block.bits <= 128) {
            bool supported = true;
            for (unsigned i = 0; i < 4; i++) {
                const struct util_format_channel_description *channel = &desc->channel[i];
                if (channel->type != UTIL_FORMAT_TYPE_VOID &&
                    ((channel->type != UTIL_FORMAT_TYPE_UNSIGNED && channel->type != UTIL_FORMAT_TYPE_SIGNED &&
                      channel->type != UTIL_FORMAT_TYPE_FLOAT) || channel->size > 32))
                    supported = false;
            }
            if (supported)
                return true;
        }
    }
    if (format == PIPE_FORMAT_R8G8B8A8_UNORM || format == PIPE_FORMAT_B8G8R8A8_UNORM || format == PIPE_FORMAT_B8G8R8X8_UNORM || format == PIPE_FORMAT_R32G32B32A32_FLOAT || format == PIPE_FORMAT_R32G32_FLOAT || format == PIPE_FORMAT_R32G32B32_FLOAT || format == PIPE_FORMAT_Z16_UNORM || format == PIPE_FORMAT_Z24X8_UNORM || format == PIPE_FORMAT_Z24_UNORM_S8_UINT || format == PIPE_FORMAT_S8_UINT_Z24_UNORM || format == PIPE_FORMAT_Z32_FLOAT)
        return true;
    return util_format_get_blocksize(format) > 0 && util_format_is_unorm(format);
}

static bool resource_publish(struct mxgpu_resource *res);

static bool resource_cpu_sync(struct mxgpu_resource *res, unsigned usage, bool end)
{
    struct dma_buf_sync sync = {0};
    unsigned attempt;
    bool private_buffer = res->base.target == PIPE_BUFFER && !res->external &&
                          !res->imported && !res->dumb && !res->framebuffer;
    if (!end && (usage & PIPE_MAP_WRITE) && !private_buffer && mxgpu_device_flush())
        return false;
    if (!end && !res->framebuffer_publishing && !res->framebuffer_discarding && !resource_publish(res))
        return false;
    if (end && (usage & PIPE_MAP_WRITE) && !res->framebuffer_publishing && !res->framebuffer_discarding)
        res->cpu_revision++;
    if (!res->import_map)
        return true;
    if (!end && atomic_load(&res->sync_failed))
        return false;
    if (usage & PIPE_MAP_READ)
        sync.flags |= DMA_BUF_SYNC_READ;
    if (usage & PIPE_MAP_WRITE)
        sync.flags |= DMA_BUF_SYNC_WRITE;
    if (!(sync.flags & DMA_BUF_SYNC_RW))
        return false;
    sync.flags |= end ? DMA_BUF_SYNC_END : DMA_BUF_SYNC_START;
    for (attempt = 0; attempt < 8; attempt++) {
        if (!ioctl(res->import_fd, DMA_BUF_IOCTL_SYNC, &sync))
            return true;
        if (errno != EINTR && errno != EAGAIN)
            break;
    }
    atomic_store(&res->sync_failed, true);
    return false;
}

static void mxgpu_resource_destroy(struct pipe_screen *screen, struct pipe_resource *resource)
{
    struct mxgpu_resource *res = res_of(resource);
    struct mxgpu_screen *mx = screen_of(screen);
    struct mxgpu_resource **link = &mx->resources;
    struct mxgpu_resource *peer;
    resource_publish(res);
    mxgpu_framebuffer_destroy(res->framebuffer);
    free(res->framebuffer_pixels);
    simple_mtx_lock(&mx->resource_mutex);
    while (*link && *link != res)
        link = &(*link)->next;
    if (*link)
        *link = res->next;
    if (res->import_map) {
        munmap(res->import_map, res->import_size);
        close(res->import_fd);
    }
    else if (res->dumb)
        munmap(res->data, res->size);
    else
        FREE(res->data);
    for (peer = mx->resources; peer; peer = peer->next) {
        if (res->gem && peer->gem == res->gem) {
            peer->gem_dumb |= res->gem_dumb;
            break;
        }
    }
    if (res->gem && !peer && mx->fd >= 0) {
        if (res->gem_dumb) {
            struct drm_mode_destroy_dumb destroy = { .handle = res->gem };
            ioctl(mx->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
        } else {
            struct drm_gem_close close_arg = { .handle = res->gem };
            ioctl(mx->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        }
    }
    while (res->view_identities) {
        struct mxgpu_view_identity *entry = res->view_identities;
        res->view_identities = entry->next;
        FREE(entry);
    }
    simple_mtx_unlock(&mx->resource_mutex);
    FREE(res);
}

static bool mxgpu_get_handle(struct pipe_screen *screen, struct pipe_context *context, struct pipe_resource *resource, struct winsys_handle *handle, unsigned usage)
{
    struct mxgpu_resource *res = res_of(resource);
    struct mxgpu_screen *mx = screen_of(screen);
    (void)context;
    (void)usage;
    if (mxgpu_device_flush() || !resource_publish(res))
        return false;
    res->cpu_revision++;
    if (!res->gem || !handle || mx->fd < 0 || handle->plane || handle->layer)
        return false;
    if (handle->type == WINSYS_HANDLE_TYPE_FD) {
        struct drm_prime_handle prime;
        memset(&prime, 0, sizeof prime);
        prime.handle = res->gem;
        prime.flags = DRM_CLOEXEC | DRM_RDWR;
        if (ioctl(mx->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime))
            return false;
        handle->handle = prime.fd;
    } else if (handle->type == WINSYS_HANDLE_TYPE_KMS) {
        handle->handle = res->gem;
    } else if (handle->type == WINSYS_HANDLE_TYPE_SHARED) {
        struct drm_gem_flink flink_arg = { .handle = res->gem };
        if (ioctl(mx->fd, DRM_IOCTL_GEM_FLINK, &flink_arg))
            return false;
        handle->handle = flink_arg.name;
    } else {
        return false;
    }
    handle->stride = res->stride;
    handle->size = res->import_map ? res->import_size : res->size;
    handle->offset = res->import_offset;
    handle->modifier = DRM_FORMAT_MOD_LINEAR;
    res->external = true;
    return true;
}

static struct pipe_resource *mxgpu_resource_from_handle(struct pipe_screen *screen,
                                                       const struct pipe_resource *templ,
                                                       struct winsys_handle *handle,
                                                       unsigned usage)
{
    struct mxgpu_screen *mx = screen_of(screen);
    struct mxgpu_resource *res;
    struct drm_prime_handle prime = {0};
    struct stat statbuf;
    uint64_t bytes;
    unsigned rows, row_bytes;
    void *mapping;
    int import_fd;
    (void)usage;
    if (!handle || handle->type != WINSYS_HANDLE_TYPE_FD || mx->fd < 0 ||
        handle->plane || templ->last_level || templ->depth0 > 1 ||
        templ->array_size > 1 || templ->nr_samples > 1 ||
        (templ->target != PIPE_TEXTURE_2D && templ->target != PIPE_TEXTURE_RECT) ||
        (handle->modifier != DRM_FORMAT_MOD_LINEAR &&
         handle->modifier != DRM_FORMAT_MOD_INVALID))
        return NULL;
    row_bytes = util_format_get_stride(templ->format, templ->width0);
    rows = util_format_get_nblocksy(templ->format, templ->height0);
    if (!rows || !row_bytes || handle->stride < row_bytes)
        return NULL;
    bytes = (uint64_t)handle->stride * rows;
    if (bytes > UINT_MAX || fstat(handle->handle, &statbuf) ||
        statbuf.st_size <= 0 || (uint64_t)statbuf.st_size > SIZE_MAX ||
        (uint64_t)handle->offset + bytes > (uint64_t)statbuf.st_size)
        return NULL;
    import_fd = fcntl(handle->handle, F_DUPFD_CLOEXEC, 0);
    if (import_fd < 0)
        return NULL;
    mapping = mmap(NULL, statbuf.st_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   import_fd, 0);
    if (mapping == MAP_FAILED) {
        close(import_fd);
        return NULL;
    }
    res = CALLOC_STRUCT(mxgpu_resource);
    if (!res) {
        munmap(mapping, statbuf.st_size);
        close(import_fd);
        return NULL;
    }
    prime.fd = import_fd;
    simple_mtx_lock(&mx->resource_mutex);
    if (ioctl(mx->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime)) {
        simple_mtx_unlock(&mx->resource_mutex);
        munmap(mapping, statbuf.st_size);
        close(import_fd);
        FREE(res);
        return NULL;
    }
    res->base = *templ;
    res->base.screen = screen;
    pipe_reference_init(&res->base.reference, 1);
    res->import_map = mapping;
    res->imported = true;
    res->import_fd = import_fd;
    atomic_init(&res->sync_failed, false);
    res->import_size = statbuf.st_size;
    res->import_offset = handle->offset;
    res->data = (unsigned char *)mapping + handle->offset;
    res->stride = handle->stride;
    res->size = bytes;
    res->level_stride[0] = handle->stride;
    res->gem = prime.handle;
    res->next = mx->resources;
    mx->resources = res;
    simple_mtx_unlock(&mx->resource_mutex);
    return &res->base;
}

static bool allocate_render_gem(struct mxgpu_screen *screen, struct mxgpu_resource *res)
{
    uint8_t record[MXGPU_DRM_HEADER_BYTES + 8u];
    uint32_t size, handle;
    uint64_t allocation = ((uint64_t)res->size + MXGPU_DRM_GEM_ALIGN - 1u) &
                          ~((uint64_t)MXGPU_DRM_GEM_ALIGN - 1u);
    struct mxgpu_drm_user user = {0};
    struct drm_prime_handle prime = {0};
    struct drm_gem_close close_arg;
    struct stat statbuf;
    void *mapping;
    if (!allocation || allocation > UINT_MAX ||
        mxgpu_drm_gem_create_encode(allocation, record, sizeof record, &size) != MXGPU_DRM_OK)
        return false;
    user.pointer = (uint64_t)(uintptr_t)record;
    user.size = size;
    user.capacity = sizeof record;
    if (ioctl(screen->fd, DRM_IOWR(DRM_COMMAND_BASE + 3, struct mxgpu_drm_user), &user) ||
        user.size > sizeof record ||
        mxgpu_drm_gem_create_response_decode(record, user.size, &handle) != MXGPU_DRM_OK)
        return false;
    close_arg.handle = handle;
    close_arg.pad = 0;
    prime.handle = handle;
    prime.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(screen->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime)) {
        ioctl(screen->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        return false;
    }
    if (fstat(prime.fd, &statbuf) || statbuf.st_size < (int64_t)allocation ||
        (uint64_t)statbuf.st_size > SIZE_MAX) {
        close(prime.fd);
        ioctl(screen->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        return false;
    }
    mapping = mmap(NULL, statbuf.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, prime.fd, 0);
    if (mapping == MAP_FAILED) {
        close(prime.fd);
        ioctl(screen->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        return false;
    }
    res->gem = handle;
    res->import_fd = prime.fd;
    res->import_map = mapping;
    res->import_size = statbuf.st_size;
    res->data = mapping;
    atomic_init(&res->sync_failed, false);
    return true;
}

static struct pipe_resource *mxgpu_resource_create(struct pipe_screen *screen, const struct pipe_resource *templ)
{
    struct mxgpu_resource *res = CALLOC_STRUCT(mxgpu_resource);
    unsigned block;
    if (!res)
        return NULL;
    res->base = *templ;
    res->base.screen = screen;
    pipe_reference_init(&res->base.reference, 1);
    block = util_format_get_blocksize(templ->format);
    if (templ->target == PIPE_BUFFER) {
        res->stride = block ? block : 1;
        res->size = templ->width0;
    } else {
        uint64_t size = 0;
        unsigned level;
        if (templ->last_level >= PIPE_MAX_TEXTURE_LEVELS) {
            FREE(res);
            return NULL;
        }
        for (level = 0; level <= templ->last_level; level++) {
            unsigned width = u_minify(templ->width0, level);
            unsigned height = u_minify(templ->height0, level);
            unsigned stride = util_format_get_stride(templ->format, width);
            uint64_t bytes = (uint64_t)stride * util_format_get_nblocksy(templ->format, height);
            unsigned layers = templ->target == PIPE_TEXTURE_3D ? u_minify(templ->depth0, level) : MAX2(templ->array_size, 1);
            if (bytes > UINT_MAX / layers) {
                FREE(res);
                return NULL;
            }
            bytes *= layers;
            if (size + bytes > UINT_MAX) {
                FREE(res);
                return NULL;
            }
            res->level_offset[level] = (unsigned)size;
            res->level_stride[level] = stride;
            size += bytes;
        }
        res->stride = res->level_stride[0];
        res->size = (unsigned)size;
    }
    if (res->size == 0)
        res->size = 1;
    if (screen_of(screen)->fd >= 0 && templ->target != PIPE_BUFFER && !templ->last_level && templ->array_size <= 1 && templ->depth0 <= 1 && block &&
        util_format_get_blockwidth(templ->format) == 1 &&
        util_format_get_blockheight(templ->format) == 1) {
        struct drm_mode_create_dumb create;
        struct drm_mode_map_dumb map;
        void *ptr;
        memset(&create, 0, sizeof create);
        create.width = templ->width0;
        create.height = templ->height0 ? templ->height0 : 1;
        create.bpp = block * 8u;
        if (!ioctl(screen_of(screen)->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create)) {
            memset(&map, 0, sizeof map);
            map.handle = create.handle;
            if (create.pitch >= res->stride && create.size <= UINT_MAX &&
                create.size >= (uint64_t)create.pitch * create.height &&
                !ioctl(screen_of(screen)->fd, DRM_IOCTL_MODE_MAP_DUMB, &map)) {
                ptr = mmap(NULL, create.size, PROT_READ | PROT_WRITE, MAP_SHARED, screen_of(screen)->fd, map.offset);
                if (ptr != MAP_FAILED) {
                    res->data = ptr;
                    res->gem = create.handle;
                    res->dumb = 1;
                    res->gem_dumb = 1;
                    res->stride = create.pitch;
                    res->level_stride[0] = create.pitch;
                    res->size = create.size;
                }
            }
            if (!res->dumb) {
                struct drm_mode_destroy_dumb destroy = { .handle = create.handle };
                ioctl(screen_of(screen)->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
            }
        }
    }
    if (!res->data && screen_of(screen)->fd >= 0 && templ->target != PIPE_BUFFER &&
        !templ->last_level && templ->depth0 <= 1 && templ->array_size <= 1 &&
        templ->nr_samples <= 1)
        allocate_render_gem(screen_of(screen), res);
    if (!res->data && (templ->bind & (PIPE_BIND_SCANOUT | PIPE_BIND_SHARED | PIPE_BIND_DISPLAY_TARGET))) {
        FREE(res);
        return NULL;
    }
    if (!res->data)
        res->data = CALLOC(1, res->size);
    if (!res->data) {
        FREE(res);
        return NULL;
    }
    simple_mtx_lock(&screen_of(screen)->resource_mutex);
    res->next = screen_of(screen)->resources;
    screen_of(screen)->resources = res;
    simple_mtx_unlock(&screen_of(screen)->resource_mutex);
    return &res->base;
}

static struct pipe_fence_handle *signaled_fence(void)
{
    static int token;
    return (struct pipe_fence_handle *)&token;
}

static void mxgpu_fence_reference(struct pipe_screen *screen, struct pipe_fence_handle **ptr, struct pipe_fence_handle *fence)
{
    (void)screen;
    if (ptr)
        *ptr = fence;
}

static bool mxgpu_fence_finish(struct pipe_screen *screen, struct pipe_context *ctx, struct pipe_fence_handle *fence, uint64_t timeout)
{
    (void)screen;
    (void)ctx;
    (void)fence;
    (void)timeout;
    return true;
}

static void *map_resource(struct pipe_context *pipe, struct pipe_resource *resource, unsigned level, unsigned usage, const struct pipe_box *box, struct pipe_transfer **out)
{
    struct mxgpu_resource *res = res_of(resource);
    struct pipe_transfer *xfer;
    uint64_t offset, span;
    unsigned rows, row_bytes;
    unsigned width, height, stride;
    (void)pipe;
    *out = NULL;
    if (level > resource->last_level || level >= PIPE_MAX_TEXTURE_LEVELS || !box || box->x < 0 || box->y < 0 || box->z < 0 ||
        box->width <= 0 || box->height <= 0 || box->depth <= 0 || !res->data)
        return NULL;
    width = u_minify(resource->width0, level);
    height = u_minify(resource->height0, level);
    stride = resource->target == PIPE_BUFFER ? res->stride : res->level_stride[level];
    if ((unsigned)box->x > width ||
        (unsigned)box->width > width - (unsigned)box->x)
        return NULL;
    if (resource->target == PIPE_BUFFER) {
        if (box->y || box->height != 1 || box->z || box->depth != 1)
            return NULL;
        offset = (unsigned)box->x;
        span = (unsigned)box->width;
    } else {
        if ((unsigned)box->y > height ||
            (unsigned)box->height > height - (unsigned)box->y ||
            (unsigned)box->x % util_format_get_blockwidth(resource->format) ||
            (unsigned)box->y % util_format_get_blockheight(resource->format))
            return NULL;
        unsigned layers = resource->target == PIPE_TEXTURE_3D ? u_minify(resource->depth0, level) : MAX2(resource->array_size, 1);
        if ((unsigned)box->z >= layers || (unsigned)box->depth > layers - (unsigned)box->z)
            return NULL;
        rows = util_format_get_nblocksy(resource->format, box->height);
        row_bytes = util_format_get_stride(resource->format, box->width);
        offset = res->level_offset[level] + (uint64_t)box->z * stride * util_format_get_nblocksy(resource->format, height) +
                 (uint64_t)util_format_get_nblocksy(resource->format, box->y) * stride +
                 util_format_get_stride(resource->format, box->x);
        span = (uint64_t)(box->depth - 1) * stride * util_format_get_nblocksy(resource->format, height) +
               (uint64_t)(rows - 1u) * stride + row_bytes;
    }
    if (offset > res->size || span > res->size - offset)
        return NULL;
    xfer = CALLOC_STRUCT(pipe_transfer);
    if (!xfer)
        return NULL;
    if (!resource_cpu_sync(res, usage, false)) {
        FREE(xfer);
        return NULL;
    }
    pipe_resource_reference(&xfer->resource, resource);
    xfer->level = level;
    xfer->usage = usage;
    xfer->box = *box;
    xfer->stride = stride;
    xfer->layer_stride = resource->target == PIPE_BUFFER ? 0 : stride * util_format_get_nblocksy(resource->format, height);
    *out = xfer;
    return res->data + (size_t)offset;
}

static void unmap_resource(struct pipe_context *pipe, struct pipe_transfer *xfer)
{
    (void)pipe;
    resource_cpu_sync(res_of(xfer->resource), xfer->usage, true);
    pipe_resource_reference(&xfer->resource, NULL);
    FREE(xfer);
}

static void *blob_state(const void *src, size_t bytes)
{
    void *dst = MALLOC(bytes);
    if (dst && src)
        memcpy(dst, src, bytes);
    return dst;
}

static void bind_nop(struct pipe_context *pipe, void *state)
{
    (void)pipe;
    (void)state;
}

static void bind_samplers(struct pipe_context *pipe, mesa_shader_stage shader, unsigned start, unsigned count, void **samplers)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    if (shader != MESA_SHADER_FRAGMENT || start >= MXGPU_SHADER_TEXTURES)
        return;
    count = MIN2(count, MXGPU_SHADER_TEXTURES - start);
    for (unsigned i = 0; i < count; i++) {
        ctx->sampler_bound[start + i] = samplers && samplers[i];
        if (ctx->sampler_bound[start + i])
            ctx->samplers[start + i] = *(struct pipe_sampler_state *)samplers[i];
    }
}

static void delete_blob(struct pipe_context *pipe, void *state)
{
    (void)pipe;
    FREE(state);
}

static void *create_blend(struct pipe_context *pipe, const struct pipe_blend_state *state)
{
    (void)pipe;
    return blob_state(state, sizeof *state);
}

static void bind_blend(struct pipe_context *pipe, void *state)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    ctx->blend_bound = state != NULL;
    if (state)
        ctx->blend = *(struct pipe_blend_state *)state;
    else
        memset(&ctx->blend, 0, sizeof ctx->blend);
}

static bool premultiplied_blend(const struct pipe_rt_blend_state *rt)
{
    return rt->rgb_func == PIPE_BLEND_ADD && rt->alpha_func == PIPE_BLEND_ADD &&
           rt->rgb_src_factor == PIPE_BLENDFACTOR_ONE &&
           rt->alpha_src_factor == PIPE_BLENDFACTOR_ONE &&
           rt->rgb_dst_factor == PIPE_BLENDFACTOR_INV_SRC_ALPHA &&
           rt->alpha_dst_factor == PIPE_BLENDFACTOR_INV_SRC_ALPHA;
}

static void compose_pixels(unsigned char *pixels, const unsigned char *background,
                           unsigned width, unsigned height, unsigned mask, bool source_over)
{
    size_t i, size = (size_t)width * height * 4u;
    for (i = 0; i < size; i += 4u) {
        unsigned alpha = pixels[i + 3u];
        unsigned channel;
        for (channel = 0; channel < 4; channel++) {
            unsigned value;
            if (!(mask & (1u << channel))) {
                pixels[i + channel] = background[i + channel];
                continue;
            }
            if (!source_over)
                continue;
            value = pixels[i + channel] +
                    (background[i + channel] * (255u - alpha) + 127u) / 255u;
            pixels[i + channel] = (unsigned char)MIN2(value, 255u);
        }
    }
}

static void *create_rast(struct pipe_context *pipe, const struct pipe_rasterizer_state *state)
{
    (void)pipe;
    return blob_state(state, sizeof *state);
}

static void *create_dsa(struct pipe_context *pipe, const struct pipe_depth_stencil_alpha_state *state)
{
    (void)pipe;
    return blob_state(state, sizeof *state);
}

static void bind_dsa(struct pipe_context *pipe, void *state)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    ctx->dsa_bound = state != NULL;
    if (state)
        ctx->dsa = *(const struct pipe_depth_stencil_alpha_state *)state;
    else
        memset(&ctx->dsa, 0, sizeof ctx->dsa);
}

static unsigned native_compare(unsigned func)
{
    switch (func) {
    case PIPE_FUNC_NEVER: return MXGPU_COMPARE_NEVER;
    case PIPE_FUNC_LESS: return MXGPU_COMPARE_LESS;
    case PIPE_FUNC_EQUAL: return MXGPU_COMPARE_EQUAL;
    case PIPE_FUNC_LEQUAL: return MXGPU_COMPARE_LESS_EQUAL;
    case PIPE_FUNC_GREATER: return MXGPU_COMPARE_GREATER;
    case PIPE_FUNC_NOTEQUAL: return MXGPU_COMPARE_NOT_EQUAL;
    case PIPE_FUNC_GEQUAL: return MXGPU_COMPARE_GREATER_EQUAL;
    case PIPE_FUNC_ALWAYS: return MXGPU_COMPARE_ALWAYS;
    default: return 0;
    }
}

static unsigned native_stencil_op(unsigned op)
{
    switch (op) {
    case PIPE_STENCIL_OP_KEEP: return MXGPU_STENCIL_KEEP;
    case PIPE_STENCIL_OP_ZERO: return MXGPU_STENCIL_ZERO;
    case PIPE_STENCIL_OP_REPLACE: return MXGPU_STENCIL_REPLACE;
    case PIPE_STENCIL_OP_INCR: return MXGPU_STENCIL_INCREMENT_CLAMP;
    case PIPE_STENCIL_OP_DECR: return MXGPU_STENCIL_DECREMENT_CLAMP;
    case PIPE_STENCIL_OP_INCR_WRAP: return MXGPU_STENCIL_INCREMENT_WRAP;
    case PIPE_STENCIL_OP_DECR_WRAP: return MXGPU_STENCIL_DECREMENT_WRAP;
    case PIPE_STENCIL_OP_INVERT: return MXGPU_STENCIL_INVERT;
    default: return 0;
    }
}

static bool native_depth_stencil_state(struct mxgpu_context *ctx,
                                       struct mxgpu_depth_stencil_state *state,
                                       uint32_t *reference)
{
    const struct pipe_depth_stencil_alpha_state *dsa = &ctx->dsa;
    const struct pipe_stencil_state *front = &dsa->stencil[0];
    const struct pipe_stencil_state *back = dsa->stencil[1].enabled ? &dsa->stencil[1] : front;
    memset(state, 0, sizeof *state);
    state->depth_compare = MXGPU_COMPARE_ALWAYS;
    state->front.compare = state->back.compare = MXGPU_COMPARE_ALWAYS;
    state->front.fail = state->front.depth_fail = state->front.pass = MXGPU_STENCIL_KEEP;
    state->back = state->front;
    *reference = ctx->stencil_ref.ref_value[0];
    if (!ctx->dsa_bound)
        return true;
    if (dsa->alpha_enabled || dsa->depth_bounds_test)
        return false;
    state->depth_test_enable = dsa->depth_enabled;
    state->depth_write_enable = dsa->depth_enabled && dsa->depth_writemask;
    state->depth_compare = native_compare(dsa->depth_func);
    if (!state->depth_compare)
        return false;
    state->stencil_enable = front->enabled;
    if (!front->enabled)
        return true;
    if (front->valuemask != back->valuemask || front->writemask != back->writemask ||
        (dsa->stencil[1].enabled && ctx->stencil_ref.ref_value[0] != ctx->stencil_ref.ref_value[1]))
        return false;
    state->stencil_read_mask = front->valuemask;
    state->stencil_write_mask = front->writemask;
    for (unsigned i = 0; i < 2; i++) {
        const struct pipe_stencil_state *src = i ? back : front;
        struct mxgpu_stencil_face *dst = i ? &state->back : &state->front;
        dst->compare = native_compare(src->func);
        dst->fail = native_stencil_op(src->fail_op);
        dst->depth_fail = native_stencil_op(src->zfail_op);
        dst->pass = native_stencil_op(src->zpass_op);
        if (!dst->compare || !dst->fail || !dst->depth_fail || !dst->pass)
            return false;
    }
    return true;
}

static void *create_sampler(struct pipe_context *pipe, const struct pipe_sampler_state *state)
{
    (void)pipe;
    return blob_state(state, sizeof *state);
}

static void set_blend_color(struct pipe_context *pipe, const struct pipe_blend_color *color)
{
    if (color)
        ctx_of(pipe)->blend_color = *color;
}

static void set_stencil_ref(struct pipe_context *pipe, const struct pipe_stencil_ref ref)
{
    ctx_of(pipe)->stencil_ref = ref;
}

static void set_sample_mask(struct pipe_context *pipe, unsigned mask)
{
    (void)pipe;
    (void)mask;
}

static void set_clip(struct pipe_context *pipe, const struct pipe_clip_state *state)
{
    (void)pipe;
    (void)state;
}

static void set_viewport(struct pipe_context *pipe, unsigned start, unsigned count, const struct pipe_viewport_state *state)
{
    if (!start && count && state) {
        ctx_of(pipe)->viewport = state[0];
        ctx_of(pipe)->viewport_bound = true;
    }
}

static void set_scissor(struct pipe_context *pipe, unsigned start, unsigned count, const struct pipe_scissor_state *state)
{
    if (!start && count && state)
        ctx_of(pipe)->scissor = state[0];
}

static void bind_rasterizer(struct pipe_context *pipe, void *state)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    ctx->rasterizer_bound = state != NULL;
    if (state)
        ctx->rasterizer = *(struct pipe_rasterizer_state *)state;
    else
        memset(&ctx->rasterizer, 0, sizeof ctx->rasterizer);
    ctx->scissor_enabled = state && ctx->rasterizer.scissor;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static uint8_t native_blend_factor(unsigned factor)
{
    switch (factor) {
    case PIPE_BLENDFACTOR_ZERO: return MXGPU_BLEND_ZERO;
    case PIPE_BLENDFACTOR_ONE: return MXGPU_BLEND_ONE;
    case PIPE_BLENDFACTOR_SRC_COLOR: return MXGPU_BLEND_SRC_COLOR;
    case PIPE_BLENDFACTOR_INV_SRC_COLOR: return MXGPU_BLEND_INV_SRC_COLOR;
    case PIPE_BLENDFACTOR_SRC_ALPHA: return MXGPU_BLEND_SRC_ALPHA;
    case PIPE_BLENDFACTOR_INV_SRC_ALPHA: return MXGPU_BLEND_INV_SRC_ALPHA;
    case PIPE_BLENDFACTOR_DST_ALPHA: return MXGPU_BLEND_DST_ALPHA;
    case PIPE_BLENDFACTOR_INV_DST_ALPHA: return MXGPU_BLEND_INV_DST_ALPHA;
    case PIPE_BLENDFACTOR_DST_COLOR: return MXGPU_BLEND_DST_COLOR;
    case PIPE_BLENDFACTOR_INV_DST_COLOR: return MXGPU_BLEND_INV_DST_COLOR;
    case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE: return MXGPU_BLEND_SRC_ALPHA_SATURATE;
    case PIPE_BLENDFACTOR_CONST_COLOR: return MXGPU_BLEND_CONSTANT_COLOR;
    case PIPE_BLENDFACTOR_INV_CONST_COLOR: return MXGPU_BLEND_INV_CONSTANT_COLOR;
    case PIPE_BLENDFACTOR_CONST_ALPHA: return MXGPU_BLEND_CONSTANT_ALPHA;
    case PIPE_BLENDFACTOR_INV_CONST_ALPHA: return MXGPU_BLEND_INV_CONSTANT_ALPHA;
    default: return 0;
    }
}

static uint8_t native_blend_operation(unsigned operation)
{
    switch (operation) {
    case PIPE_BLEND_ADD: return MXGPU_BLEND_ADD;
    case PIPE_BLEND_SUBTRACT: return MXGPU_BLEND_SUBTRACT;
    case PIPE_BLEND_REVERSE_SUBTRACT: return MXGPU_BLEND_REVERSE_SUBTRACT;
    case PIPE_BLEND_MIN: return MXGPU_BLEND_MIN;
    case PIPE_BLEND_MAX: return MXGPU_BLEND_MAX;
    default: return 0;
    }
}

static uint8_t native_address(unsigned wrap)
{
    switch (wrap) {
    case PIPE_TEX_WRAP_REPEAT: return MXGPU_ADDRESS_REPEAT;
    case PIPE_TEX_WRAP_MIRROR_REPEAT: return MXGPU_ADDRESS_MIRRORED_REPEAT;
    case PIPE_TEX_WRAP_CLAMP_TO_EDGE: return MXGPU_ADDRESS_CLAMP_TO_EDGE;
    case PIPE_TEX_WRAP_CLAMP_TO_BORDER: return MXGPU_ADDRESS_CLAMP_TO_BORDER;
    default: return 0;
    }
}

static bool native_sampler_state(const struct pipe_sampler_state *sampler, struct mxgpu_sampler_state *out)
{
    uint8_t bytes[MXGPU_SAMPLER_STATE_SIZE];
    memset(out, 0, sizeof *out);
    out->sampler_id = 1;
    out->max_anisotropy = 1;
    out->min_filter = out->mag_filter = out->mip_filter = MXGPU_FILTER_NEAREST;
    out->address_u = out->address_v = out->address_w = MXGPU_ADDRESS_REPEAT;
    if (sampler) {
        if ((sampler->compare_mode != PIPE_TEX_COMPARE_NONE && sampler->compare_mode != PIPE_TEX_COMPARE_R_TO_TEXTURE) || sampler->unnormalized_coords)
            return false;
        if (sampler->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE)
            out->compare = native_compare(sampler->compare_func);
        out->min_filter = sampler->min_img_filter == PIPE_TEX_FILTER_LINEAR ? MXGPU_FILTER_LINEAR : MXGPU_FILTER_NEAREST;
        out->mag_filter = sampler->mag_img_filter == PIPE_TEX_FILTER_LINEAR ? MXGPU_FILTER_LINEAR : MXGPU_FILTER_NEAREST;
        out->mip_filter = sampler->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR ? MXGPU_FILTER_LINEAR : MXGPU_FILTER_NEAREST;
        out->address_u = native_address(sampler->wrap_s);
        out->address_v = native_address(sampler->wrap_t);
        out->address_w = native_address(sampler->wrap_r);
        if (!out->address_u || !out->address_v || !out->address_w)
            return false;
        out->max_anisotropy = MAX2(1, sampler->max_anisotropy);
        out->mip_lod_bias = float_bits(sampler->lod_bias);
        if (sampler->min_mip_filter != PIPE_TEX_MIPFILTER_NONE) {
            out->min_lod = float_bits(sampler->min_lod);
            out->max_lod = float_bits(sampler->max_lod);
        }
        for (unsigned i = 0; i < 4; i++)
            out->border_color[i] = float_bits(sampler->border_color.f[i]);
    }
    return mxgpu_sampler_state_encode(out, bytes, sizeof bytes, NULL) == MX_OK;
}

static bool native_render_state(struct mxgpu_context *ctx, unsigned width, unsigned height,
                                struct mxgpu_native_render_state *state)
{
    struct mxgpu_blend_target *target;
    struct mxgpu_rasterizer_state *raster;
    const struct pipe_rasterizer_state *r = &ctx->rasterizer;
    unsigned i;
    memset(state, 0, sizeof *state);
    state->blend.state_id = 1;
    state->blend.target_count = 1;
    target = &state->blend.targets[0];
    target->write_mask = 15;
    target->src_color = target->src_alpha = MXGPU_BLEND_ONE;
    target->dst_color = target->dst_alpha = MXGPU_BLEND_ZERO;
    target->color_op = target->alpha_op = MXGPU_BLEND_ADD;
    if (ctx->blend_bound) {
        const struct pipe_rt_blend_state *rt = &ctx->blend.rt[0];
        if (ctx->blend.logicop_enable || ctx->blend.advanced_blend_func ||
            ctx->blend.alpha_to_coverage || ctx->blend.alpha_to_one)
            return false;
        target->enable = rt->blend_enable;
        target->write_mask = rt->colormask;
        if (target->enable) {
            target->src_color = native_blend_factor(rt->rgb_src_factor);
            target->dst_color = native_blend_factor(rt->rgb_dst_factor);
            target->color_op = native_blend_operation(rt->rgb_func);
            target->src_alpha = native_blend_factor(rt->alpha_src_factor);
            target->dst_alpha = native_blend_factor(rt->alpha_dst_factor);
            target->alpha_op = native_blend_operation(rt->alpha_func);
            if (!target->src_color || !target->dst_color || !target->color_op ||
                !target->src_alpha || !target->dst_alpha || !target->alpha_op)
                return false;
        }
    }
    for (i = 0; i < 4; i++) {
        state->blend_factor[i] = float_bits(ctx->blend_color.color[i]);
        if (!mxgpu_f32_finite(state->blend_factor[i])) return false;
    }
    raster = &state->rasterizer;
    raster->state_id = 1;
    raster->fill_mode = MXGPU_FILL_SOLID;
    raster->cull_mode = MXGPU_CULL_NONE;
    raster->front_face = MXGPU_FRONT_COUNTERCLOCKWISE;
    raster->depth_clip_enable = 1;
    if (ctx->rasterizer_bound) {
        if (r->fill_front != r->fill_back || r->fill_front == PIPE_POLYGON_MODE_POINT ||
            r->cull_face == PIPE_FACE_FRONT_AND_BACK || r->depth_clip_near != r->depth_clip_far ||
            r->offset_tri || r->poly_smooth || r->poly_stipple_enable)
            return false;
        raster->fill_mode = r->fill_front == PIPE_POLYGON_MODE_LINE ? MXGPU_FILL_WIREFRAME : MXGPU_FILL_SOLID;
        raster->cull_mode = r->cull_face == PIPE_FACE_FRONT ? MXGPU_CULL_FRONT :
                            r->cull_face == PIPE_FACE_BACK ? MXGPU_CULL_BACK : MXGPU_CULL_NONE;
        raster->front_face = r->front_ccw ? MXGPU_FRONT_COUNTERCLOCKWISE : MXGPU_FRONT_CLOCKWISE;
        raster->depth_clip_enable = r->depth_clip_near;
        raster->multisample_enable = r->multisample;
    }
    raster->scissor_enable = ctx->scissor_enabled;
    if (ctx->viewport_bound) {
        const struct pipe_viewport_state *v = &ctx->viewport;
        state->viewport.x = float_bits(v->translate[0] - v->scale[0]);
        state->viewport.y = float_bits(v->translate[1] + v->scale[1]);
        state->viewport.width = float_bits(2.f * v->scale[0]);
        state->viewport.height = float_bits(-2.f * v->scale[1]);
        state->viewport.min_depth = float_bits(v->translate[2] - v->scale[2]);
        state->viewport.max_depth = float_bits(v->translate[2] + v->scale[2]);
    } else {
        state->viewport.width = float_bits((float)width);
        state->viewport.height = float_bits((float)height);
        state->viewport.max_depth = float_bits(1.f);
    }
    state->scissor.right = width;
    state->scissor.bottom = height;
    if (ctx->scissor_enabled) {
        state->scissor.left = MIN2(ctx->scissor.minx, width);
        state->scissor.right = MIN2(ctx->scissor.maxx, width);
        state->scissor.top = MIN2(ctx->scissor.miny, height);
        state->scissor.bottom = MIN2(ctx->scissor.maxy, height);
    }
    return true;
}

static void restore_scissor_pixels(unsigned char *pixels, const unsigned char *preserved,
                                   unsigned width, unsigned height,
                                   const struct pipe_scissor_state *scissor)
{
    unsigned minx = MIN2(scissor->minx, width);
    unsigned maxx = MIN2(scissor->maxx, width);
    unsigned miny = MIN2(scissor->miny, height);
    unsigned maxy = MIN2(scissor->maxy, height);
    unsigned y;
    size_t stride = (size_t)width * 4u;
    for (y = 0; y < height; y++) {
        size_t offset = (size_t)(height - 1u - y) * stride;
        if (y < miny || y >= maxy || maxx <= minx) {
            memcpy(pixels + offset, preserved + offset, stride);
        } else {
            memcpy(pixels + offset, preserved + offset, (size_t)minx * 4u);
            memcpy(pixels + offset + (size_t)maxx * 4u,
                   preserved + offset + (size_t)maxx * 4u, (size_t)(width - maxx) * 4u);
        }
    }
}

static void set_constant(struct pipe_context *pipe, mesa_shader_stage shader, uint index, const struct pipe_constant_buffer *buf)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    struct pipe_constant_buffer next = {0};
    void *copy = NULL;
    unsigned stage;
    bool invalid = false;
    if (index >= MXGPU_UNIFORM_BUFFERS || (shader != MESA_SHADER_VERTEX && shader != MESA_SHADER_FRAGMENT))
        return;
    stage = shader == MESA_SHADER_FRAGMENT;
    if (buf && buf->buffer_size) {
        next.buffer_offset = buf->buffer_offset;
        next.buffer_size = buf->buffer_size;
        if (buf->buffer) {
            pipe_resource_reference(&next.buffer, buf->buffer);
        } else if (buf->user_buffer &&
                   (uintptr_t)buf->user_buffer <= UINTPTR_MAX - buf->buffer_offset &&
                   (uintptr_t)buf->user_buffer + buf->buffer_offset <= UINTPTR_MAX - buf->buffer_size) {
            copy = malloc(buf->buffer_size);
            if (copy) {
                memcpy(copy, (const unsigned char *)buf->user_buffer + buf->buffer_offset,
                       buf->buffer_size);
                next.user_buffer = copy;
                next.buffer_offset = 0;
            } else {
                invalid = true;
            }
        } else {
            invalid = true;
        }
    }
    pipe_resource_reference(&ctx->constants[stage][index].buffer, NULL);
    FREE(ctx->constant_copies[stage][index]);
    ctx->constants[stage][index] = next;
    ctx->constant_copies[stage][index] = copy;
    ctx->constant_invalid[stage][index] = invalid;
}

static bool constant_data(struct mxgpu_context *ctx, unsigned stage, unsigned index,
                          const void **data, uint32_t *size)
{
    const struct pipe_constant_buffer *buf = &ctx->constants[stage][index];
    *data = NULL;
    *size = 0;
    if (ctx->constant_invalid[stage][index])
        return false;
    if (!buf->buffer_size)
        return true;
    if (buf->buffer) {
        struct mxgpu_resource *res = res_of(buf->buffer);
        if (!res->data || buf->buffer_offset > res->size ||
            buf->buffer_size > res->size - buf->buffer_offset)
            return false;
        *data = res->data + buf->buffer_offset;
    } else {
        *data = buf->user_buffer;
    }
    if (!*data)
        return false;
    *size = buf->buffer_size;
    return true;
}

static bool copy_constant_bytes(struct mxgpu_context *ctx, unsigned stage, unsigned index,
                                 unsigned char *destination, unsigned bytes)
{
    const void *data;
    uint32_t size;
    if (!constant_data(ctx, stage, index, &data, &size) || bytes > size || (bytes && !data))
        return false;
    struct pipe_resource *buffer = ctx->constants[stage][index].buffer;
    if (buffer && !resource_cpu_sync(res_of(buffer), PIPE_MAP_READ, false))
        return false;
    if (bytes)
        memcpy(destination, data, bytes);
    return !buffer || resource_cpu_sync(res_of(buffer), PIPE_MAP_READ, true);
}

static bool collect_stage_uniforms(struct mxgpu_context *ctx, unsigned stage,
                                    const struct mxgpu_shader *shader,
                                    const void **data, uint32_t *size)
{
    *data = NULL;
    *size = 0;
    if (!shader || !shader->uses_uniforms)
        return true;
    if (!shader->uniform_count || shader->uniform_count > UINT32_MAX / 16)
        return false;
    unsigned bytes = shader->uniform_count * 16;
    unsigned legacy_bytes = bytes;
    for (unsigned i = 0; i < shader->uniform_buffer_count; i++)
        legacy_bytes = MIN2(legacy_bytes, shader->uniform_buffers[i].offset);
    unsigned char *packed = calloc(1, bytes);
    if (!packed)
        return false;
    if (legacy_bytes) {
        const void *legacy;
        uint32_t legacy_size;
        if (!constant_data(ctx, stage, 0, &legacy, &legacy_size) || !legacy_size ||
            ((uint64_t)legacy_size + 15) / 16 < legacy_bytes / 16 ||
            !copy_constant_bytes(ctx, stage, 0, packed, MIN2(legacy_bytes, legacy_size)))
            goto fail;
    }
    for (unsigned i = 0; i < shader->uniform_buffer_count; i++) {
        const struct mxgpu_uniform_buffer *ubo = &shader->uniform_buffers[i];
        if (!ubo->constant_slot || ubo->constant_slot >= MXGPU_UNIFORM_BUFFERS ||
            ubo->offset > bytes || ubo->size > bytes - ubo->offset ||
            !copy_constant_bytes(ctx, stage, ubo->constant_slot, packed + ubo->offset, ubo->size))
            goto fail;
    }
    *data = packed;
    *size = bytes;
    return true;
fail:
    free(packed);
    return false;
}

static bool lower_gl_position(nir_shader *nir)
{
    nir_foreach_function_impl(impl, nir) {
        nir_builder b = nir_builder_create(impl);
        nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
                if (instr->type != nir_instr_type_intrinsic)
                    continue;
                nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
                if (store->intrinsic != nir_intrinsic_store_deref)
                    continue;
                nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(store->src[0]));
                if (!var || var->data.mode != nir_var_shader_out || var->data.location != VARYING_SLOT_POS)
                    continue;
                nir_def *position = store->src[1].ssa;
                if (position->num_components != 4 || position->bit_size != 32 ||
                    nir_intrinsic_write_mask(store) != 15)
                    return false;
                b.cursor = nir_before_instr(instr);
                nir_def *z = nir_fmul_imm(&b, nir_fadd(&b, nir_channel(&b, position, 2),
                                                     nir_channel(&b, position, 3)), 0.5f);
                nir_def *converted = nir_vec4(&b, nir_channel(&b, position, 0),
                                               nir_channel(&b, position, 1), z,
                                               nir_channel(&b, position, 3));
                nir_src_rewrite(&store->src[1], converted);
            }
        }
        nir_progress(true, impl, nir_metadata_control_flow);
    }
    return true;
}

static void *create_shader(struct pipe_context *pipe, const struct pipe_shader_state *state, bool fragment)
{
    struct mxgpu_shader *shader;
    struct mxgpu_gallium_shader *storage;
    int result;
    (void)pipe;
    if (!state || state->type != PIPE_SHADER_IR_NIR || !state->ir.nir)
        return NULL;
    storage = CALLOC_STRUCT(mxgpu_gallium_shader);
    shader = storage ? &storage->shader : NULL;
    if (!shader) {
        ralloc_free(state->ir.nir);
        return NULL;
    }
    if (!fragment) {
        nir_shader *gl = nir_shader_clone(NULL, state->ir.nir);
        storage->gl_position = CALLOC_STRUCT(mxgpu_shader);
        if (!gl || !storage->gl_position || !lower_gl_position(gl) ||
            mxgpu_compile_nir(gl, false, storage->gl_position)) {
            FREE(storage->gl_position);
            storage->gl_position = NULL;
        }
        ralloc_free(gl);
    }
    result = mxgpu_compile_nir(state->ir.nir, fragment, shader);
    ralloc_free(state->ir.nir);
    if (result != 0) {
        FREE(storage->gl_position);
        FREE(shader);
        return NULL;
    }
    return shader;
}

static void *create_fs(struct pipe_context *pipe, const struct pipe_shader_state *state)
{
    return create_shader(pipe, state, true);
}

static void *create_vs(struct pipe_context *pipe, const struct pipe_shader_state *state)
{
    return create_shader(pipe, state, false);
}

static void bind_fs(struct pipe_context *pipe, void *state)
{
    ctx_of(pipe)->fs = state;
}

static void bind_vs(struct pipe_context *pipe, void *state)
{
    ctx_of(pipe)->vs = state;
}

static void delete_shader(struct pipe_context *pipe, void *state)
{
    (void)pipe;
    if (state)
        FREE(((struct mxgpu_gallium_shader *)state)->gl_position);
    FREE(state);
}

static void view_identity_key(const struct pipe_sampler_view *view, uint32_t key[16])
{
    memset(key, 0, 16 * sizeof(*key));
    key[0] = view->format;
    key[1] = view->target;
    key[2] = view->astc_decode_format;
    key[3] = view->is_tex2d_from_buf;
    key[4] = view->swizzle_r;
    key[5] = view->swizzle_g;
    key[6] = view->swizzle_b;
    key[7] = view->swizzle_a;
    if (view->is_tex2d_from_buf) {
        key[8] = view->u.tex2d_from_buf.offset;
        key[9] = view->u.tex2d_from_buf.row_stride;
        key[10] = view->u.tex2d_from_buf.width;
        key[11] = view->u.tex2d_from_buf.height;
    } else if (view->target == PIPE_BUFFER) {
        key[8] = view->u.buf.offset;
        key[9] = view->u.buf.size;
    } else {
        key[8] = view->u.tex.first_layer;
        key[9] = view->u.tex.last_layer;
        key[10] = view->u.tex.first_level;
        key[11] = view->u.tex.last_level;
        memcpy(&key[12], &view->u.tex.min_lod_clamp, sizeof(key[12]));
    }
}

static struct pipe_sampler_view *create_view(struct pipe_context *pipe, struct pipe_resource *texture, const struct pipe_sampler_view *templ)
{
    struct mxgpu_sampler_view *mxview = CALLOC_STRUCT(mxgpu_sampler_view);
    struct mxgpu_resource *res = res_of(texture);
    struct mxgpu_screen *screen = screen_of(texture->screen);
    struct mxgpu_view_identity *entry;
    uint32_t key[16];
    if (!mxview)
        return NULL;
    view_identity_key(templ, key);
    simple_mtx_lock(&screen->resource_mutex);
    for (entry = res->view_identities; entry; entry = entry->next)
        if (!memcmp(entry->key, key, sizeof(key)))
            break;
    if (!entry) {
        uint_fast64_t identity = atomic_load_explicit(&next_view_identity, memory_order_relaxed);
        entry = CALLOC_STRUCT(mxgpu_view_identity);
        if (!entry)
            goto fail;
        do {
            if (identity == UINT64_MAX) {
                FREE(entry);
                goto fail;
            }
        } while (!atomic_compare_exchange_weak_explicit(&next_view_identity, &identity, identity + 1,
                                                        memory_order_relaxed, memory_order_relaxed));
        memcpy(entry->key, key, sizeof(key));
        entry->identity = identity;
        entry->next = res->view_identities;
        res->view_identities = entry;
    }
    mxview->identity = entry->identity;
    simple_mtx_unlock(&screen->resource_mutex);
    struct pipe_sampler_view *view = &mxview->base;
    *view = *templ;
    view->context = pipe;
    view->texture = NULL;
    pipe_reference_init(&view->reference, 1);
    pipe_resource_reference(&view->texture, texture);
    return view;
fail:
    simple_mtx_unlock(&screen->resource_mutex);
    FREE(mxview);
    return NULL;
}

static void destroy_view(struct pipe_context *pipe, struct pipe_sampler_view *view)
{
    (void)pipe;
    pipe_resource_reference(&view->texture, NULL);
    FREE(view);
}

static void set_views(struct pipe_context *pipe, mesa_shader_stage shader, unsigned start, unsigned count, unsigned unbind, struct pipe_sampler_view **views)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    if (shader != MESA_SHADER_FRAGMENT || start >= MXGPU_SHADER_TEXTURES)
        return;
    count = MIN2(count, MXGPU_SHADER_TEXTURES - start);
    for (unsigned i = 0; i < count; i++)
        pipe_sampler_view_reference(&ctx->views[start + i], views ? views[i] : NULL);
    for (unsigned i = start + count; i < MXGPU_SHADER_TEXTURES && unbind; i++, unbind--)
        pipe_sampler_view_reference(&ctx->views[i], NULL);
    ctx->view = ctx->views[0];
}

static void *create_velems(struct pipe_context *pipe, unsigned count, const struct pipe_vertex_element *el)
{
    struct mxgpu_velem_state *state = CALLOC_STRUCT(mxgpu_velem_state);
    (void)pipe;
    if (!state)
        return NULL;
    state->count = count;
    if (count > PIPE_MAX_ATTRIBS)
        count = PIPE_MAX_ATTRIBS;
    memcpy(state->el, el, count * sizeof *el);
    state->count = count;
    return state;
}

static void bind_velems(struct pipe_context *pipe, void *state)
{
    ctx_of(pipe)->velems = state;
}

static void set_vbs(struct pipe_context *pipe, unsigned count, const struct pipe_vertex_buffer *buffers)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    unsigned i;
    for (i = 0; i < ctx->vb_count; i++) {
        if (!ctx->vb[i].is_user_buffer)
            pipe_resource_reference(&ctx->vb[i].buffer.resource, NULL);
    }
    memset(ctx->vb, 0, sizeof ctx->vb);
    if (count > PIPE_MAX_ATTRIBS)
        count = PIPE_MAX_ATTRIBS;
    ctx->vb_count = count;
    for (i = 0; i < count; i++) {
        ctx->vb[i] = buffers[i];
        ctx->vb[i].buffer.resource = NULL;
        if (!buffers[i].is_user_buffer)
            pipe_resource_reference(&ctx->vb[i].buffer.resource, buffers[i].buffer.resource);
        else
            ctx->vb[i].buffer.user = buffers[i].buffer.user;
    }
}

static void set_fb(struct pipe_context *pipe, const struct pipe_framebuffer_state *fb)
{
    util_copy_framebuffer_state(&ctx_of(pipe)->fb, fb);
}

static bool track_framebuffer(struct mxgpu_context *ctx, struct pipe_resource *resource)
{
    struct mxgpu_pending *entry;
    for (entry = ctx->pending; entry; entry = entry->next)
        if (entry->resource == resource)
            return true;
    entry = CALLOC_STRUCT(mxgpu_pending);
    if (!entry)
        return false;
    pipe_resource_reference(&entry->resource, resource);
    entry->next = ctx->pending;
    ctx->pending = entry;
    return true;
}

static void mxgpu_flush(struct pipe_context *pipe, struct pipe_fence_handle **fence, unsigned flags)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    struct mxgpu_pending **link = &ctx->pending;
    bool success = true;
    (void)flags;
    if (mxgpu_device_flush()) {
        if (fence)
            *fence = NULL;
        return;
    }
    while (*link) {
        struct mxgpu_pending *entry = *link;
        struct mxgpu_resource *res = res_of(entry->resource);
        if (res->external || res->imported) {
            if (!resource_publish(res)) {
                success = false;
                link = &entry->next;
                continue;
            }
            res->cpu_revision++;
        }
        *link = entry->next;
        pipe_resource_reference(&entry->resource, NULL);
        FREE(entry);
    }
    if (fence)
        *fence = success ? signaled_fence() : NULL;
}

static const void *vertex_ptr(struct mxgpu_context *ctx, unsigned index, uint64_t offset, unsigned bytes)
{
    const struct pipe_vertex_buffer *vb;
    uint64_t start;
    if (index >= ctx->vb_count)
        return NULL;
    vb = &ctx->vb[index];
    if (offset > UINT64_MAX - vb->buffer_offset)
        return NULL;
    start = (uint64_t)vb->buffer_offset + offset;
    if (start > SIZE_MAX || bytes > SIZE_MAX - start)
        return NULL;
    if (vb->is_user_buffer) {
        if (!vb->buffer.user)
            return NULL;
        return (const unsigned char *)vb->buffer.user + (size_t)start;
    }
    if (!vb->buffer.resource)
        return NULL;
    if (start > res_of(vb->buffer.resource)->size ||
        bytes > res_of(vb->buffer.resource)->size - start)
        return NULL;
    return res_of(vb->buffer.resource)->data + (size_t)start;
}

static unsigned vertex_stride_floats(struct mxgpu_context *ctx)
{
    return ctx->vs && ctx->vs->vertex_attribute_count ? ctx->vs->vertex_attribute_count * 4u : 4u;
}

static bool gather_vertices(struct mxgpu_context *ctx, const unsigned *indices, unsigned start, unsigned count, float *out)
{
    unsigned i, a;
    unsigned attributes = vertex_stride_floats(ctx) / 4u;
    if (attributes > MXGPU_SHADER_VERTEX_SLOTS)
        return false;
    for (i = 0; i < count; i++) {
        uint64_t vertex = indices ? indices[i] : (uint64_t)start + i;
        if (vertex > UINT_MAX)
            return false;
        for (a = 0; a < attributes; a++) {
            float *slot = out + ((size_t)i * attributes + a) * 4u;
            slot[0] = slot[1] = slot[2] = 0.f;
            slot[3] = 1.f;
            if (ctx->vs && ctx->vs->vertex_builtins && a == ctx->vs->vertex_builtin_slot) {
                uint32_t values[4] = {(uint32_t)vertex, ctx->draw_instance,
                                      (uint32_t)ctx->draw_base_vertex, ctx->draw_base_instance};
                memcpy(slot, values, sizeof values);
                continue;
            }
            if (!ctx->velems || a >= ctx->velems->count) {
                if (ctx->vs && ctx->vs->vertex_input_locations[a] != UINT32_MAX)
                    return false;
                continue;
            }
            const struct pipe_vertex_element *el = &ctx->velems->el[a];
            uint64_t element = el->instance_divisor ? (uint64_t)ctx->draw_base_instance +
                               ctx->draw_instance / el->instance_divisor : vertex;
            if (el->src_stride && element > (UINT64_MAX - el->src_offset) / el->src_stride)
                return false;
            const void *src = vertex_ptr(ctx, el->vertex_buffer_index,
                                         element * el->src_stride + el->src_offset,
                                         util_format_get_blocksize((enum pipe_format)el->src_format));
            if (!src)
                return false;
            util_format_unpack_rgba((enum pipe_format)el->src_format, slot, src, 1);
        }
    }
    return true;
}

static unsigned color_row_fast_format(enum pipe_format format);
static void color_row_copy(unsigned char *dst, const unsigned char *src, unsigned width, unsigned fast);

static bool texture_framebuffer_source(struct pipe_sampler_view *view,
                                       struct pipe_resource *destination,
                                       struct mxgpu_texture_input *input)
{
    struct mxgpu_resource *res;
    if (!view || !view->texture || view->texture == destination ||
        view->target != PIPE_TEXTURE_2D || view->texture->target != PIPE_TEXTURE_2D ||
        view->texture->array_size > 1 || view->is_tex2d_from_buf ||
        (view->format != PIPE_FORMAT_R8G8B8A8_UNORM && view->format != PIPE_FORMAT_B8G8R8A8_UNORM) ||
        view->texture->format != view->format ||
        view->u.tex.first_level || view->u.tex.last_level ||
        view->u.tex.first_layer || view->u.tex.last_layer || view->u.tex.min_lod_clamp != 0.0f ||
        view->swizzle_r != PIPE_SWIZZLE_X || view->swizzle_g != PIPE_SWIZZLE_Y ||
        view->swizzle_b != PIPE_SWIZZLE_Z || view->swizzle_a != PIPE_SWIZZLE_W)
        return false;
    res = res_of(view->texture);
    if (res->imported || res->external || !res->framebuffer || !res->framebuffer_rendered ||
        res->framebuffer_revision != res->cpu_revision)
        return false;
    input->framebuffer = res->framebuffer;
    input->framebuffer_revision = res->framebuffer_revision;
    input->width = res->base.width0;
    input->height = res->base.height0;
    return true;
}

static bool read_texture_slice(struct pipe_sampler_view *view, unsigned level, unsigned layer, bool depth, unsigned char **out_data, unsigned *w, unsigned *h)
{
    struct mxgpu_resource *res;
    const struct util_format_unpack_description *unpack;
    unsigned x, y, width, height, block_height, rows, row_bytes;
    unsigned swizzles[4], fast;
    bool identity;
    uint64_t span;
    unsigned char *dst;
    float *stripe;
    *out_data = NULL;
    if (!view || !view->texture || view->texture->target == PIPE_BUFFER ||
        util_format_is_pure_integer(view->format))
        return false;
    unpack = util_format_unpack_description(view->format);
    if (!unpack || (depth ? !unpack->unpack_z_float : (!unpack->unpack_rgba && !unpack->unpack_rgba_rect)))
        return false;
    res = res_of(view->texture);
    if (level > view->texture->last_level || level >= PIPE_MAX_TEXTURE_LEVELS || !res->data)
        return false;
    width = u_minify(view->texture->width0, level);
    height = u_minify(view->texture->height0, level);
    if (!width || !height || width > 4096 || height > 4096)
        return false;
    block_height = util_format_get_blockheight(view->format);
    rows = util_format_get_nblocksy(view->format, height);
    row_bytes = util_format_get_stride(view->format, width);
    if (!block_height || !rows || !row_bytes || res->level_stride[level] < row_bytes)
        return false;
    span = (uint64_t)(rows - 1) * res->level_stride[level] + row_bytes;
    unsigned layers = view->texture->target == PIPE_TEXTURE_3D ? u_minify(view->texture->depth0, level) : MAX2(view->texture->array_size, 1);
    uint64_t layer_offset = (uint64_t)layer * res->level_stride[level] * util_format_get_nblocksy(view->format, height);
    uint64_t offset = (uint64_t)res->level_offset[level] + layer_offset;
    if (layer >= layers || offset > res->size || span > res->size - offset)
        return false;
    swizzles[0] = view->swizzle_r;
    swizzles[1] = view->swizzle_g;
    swizzles[2] = view->swizzle_b;
    swizzles[3] = view->swizzle_a;
    for (x = 0; x < 4; x++) {
        if (swizzles[x] > PIPE_SWIZZLE_1)
            return false;
    }
    fast = depth ? 0 : color_row_fast_format(view->format);
    identity = swizzles[0] == PIPE_SWIZZLE_X && swizzles[1] == PIPE_SWIZZLE_Y &&
               swizzles[2] == PIPE_SWIZZLE_Z && swizzles[3] == PIPE_SWIZZLE_W;
    dst = malloc((size_t)width * height * 4u);
    stripe = fast ? NULL : malloc((size_t)width * block_height * 4u * sizeof(float));
    if (!dst || (!fast && !stripe)) {
        free(dst);
        free(stripe);
        return false;
    }
    if (!resource_cpu_sync(res, PIPE_MAP_READ, false)) {
        free(dst);
        free(stripe);
        return false;
    }
    for (y = 0; y < height; y += block_height) {
        unsigned row, stripe_height = MIN2(block_height, height - y);
        const unsigned char *src = res->data + offset +
            (size_t)(y / block_height) * res->level_stride[level];
        if (fast) {
            unsigned char *out = dst + (size_t)y * width * 4u;
            color_row_copy(out, src, width, fast);
            if (!identity) {
                for (x = 0; x < width; x++) {
                    unsigned char pixel[4];
                    memcpy(pixel, out + (size_t)x * 4u, sizeof pixel);
                    for (unsigned c = 0; c < 4; c++)
                        out[(size_t)x * 4u + c] = swizzles[c] <= PIPE_SWIZZLE_W ? pixel[swizzles[c]] :
                                                   swizzles[c] == PIPE_SWIZZLE_1 ? 255 : 0;
                }
            }
            continue;
        }
        if (depth) {
            unpack->unpack_z_float(stripe, width * sizeof(float), src, res->level_stride[level], width, stripe_height);
            for (unsigned pixel = 0; pixel < width * stripe_height; pixel++) {
                uint32_t bits = float_bits(stripe[pixel]);
                unsigned char *out = dst + ((size_t)y * width + pixel) * 4u;
                for (unsigned byte = 0; byte < 4; byte++)
                    out[byte] = (unsigned char)(bits >> (byte * 8));
            }
            continue;
        }
        util_format_unpack_rgba_rect(view->format, stripe, width * 4u * sizeof(float),
                                     src, res->level_stride[level], width, stripe_height);
        for (row = 0; row < stripe_height; row++) {
            for (x = 0; x < width; x++) {
                const float *pixel = stripe + ((size_t)row * width + x) * 4u;
                unsigned char *out = dst + ((size_t)(y + row) * width + x) * 4u;
                unsigned c;
                for (c = 0; c < 4; c++) {
                    unsigned swizzle = swizzles[c];
                    float value = swizzle <= PIPE_SWIZZLE_W ? pixel[swizzle] :
                                  swizzle == PIPE_SWIZZLE_1 ? 1.f : 0.f;
                    out[c] = !(value > 0.f) ? 0 : value >= 1.f ? 255 : (unsigned char)(value * 255.f + 0.5f);
                }
            }
        }
    }
    free(stripe);
    if (!resource_cpu_sync(res, PIPE_MAP_READ, true)) {
        free(dst);
        return false;
    }
    *out_data = dst;
    *w = width;
    *h = height;
    return true;
}

static unsigned color_row_fast_format(enum pipe_format format)
{
#if UTIL_ARCH_LITTLE_ENDIAN
    switch (format) {
    case PIPE_FORMAT_R8G8B8A8_UNORM: return 1;
    case PIPE_FORMAT_B8G8R8A8_UNORM: return 2;
    case PIPE_FORMAT_R8G8B8X8_UNORM: return 3;
    case PIPE_FORMAT_B8G8R8X8_UNORM: return 4;
    default: break;
    }
#endif
    return 0;
}

static bool color_rows_fit(const struct mxgpu_resource *res, unsigned level,
                           unsigned width, unsigned height, unsigned block)
{
    uint64_t row = (uint64_t)width * block;
    uint64_t end;
    if (!res->data || !width || !height || !block || row > res->level_stride[level])
        return false;
    end = res->level_offset[level] + (uint64_t)(height - 1) * res->level_stride[level] + row;
    if (end > res->size)
        return false;
    return !res->import_map || (res->import_offset <= res->import_size &&
                              end <= res->import_size - res->import_offset);
}

static void color_row_copy(unsigned char *dst, const unsigned char *src,
                           unsigned width, unsigned fast)
{
    if (fast == 1) {
        memcpy(dst, src, (size_t)width * 4);
        return;
    }
#if UTIL_ARCH_LITTLE_ENDIAN
    size_t bytes = (size_t)width * 4;
    uintptr_t destination = (uintptr_t)dst, source = (uintptr_t)src;
    if (fast >= 2 && fast <= 4 &&
        (destination == source || (destination < source ? source - destination >= bytes :
                                    destination - source >= bytes))) {
        bool swap = fast == 2 || fast == 4;
        uint64_t keep = swap ? UINT64_C(0xff00ff00ff00ff00) : UINT64_MAX;
        uint64_t low = swap ? UINT64_C(0x000000ff000000ff) : 0;
        uint64_t high = swap ? UINT64_C(0x00ff000000ff0000) : 0;
        uint64_t opaque = fast >= 3 ? UINT64_C(0xff000000ff000000) : 0;
        unsigned x = 0;
        for (; width - x >= 2; x += 2) {
            uint64_t value;
            memcpy(&value, src, sizeof value);
            value = (value & keep) | ((value & low) << 16) | ((value & high) >> 16) | opaque;
            memcpy(dst, &value, sizeof value);
            src += sizeof value;
            dst += sizeof value;
        }
        if (x < width) {
            uint32_t value;
            memcpy(&value, src, sizeof value);
            value = (value & (uint32_t)keep) | ((value & (uint32_t)low) << 16) |
                    ((value & (uint32_t)high) >> 16) | (uint32_t)opaque;
            memcpy(dst, &value, sizeof value);
        }
        return;
    }
#endif
    for (unsigned x = 0; x < width; x++) {
        unsigned char r = src[0], g = src[1], b = src[2], a = src[3];
        dst[0] = (fast == 2 || fast == 4) ? b : r;
        dst[1] = g;
        dst[2] = (fast == 2 || fast == 4) ? r : b;
        dst[3] = fast >= 3 ? 255 : a;
        src += 4;
        dst += 4;
    }
}

static bool read_texture_level(struct pipe_sampler_view *view, unsigned level, unsigned char **out_data, unsigned *w, unsigned *h)
{
    return view && read_texture_slice(view, level, view->u.tex.first_layer, false, out_data, w, h);
}

static bool read_texture_faces(struct pipe_sampler_view *view, unsigned level, bool cube, bool depth,
                               unsigned char **out_data, unsigned *width, unsigned *height)
{
    unsigned faces = cube ? 6 : 1;
    unsigned first = view->u.tex.first_layer;
    unsigned char *packed = NULL;
    *out_data = NULL;
    if (cube && (view->texture->target != PIPE_TEXTURE_CUBE || view->u.tex.last_layer < first ||
                 view->u.tex.last_layer - first != 5))
        return false;
    for (unsigned face = 0; face < faces; face++) {
        unsigned char *copy = NULL;
        unsigned w, h;
        if (!read_texture_slice(view, level, first + face, depth, &copy, &w, &h)) {
            free(packed);
            return false;
        }
        if (!face) {
            if (cube && w != h) {
                free(copy);
                return false;
            }
            *width = w;
            *height = h;
            packed = malloc((size_t)w * h * 4u * faces);
            if (!packed) {
                free(copy);
                return false;
            }
        }
        memcpy(packed + (size_t)face * w * h * 4u, copy, (size_t)w * h * 4u);
        free(copy);
    }
    *out_data = packed;
    return true;
}

static bool read_texture(struct pipe_sampler_view *view, unsigned char **out_data, unsigned *w, unsigned *h)
{
    return view && read_texture_level(view, view->u.tex.first_level, out_data, w, h);
}

static bool read_color_rows(struct pipe_surface *surf, unsigned char *rgba,
                       unsigned width, unsigned height, bool reversed)
{
    struct mxgpu_resource *res;
    unsigned x, y, block, fast;
    if (!surf || !surf->texture || !rgba || !width || !height ||
        surf->level > surf->texture->last_level || surf->level >= PIPE_MAX_TEXTURE_LEVELS ||
        width > u_minify(surf->texture->width0, surf->level) ||
        height > u_minify(surf->texture->height0, surf->level))
        return false;
    res = res_of(surf->texture);
    block = util_format_get_blocksize(surf->format);
    if (!color_rows_fit(res, surf->level, width, height, block))
        return false;
    fast = color_row_fast_format(surf->format);
    if (!resource_cpu_sync(res, PIPE_MAP_READ, false))
        return false;
    for (y = 0; y < height; y++) {
        if (fast) {
            color_row_copy(rgba + (size_t)(reversed ? height - 1u - y : y) * width * 4,
                           res->data + res->level_offset[surf->level] +
                           (size_t)y * res->level_stride[surf->level], width, fast);
            continue;
        }
        for (x = 0; x < width; x += 1024) {
            float row[1024 * 4];
            unsigned count = MIN2(1024, width - x);
            unsigned i, c;
            util_format_unpack_rgba(surf->format, row,
                                   res->data + res->level_offset[surf->level] +
                                   (size_t)y * res->level_stride[surf->level] + (size_t)x * block,
                                   count);
            for (i = 0; i < count; i++) {
                unsigned char *p = rgba + ((size_t)(reversed ? height - 1u - y : y) * width + x + i) * 4u;
                for (c = 0; c < 4; c++) {
                    float v = row[i * 4 + c];
                    p[c] = !(v > 0.f) ? 0 : v >= 1.f ? 255 : (unsigned char)(v * 255.f + 0.5f);
                }
            }
        }
    }
    return resource_cpu_sync(res, PIPE_MAP_READ, true);
}

static bool write_color_rows(struct pipe_surface *surf, const unsigned char *rgba,
                        unsigned width, unsigned height, bool reversed)
{
    struct mxgpu_resource *res;
    unsigned x, y, copy_width, copy_height, block, fast;
    if (!surf || !surf->texture || !rgba || !width || !height ||
        surf->level > surf->texture->last_level || surf->level >= PIPE_MAX_TEXTURE_LEVELS)
        return false;
    res = res_of(surf->texture);
    copy_width = MIN2(width, u_minify(surf->texture->width0, surf->level));
    copy_height = MIN2(height, u_minify(surf->texture->height0, surf->level));
    block = util_format_get_blocksize(surf->format);
    if (!color_rows_fit(res, surf->level, copy_width, copy_height, block))
        return false;
    fast = color_row_fast_format(surf->format);
    if (!resource_cpu_sync(res, PIPE_MAP_WRITE, false))
        return false;
    for (y = 0; y < copy_height; y++) {
        if (fast) {
            color_row_copy(res->data + res->level_offset[surf->level] +
                           (size_t)y * res->level_stride[surf->level],
                           rgba + (size_t)(reversed ? height - 1u - y : y) * width * 4,
                           copy_width, fast);
            continue;
        }
        for (x = 0; x < copy_width; x += 1024) {
            float row[1024 * 4];
            unsigned count = MIN2(1024, copy_width - x);
            unsigned i;
            for (i = 0; i < count; i++) {
                const unsigned char *p = rgba + ((size_t)(reversed ? height - 1u - y : y) * width + x + i) * 4u;
                row[i * 4 + 0] = (float)p[0] / 255.f;
                row[i * 4 + 1] = (float)p[1] / 255.f;
                row[i * 4 + 2] = (float)p[2] / 255.f;
                row[i * 4 + 3] = (float)p[3] / 255.f;
            }
            util_format_pack_rgba(surf->format,
                                 res->data + res->level_offset[surf->level] +
                                 (size_t)y * res->level_stride[surf->level] + (size_t)x * block,
                                 row, count);
        }
    }
    return resource_cpu_sync(res, PIPE_MAP_WRITE, true);
}

static bool resource_publish(struct mxgpu_resource *res)
{
    struct pipe_surface surface = {0};
    int changed;
    bool success;
    if (!res->framebuffer_dirty)
        return true;
    if (mxgpu_framebuffer_sync(res->framebuffer, res->framebuffer_pixels, &changed))
        return false;
    surface.texture = &res->base;
    surface.format = res->base.format;
    res->framebuffer_publishing = true;
    success = write_color_rows(&surface, res->framebuffer_pixels,
                              res->base.width0, res->base.height0, false);
    res->framebuffer_publishing = false;
    if (success)
        res->framebuffer_dirty = false;
    return success;
}

static unsigned native_depth_format(enum pipe_format format)
{
#if UTIL_ARCH_LITTLE_ENDIAN
    switch (format) {
    case PIPE_FORMAT_Z16_UNORM:
    case PIPE_FORMAT_Z32_FLOAT: return MXGPU_FMT_DEPTH32_FLOAT;
    case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT: return MXGPU_FMT_DEPTH32_FLOAT_STENCIL8;
    case PIPE_FORMAT_S8_UINT_Z24_UNORM:
    case PIPE_FORMAT_Z24_UNORM_S8_UINT:
    case PIPE_FORMAT_Z24X8_UNORM: return MXGPU_FMT_DEPTH24_UNORM_STENCIL8;
    default: break;
    }
#endif
    return 0;
}

static unsigned native_depth_block(unsigned format)
{
    return format == MXGPU_FMT_DEPTH32_FLOAT_STENCIL8 ? 8u : 4u;
}

static bool depth_rows_copy(struct pipe_surface *surf, unsigned char *pixels,
                            unsigned width, unsigned height, bool write)
{
    struct mxgpu_resource *res;
    unsigned block, wire_block, flags;
    if (!surf || !surf->texture || !pixels || !native_depth_format(surf->format) ||
        surf->level > surf->texture->last_level || surf->level >= PIPE_MAX_TEXTURE_LEVELS ||
        width > u_minify(surf->texture->width0, surf->level) ||
        height > u_minify(surf->texture->height0, surf->level))
        return false;
    res = res_of(surf->texture);
    block = util_format_get_blocksize(surf->format);
    if (!color_rows_fit(res, surf->level, width, height, block))
        return false;
    wire_block = native_depth_block(native_depth_format(surf->format));
    if (write && surf->format == PIPE_FORMAT_Z16_UNORM) {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            float depth;
            memcpy(&depth, pixels + i * wire_block, sizeof depth);
            if (!isfinite(depth))
                return false;
        }
    }
    flags = write ? PIPE_MAP_WRITE : PIPE_MAP_READ;
    if (!resource_cpu_sync(res, flags, false))
        return false;
    for (unsigned y = 0; y < height; y++) {
        unsigned char *resource_row = res->data + res->level_offset[surf->level] +
                                      (size_t)y * res->level_stride[surf->level];
        unsigned char *packed_row = pixels + (size_t)y * width * wire_block;
        if (surf->format == PIPE_FORMAT_Z16_UNORM) {
            for (unsigned x = 0; x < width; x++) {
                uint16_t z;
                float depth;
                if (write) {
                    memcpy(&depth, packed_row + (size_t)x * wire_block, sizeof depth);
                    z = depth <= 0.f ? 0 : depth >= 1.f ? UINT16_MAX :
                        (uint16_t)((double)depth * UINT16_MAX + 0.5);
                    memcpy(resource_row + (size_t)x * block, &z, sizeof z);
                } else {
                    memcpy(&z, resource_row + (size_t)x * block, sizeof z);
                    depth = (float)z / (float)UINT16_MAX;
                    memcpy(packed_row + (size_t)x * wire_block, &depth, sizeof depth);
                }
            }
        } else if (surf->format == PIPE_FORMAT_S8_UINT_Z24_UNORM) {
            for (unsigned x = 0; x < width; x++) {
                uint32_t value;
                if (write) {
                    memcpy(&value, packed_row + (size_t)x * wire_block, sizeof value);
                    value = (value << 8) | (value >> 24);
                    memcpy(resource_row + (size_t)x * block, &value, sizeof value);
                } else {
                    memcpy(&value, resource_row + (size_t)x * block, sizeof value);
                    value = (value >> 8) | (value << 24);
                    memcpy(packed_row + (size_t)x * wire_block, &value, sizeof value);
                }
            }
        } else if (write && surf->format == PIPE_FORMAT_Z24X8_UNORM) {
            for (unsigned x = 0; x < width; x++)
                memcpy(resource_row + (size_t)x * block, packed_row + (size_t)x * block, 3);
        } else if (write && surf->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
            for (unsigned x = 0; x < width; x++)
                memcpy(resource_row + (size_t)x * block, packed_row + (size_t)x * block, 5);
        } else if (write)
            memcpy(resource_row, packed_row, (size_t)width * block);
        else
            memcpy(packed_row, resource_row, (size_t)width * block);
        if (surf->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT && !write)
            for (unsigned x = 0; x < width; x++)
                memset(packed_row + (size_t)x * block + 5, 0, 3);
        if (surf->format == PIPE_FORMAT_Z24X8_UNORM && !write)
            for (unsigned x = 0; x < width; x++)
                packed_row[(size_t)x * block + 3] = 0;
    }
    return resource_cpu_sync(res, flags, true);
}

static int64_t div_floor(int64_t num, int64_t den)
{
    if (den < 0) {
        num = -num;
        den = -den;
    }
    if (den == 0)
        return 0;
    if (num >= 0)
        return num / den;
    return -(((-num) + den - 1) / den);
}

static void mxgpu_blit_cpu(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
    struct mxgpu_resource *src, *dst;
    unsigned src_bpp, dst_bpp, src_width, src_height, dst_width, dst_height;
    int64_t dw, dh, left, right, top, bottom, dx, dy;
    const struct util_format_unpack_description *src_unpack, *dst_unpack;
    const struct util_format_pack_description *dst_pack;
    unsigned mask;
    bool depth_stencil, linear;
    unsigned char *snapshot = NULL;
    const unsigned char *source_data;
    (void)pipe;
    if (!info || !info->src.resource || !info->dst.resource)
        return;
    if (info->src.level > info->src.resource->last_level ||
        info->dst.level > info->dst.resource->last_level ||
        info->src.level >= PIPE_MAX_TEXTURE_LEVELS || info->dst.level >= PIPE_MAX_TEXTURE_LEVELS)
        return;
    src_width = u_minify(info->src.resource->width0, info->src.level);
    src_height = u_minify(info->src.resource->height0, info->src.level);
    dst_width = u_minify(info->dst.resource->width0, info->dst.level);
    dst_height = u_minify(info->dst.resource->height0, info->dst.level);
    src = res_of(info->src.resource);
    dst = res_of(info->dst.resource);
    dw = info->dst.box.width;
    dh = info->dst.box.height;
    if (dw < 0)
        dw = -dw;
    if (dh < 0)
        dh = -dh;
    if (!src->data || !dst->data || !dw || !dh ||
        !info->src.box.width || !info->src.box.height)
        return;
    left = info->dst.box.width < 0 ? (int64_t)info->dst.box.x - dw : info->dst.box.x;
    top = info->dst.box.height < 0 ? (int64_t)info->dst.box.y - dh : info->dst.box.y;
    right = MIN2(left + dw, dst_width);
    bottom = MIN2(top + dh, dst_height);
    left = MAX2(left, 0);
    top = MAX2(top, 0);
    if (info->scissor_enable) {
        left = MAX2(left, info->scissor.minx);
        top = MAX2(top, info->scissor.miny);
        right = MIN2(right, info->scissor.maxx);
        bottom = MIN2(bottom, info->scissor.maxy);
    }
    src_bpp = util_format_get_blocksize(info->src.format);
    dst_bpp = util_format_get_blocksize(info->dst.format);
    mask = info->mask;
    depth_stencil = util_format_is_depth_or_stencil(info->src.format) ||
                    util_format_is_depth_or_stencil(info->dst.format);
    linear = info->filter == PIPE_TEX_FILTER_LINEAR;
    if (info->filter != PIPE_TEX_FILTER_NEAREST && !linear)
        return;
    if (linear && (depth_stencil || util_format_is_pure_integer(info->src.format) ||
                   util_format_is_pure_integer(info->dst.format)))
        return;
    src_unpack = util_format_unpack_description(info->src.format);
    dst_unpack = util_format_unpack_description(info->dst.format);
    dst_pack = util_format_pack_description(info->dst.format);
    if (!mask || !src_bpp || !dst_bpp || src_bpp > 16 || dst_bpp > 16 ||
        util_format_get_blockwidth(info->src.format) != 1 || util_format_get_blockheight(info->src.format) != 1 ||
        util_format_get_blockwidth(info->dst.format) != 1 || util_format_get_blockheight(info->dst.format) != 1 ||
        !src_unpack || !dst_unpack || !dst_pack ||
        !color_rows_fit(src, info->src.level, src_width, src_height, src_bpp) ||
        !color_rows_fit(dst, info->dst.level, dst_width, dst_height, dst_bpp))
        return;
    if (depth_stencil) {
        if (!(mask & PIPE_MASK_ZS) ||
            ((mask & PIPE_MASK_Z) && (!src_unpack->unpack_z_float || !dst_pack->pack_z_float)) ||
            ((mask & PIPE_MASK_S) && (!src_unpack->unpack_s_8uint || !dst_pack->pack_s_8uint)))
            return;
    } else if (!(mask & PIPE_MASK_RGBA) || !src_unpack->unpack_rgba ||
               !dst_unpack->unpack_rgba ||
               (util_format_is_pure_uint(info->dst.format) ? !dst_pack->pack_rgba_uint :
                util_format_is_pure_sint(info->dst.format) ? !dst_pack->pack_rgba_sint : !dst_pack->pack_rgba_float)) {
        return;
    }
    source_data = src->data;
    if (src == dst) {
        snapshot = malloc(src->size);
        if (!snapshot)
            return;
        memcpy(snapshot, src->data, src->size);
        source_data = snapshot;
    }
    for (dy = top; dy < bottom; dy++) {
        int64_t y = info->dst.box.height < 0 ? (int64_t)info->dst.box.y - 1 - dy : dy - info->dst.box.y;
        int64_t sy = (int64_t)info->src.box.y + div_floor((2 * y + 1) * info->src.box.height, 2 * dh);
        for (dx = left; dx < right; dx++) {
            int64_t x = info->dst.box.width < 0 ? (int64_t)info->dst.box.x - 1 - dx : dx - info->dst.box.x;
            int64_t sx = (int64_t)info->src.box.x + div_floor((2 * x + 1) * info->src.box.width, 2 * dw);
            unsigned char *dp = dst->data + dst->level_offset[info->dst.level] +
                                (size_t)dy * dst->level_stride[info->dst.level] + (size_t)dx * dst_bpp;
            if (sx < 0 || sy < 0 || (uint64_t)sx >= src_width || (uint64_t)sy >= src_height)
                continue;
            const unsigned char *sp = source_data + src->level_offset[info->src.level] +
                                      (size_t)sy * src->level_stride[info->src.level] + (size_t)sx * src_bpp;
            if (depth_stencil) {
                unsigned char previous[16];
                memcpy(previous, dp, dst_bpp);
                if ((mask & PIPE_MASK_Z) && info->src.format == info->dst.format) {
                    unsigned char merged[16];
                    memcpy(merged, sp, src_bpp);
                    if (info->dst.format == PIPE_FORMAT_Z24X8_UNORM)
                        merged[3] = dp[3];
                    if (info->dst.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)
                        memcpy(merged + 5, dp + 5, 3);
                    if (!(mask & PIPE_MASK_S) && dst_pack->pack_s_8uint && dst_unpack->unpack_s_8uint) {
                        uint8_t stencil;
                        util_format_unpack_s_8uint(info->dst.format, &stencil, dp, 1);
                        util_format_pack_s_8uint(info->dst.format, merged, &stencil, 1);
                    }
                    memcpy(dp, merged, dst_bpp);
                } else if (mask & PIPE_MASK_Z) {
                    float depth;
                    util_format_unpack_z_float(info->src.format, &depth, sp, 1);
                    util_format_pack_z_float(info->dst.format, dp, &depth, 1);
                }
                if (mask & PIPE_MASK_S) {
                    uint8_t stencil;
                    util_format_unpack_s_8uint(info->src.format, &stencil, sp, 1);
                    util_format_pack_s_8uint(info->dst.format, dp, &stencil, 1);
                }
                if (info->dst.format == PIPE_FORMAT_Z24X8_UNORM)
                    dp[3] = previous[3];
                if (info->dst.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)
                    memcpy(dp + 5, previous + 5, 3);
            } else {
                union pipe_color_union pixel, previous;
                if (linear) {
                    double sample_x = (double)info->src.box.x +
                        ((double)x + 0.5) * info->src.box.width / (double)dw - 0.5;
                    double sample_y = (double)info->src.box.y +
                        ((double)y + 0.5) * info->src.box.height / (double)dh - 0.5;
                    int64_t x0 = (int64_t)floor(sample_x), y0 = (int64_t)floor(sample_y);
                    double fx = sample_x - x0, fy = sample_y - y0;
                    float samples[4][4];
                    for (unsigned row = 0; row < 2; row++) {
                        int64_t py = CLAMP(y0 + row, 0, (int64_t)src_height - 1);
                        for (unsigned col = 0; col < 2; col++) {
                            int64_t px = CLAMP(x0 + col, 0, (int64_t)src_width - 1);
                            const unsigned char *sample = source_data + src->level_offset[info->src.level] +
                                (size_t)py * src->level_stride[info->src.level] + (size_t)px * src_bpp;
                            util_format_unpack_rgba(info->src.format, samples[row * 2 + col], sample, 1);
                        }
                    }
                    for (unsigned c = 0; c < 4; c++) {
                        double top_value = samples[0][c] * (1.0 - fx) + samples[1][c] * fx;
                        double bottom_value = samples[2][c] * (1.0 - fx) + samples[3][c] * fx;
                        pixel.f[c] = top_value * (1.0 - fy) + bottom_value * fy;
                    }
                } else {
                    util_format_unpack_rgba(info->src.format, &pixel, sp, 1);
                }
                if ((mask & PIPE_MASK_RGBA) != PIPE_MASK_RGBA) {
                    util_format_unpack_rgba(info->dst.format, &previous, dp, 1);
                    for (unsigned c = 0; c < 4; c++)
                        if (!(mask & (1u << c)))
                            pixel.ui[c] = previous.ui[c];
                }
                util_format_pack_rgba(info->dst.format, dp, &pixel, 1);
            }
        }
    }
    free(snapshot);
}

static void mxgpu_blit(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
    struct mxgpu_resource *src, *dst;
    if (!info || !info->src.resource || !info->dst.resource)
        return;
    src = res_of(info->src.resource);
    dst = res_of(info->dst.resource);
    if (src == dst) {
        if (!resource_cpu_sync(src, PIPE_MAP_READ | PIPE_MAP_WRITE, false))
            return;
        mxgpu_blit_cpu(pipe, info);
        resource_cpu_sync(src, PIPE_MAP_READ | PIPE_MAP_WRITE, true);
        return;
    }
    if (!resource_cpu_sync(src, PIPE_MAP_READ, false))
        return;
    if (!resource_cpu_sync(dst, PIPE_MAP_READ | PIPE_MAP_WRITE, false)) {
        resource_cpu_sync(src, PIPE_MAP_READ, true);
        return;
    }
    mxgpu_blit_cpu(pipe, info);
    resource_cpu_sync(dst, PIPE_MAP_READ | PIPE_MAP_WRITE, true);
    resource_cpu_sync(src, PIPE_MAP_READ, true);
}

static void mxgpu_copy_region(struct pipe_context *pipe, struct pipe_resource *dst, unsigned dst_level, unsigned dstx, unsigned dsty, unsigned dstz, struct pipe_resource *src, unsigned src_level, const struct pipe_box *box)
{
    unsigned src_width, src_height, dst_width, dst_height;
    if (!box || !dst || !src)
        return;
    if (src->target == PIPE_BUFFER || dst->target == PIPE_BUFFER) {
        struct mxgpu_resource *source = res_of(src);
        struct mxgpu_resource *dest = res_of(dst);
        if (src->target != PIPE_BUFFER || dst->target != PIPE_BUFFER ||
            src_level || dst_level || dsty || dstz || box->y || box->z ||
            box->height != 1 || box->depth != 1 || box->x < 0 || box->width <= 0 ||
            (unsigned)box->x > source->size ||
            (unsigned)box->width > source->size - (unsigned)box->x ||
            dstx > dest->size || (unsigned)box->width > dest->size - dstx ||
            !source->data || !dest->data)
            return;
        memmove(dest->data + dstx, source->data + (unsigned)box->x, (unsigned)box->width);
        return;
    }
    if (dst_level > dst->last_level || src_level > src->last_level ||
        dst_level >= PIPE_MAX_TEXTURE_LEVELS || src_level >= PIPE_MAX_TEXTURE_LEVELS)
        return;
    src_width = u_minify(src->width0, src_level);
    src_height = u_minify(src->height0, src_level);
    dst_width = u_minify(dst->width0, dst_level);
    dst_height = u_minify(dst->height0, dst_level);
    if (dstz || box->z || box->depth != 1 ||
        box->x < 0 || box->y < 0 || box->width <= 0 || box->height <= 0 ||
        (unsigned)box->x > src_width || (unsigned)box->y > src_height ||
        (unsigned)box->width > src_width - (unsigned)box->x ||
        (unsigned)box->height > src_height - (unsigned)box->y ||
        dstx > dst_width || dsty > dst_height ||
        (unsigned)box->width > dst_width - dstx ||
        (unsigned)box->height > dst_height - dsty ||
        util_format_get_blocksize(src->format) != util_format_get_blocksize(dst->format) ||
        util_format_get_blockwidth(src->format) != util_format_get_blockwidth(dst->format) ||
        util_format_get_blockheight(src->format) != util_format_get_blockheight(dst->format))
        return;
    if ((unsigned)box->x % util_format_get_blockwidth(src->format) ||
        (unsigned)box->y % util_format_get_blockheight(src->format) ||
        dstx % util_format_get_blockwidth(dst->format) ||
        dsty % util_format_get_blockheight(dst->format))
        return;
    util_resource_copy_region(pipe, dst, dst_level, dstx, dsty, dstz, src, src_level, box);
}

static void set_stipple(struct pipe_context *pipe, const struct pipe_poly_stipple *stipple)
{
    (void)pipe;
    (void)stipple;
}

static void set_windows(struct pipe_context *pipe, bool include, unsigned num, const struct pipe_scissor_state *rects)
{
    (void)pipe;
    (void)include;
    (void)num;
    (void)rects;
}

static void set_locations(struct pipe_context *pipe, size_t size, const uint8_t *locations)
{
    (void)pipe;
    (void)size;
    (void)locations;
}

static void set_min_samples(struct pipe_context *pipe, unsigned samples)
{
    (void)pipe;
    (void)samples;
}

static void set_tess(struct pipe_context *pipe, const float outer[4], const float inner[2])
{
    (void)pipe;
    (void)outer;
    (void)inner;
}

static void set_patch(struct pipe_context *pipe, uint8_t count)
{
    (void)pipe;
    (void)count;
}

static void set_inlinable(struct pipe_context *pipe, mesa_shader_stage shader, uint count, uint32_t *values)
{
    (void)pipe;
    (void)shader;
    (void)count;
    (void)values;
}

static void set_ssbo(struct pipe_context *pipe, mesa_shader_stage shader, unsigned start, unsigned count, const struct pipe_shader_buffer *buffers, unsigned writable)
{
    (void)pipe;
    (void)shader;
    (void)start;
    (void)count;
    (void)buffers;
    (void)writable;
}

static void set_atomics(struct pipe_context *pipe, unsigned start, unsigned count, const struct pipe_shader_buffer *buffers)
{
    (void)pipe;
    (void)start;
    (void)count;
    (void)buffers;
}

static void set_images(struct pipe_context *pipe, mesa_shader_stage shader, unsigned start, unsigned count, unsigned unbind, const struct pipe_image_view *images)
{
    (void)pipe;
    (void)shader;
    (void)start;
    (void)count;
    (void)unbind;
    (void)images;
}

static void barrier_nop(struct pipe_context *pipe, unsigned flags)
{
    (void)pipe;
    (void)flags;
}

static void resource_nop(struct pipe_context *pipe, struct pipe_resource *resource)
{
    (void)pipe;
    if (!mxgpu_device_flush() && resource && resource_publish(res_of(resource)))
        res_of(resource)->cpu_revision++;
}

static void sample_pos(struct pipe_context *pipe, unsigned count, unsigned index, float *out)
{
    (void)pipe;
    (void)count;
    (void)index;
    if (out) {
        out[0] = 0.5f;
        out[1] = 0.5f;
    }
}

static enum pipe_reset_status reset_status(struct pipe_context *pipe)
{
    (void)pipe;
    return mxgpu_device_lost() ? PIPE_UNKNOWN_CONTEXT_RESET : PIPE_NO_RESET;
}

static bool read_indices_with_restart(const struct pipe_draw_info *info,
                         const struct pipe_draw_start_count_bias *draw, unsigned *indices, bool *restarts)
{
    const unsigned char *base;
    uint64_t offset, bytes;
    unsigned n;
    if (info->index_size != 1 && info->index_size != 2 && info->index_size != 4)
        return false;
    offset = (uint64_t)draw->start * info->index_size;
    bytes = (uint64_t)draw->count * info->index_size;
    if (offset > SIZE_MAX || bytes > SIZE_MAX - offset)
        return false;
    if (info->has_user_indices) {
        base = info->index.user;
    } else {
        struct mxgpu_resource *res;
        if (!info->index.resource)
            return false;
        res = res_of(info->index.resource);
        if (offset > res->size || bytes > res->size - offset)
            return false;
        base = res->data;
    }
    if (!base)
        return false;
    base += (size_t)offset;
    for (n = 0; n < draw->count; n++) {
        uint32_t index = 0;
        int64_t vertex;
        if (info->index_size == 1)
            index = base[n];
        else if (info->index_size == 2) {
            uint16_t value;
            memcpy(&value, base + (size_t)n * 2, sizeof value);
            index = value;
        } else {
            memcpy(&index, base + (size_t)n * 4, sizeof index);
        }
        if (restarts) {
            restarts[n] = info->primitive_restart && index == info->restart_index;
            if (restarts[n]) {
                indices[n] = 0;
                continue;
            }
        }
        vertex = (int64_t)index + draw->index_bias;
        if (vertex < 0 || vertex > UINT_MAX)
            return false;
        indices[n] = (unsigned)vertex;
    }
    return true;
}

static bool read_indices(const struct pipe_draw_info *info,
                         const struct pipe_draw_start_count_bias *draw, unsigned *indices)
{
    return read_indices_with_restart(info, draw, indices, NULL);
}

static float *triangulate_vertices_stride(float *vertices, unsigned count, enum mesa_prim mode, unsigned *out_count, unsigned stride)
{
    float *triangles;
    uint64_t expanded;
    unsigned primitive, n;
    *out_count = 0;
    if (mode == MESA_PRIM_TRIANGLES) {
        *out_count = count - count % 3u;
        if (*out_count)
            return vertices;
        free(vertices);
        return NULL;
    }
    if (mode == MESA_PRIM_TRIANGLE_STRIP || mode == MESA_PRIM_TRIANGLE_FAN)
        expanded = count >= 3 ? (uint64_t)(count - 2) * 3u : 0;
    else if (mode == MESA_PRIM_QUADS)
        expanded = (uint64_t)(count / 4u) * 6u;
    else if (mode == MESA_PRIM_QUAD_STRIP)
        expanded = count >= 4 ? (uint64_t)(count / 2u - 1) * 6u : 0;
    else
        expanded = 0;
    if (!expanded || expanded > UINT_MAX || expanded > SIZE_MAX / (stride * sizeof(float))) {
        free(vertices);
        return NULL;
    }
    triangles = malloc((size_t)expanded * stride * sizeof(float));
    if (!triangles) {
        free(vertices);
        return NULL;
    }
    for (primitive = 0; primitive < expanded / 3u; primitive++) {
        unsigned indices[3];
        if (mode == MESA_PRIM_TRIANGLE_STRIP) {
            indices[0] = primitive + (primitive & 1u);
            indices[1] = primitive + 1u - (primitive & 1u);
            indices[2] = primitive + 2u;
        } else if (mode == MESA_PRIM_TRIANGLE_FAN) {
            indices[0] = 0;
            indices[1] = primitive + 1u;
            indices[2] = primitive + 2u;
        } else if (mode == MESA_PRIM_QUADS) {
            unsigned base = primitive / 2u * 4u;
            indices[0] = base;
            indices[1] = base + (primitive & 1u ? 2u : 1u);
            indices[2] = base + (primitive & 1u ? 3u : 2u);
        } else {
            unsigned base = primitive / 2u * 2u;
            indices[0] = base;
            indices[1] = base + (primitive & 1u ? 3u : 1u);
            indices[2] = base + (primitive & 1u ? 2u : 3u);
        }
        for (n = 0; n < 3; n++)
            memcpy(triangles + (size_t)(primitive * 3u + n) * stride,
                   vertices + (size_t)indices[n] * stride, stride * sizeof(float));
    }
    free(vertices);
    *out_count = (unsigned)expanded;
    return triangles;
}

static float *gather_restart_triangles(struct mxgpu_context *ctx, const unsigned *indices,
                                      const bool *restarts, unsigned count, enum mesa_prim mode,
                                      unsigned *out_count)
{
    float *result;
    unsigned cursor = 0, total = 0;
    unsigned stride = vertex_stride_floats(ctx);
    uint64_t capacity = (uint64_t)count * 3u;
    *out_count = 0;
    if (!capacity || capacity > UINT_MAX || capacity > SIZE_MAX / (stride * sizeof(float)))
        return NULL;
    result = malloc((size_t)capacity * stride * sizeof(float));
    if (!result)
        return NULL;
    while (cursor < count) {
        unsigned begin, length, triangles;
        float *segment;
        while (cursor < count && restarts[cursor])
            cursor++;
        begin = cursor;
        while (cursor < count && !restarts[cursor])
            cursor++;
        length = cursor - begin;
        if (length < ((mode == MESA_PRIM_QUADS || mode == MESA_PRIM_QUAD_STRIP) ? 4u : 3u))
            continue;
        segment = malloc((size_t)length * stride * sizeof(float));
        if (!segment || !gather_vertices(ctx, indices + begin, 0, length, segment)) {
            free(segment);
            free(result);
            return NULL;
        }
        segment = triangulate_vertices_stride(segment, length, mode, &triangles, stride);
        if (!segment) {
            free(result);
            return NULL;
        }
        memcpy(result + (size_t)total * stride, segment, (size_t)triangles * stride * sizeof(float));
        total += triangles;
        free(segment);
    }
    if (!total) {
        free(result);
        return NULL;
    }
    *out_count = total;
    return result;
}

static void mxgpu_draw(struct pipe_context *pipe, const struct pipe_draw_info *info, unsigned drawid_offset, const struct pipe_draw_indirect_info *indirect, const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    unsigned char *texels = NULL;
    struct mxgpu_texture_input textures[MXGPU_SHADER_TEXTURES] = {0};
    unsigned char *texture_copies[MXGPU_SHADER_TEXTURES] = {0};
    struct mxgpu_texture_level texture_levels[MXGPU_SHADER_TEXTURES][PIPE_MAX_TEXTURE_LEVELS] = {0};
    struct mxgpu_shader *vs = ctx->vs;
    bool has_color = ctx->fb.nr_cbufs && ctx->fb.cbufs[0].texture;
    struct mxgpu_shader *fs = ctx->fs;
    unsigned texture_count = 0;
    bool bound_sampler = false;
    uint32_t sampler_compare[MXGPU_SHADER_TEXTURES] = {0};
    unsigned char *pixels;
    unsigned char *depth_pixels = NULL;
    unsigned char *preserved = NULL;
    unsigned tw = 0, th = 0, cw, ch;
    unsigned draw, color_mask = 15u;
    unsigned vertex_stride = vertex_stride_floats(ctx);
    bool source_over = false;
    bool native = false;
    bool deferred = false;
    struct mxgpu_resource *target = NULL;
    struct mxgpu_native_render_state native_state;
    const void *vs_constants, *fs_constants;
    uint32_t vs_constant_size, fs_constant_size;
    (void)drawid_offset;
    (void)indirect;
    if (!ctx->vs || !ctx->vs->count || (fs ? !fs->count : has_color || !ctx->fb.zsbuf.texture) || (!has_color && !ctx->fb.zsbuf.texture)) {
        static int reported;
        if (!reported) {
            fprintf(stderr, "mxgpu draw skip vs %u fs %u cbufs %u\n",
                    ctx->vs ? ctx->vs->count : 0, ctx->fs ? ctx->fs->count : 0, ctx->fb.nr_cbufs);
            reported = 1;
        }
        return;
    }
    if (ctx->blend_bound) {
        const struct pipe_rt_blend_state *rt = &ctx->blend.rt[0];
        if (ctx->blend.logicop_enable || ctx->blend.advanced_blend_func)
            return;
        color_mask = rt->colormask;
        source_over = rt->blend_enable;

    }
    vs_constants = NULL;
    fs_constants = NULL;
    vs_constant_size = 0;
    fs_constant_size = 0;
    cw = ctx->fb.width;
    ch = ctx->fb.height;
    if (cw == 0 || ch == 0 || cw > 2048 || ch > 2048)
        return;
    if (ctx->scissor_enabled &&
        (ctx->scissor.minx >= MIN2(ctx->scissor.maxx, cw) ||
         ctx->scissor.miny >= MIN2(ctx->scissor.maxy, ch)))
        return;
    if (mxgpu_native_render_available(screen_of(pipe->screen)->fd)) {
        if (!native_render_state(ctx, cw, ch, &native_state))
            return;
        native = true;
        if (!ctx->rasterizer_bound || !ctx->rasterizer.clip_halfz) {
            vs = ((struct mxgpu_gallium_shader *)ctx->vs)->gl_position;
            if (!vs)
                return;
        } else if (ctx->viewport_bound) {
            native_state.viewport.min_depth = float_bits(ctx->viewport.translate[2]);
            native_state.viewport.max_depth = float_bits(ctx->viewport.translate[2] + ctx->viewport.scale[2]);
        }
        if (!has_color)
            native_state.blend.targets[0].write_mask = 0;
    }
    if (ctx->dsa_bound && (ctx->dsa.alpha_enabled || ctx->dsa.depth_bounds_test))
        return;
    if (ctx->fb.zsbuf.texture && ctx->dsa_bound &&
        (ctx->dsa.depth_enabled || ctx->dsa.stencil[0].enabled)) {
        unsigned format = native_depth_format(ctx->fb.zsbuf.format);
        unsigned block = native_depth_block(format);
        if (!native || !format || !mxgpu_native_depth_available(screen_of(pipe->screen)->fd, format) ||
            !native_depth_stencil_state(ctx, &native_state.depth_stencil, &native_state.stencil_reference))
            return;
        depth_pixels = malloc((size_t)cw * ch * block);
        if (!depth_pixels)
            return;
        if (!depth_rows_copy(&ctx->fb.zsbuf, depth_pixels, cw, ch, false))
            goto free_textures;
        native_state.depth_enabled = 1;
        native_state.depth.pixels = depth_pixels;
        native_state.depth.format = format;
        native_state.depth.width = cw;
        native_state.depth.height = ch;
    }
    if (!native && (!has_color || !color_mask || (source_over && !premultiplied_blend(&ctx->blend.rt[0]))))
        goto free_textures;
    if (!collect_stage_uniforms(ctx, 0, vs, &vs_constants, &vs_constant_size) ||
        !collect_stage_uniforms(ctx, 1, fs, &fs_constants, &fs_constant_size))
        goto free_textures;
    texture_count = fs ? fs->texture_count : 0;
    bound_sampler = native && texture_count && mxgpu_native_sampler_available(screen_of(pipe->screen)->fd);
    if (texture_count > MXGPU_SHADER_TEXTURES)
        goto free_textures;
    for (unsigned i = 0; i < texture_count; i++) {
        const struct mxgpu_texture_binding *binding = &fs->textures[i];
        unsigned slot;
        if (binding->set || binding->binding >= MXGPU_SHADER_TEXTURES ||
            binding->element >= MXGPU_SHADER_TEXTURES - binding->binding)
            goto free_textures;
        slot = binding->binding + binding->element;
        bool cube = binding->binding_kind == MXSB_BINDING_TEXTURE_CUBE;
        bool shadow = binding->shadow;
        if (!ctx->views[slot] || (cube && (!native || !mxgpu_native_cube_caps(screen_of(pipe->screen)->fd))) ||
            (shadow && (!native || !mxgpu_native_shadow_caps(screen_of(pipe->screen)->fd))))
            goto free_textures;
        bool lease_source = native && !cube && !shadow &&
            texture_framebuffer_source(ctx->views[slot], has_color ? ctx->fb.cbufs[0].texture : NULL, &textures[i]);
        if (!lease_source &&
            !read_texture_faces(ctx->views[slot], ctx->views[slot]->u.tex.first_level, cube, shadow,
                                &texture_copies[i], &textures[i].width, &textures[i].height))
            goto free_textures;
        textures[i].binding_kind = cube ? MXGPU_BIND_KIND_TEXTURE_CUBE : MXGPU_BIND_KIND_TEXTURE_2D;
        textures[i].identity = ((struct mxgpu_sampler_view *)ctx->views[slot])->identity;
        textures[i].array_layers = cube ? 6 : 1;
        textures[i].format = shadow ? MXGPU_FMT_DEPTH32_FLOAT : MXGPU_FMT_RGBA8_UNORM;
        if (!lease_source && native && mxgpu_native_mip_caps(screen_of(pipe->screen)->fd)) {
            unsigned first = ctx->views[slot]->u.tex.first_level;
            unsigned last = ctx->views[slot]->u.tex.last_level;
            if (last < first || last >= PIPE_MAX_TEXTURE_LEVELS || last > ctx->views[slot]->texture->last_level)
                goto free_textures;
            textures[i].mip_count = last - first + 1;
            textures[i].levels = texture_levels[i];
            texture_levels[i][0] = (struct mxgpu_texture_level){texture_copies[i], textures[i].width, textures[i].height};
            for (unsigned mip = 1; mip < textures[i].mip_count; mip++) {
                unsigned char *copy = NULL;
                if (!read_texture_faces(ctx->views[slot], first + mip, cube, shadow, &copy,
                                        &texture_levels[i][mip].width, &texture_levels[i][mip].height))
                    goto free_textures;
                texture_levels[i][mip].pixels = copy;
            }
        }
        textures[i].pixels = texture_copies[i];
        textures[i].texture_slot = binding->texture_slot;
        textures[i].sampler_slot = binding->sampler_slot;
        if (bound_sampler && !native_sampler_state(ctx->sampler_bound[slot] ? &ctx->samplers[slot] : NULL,
                                                   &textures[i].sampler))
            goto free_textures;
        sampler_compare[i] = textures[i].sampler.compare;
    }
    if (texture_count) {
        texels = texture_copies[0];
        tw = textures[0].width;
        th = textures[0].height;
        native_state.sampler_enabled = bound_sampler;
    } else {
        tw = th = 1;
        texels = malloc(4);
        if (!texels)
            goto free_textures;
        memset(texels, 255, 4);
    }
    if (native && has_color && !native_state.depth_enabled &&
        !ctx->fb.cbufs[0].level && !ctx->fb.cbufs[0].first_layer &&
        !ctx->fb.cbufs[0].last_layer &&
        ctx->fb.cbufs[0].format == ctx->fb.cbufs[0].texture->format &&
        cw == ctx->fb.cbufs[0].texture->width0 && ch == ctx->fb.cbufs[0].texture->height0 &&
        color_row_fast_format(ctx->fb.cbufs[0].format)) {
        target = res_of(ctx->fb.cbufs[0].texture);
        if (!target->framebuffer)
            target->framebuffer = mxgpu_framebuffer_create(cw, ch);
        if (!target->framebuffer_pixels)
            target->framebuffer_pixels = malloc((size_t)cw * ch * 4u);
        if (!target->framebuffer || !target->framebuffer_pixels ||
            !track_framebuffer(ctx, &target->base))
            goto free_textures;
        deferred = true;
    }
    pixels = deferred ? target->framebuffer_pixels : malloc((size_t)cw * (size_t)ch * 4u);
    if (!pixels) {
        goto free_textures;
    }
    if (!has_color)
        memset(pixels, 0, (size_t)cw * ch * 4u);
    if (has_color && !(deferred && target->framebuffer_dirty) && !read_color_rows(&ctx->fb.cbufs[0], pixels, cw, ch, !native)) {
        if (!deferred) free(pixels);
        goto free_textures;
    }
    if (!native && (ctx->scissor_enabled || source_over || color_mask != 15u)) {
        preserved = malloc((size_t)cw * ch * 4u);
        if (!preserved) {
            if (!deferred) free(pixels);
            goto free_textures;
        }
    }
    ctx->draw_base_instance = info->start_instance;
    for (ctx->draw_instance = 0; ctx->draw_instance < info->instance_count; ctx->draw_instance++) {
    for (draw = 0; draw < num_draws; draw++) {
        ctx->draw_base_vertex = info->index_size ? draws[draw].index_bias : draws[draw].start;
        uint8_t module[MXGPU_LINK_MODULE_CAPACITY];
        uint32_t module_len = 0;
        unsigned count = draws[draw].count;
        unsigned *indices = NULL;
        bool *restarts = NULL;
        float *verts;
        if (count < 3 || count > 65536)
            continue;
        if (info->index_size) {
            indices = malloc(sizeof(unsigned) * count);
            if (!indices)
                continue;
            if (info->primitive_restart)
                restarts = malloc(sizeof(bool) * count);
            if ((info->primitive_restart && !restarts) ||
                (restarts ? !read_indices_with_restart(info, &draws[draw], indices, restarts) :
                            !read_indices(info, &draws[draw], indices))) {
                free(restarts);
                free(indices);
                continue;
            }
        }
        if (restarts) {
            verts = gather_restart_triangles(ctx, indices, restarts, count, info->mode, &count);
            free(restarts);
            free(indices);
        } else {
            verts = malloc(sizeof(float) * vertex_stride * count);
            if (!verts || !gather_vertices(ctx, indices, draws[draw].start, count, verts)) {
                free(indices);
                free(verts);
                continue;
            }
            free(indices);
            verts = triangulate_vertices_stride(verts, count, info->mode, &count, vertex_stride);
        }
        if (!verts)
            continue;
        if (mxgpu_link_shaders_draw_samplers(vs, ctx->fs, count, bound_sampler, module, sizeof module, &module_len, sampler_compare) != 0) {
            static int link_noted;
            if (!link_noted) {
                fprintf(stderr, "mxgpu link failed vs %u fs %u\n", ctx->vs ? ctx->vs->count : 0, ctx->fs ? fs->count : 0);
                link_noted = 1;
            }
            free(verts);
            continue;
        }
        {
            unsigned first, step = source_over && !native ? 3u : count;
            bool updated = false;
            for (first = 0; first < count; first += step) {
                if (preserved) {
                    memcpy(preserved, pixels, (size_t)cw * ch * 4u);
                    if (source_over)
                        memset(pixels, 0, (size_t)cw * ch * 4u);
                }
                int readback_complete = 0;
                int result;
                if (deferred) {
                    target->framebuffer_dirty = true;
                    result = mxgpu_execute_module_transaction_native_resources_deferred(screen_of(pipe->screen)->fd, module, module_len,
                        verts + (size_t)first * vertex_stride, (int)step, texels, (int)tw, (int)th, pixels, pixels, (int)cw, (int)ch,
                        vs_constants, vs_constant_size, fs_constants, fs_constant_size, &readback_complete, &native_state,
                        vertex_stride * sizeof(float), textures, texture_count, target->framebuffer, target->cpu_revision);
                } else
                    result = mxgpu_execute_module_transaction_native_resources(screen_of(pipe->screen)->fd, module, module_len,
                                                      verts + (size_t)first * vertex_stride, (int)step,
                                                      texels, (int)tw, (int)th, pixels, pixels, (int)cw, (int)ch,
                                                      vs_constants, vs_constant_size,
                                                      fs_constants, fs_constant_size,
                                                      &readback_complete, native ? &native_state : NULL, vertex_stride * sizeof(float), textures, texture_count);
                if (result != 0 || (!deferred && !readback_complete)) {
                    if (preserved)
                        memcpy(pixels, preserved, (size_t)cw * ch * 4u);
                    break;
                }
                if (preserved) {
                    compose_pixels(pixels, preserved, cw, ch, color_mask, source_over);
                    if (ctx->scissor_enabled)
                        restore_scissor_pixels(pixels, preserved, cw, ch, &ctx->scissor);
                }
                if (deferred) {
                    target->framebuffer_dirty = true;
                    target->framebuffer_revision = target->cpu_revision;
                    target->framebuffer_rendered = true;
                }
                updated = true;
            }
            if (updated) {
                if (has_color && !deferred)
                    write_color_rows(&ctx->fb.cbufs[0], pixels, cw, ch, !native);
                if (depth_pixels && !depth_rows_copy(&ctx->fb.zsbuf, depth_pixels, cw, ch, true)) {
                    free(verts);
                    break;
                }
            }
        }
        free(verts);
    }
    }
    free(preserved);
    if (!deferred) free(pixels);
free_textures:
    free((void *)vs_constants);
    free((void *)fs_constants);
    free(depth_pixels);
    for (unsigned i = 0; i < texture_count; i++) {
        for (unsigned mip = 1; mip < textures[i].mip_count; mip++)
            free((void *)texture_levels[i][mip].pixels);
        free(texture_copies[i]);
    }
    if (!texture_count)
        free(texels);
}

static void mxgpu_clear_depth_stencil(struct mxgpu_context *ctx, unsigned buffers,
                                      uint8_t stencil_mask, const struct pipe_scissor_state *scissor,
                                      double depth, unsigned stencil)
{
    struct pipe_surface *surf = &ctx->fb.zsbuf;
    struct mxgpu_resource *res;
    const struct util_format_pack_description *pack;
    float depth_row[1024];
    uint8_t stencil_row[1024];
    unsigned minx = 0, miny = 0, maxx, maxy, x, y, i, block;
    if (!surf->texture || !(buffers & PIPE_CLEAR_DEPTHSTENCIL))
        return;
    res = res_of(surf->texture);
    pack = util_format_pack_description(surf->format);
    maxx = MIN2(ctx->fb.width, u_minify(surf->texture->width0, surf->level));
    maxy = MIN2(ctx->fb.height, u_minify(surf->texture->height0, surf->level));
    if (scissor) {
        minx = MIN2(scissor->minx, maxx);
        miny = MIN2(scissor->miny, maxy);
        maxx = MIN2(scissor->maxx, maxx);
        maxy = MIN2(scissor->maxy, maxy);
    }
    block = util_format_get_blocksize(surf->format);
    for (i = 0; i < 1024; i++)
        depth_row[i] = (float)depth;
    if (!resource_cpu_sync(res, PIPE_MAP_READ | PIPE_MAP_WRITE, false))
        return;
    for (y = miny; y < maxy; y++) {
        for (x = minx; x < maxx; x += 1024) {
            unsigned width = MIN2(1024, maxx - x);
            unsigned char *dst = res->data + res->level_offset[surf->level] +
                                 (size_t)y * res->level_stride[surf->level] + (size_t)x * block;
            if ((buffers & PIPE_CLEAR_DEPTH) && pack->pack_z_float)
                util_format_pack_z_float(surf->format, dst, depth_row, width);
            if ((buffers & PIPE_CLEAR_STENCIL) && stencil_mask && pack->pack_s_8uint) {
                if (stencil_mask != 255u)
                    util_format_unpack_s_8uint(surf->format, stencil_row, dst, width);
                for (i = 0; i < width; i++) {
                    stencil_row[i] = stencil_mask == 255u ? (uint8_t)stencil :
                        (stencil_row[i] & (uint8_t)~stencil_mask) | ((uint8_t)stencil & stencil_mask);
                }
                util_format_pack_s_8uint(surf->format, dst, stencil_row, width);
            }
        }
    }
    resource_cpu_sync(res, PIPE_MAP_READ | PIPE_MAP_WRITE, true);
}

static bool full_color_clear_discard(const struct pipe_surface *surface,
                                     const struct mxgpu_resource *res,
                                     unsigned mask, unsigned minx, unsigned miny,
                                     unsigned maxx, unsigned maxy)
{
    return res->framebuffer && !res->imported && !res->external && mask == 15u &&
        res->base.target == PIPE_TEXTURE_2D && res->base.array_size <= 1 && !res->base.last_level &&
        !surface->level && !surface->first_layer && !surface->last_layer &&
        surface->format == res->base.format && color_row_fast_format(surface->format) &&
        !minx && !miny && maxx == res->base.width0 && maxy == res->base.height0 &&
        color_rows_fit(res, 0, maxx, maxy, util_format_get_blocksize(surface->format));
}

static void mxgpu_clear(struct pipe_context *pipe, unsigned buffers, uint32_t color_clear_mask, uint8_t stencil_clear_mask, const struct pipe_scissor_state *scissor, const union pipe_color_union *color, double depth, unsigned stencil)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    union pipe_color_union row[1024];
    unsigned attachment;
    mxgpu_clear_depth_stencil(ctx, buffers, stencil_clear_mask, scissor, depth, stencil);
    if (!color)
        return;
    for (attachment = 0; attachment < ctx->fb.nr_cbufs; attachment++) {
        struct pipe_surface *surf = &ctx->fb.cbufs[attachment];
        struct mxgpu_resource *res;
        unsigned mask = (color_clear_mask >> (attachment * 4u)) & 15u;
        unsigned minx = 0, miny = 0, maxx, maxy, x, y, block;
        if (!(buffers & (PIPE_CLEAR_COLOR0 << attachment)) || !surf->texture || !mask)
            continue;
        res = res_of(surf->texture);
        maxx = MIN2(ctx->fb.width, u_minify(surf->texture->width0, surf->level));
        maxy = MIN2(ctx->fb.height, u_minify(surf->texture->height0, surf->level));
        if (scissor) {
            minx = MIN2(scissor->minx, maxx);
            miny = MIN2(scissor->miny, maxy);
            maxx = MIN2(scissor->maxx, maxx);
            maxy = MIN2(scissor->maxy, maxy);
        }
        block = util_format_get_blocksize(surf->format);
        bool discard = full_color_clear_discard(surf, res, mask, minx, miny, maxx, maxy);
        res->framebuffer_discarding = discard;
        if (!resource_cpu_sync(res, discard ? PIPE_MAP_WRITE : PIPE_MAP_READ | PIPE_MAP_WRITE, false)) {
            res->framebuffer_discarding = false;
            continue;
        }
        for (y = miny; y < maxy; y++) {
            for (x = minx; x < maxx; x += 1024) {
                unsigned width = MIN2(1024, maxx - x);
                unsigned i, channel;
                unsigned char *dst = res->data + res->level_offset[surf->level] +
                                 (size_t)y * res->level_stride[surf->level] + (size_t)x * block;
                if (mask != 15u)
                    util_format_unpack_rgba(surf->format, row, dst, width);
                for (i = 0; i < width; i++) {
                    for (channel = 0; channel < 4; channel++) {
                        if (mask & (1u << channel))
                            row[i].ui[channel] = color->ui[channel];
                    }
                }
                util_format_pack_rgba(surf->format, dst, row, width);
            }
        }
        bool complete = resource_cpu_sync(res, discard ? PIPE_MAP_WRITE : PIPE_MAP_READ | PIPE_MAP_WRITE, true);
        res->framebuffer_discarding = false;
        if (discard && complete && !mxgpu_framebuffer_discard(res->framebuffer)) {
            res->framebuffer_dirty = false;
            res->framebuffer_rendered = false;
            res->cpu_revision++;
        }
    }
}

static void mxgpu_destroy_context(struct pipe_context *pipe)
{
    struct mxgpu_context *ctx = ctx_of(pipe);
    unsigned i;
    mxgpu_flush(pipe, NULL, 0);
    while (ctx->pending) {
        struct mxgpu_pending *entry = ctx->pending;
        ctx->pending = entry->next;
        pipe_resource_reference(&entry->resource, NULL);
        FREE(entry);
    }
    if (pipe->stream_uploader)
        u_upload_destroy(pipe->stream_uploader);
    for (i = 0; i < ctx->vb_count; i++) {
        if (!ctx->vb[i].is_user_buffer)
            pipe_resource_reference(&ctx->vb[i].buffer.resource, NULL);
    }
    for (i = 0; i < 2; i++) {
        for (unsigned index = 0; index < MXGPU_UNIFORM_BUFFERS; index++) {
            pipe_resource_reference(&ctx->constants[i][index].buffer, NULL);
            FREE(ctx->constant_copies[i][index]);
        }
    }
    util_unreference_framebuffer_state(&ctx->fb);
    for (i = 0; i < MXGPU_SHADER_TEXTURES; i++)
        pipe_sampler_view_reference(&ctx->views[i], NULL);
    FREE(ctx);
}

static struct pipe_context *mxgpu_context_create(struct pipe_screen *screen, void *priv, unsigned flags)
{
    struct mxgpu_context *ctx = CALLOC_STRUCT(mxgpu_context);
    (void)flags;
    if (!ctx)
        return NULL;
    ctx->base.screen = screen;
    ctx->base.priv = priv;
    ctx->base.destroy = mxgpu_destroy_context;
    ctx->base.draw_vbo = mxgpu_draw;
    ctx->base.clear = mxgpu_clear;
    ctx->base.flush = mxgpu_flush;
    ctx->base.set_framebuffer_state = set_fb;
    ctx->base.set_debug_callback = u_default_set_debug_callback;
    ctx->base.buffer_map = map_resource;
    ctx->base.texture_map = map_resource;
    ctx->base.buffer_unmap = unmap_resource;
    ctx->base.texture_unmap = unmap_resource;
    ctx->base.transfer_flush_region = u_default_transfer_flush_region;
    ctx->base.buffer_subdata = u_default_buffer_subdata;
    ctx->base.texture_subdata = u_default_texture_subdata;
    ctx->base.create_blend_state = create_blend;
    ctx->base.bind_blend_state = bind_blend;
    ctx->base.delete_blend_state = delete_blob;
    ctx->base.create_rasterizer_state = create_rast;
    ctx->base.bind_rasterizer_state = bind_rasterizer;
    ctx->base.delete_rasterizer_state = delete_blob;
    ctx->base.create_depth_stencil_alpha_state = create_dsa;
    ctx->base.bind_depth_stencil_alpha_state = bind_dsa;
    ctx->base.delete_depth_stencil_alpha_state = delete_blob;
    ctx->base.set_blend_color = set_blend_color;
    ctx->base.set_stencil_ref = set_stencil_ref;
    ctx->base.set_sample_mask = set_sample_mask;
    ctx->base.set_min_samples = set_min_samples;
    ctx->base.set_clip_state = set_clip;
    ctx->base.set_viewport_states = set_viewport;
    ctx->base.set_scissor_states = set_scissor;
    ctx->base.set_polygon_stipple = set_stipple;
    ctx->base.set_window_rectangles = set_windows;
    ctx->base.set_sample_locations = set_locations;
    ctx->base.set_tess_state = set_tess;
    ctx->base.set_patch_vertices = set_patch;
    ctx->base.set_constant_buffer = set_constant;
    ctx->base.set_inlinable_constants = set_inlinable;
    ctx->base.set_shader_buffers = set_ssbo;
    ctx->base.set_hw_atomic_buffers = set_atomics;
    ctx->base.set_shader_images = set_images;
    ctx->base.texture_barrier = barrier_nop;
    ctx->base.memory_barrier = barrier_nop;
    ctx->base.flush_resource = resource_nop;
    ctx->base.invalidate_resource = resource_nop;
    ctx->base.resource_release = u_default_resource_release;
    ctx->base.get_sample_position = sample_pos;
    ctx->base.get_device_reset_status = reset_status;
    ctx->base.blit = mxgpu_blit;
    ctx->base.resource_copy_region = mxgpu_copy_region;
    ctx->base.create_tcs_state = create_vs;
    ctx->base.bind_tcs_state = bind_nop;
    ctx->base.delete_tcs_state = delete_shader;
    ctx->base.create_tes_state = create_vs;
    ctx->base.bind_tes_state = bind_nop;
    ctx->base.delete_tes_state = delete_shader;
    ctx->base.bind_compute_state = bind_nop;
    ctx->base.delete_compute_state = delete_shader;
    ctx->base.create_fs_state = create_fs;
    ctx->base.bind_fs_state = bind_fs;
    ctx->base.delete_fs_state = delete_shader;
    ctx->base.create_vs_state = create_vs;
    ctx->base.bind_vs_state = bind_vs;
    ctx->base.delete_vs_state = delete_shader;
    ctx->base.create_gs_state = create_vs;
    ctx->base.bind_gs_state = bind_nop;
    ctx->base.delete_gs_state = delete_shader;
    ctx->base.create_sampler_state = create_sampler;
    ctx->base.bind_sampler_states = bind_samplers;
    ctx->base.delete_sampler_state = delete_blob;
    ctx->base.create_sampler_view = create_view;
    ctx->base.sampler_view_destroy = destroy_view;
    ctx->base.sampler_view_release = u_default_sampler_view_release;
    ctx->base.set_sampler_views = set_views;
    ctx->base.create_vertex_elements_state = create_velems;
    ctx->base.bind_vertex_elements_state = bind_velems;
    ctx->base.delete_vertex_elements_state = delete_blob;
    ctx->base.set_vertex_buffers = set_vbs;
    ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
    ctx->base.const_uploader = ctx->base.stream_uploader;
    return &ctx->base;
}

static const nir_shader_compiler_options mxgpu_nir_options = { 0 };

static struct pipe_screen *screen_create(struct sw_winsys *winsys, int fd)
{
    struct mxgpu_screen *screen = CALLOC_STRUCT(mxgpu_screen);
    struct pipe_caps *caps;
    unsigned stage;
    if (!screen)
        return NULL;
    simple_mtx_init(&screen->resource_mutex, mtx_plain);
    screen->winsys = winsys;
    screen->fd = fd;
    screen->base.destroy = mxgpu_destroy_screen;
    screen->base.get_name = mxgpu_get_name;
    screen->base.get_vendor = mxgpu_get_vendor;
    screen->base.get_device_vendor = mxgpu_get_vendor;
    screen->base.get_screen_fd = mxgpu_get_fd;
    screen->base.get_timestamp = u_default_get_timestamp;
    screen->base.is_format_supported = mxgpu_format_ok;
    screen->base.context_create = mxgpu_context_create;
    screen->base.resource_create = mxgpu_resource_create;
    screen->base.resource_from_handle = mxgpu_resource_from_handle;
    screen->base.resource_destroy = mxgpu_resource_destroy;
    screen->base.resource_get_handle = mxgpu_get_handle;
    screen->base.fence_reference = mxgpu_fence_reference;
    screen->base.fence_finish = mxgpu_fence_finish;
    u_init_pipe_screen_caps(&screen->base, 1);
    caps = (struct pipe_caps *)&screen->base.caps;
    caps->vendor_id = MX_PCI_VENDOR_ID;
    caps->device_id = MXGPU_PCI_DEVICE_ID;
    caps->npot_textures = true;
    caps->blend_equation_separate = true;
    caps->indep_blend_enable = false;
    caps->indep_blend_func = false;
    caps->mixed_framebuffer_sizes = true;
    caps->mixed_color_depth_bits = true;
    caps->texture_swizzle = true;
    caps->user_vertex_buffers = true;
    caps->vs_instanceid = true;
    caps->vertex_element_instance_divisor = true;
    caps->primitive_restart = true;
    caps->primitive_restart_fixed_index = true;
    caps->occlusion_query = false;
    caps->fragment_shader_texture_lod = false;
    caps->fragment_shader_derivatives = fd >= 0 && mxgpu_native_render_caps(fd);
    caps->device_reset_status_query = fd >= 0;
    caps->max_render_targets = 1;
    caps->max_dual_source_render_targets = 0;
    caps->max_texture_2d_size = 4096;
    caps->max_texture_cube_levels = fd >= 0 && mxgpu_native_cube_caps(fd) ? 13 : 0;
    caps->glsl_feature_level = 140;
    caps->glsl_feature_level_compatibility = 140;
    caps->fragment_color_clamped = true;
    caps->fs_coord_origin_upper_left = true;
    caps->fs_coord_pixel_center_half_integer = true;
    caps->constant_buffer_offset_alignment = 16;
    caps->max_constant_buffer_size = 65536;
    caps->min_map_buffer_alignment = 64;
    caps->max_vertex_attrib_stride = 2048;
    caps->max_viewports = 1;
    caps->point_size_fixed = PIPE_POINT_SIZE_LOWER_NEVER;
    caps->uma = true;
    caps->dmabuf = DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT;
    for (stage = 0; stage < MESA_SHADER_MESH_STAGES; stage++) {
        if (stage != MESA_SHADER_VERTEX && stage != MESA_SHADER_FRAGMENT)
            continue;
        struct pipe_shader_caps *sc = (struct pipe_shader_caps *)&screen->base.shader_caps[stage];
        tgsi_exec_init_shader_caps(sc);
        sc->supported_irs = 1u << PIPE_SHADER_IR_NIR;
        sc->max_const_buffers = MXGPU_UNIFORM_BUFFERS;
        sc->max_texture_samplers = stage == MESA_SHADER_FRAGMENT ? MXGPU_SHADER_TEXTURES : 0;
        sc->max_sampler_views = stage == MESA_SHADER_FRAGMENT ? MXGPU_SHADER_TEXTURES : 0;
        screen->base.nir_options[stage] = &mxgpu_nir_options;
    }
    return &screen->base;
}

struct pipe_screen *mxgpu_create_screen(struct sw_winsys *winsys)
{
    return screen_create(winsys, -1);
}

struct pipe_screen *mxgpu_drm_screen_create(int fd)
{
    return screen_create(NULL, fd);
}
