/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_DRIVER_H
#define MXGPU_DRIVER_H

#include <stdint.h>
#include "mxgpu_wire.h"
#include "mxgpu_drm_uapi.h"
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>

static inline int mxgpu_ioctl(int fd, unsigned long request, void *arg)
{
    unsigned char saved[512];
    size_t size = _IOC_SIZE(request);
    int result;

    if (size > sizeof saved)
        return ioctl(fd, request, arg);
    memcpy(saved, arg, size);
    while ((result = ioctl(fd, request, arg)) == -1 && (errno == EINTR || errno == EAGAIN))
        memcpy(arg, saved, size);
    return result;
}

#define MXGPU_FB_SIZE 64

int mxgpu_device_open(void);
int mxgpu_device_open_fd(int fd);
void mxgpu_device_close(void);


int mxgpu_debug_illegal_then_legal(void);
int mxgpu_device_available(void);
int mxgpu_device_lost(void);
int mxgpu_device_flush(void);
const char *mxgpu_readback_reason(const char *reason);
unsigned mxgpu_last_submits(void);

int mxgpu_readback_ready(void);
void mxgpu_flush_frame(void);
int mxgpu_seed_color(const unsigned char *pixels, unsigned width, unsigned height);
int mxgpu_execute_module(const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         unsigned char *color, int cw, int ch);
int mxgpu_execute_module_uniforms(const uint8_t *module, uint32_t module_len,
                                  const float *vertices, int vertex_count,
                                  const unsigned char *texels, int tw, int th,
                                  unsigned char *color, int cw, int ch,
                                  const void *vertex_uniforms, uint32_t vertex_uniform_size,
                                  const void *fragment_uniforms, uint32_t fragment_uniform_size);
int mxgpu_execute_module_transaction(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete);
#define MXGPU_TEXTURE_INPUTS 16u
struct mxgpu_texture_level {
    const unsigned char *pixels;
    uint32_t width, height;
};
struct mxgpu_texture_input {
    const unsigned char *pixels;
    uint32_t width, height, texture_slot, sampler_slot;
    struct mxgpu_sampler_state sampler;
    uint32_t mip_count;
    const struct mxgpu_texture_level *levels;
    uint32_t binding_kind, format, array_layers;
    uint64_t identity;
    struct mxgpu_framebuffer *framebuffer;
    uint64_t framebuffer_revision;
};
struct mxgpu_framebuffer;
struct mxgpu_depth_input {
    unsigned char *pixels;
    uint32_t format, width, height;
    struct mxgpu_framebuffer *framebuffer;
    uint64_t cpu_revision;
};
static inline int mxgpu_depth_stencil_writes(const struct mxgpu_depth_stencil_state *state)
{
    const struct mxgpu_stencil_face *faces[2] = {&state->front, &state->back};
    if (state->depth_write_enable)
        return 1;
    if (!state->stencil_enable || !state->stencil_write_mask)
        return 0;
    for (unsigned i = 0; i < 2; i++)
        if (faces[i]->fail != MXGPU_STENCIL_KEEP || faces[i]->depth_fail != MXGPU_STENCIL_KEEP ||
            faces[i]->pass != MXGPU_STENCIL_KEEP)
            return 1;
    return 0;
}
struct mxgpu_native_render_state {
    struct mxgpu_blend_state blend;
    struct mxgpu_rasterizer_state rasterizer;
    struct mxgpu_viewport viewport;
    struct mxgpu_scissor scissor;
    uint32_t blend_factor[4];
    int sampler_enabled;
    struct mxgpu_sampler_state sampler;
    int depth_enabled;
    struct mxgpu_depth_stencil_state depth_stencil;
    struct mxgpu_depth_input depth;
    uint32_t stencil_reference;
    int bounds_valid;
    struct mxgpu_scissor bounds;
    uint32_t color_format;
};
struct mxgpu_framebuffer;
struct mxgpu_framebuffer *mxgpu_framebuffer_create(uint32_t width, uint32_t height);
struct mxgpu_framebuffer *mxgpu_depth_framebuffer_create(uint32_t width, uint32_t height, uint32_t format);
int mxgpu_framebuffer_current(struct mxgpu_framebuffer *framebuffer, uint64_t cpu_revision);
int mxgpu_framebuffer_destroy(struct mxgpu_framebuffer *framebuffer);
int mxgpu_framebuffer_discard(struct mxgpu_framebuffer *framebuffer);
int mxgpu_framebuffer_clear(struct mxgpu_framebuffer *framebuffer, const unsigned char rgba[4], uint64_t cpu_revision);
int mxgpu_depth_framebuffer_clear(struct mxgpu_framebuffer *framebuffer, int clear_depth, uint32_t depth_bits,
                                  int clear_stencil, uint32_t stencil, const unsigned char wire[8], uint64_t cpu_revision);
typedef void (*mxgpu_row_convert)(unsigned char *destination, const unsigned char *source, unsigned width, unsigned arg);
int mxgpu_framebuffer_sync(struct mxgpu_framebuffer *framebuffer, unsigned char *destination,
                           uint32_t destination_stride, mxgpu_row_convert convert, unsigned convert_arg,
                           int all_rows, int *changed);
int mxgpu_framebuffer_refresh(struct mxgpu_framebuffer *framebuffer, const unsigned char *source,
                              uint32_t source_stride, mxgpu_row_convert convert, unsigned convert_arg,
                              uint64_t cpu_revision);
int mxgpu_native_render_caps(int fd);
int mxgpu_adapter_limits(int fd, struct mxgpu_adapter_info *info);
int mxgpu_format_caps(int fd, struct mxgpu_format_capabilities *caps);
int mxgpu_sampled_format_supported(int fd, uint32_t format);
int mxgpu_color_target_format_supported(int fd, uint32_t format);
int mxgpu_native_mip_caps(int fd);
int mxgpu_native_cube_caps(int fd);
int mxgpu_native_shadow_caps(int fd);
int mxgpu_native_render_available(int fd);
int mxgpu_native_sampler_available(int fd);
int mxgpu_native_depth_available(int fd, uint32_t format);
int mxgpu_execute_module_transaction_native(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state);
int mxgpu_execute_module_transaction_native_stride(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes);
int mxgpu_execute_module_transaction_native_resources(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes, const struct mxgpu_texture_input *textures, uint32_t texture_count);
int mxgpu_execute_module_transaction_native_resources_deferred(int fd, const uint8_t *module, uint32_t module_len,
                         const float *vertices, int vertex_count,
                         const unsigned char *texels, int tw, int th,
                         const unsigned char *initial_color, unsigned char *color, int cw, int ch,
                         const void *vertex_uniforms, uint32_t vertex_uniform_size,
                         const void *fragment_uniforms, uint32_t fragment_uniform_size,
                         int *readback_complete, const struct mxgpu_native_render_state *state, uint32_t vertex_stride_bytes, const struct mxgpu_texture_input *textures, uint32_t texture_count,
                         struct mxgpu_framebuffer *framebuffer, uint64_t cpu_revision);
int mxgpu_execute_scene(const float *vertices, int vertex_count,
                        const unsigned char *texels, int tw, int th,
                        unsigned char *color, int cw, int ch);

struct mxgpu_compute_binding {
    uint16_t slot;
    uint16_t access;
    uint32_t buffer;
    uint64_t offset;
    uint64_t size;
};
int mxgpu_compute_available(void);
int mxgpu_compute_limits(struct mxgpu_drm_compute_limits *limits);
uint32_t mxgpu_storage_buffer_create(uint32_t size);
int mxgpu_storage_buffer_upload(uint32_t buffer, uint32_t offset, const void *data, uint32_t size);
int mxgpu_storage_buffer_read(uint32_t buffer, uint32_t offset, void *data, uint32_t size);
int mxgpu_storage_buffer_destroy(uint32_t buffer);
uint32_t mxgpu_compute_pipeline_create(const uint8_t *module, uint32_t module_len, uint32_t entry);
int mxgpu_compute_pipeline_destroy(uint32_t pipeline);
int mxgpu_compute_dispatch(uint32_t pipeline, uint16_t dispatch_kind, const uint32_t dimensions[3],
                           const struct mxgpu_compute_binding *bindings, uint32_t binding_count);


#endif
