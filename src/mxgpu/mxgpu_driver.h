/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_DRIVER_H
#define MXGPU_DRIVER_H

#include <stdint.h>
#include "mxgpu_wire.h"
#include "mxgpu_drm_uapi.h"

#define MXGPU_FB_SIZE 64

int mxgpu_device_open(void);
int mxgpu_device_open_fd(int fd);
void mxgpu_device_close(void);


int mxgpu_debug_illegal_then_legal(void);
int mxgpu_device_available(void);
int mxgpu_device_lost(void);
int mxgpu_device_flush(void);
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
#define MXGPU_TEXTURE_INPUTS 8u
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
struct mxgpu_depth_input {
    unsigned char *pixels;
    uint32_t format, width, height;
};
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
};
struct mxgpu_framebuffer;
struct mxgpu_framebuffer *mxgpu_framebuffer_create(uint32_t width, uint32_t height);
int mxgpu_framebuffer_destroy(struct mxgpu_framebuffer *framebuffer);
int mxgpu_framebuffer_discard(struct mxgpu_framebuffer *framebuffer);
int mxgpu_framebuffer_sync(struct mxgpu_framebuffer *framebuffer, unsigned char *pixels, int *changed);
int mxgpu_native_render_caps(int fd);
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


#endif
