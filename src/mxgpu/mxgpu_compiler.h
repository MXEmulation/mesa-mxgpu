/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_COMPILER_H
#define MXGPU_COMPILER_H
#include <stdbool.h>
#include <stdint.h>
#define MX_LOW_INSNS 2048u
#define MXGPU_SHADER_INSTRUCTION_WORDS 14u
#define MXGPU_LINK_MODULE_CAPACITY (MX_LOW_INSNS * MXGPU_SHADER_INSTRUCTION_WORDS * 8u + 4096u)
struct mxgpu_uniform_buffer {
    unsigned set, binding, element, offset, size;
    unsigned constant_slot;
    bool push_constant;
    uint32_t push_constant_words;
};
#define MXGPU_UNIFORM_BUFFERS 16u
struct mxgpu_texture_binding {
    unsigned set, binding, element, texture_id, texture_slot, sampler_id, sampler_slot;
    unsigned binding_kind;
    bool shadow;
};
#define MXGPU_SHADER_TEXTURES 8u
#define MXGPU_SHADER_VERTEX_SLOTS 33u
#define MXGPU_SHADER_STORAGE_BUFFERS 15u
#define MXGPU_COMPUTE_UNIFORM_BINDING 1u
struct mxgpu_storage_binding {
    unsigned set, binding, element, binding_id, slot, access;
};
struct mxgpu_shader {
    bool samples;
    bool uses_uniforms;
    unsigned uniform_count;
    unsigned uniform_buffer_count;
    struct mxgpu_uniform_buffer uniform_buffers[MXGPU_UNIFORM_BUFFERS];
    uint32_t words[MX_LOW_INSNS][MXGPU_SHADER_INSTRUCTION_WORDS];
    uint32_t lens[MX_LOW_INSNS];
    unsigned count;
    unsigned vertex_attribute_count;
    bool vertex_builtins;
    unsigned vertex_builtin_slot;
    unsigned vertex_input_locations[MXGPU_SHADER_VERTEX_SLOTS];
    unsigned texture_binding;
    unsigned texture_set;
    unsigned texture_element;
    unsigned texture_count;
    struct mxgpu_texture_binding textures[MXGPU_SHADER_TEXTURES];
    bool compute;
    uint32_t workgroup_size[3];
    uint32_t workgroup_bytes;
    unsigned uniform_slot;
    unsigned storage_count;
    struct mxgpu_storage_binding storage[MXGPU_SHADER_STORAGE_BUFFERS];
};

struct nir_shader;
int mxgpu_compile_nir(struct nir_shader *nir, bool fragment, struct mxgpu_shader *shader);
int mxgpu_link_shaders_draw(const struct mxgpu_shader *vs, const struct mxgpu_shader *fs,
                       unsigned vertex_count, bool bound_sampler, uint8_t *out, uint32_t cap, uint32_t *out_len);
int mxgpu_link_shaders(const struct mxgpu_shader *vs, const struct mxgpu_shader *fs,
                       uint8_t *out, uint32_t cap, uint32_t *out_len);
int mxgpu_link_compute(const struct mxgpu_shader *cs, uint8_t *out, uint32_t cap, uint32_t *out_len);
int mxgpu_link_shaders_draw_samplers(const struct mxgpu_shader *vs, const struct mxgpu_shader *fs, unsigned vertex_count, bool bound_sampler, uint8_t *out, uint32_t cap, uint32_t *out_len, const uint32_t *sampler_compare);
#endif
