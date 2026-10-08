/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_compiler.h"
#include "mxsb.h"
#include "nir.h"
#include <string.h>
#include <stdlib.h>
#define MX_LOW_DEFS 1024u
#define MX_LOW_IDS 4096u

struct mx_descriptor_ref {
    bool valid;
    unsigned set, binding, element;
};

struct mx_low {
    uint32_t words[MX_LOW_INSNS][MXGPU_SHADER_INSTRUCTION_WORDS];
    uint32_t lens[MX_LOW_INSNS];
    unsigned count;
    unsigned next;
    uint32_t id_type[MX_LOW_IDS];
    unsigned id_of[MX_LOW_DEFS];
    unsigned comps[MX_LOW_DEFS];
    unsigned comp_id[MX_LOW_DEFS][4];
    unsigned vid;
    unsigned vload[MXGPU_SHADER_VERTEX_SLOTS];
    unsigned vertex_attribute_count;
    bool vertex_builtins;
    unsigned vertex_builtin_slot;
    unsigned vertex_input_locations[MXGPU_SHADER_VERTEX_SLOTS];
    int fail;
    unsigned texture_binding, texture_set;
    unsigned texture_element;
    unsigned texture_count;
    struct mxgpu_texture_binding textures[MXGPU_SHADER_TEXTURES];
    bool texture_live;
    unsigned uniform_binding;
    unsigned uniform_count;
    bool uses_uniforms;
    struct mx_descriptor_ref descriptors[MX_LOW_DEFS];
    unsigned uniform_buffer_count;
    struct mxgpu_uniform_buffer uniform_buffers[MXGPU_UNIFORM_BUFFERS];
    bool compute;
    nir_block *predicate_block;
    unsigned predicate;
    unsigned storage_count;
    struct mxgpu_storage_binding storage[MXGPU_SHADER_STORAGE_BUFFERS];
    unsigned workgroup_bytes;
};

static unsigned low_id(struct mx_low *low)
{
    unsigned id = low->next++;
    if (id == 0 || id >= MX_LOW_IDS)
        low->fail = 1;
    return id;
}

static void low_emit(struct mx_low *low, const uint32_t *words, unsigned count)
{
    if (low->fail || low->count >= MX_LOW_INSNS || count == 0 || count > MXGPU_SHADER_INSTRUCTION_WORDS) {
        low->fail = 1;
        return;
    }
    uint32_t opcode = words[0] & 0xffffu;
    if (count >= 3 && opcode != MXSB_OP_STAGE_OUTPUT && opcode != MXSB_OP_RETURN_VALUE &&
        opcode != MXSB_OP_BUFFER_STORE && opcode != MXSB_OP_TEXTURE_STORE &&
        opcode != MXSB_OP_WORKGROUP_STORE && opcode != MXSB_OP_WORKGROUP_ATOMIC_STORE &&
        opcode != MXSB_OP_BUFFER_ATOMIC_STORE && words[1] < MX_LOW_IDS)
        low->id_type[words[1]] = words[2];
    memcpy(low->words[low->count], words, count * sizeof(uint32_t));
    low->lens[low->count] = count;
    low->count++;
}

static unsigned low_cast(struct mx_low *low, unsigned value, uint32_t type)
{
    uint32_t rec[4];
    unsigned id;
    if (!value || value >= MX_LOW_IDS) {
        low->fail = 1;
        return 0;
    }
    if (low->id_type[value] == type)
        return value;
    if ((type != MXSB_TYPE_U32 && type != MXSB_TYPE_F32) ||
        (low->id_type[value] != MXSB_TYPE_U32 && low->id_type[value] != MXSB_TYPE_F32)) {
        low->fail = 1;
        return 0;
    }
    id = low_id(low);
    rec[0] = (4u << 16) | MXSB_OP_BITCAST;
    rec[1] = id;
    rec[2] = type;
    rec[3] = value;
    low_emit(low, rec, 4);
    return id;
}

static void low_remember(struct mx_low *low, unsigned index, unsigned id, unsigned comps)
{
    unsigned c;
    if (index >= MX_LOW_DEFS || id == 0 || comps == 0 || comps > 4) {
        low->fail = 1;
        return;
    }
    low->id_of[index] = id;
    low->comps[index] = comps;
    for (c = 0; c < 4; c++)
        low->comp_id[index][c] = 0;
    if (comps == 1)
        low->comp_id[index][0] = id;
}

static unsigned low_component(struct mx_low *low, nir_def *def, unsigned lane)
{
    unsigned index, id;
    uint32_t rec[5];
    if (!def || def->index >= MX_LOW_DEFS || lane >= low->comps[def->index]) {
        low->fail = 1;
        return 0;
    }
    index = def->index;
    if (low->comp_id[index][lane])
        return low->comp_id[index][lane];
    id = low_id(low);
    rec[0] = (5u << 16) | MXSB_OP_EXTRACT;
    rec[1] = id;
    rec[2] = MXSB_TYPE_F32;
    rec[3] = low->id_of[index];
    rec[4] = lane;
    low_emit(low, rec, 5);
    low->comp_id[index][lane] = id;
    return id;
}

static uint32_t low_vec_type(unsigned comps)
{
    if (comps == 2)
        return MXSB_TYPE_F32X2;
    if (comps == 3)
        return MXSB_TYPE_F32X3;
    return MXSB_TYPE_F32X4;
}

static void low_keep_comps(struct mx_low *low, unsigned index, const unsigned *ids, unsigned comps)
{
    unsigned i;
    if (index >= MX_LOW_DEFS)
        return;
    for (i = 0; i < comps && i < 4; i++)
        low->comp_id[index][i] = ids[i];
}

static void low_const(struct mx_low *low, nir_load_const_instr *lc)
{
    unsigned n = lc->def.num_components;
    unsigned ids[4];
    unsigned i;
    uint32_t rec[8];
    if (n == 0 || n > 4 || (lc->def.bit_size != 1 && lc->def.bit_size != 32)) {
        low->fail = 1;
        return;
    }
    for (i = 0; i < n; i++) {
        uint32_t bits = lc->value[i].u32;
        ids[i] = low_id(low);
        rec[0] = (4u << 16) | MXSB_OP_CONSTANT;
        rec[1] = ids[i];
        if (lc->def.bit_size == 1) {
            rec[2] = MXSB_TYPE_BOOL;
            rec[3] = bits ? 1u : 0u;
        } else {
            rec[2] = MXSB_TYPE_U32;
            rec[3] = bits;
        }
        low_emit(low, rec, 4);
    }
    if (n == 1 || lc->def.bit_size == 1) {
        low_remember(low, lc->def.index, ids[0], n);
        low_keep_comps(low, lc->def.index, ids, n);
        return;
    }
    rec[0] = ((4u + n) << 16) | MXSB_OP_CONSTRUCT;
    rec[1] = low_id(low);
    rec[2] = low_vec_type(n);
    rec[3] = n;
    for (i = 0; i < n; i++)
        rec[4 + i] = low_cast(low, ids[i], MXSB_TYPE_F32);
    low_emit(low, rec, 4 + n);
    low_remember(low, lc->def.index, rec[1], n);
    low_keep_comps(low, lc->def.index, ids, n);
}

static unsigned low_unop_ty(struct mx_low *low, uint16_t opcode, uint32_t type, unsigned value)
{
    uint32_t rec[4];
    unsigned id = low_id(low);
    rec[0] = (4u << 16) | opcode;
    rec[1] = id;
    rec[2] = type;
    rec[3] = value;
    low_emit(low, rec, 4);
    return id;
}

static unsigned low_unop(struct mx_low *low, uint16_t opcode, unsigned value)
{
    return low_unop_ty(low, opcode, MXSB_TYPE_F32, value);
}

static unsigned low_binop_ty(struct mx_low *low, uint16_t opcode, uint32_t type, unsigned left, unsigned right)
{
    uint32_t rec[5];
    unsigned id = low_id(low);
    rec[0] = (5u << 16) | opcode;
    rec[1] = id;
    rec[2] = type;
    rec[3] = left;
    rec[4] = right;
    low_emit(low, rec, 5);
    return id;
}

static unsigned low_binop(struct mx_low *low, uint16_t opcode, unsigned left, unsigned right)
{
    return low_binop_ty(low, opcode, MXSB_TYPE_F32, left, right);
}

static unsigned low_cmp(struct mx_low *low, uint16_t opcode, unsigned left, unsigned right)
{
    uint32_t rec[5];
    unsigned id = low_id(low);
    rec[0] = (5u << 16) | opcode;
    rec[1] = id;
    rec[2] = MXSB_TYPE_BOOL;
    rec[3] = left;
    rec[4] = right;
    low_emit(low, rec, 5);
    return id;
}

static unsigned low_select(struct mx_low *low, uint32_t type, unsigned cond, unsigned on_true, unsigned on_false)
{
    uint32_t rec[6];
    unsigned id = low_id(low);
    rec[0] = (6u << 16) | MXSB_OP_SELECT;
    rec[1] = id;
    rec[2] = type;
    rec[3] = cond;
    rec[4] = on_true;
    rec[5] = on_false;
    low_emit(low, rec, 6);
    return id;
}

static unsigned low_select_f32(struct mx_low *low, unsigned cond, unsigned on_true, unsigned on_false)
{
    return low_select(low, MXSB_TYPE_F32, cond,
                      low_cast(low, on_true, MXSB_TYPE_F32), low_cast(low, on_false, MXSB_TYPE_F32));
}

static unsigned low_bool(struct mx_low *low, int value)
{
    uint32_t rec[4];
    unsigned id = low_id(low);
    rec[0] = (4u << 16) | MXSB_OP_CONSTANT;
    rec[1] = id;
    rec[2] = MXSB_TYPE_BOOL;
    rec[3] = value ? 1u : 0u;
    low_emit(low, rec, 4);
    return id;
}

static void low_finish_lanes(struct mx_low *low, unsigned def_index, const unsigned *ids, unsigned comps)
{
    if (comps == 0 || comps > 4 || !ids || !ids[0]) {
        low->fail = 1;
        return;
    }
    low_remember(low, def_index, ids[0], comps);
    if (comps > 1)
        low_keep_comps(low, def_index, ids, comps);
}

static unsigned low_u32(struct mx_low *low, uint32_t value)
{
    uint32_t rec[4];
    unsigned id = low_id(low);
    rec[0] = (4u << 16) | MXSB_OP_CONSTANT;
    rec[1] = id;
    rec[2] = MXSB_TYPE_U32;
    rec[3] = value;
    low_emit(low, rec, 4);
    return id;
}

static unsigned low_f32(struct mx_low *low, float value)
{
    uint32_t bits;
    uint32_t rec[4];
    unsigned id = low_id(low);
    memcpy(&bits, &value, sizeof bits);
    rec[0] = (4u << 16) | MXSB_OP_CONSTANT;
    rec[1] = id;
    rec[2] = MXSB_TYPE_F32;
    rec[3] = bits;
    low_emit(low, rec, 4);
    return id;
}

static unsigned low_uniform_vec4_id(struct mx_low *low, unsigned index_id)
{
    uint32_t rec[5];
    unsigned vec = low_id(low);
    rec[0] = (5u << 16) | MXSB_OP_BUFFER_LOAD;
    rec[1] = vec;
    rec[2] = MXSB_TYPE_F32X4;
    rec[3] = low->uniform_binding;
    rec[4] = index_id;
    low->uses_uniforms = true;
    low_emit(low, rec, 5);
    return vec;
}

static unsigned low_uniform_vec4(struct mx_low *low, unsigned slot)
{
    if (slot == UINT32_MAX) {
        low->fail = 1;
        return 0;
    }
    if (low->uniform_count <= slot)
        low->uniform_count = slot + 1;
    return low_uniform_vec4_id(low, low_u32(low, slot));
}

static unsigned low_uniform_index(struct mx_low *low, int base, nir_src *src)
{
    unsigned dyn;
    unsigned bits;
    nir_def *def;
    if (!src)
        return 0;
    if (nir_src_is_const(*src)) {
        uint64_t index = (uint64_t)(unsigned)base + nir_src_as_uint(*src);
        if (index >= UINT32_MAX) {
            low->fail = 1;
            return 0;
        }
        if (low->uniform_count <= index)
            low->uniform_count = (unsigned)index + 1;
        return low_u32(low, (unsigned)index);
    }
    def = src->ssa;
    if (!def || def->index >= MX_LOW_DEFS || !low->id_of[def->index])
        return 0;
    dyn = low_component(low, def, 0);
    if (!dyn || low->fail)
        return 0;
    bits = low_unop_ty(low, MXSB_OP_BITCAST, MXSB_TYPE_U32, dyn);
    if (base == 0)
        return bits;
    return low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, low_u32(low, (unsigned)base), bits);
}

static unsigned low_lane(struct mx_low *low, nir_alu_instr *alu, unsigned src, unsigned lane)
{
    unsigned id = low_component(low, alu->src[src].src.ssa, alu->src[src].swizzle[lane]);
    nir_alu_type base = nir_alu_type_get_base_type(nir_op_infos[alu->op].input_types[src]);
    if (base == nir_type_float)
        return low_cast(low, id, MXSB_TYPE_F32);
    if ((base == nir_type_int || base == nir_type_uint) && alu->src[src].src.ssa->bit_size == 32)
        return low_cast(low, id, MXSB_TYPE_U32);
    return id;
}

static void low_finish_vec(struct mx_low *low, nir_alu_instr *alu, const unsigned *ids, unsigned comps)
{
    uint32_t rec[8];
    unsigned i;
    if (comps == 1) {
        low_remember(low, alu->def.index, ids[0], 1);
        return;
    }
    rec[0] = ((4u + comps) << 16) | MXSB_OP_CONSTRUCT;
    rec[1] = low_id(low);
    rec[2] = low_vec_type(comps);
    rec[3] = comps;
    for (i = 0; i < comps; i++)
        rec[4 + i] = low_cast(low, ids[i], MXSB_TYPE_F32);
    low_emit(low, rec, 4 + comps);
    low_remember(low, alu->def.index, rec[1], comps);
    low_keep_comps(low, alu->def.index, ids, comps);
}

static void low_alu(struct mx_low *low, nir_alu_instr *alu)
{
    unsigned comps, i;
    unsigned ids[4];
    uint16_t bin = 0;
    uint16_t un = 0;
    comps = alu->def.num_components;
    if (comps == 0 || comps > 4) {
        low->fail = 1;
        return;
    }
    if (alu->op == nir_op_mov || alu->op == nir_op_vec2 || alu->op == nir_op_vec3 || alu->op == nir_op_vec4) {
        for (i = 0; i < comps; i++) {
            unsigned src = alu->op == nir_op_mov ? 0 : i;
            unsigned lane = alu->op == nir_op_mov ? i : 0;
            ids[i] = low_lane(low, alu, src, lane);
        }
        low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->op == nir_op_fdot2 || alu->op == nir_op_fdot3 || alu->op == nir_op_fdot4) {
        unsigned n = alu->op == nir_op_fdot2 ? 2u : alu->op == nir_op_fdot3 ? 3u : 4u;
        unsigned acc = 0;
        for (i = 0; i < n; i++) {
            unsigned prod = low_binop(low, MXSB_OP_MUL, low_lane(low, alu, 0, i), low_lane(low, alu, 1, i));
            acc = i == 0 ? prod : low_binop(low, MXSB_OP_ADD, acc, prod);
        }
        ids[0] = acc;
        low_finish_vec(low, alu, ids, 1);
        return;
    }
    if (alu->op == nir_op_ffma) {
        for (i = 0; i < comps; i++) {
            unsigned prod = low_binop(low, MXSB_OP_MUL, low_lane(low, alu, 0, i), low_lane(low, alu, 1, i));
            ids[i] = low_binop(low, MXSB_OP_ADD, prod, low_lane(low, alu, 2, i));
        }
        low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->op == nir_op_flrp) {
        unsigned one = low_f32(low, 1.f);
        for (i = 0; i < comps; i++) {
            unsigned a = low_lane(low, alu, 0, i);
            unsigned b = low_lane(low, alu, 1, i);
            unsigned t = low_lane(low, alu, 2, i);
            unsigned kept = low_binop(low, MXSB_OP_MUL, a, low_binop(low, MXSB_OP_SUB, one, t));
            unsigned mixed = low_binop(low, MXSB_OP_MUL, b, t);
            ids[i] = low_binop(low, MXSB_OP_ADD, kept, mixed);
        }
        low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->def.bit_size == 1 && alu->src[0].src.ssa->bit_size == 1 && (alu->op == nir_op_inot || alu->op == nir_op_iand || alu->op == nir_op_ior ||
                                   alu->op == nir_op_ixor || alu->op == nir_op_ieq || alu->op == nir_op_ine)) {
        unsigned yes = low_bool(low, 1);
        unsigned no = low_bool(low, 0);
        for (i = 0; i < comps; i++) {
            unsigned a = low_lane(low, alu, 0, i);
            if (alu->op == nir_op_inot) {
                ids[i] = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, a);
            } else {
                unsigned b = low_lane(low, alu, 1, i);
                if (alu->op == nir_op_iand)
                    ids[i] = low_select(low, MXSB_TYPE_BOOL, a, b, no);
                else if (alu->op == nir_op_ior)
                    ids[i] = low_select(low, MXSB_TYPE_BOOL, a, yes, b);
                else if (alu->op == nir_op_ixor || alu->op == nir_op_ine) {
                    unsigned nb = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, b);
                    ids[i] = low_select(low, MXSB_TYPE_BOOL, a, nb, b);
                } else {
                    unsigned nb = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, b);
                    ids[i] = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, low_select(low, MXSB_TYPE_BOOL, a, nb, b));
                }
            }
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->op == nir_op_b2b1) {
        for (i = 0; i < comps; i++) {
            unsigned value = low_component(low, alu->src[0].src.ssa, alu->src[0].swizzle[i]);
            if (low->id_type[value] == MXSB_TYPE_BOOL)
                ids[i] = value;
            else {
                value = low_cast(low, value, MXSB_TYPE_U32);
                ids[i] = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL,
                    low_cmp(low, MXSB_OP_EQ, value, low_u32(low, 0)));
            }
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->op == nir_op_b2i32 || alu->op == nir_op_b2f32) {
        int as_float = alu->op == nir_op_b2f32;
        unsigned one = as_float ? low_f32(low, 1.f) : low_u32(low, 1u);
        unsigned zero = as_float ? low_f32(low, 0.f) : low_u32(low, 0u);
        uint32_t type = as_float ? MXSB_TYPE_F32 : MXSB_TYPE_U32;
        for (i = 0; i < comps; i++)
            ids[i] = low_select(low, type, low_lane(low, alu, 0, i), one, zero);
        if (as_float)
            low_finish_vec(low, alu, ids, comps);
        else
            low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->def.bit_size == 32 && (alu->op == nir_op_iadd || alu->op == nir_op_isub || alu->op == nir_op_imul || alu->op == nir_op_ineg)) {
        for (i = 0; i < comps; i++) {
            unsigned a = low_lane(low, alu, 0, i);
            if (alu->op == nir_op_ineg)
                ids[i] = low_unop_ty(low, MXSB_OP_NEG, MXSB_TYPE_U32, a);
            else if (alu->op == nir_op_isub)
                ids[i] = low_binop_ty(low, MXSB_OP_SUB, MXSB_TYPE_U32, a, low_lane(low, alu, 1, i));
            else if (alu->op == nir_op_imul)
                ids[i] = low_binop_ty(low, MXSB_OP_MUL, MXSB_TYPE_U32, a, low_lane(low, alu, 1, i));
            else
                ids[i] = low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, a, low_lane(low, alu, 1, i));
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->def.bit_size == 32 && (alu->op == nir_op_ishl || alu->op == nir_op_ushr || alu->op == nir_op_ishr)) {
        for (i = 0; i < comps; i++) {
            unsigned value = low_lane(low, alu, 0, i);
            unsigned count = low_binop_ty(low, MXSB_OP_BIT_AND, MXSB_TYPE_U32,
                                           low_lane(low, alu, 1, i), low_u32(low, 31));
            unsigned shifted = low_binop_ty(low, alu->op == nir_op_ishl ? MXSB_OP_SHL : MXSB_OP_SHR,
                                             MXSB_TYPE_U32, value, count);
            if (alu->op == nir_op_ishr) {
                unsigned sign = low_binop_ty(low, MXSB_OP_BIT_AND, MXSB_TYPE_U32, value, low_u32(low, 0x80000000u));
                unsigned negative = low_cmp(low, MXSB_OP_LT, low_u32(low, 0), sign);
                unsigned fill = low_unop_ty(low, MXSB_OP_BIT_NOT, MXSB_TYPE_U32,
                    low_binop_ty(low, MXSB_OP_SHR, MXSB_TYPE_U32, low_u32(low, UINT32_MAX), count));
                shifted = low_binop_ty(low, MXSB_OP_BIT_OR, MXSB_TYPE_U32, shifted,
                    low_select(low, MXSB_TYPE_U32, negative, fill, low_u32(low, 0)));
            }
            ids[i] = shifted;
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->def.bit_size == 32 && (alu->op == nir_op_inot || alu->op == nir_op_iand || alu->op == nir_op_ior || alu->op == nir_op_ixor)) {
        for (i = 0; i < comps; i++) {
            unsigned a = low_lane(low, alu, 0, i);
            if (alu->op == nir_op_inot)
                ids[i] = low_unop_ty(low, MXSB_OP_BIT_NOT, MXSB_TYPE_U32, a);
            else if (alu->op == nir_op_iand)
                ids[i] = low_binop_ty(low, MXSB_OP_BIT_AND, MXSB_TYPE_U32, a, low_lane(low, alu, 1, i));
            else if (alu->op == nir_op_ior)
                ids[i] = low_binop_ty(low, MXSB_OP_BIT_OR, MXSB_TYPE_U32, a, low_lane(low, alu, 1, i));
            else
                ids[i] = low_binop_ty(low, MXSB_OP_BIT_XOR, MXSB_TYPE_U32, a, low_lane(low, alu, 1, i));
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->op == nir_op_feq || alu->op == nir_op_ieq || alu->op == nir_op_ine || alu->op == nir_op_fneu ||
        alu->op == nir_op_flt || alu->op == nir_op_fge ||
        alu->op == nir_op_ilt || alu->op == nir_op_ult || alu->op == nir_op_ige || alu->op == nir_op_uge) {
        for (i = 0; i < comps; i++) {
            unsigned left = low_lane(low, alu, 0, i);
            unsigned right = low_lane(low, alu, 1, i);
            if (alu->op == nir_op_ilt || alu->op == nir_op_ige) {
                unsigned sign = low_u32(low, 0x80000000u);
                left = low_binop_ty(low, MXSB_OP_BIT_XOR, MXSB_TYPE_U32, left, sign);
                right = low_binop_ty(low, MXSB_OP_BIT_XOR, MXSB_TYPE_U32, right, sign);
            }
            if (alu->op == nir_op_flt || alu->op == nir_op_ilt || alu->op == nir_op_ult)
                ids[i] = low_cmp(low, MXSB_OP_LT, left, right);
            else if (alu->op == nir_op_fge || alu->op == nir_op_ige || alu->op == nir_op_uge)
                ids[i] = low_cmp(low, MXSB_OP_LE, right, left);
            else if (alu->op == nir_op_ine || alu->op == nir_op_fneu)
                ids[i] = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, low_cmp(low, MXSB_OP_EQ, left, right));
            else
                ids[i] = low_cmp(low, MXSB_OP_EQ, left, right);
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->op == nir_op_bcsel) {
        for (i = 0; i < comps; i++) {
            unsigned left = low_lane(low, alu, 1, i);
            unsigned right = low_lane(low, alu, 2, i);
            uint32_t type = alu->def.bit_size == 1 ? MXSB_TYPE_BOOL :
                            low->id_type[left] == MXSB_TYPE_F32 && low->id_type[right] == MXSB_TYPE_F32 ?
                            MXSB_TYPE_F32 : MXSB_TYPE_U32;
            if (type != MXSB_TYPE_BOOL) {
                left = low_cast(low, left, type);
                right = low_cast(low, right, type);
            }
            ids[i] = low_select(low, type, low_lane(low, alu, 0, i), left, right);
        }
        if (alu->def.bit_size == 1)
            low_finish_lanes(low, alu->def.index, ids, comps);
        else
            low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->op == nir_op_i2f32 || alu->op == nir_op_u2f32 ||
        alu->op == nir_op_f2i32 || alu->op == nir_op_f2u32) {
        bool as_float = alu->op == nir_op_i2f32 || alu->op == nir_op_u2f32;
        for (i = 0; i < comps; i++) {
            unsigned value = low_lane(low, alu, 0, i);
            if (alu->op == nir_op_i2f32) {
                unsigned negative = low_cmp(low, MXSB_OP_LT, low_u32(low, INT32_MAX), value);
                unsigned magnitude = low_binop_ty(low, MXSB_OP_SUB, MXSB_TYPE_U32,
                                                 low_u32(low, 0), value);
                unsigned positive = low_unop_ty(low, MXSB_OP_U2F, MXSB_TYPE_F32, value);
                unsigned converted = low_unop_ty(low, MXSB_OP_U2F, MXSB_TYPE_F32, magnitude);
                ids[i] = low_select(low, MXSB_TYPE_F32, negative,
                                    low_unop(low, MXSB_OP_NEG, converted), positive);
            } else if (alu->op == nir_op_f2i32) {
                unsigned negative = low_cmp(low, MXSB_OP_LT, value, low_f32(low, 0.f));
                unsigned magnitude = low_select(low, MXSB_TYPE_F32, negative,
                                                low_unop(low, MXSB_OP_NEG, value), value);
                unsigned converted = low_unop_ty(low, MXSB_OP_F2U, MXSB_TYPE_U32, magnitude);
                unsigned negated = low_binop_ty(low, MXSB_OP_SUB, MXSB_TYPE_U32,
                                               low_u32(low, 0), converted);
                ids[i] = low_select(low, MXSB_TYPE_U32, negative, negated, converted);
            } else {
                ids[i] = low_unop_ty(low, as_float ? MXSB_OP_U2F : MXSB_OP_F2U,
                                    as_float ? MXSB_TYPE_F32 : MXSB_TYPE_U32, value);
            }
        }
        if (as_float)
            low_finish_vec(low, alu, ids, comps);
        else
            low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->op == nir_op_fsat || alu->op == nir_op_fsign || alu->op == nir_op_ftrunc) {
        if (alu->def.bit_size != 32) {
            low->fail = 1;
            return;
        }
        unsigned zero = low_f32(low, 0.f);
        unsigned one = low_f32(low, 1.f);
        for (i = 0; i < comps; i++) {
            unsigned value = low_lane(low, alu, 0, i);
            if (alu->op == nir_op_fsat) {
                unsigned positive = low_select_f32(low, low_cmp(low, MXSB_OP_LT, zero, value), value, zero);
                ids[i] = low_select_f32(low, low_cmp(low, MXSB_OP_LT, positive, one), positive, one);
            } else if (alu->op == nir_op_fsign) {
                unsigned negative = low_select_f32(low, low_cmp(low, MXSB_OP_LT, value, zero),
                                                  low_f32(low, -1.f), zero);
                unsigned signed_value = low_select_f32(low, low_cmp(low, MXSB_OP_LT, zero, value),
                                                      one, negative);
                ids[i] = low_select_f32(low, low_cmp(low, MXSB_OP_EQ, value, zero), value, signed_value);
            } else {
                unsigned floor_value = low_unop_ty(low, MXSB_OP_FLOOR, MXSB_TYPE_F32, value);
                unsigned ceil_value = low_unop_ty(low, MXSB_OP_CEIL, MXSB_TYPE_F32, value);
                ids[i] = low_select_f32(low, low_cmp(low, MXSB_OP_LT, value, zero), ceil_value, floor_value);
            }
        }
        low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->op == nir_op_fmod) {
        if (alu->def.bit_size != 32) {
            low->fail = 1;
            return;
        }
        for (i = 0; i < comps; i++) {
            unsigned value = low_lane(low, alu, 0, i);
            unsigned divisor = low_lane(low, alu, 1, i);
            unsigned quotient = low_binop(low, MXSB_OP_DIV, value, divisor);
            unsigned integral = low_unop_ty(low, MXSB_OP_FLOOR, MXSB_TYPE_F32, quotient);
            ids[i] = low_binop(low, MXSB_OP_SUB, value, low_binop(low, MXSB_OP_MUL, divisor, integral));
        }
        low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->op == nir_op_fmax || alu->op == nir_op_fmin) {
        int lesser = alu->op == nir_op_fmin;
        for (i = 0; i < comps; i++) {
            unsigned left = low_lane(low, alu, 0, i);
            unsigned right = low_lane(low, alu, 1, i);
            unsigned cond = low_cmp(low, MXSB_OP_LT, left, right);
            ids[i] = low_select_f32(low, cond, lesser ? left : right, lesser ? right : left);
        }
        low_finish_vec(low, alu, ids, comps);
        return;
    }
    if (alu->def.bit_size == 32 && (alu->op == nir_op_umin || alu->op == nir_op_umax ||
                                    alu->op == nir_op_imin || alu->op == nir_op_imax)) {
        bool lesser = alu->op == nir_op_umin || alu->op == nir_op_imin;
        bool sign = alu->op == nir_op_imin || alu->op == nir_op_imax;
        for (i = 0; i < comps; i++) {
            unsigned left = low_lane(low, alu, 0, i);
            unsigned right = low_lane(low, alu, 1, i);
            unsigned a = left, b = right;
            if (sign) {
                unsigned bias = low_u32(low, 0x80000000u);
                a = low_binop_ty(low, MXSB_OP_BIT_XOR, MXSB_TYPE_U32, left, bias);
                b = low_binop_ty(low, MXSB_OP_BIT_XOR, MXSB_TYPE_U32, right, bias);
            }
            unsigned below = low_cmp(low, MXSB_OP_LT, a, b);
            ids[i] = low_select(low, MXSB_TYPE_U32, below, lesser ? left : right, lesser ? right : left);
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->def.bit_size == 32 && (alu->op == nir_op_udiv || alu->op == nir_op_umod ||
                                    alu->op == nir_op_idiv || alu->op == nir_op_irem)) {
        bool sign = alu->op == nir_op_idiv || alu->op == nir_op_irem;
        uint16_t opcode = alu->op == nir_op_udiv || alu->op == nir_op_idiv ? MXSB_OP_IDIV : MXSB_OP_IMOD;
        for (i = 0; i < comps; i++) {
            unsigned left = low_lane(low, alu, 0, i);
            unsigned right = low_lane(low, alu, 1, i);
            if (sign) {
                left = low_unop_ty(low, MXSB_OP_BITCAST, MXSB_TYPE_I32, left);
                right = low_unop_ty(low, MXSB_OP_BITCAST, MXSB_TYPE_I32, right);
                ids[i] = low_unop_ty(low, MXSB_OP_BITCAST, MXSB_TYPE_U32,
                                     low_binop_ty(low, opcode, MXSB_TYPE_I32, left, right));
            } else {
                ids[i] = low_binop_ty(low, opcode, MXSB_TYPE_U32, left, right);
            }
        }
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->def.bit_size == 32 && alu->src[0].src.ssa->bit_size == 32 &&
        (alu->op == nir_op_u2u32 || alu->op == nir_op_i2i32)) {
        for (i = 0; i < comps; i++)
            ids[i] = low_lane(low, alu, 0, i);
        low_finish_lanes(low, alu->def.index, ids, comps);
        return;
    }
    if (alu->op == nir_op_fadd) bin = MXSB_OP_ADD;
    else if (alu->op == nir_op_fsub) bin = MXSB_OP_SUB;
    else if (alu->op == nir_op_fmul) bin = MXSB_OP_MUL;
    else if (alu->op == nir_op_fdiv) bin = MXSB_OP_DIV;
    else if (alu->op == nir_op_fneg) un = MXSB_OP_NEG;
    else if (alu->op == nir_op_fabs) un = MXSB_OP_FABS;
    else if (alu->op == nir_op_ffloor) un = MXSB_OP_FLOOR;
    else if (alu->op == nir_op_fceil) un = MXSB_OP_CEIL;
    else if (alu->op == nir_op_ffract) un = MXSB_OP_FRACT;
    else if (alu->op == nir_op_fsqrt) un = MXSB_OP_SQRT;
    else if (alu->op == nir_op_frsq) un = MXSB_OP_RSQ;
    else if (alu->op == nir_op_frcp) un = MXSB_OP_RCP;
    else if (alu->op == nir_op_fexp2) un = MXSB_OP_EXP2;
    else if (alu->op == nir_op_flog2) un = MXSB_OP_LOG2;
    else if (alu->op == nir_op_fsin) un = MXSB_OP_SIN;
    else if (alu->op == nir_op_fcos) un = MXSB_OP_COS;
    else {
        static int reported;
        if (!reported) {
            fprintf(stderr, "mxgpu unsupported nir op %s\n", nir_op_infos[alu->op].name);
            reported = 1;
        }
        low->fail = 1;
        return;
    }
    for (i = 0; i < comps; i++) {
        if (un)
            ids[i] = low_unop(low, un, low_lane(low, alu, 0, i));
        else
            ids[i] = low_binop(low, bin, low_lane(low, alu, 0, i), low_lane(low, alu, 1, i));
    }
    low_finish_vec(low, alu, ids, comps);
}

static unsigned low_vertex_load(struct mx_low *low, unsigned location)
{
    uint32_t rec[5];
    unsigned index;
    if (location >= low->vertex_attribute_count || location >= MXGPU_SHADER_VERTEX_SLOTS) {
        low->fail = 1;
        return 0;
    }
    if (low->vload[location])
        return low->vload[location];
    if (!low->vid) {
        low->vid = low_id(low);
        rec[0] = (4u << 16) | MXSB_OP_BUILTIN;
        rec[1] = low->vid;
        rec[2] = MXSB_TYPE_U32;
        rec[3] = MXSB_BUILTIN_VERTEX_ID;
        low_emit(low, rec, 4);
    }
    index = low->vertex_attribute_count == 1 ? low->vid :
        low_binop_ty(low, MXSB_OP_MUL, MXSB_TYPE_U32, low->vid, low_u32(low, low->vertex_attribute_count));
    if (location)
        index = low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, index, low_u32(low, location));
    low->vload[location] = low_id(low);
    rec[0] = (5u << 16) | MXSB_OP_BUFFER_LOAD;
    rec[1] = low->vload[location];
    rec[2] = MXSB_TYPE_F32X4;
    rec[3] = 1;
    rec[4] = index;
    low_emit(low, rec, 5);
    return low->vload[location];
}

static int vertex_slot(nir_deref_instr *leaf, unsigned *slot)
{
    nir_variable *var = nir_deref_instr_get_variable(leaf);
    uint64_t offset = 0;
    if (!var)
        return -1;
    for (nir_deref_instr *deref = leaf; deref && deref->deref_type != nir_deref_type_var;
         deref = nir_deref_instr_parent(deref)) {
        if (deref->deref_type != nir_deref_type_array || !nir_src_is_const(deref->arr.index))
            return -1;
        unsigned slots = glsl_count_attribute_slots(deref->type, false);
        offset += (uint64_t)nir_src_as_uint(deref->arr.index) * slots;
        if (!slots || offset >= 32)
            return -1;
    }
    offset += var->data.driver_location;
    if (offset >= 32)
        return -1;
    *slot = offset;
    return 0;
}

static int uniform_slot(nir_deref_instr *leaf, unsigned *slot)
{
    nir_deref_instr *deref = leaf;
    unsigned extra = 0;
    nir_variable *var;
    while (deref && deref->deref_type != nir_deref_type_var) {
        if (deref->deref_type == nir_deref_type_array) {
            unsigned step;
            if (!deref->type || !nir_src_is_const(deref->arr.index))
                return -1;
            step = glsl_count_vec4_slots(deref->type, false, false);
            extra += nir_src_as_uint(deref->arr.index) * step;
        } else if (deref->deref_type == nir_deref_type_struct) {
            nir_deref_instr *parent = nir_deref_instr_parent(deref);
            unsigned i, field = 0;
            if (!parent || !parent->type)
                return -1;
            for (i = 0; i < deref->strct.index; i++)
                field += glsl_count_vec4_slots(glsl_get_struct_field(parent->type, i), false, false);
            extra += field;
        } else {
            return -1;
        }
        deref = nir_deref_instr_parent(deref);
    }
    var = deref ? deref->var : NULL;
    if (!var || var->data.driver_location < 0)
        return -1;
    *slot = (unsigned)var->data.driver_location + extra;
    return 0;
}

static int low_ubo_load(struct mx_low *low, nir_intrinsic_instr *intr, nir_deref_instr *deref)
{
    uint64_t offset = 0;
    nir_deref_instr *root = deref;
    struct mx_descriptor_ref *ref;
    struct mxgpu_uniform_buffer *buffer = NULL;
    unsigned i, size, ids[4], comps = intr->def.num_components;
    while (root && root->deref_type != nir_deref_type_cast) {
        nir_deref_instr *parent = nir_deref_instr_parent(root);
        if (!parent)
            return -1;
        if (root->deref_type == nir_deref_type_struct) {
            int field = glsl_get_struct_field_offset(parent->type, root->strct.index);
            if (field < 0)
                return -1;
            offset += (unsigned)field;
        } else if (root->deref_type == nir_deref_type_array) {
            unsigned stride = glsl_type_is_vector(parent->type) ? 4 : glsl_get_explicit_stride(parent->type);
            if (!stride || !nir_src_is_const(root->arr.index))
                return -1;
            offset += (uint64_t)nir_src_as_uint(root->arr.index) * stride;
        } else {
            return -1;
        }
        root = parent;
    }
    if (!root || !root->parent.ssa || root->parent.ssa->index >= MX_LOW_DEFS ||
        !(ref = &low->descriptors[root->parent.ssa->index])->valid ||
        intr->def.bit_size != 32 || !comps || comps > 4 || (offset & 3))
        return -1;
    size = glsl_get_explicit_size(root->type, false);
    if (!size || size > UINT32_MAX - 15 || offset + comps * 4 > size)
        return -1;
    for (i = 0; i < low->uniform_buffer_count; i++) {
        struct mxgpu_uniform_buffer *candidate = &low->uniform_buffers[i];
        if (candidate->set == ref->set && candidate->binding == ref->binding && candidate->element == ref->element)
            buffer = candidate;
    }
    if (!buffer) {
        unsigned slots = (size + 15) / 16;
        if (low->uniform_buffer_count >= MXGPU_UNIFORM_BUFFERS || low->uniform_count > UINT32_MAX / 16 - slots)
            return -1;
        buffer = &low->uniform_buffers[low->uniform_buffer_count++];
        *buffer = (struct mxgpu_uniform_buffer){ref->set, ref->binding, ref->element, low->uniform_count * 16, size};
        low->uniform_count += slots;
    } else if (buffer->size != size) {
        return -1;
    }
    for (i = 0; i < comps; i++) {
        uint64_t byte = (uint64_t)buffer->offset + offset + i * 4;
        unsigned vec = low_uniform_vec4(low, (unsigned)(byte / 16));
        uint32_t extract[5] = {(5u << 16) | MXSB_OP_EXTRACT, low_id(low), MXSB_TYPE_F32, vec, (unsigned)(byte % 16 / 4)};
        low_emit(low, extract, 5);
        ids[i] = extract[1];
    }
    unsigned value = ids[0];
    if (comps > 1) {
        uint32_t construct[8] = {((4u + comps) << 16) | MXSB_OP_CONSTRUCT, low_id(low), low_vec_type(comps), comps};
        for (i = 0; i < comps; i++)
            construct[4 + i] = ids[i];
        low_emit(low, construct, 4 + comps);
        value = construct[1];
    }
    low_remember(low, intr->def.index, value, comps);
    low_keep_comps(low, intr->def.index, ids, comps);
    return low->fail ? -1 : 0;
}

static int low_gl_ubo_load(struct mx_low *low, nir_intrinsic_instr *intr)
{
    struct mxgpu_uniform_buffer *buffer = NULL;
    bool push = intr->intrinsic == nir_intrinsic_load_push_constant;
    unsigned source = push ? 0 : 1;
    unsigned base = push ? nir_intrinsic_base(intr) : 0;
    unsigned comps = intr->def.num_components, binding = 0;
    unsigned ids[4];
    if ((!push && !nir_src_is_const(intr->src[0])) || intr->def.bit_size != 32 || !comps || comps > 4)
        return -1;
    if (!push)
        binding = nir_src_as_uint(intr->src[0]);
    for (unsigned i = 0; i < low->uniform_buffer_count; i++)
        if (push ? low->uniform_buffers[i].push_constant : low->uniform_buffers[i].constant_slot == binding + 1)
            buffer = &low->uniform_buffers[i];
    if (!buffer)
        return -1;
    if (nir_src_is_const(intr->src[source])) {
        uint64_t offset = (uint64_t)base + nir_src_as_uint(intr->src[source]);
        if ((offset & 3) || offset + comps * 4 > buffer->size)
            return -1;
        for (unsigned i = 0; i < comps; i++) {
            uint64_t byte = buffer->offset + offset + i * 4;
            unsigned vec = low_uniform_vec4(low, byte / 16);
            uint32_t rec[5] = {(5u << 16) | MXSB_OP_EXTRACT, low_id(low), MXSB_TYPE_F32, vec, byte % 16 / 4};
            low_emit(low, rec, 5);
            ids[i] = rec[1];
        }
    } else {
        if (!push && (nir_intrinsic_align_mul(intr) < 4 || (nir_intrinsic_align_offset(intr) & 3)))
            return -1;
        unsigned byte = low_component(low, intr->src[source].ssa, 0);
        byte = low_cast(low, byte, MXSB_TYPE_U32);
        if (base)
            byte = low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, byte, low_u32(low, base));
        unsigned first_word = low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32,
            low_binop_ty(low, MXSB_OP_SHR, MXSB_TYPE_U32, byte, low_u32(low, 2)),
            low_u32(low, buffer->offset / 4));
        for (unsigned i = 0; i < comps; i++) {
            unsigned word = i ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, first_word, low_u32(low, i)) : first_word;
            unsigned index = low_binop_ty(low, MXSB_OP_SHR, MXSB_TYPE_U32, word, low_u32(low, 2));
            unsigned lane = low_binop_ty(low, MXSB_OP_BIT_AND, MXSB_TYPE_U32, word, low_u32(low, 3));
            unsigned vec = low_uniform_vec4_id(low, index), value = 0;
            for (unsigned c = 0; c < 4; c++) {
                uint32_t rec[5] = {(5u << 16) | MXSB_OP_EXTRACT, low_id(low), MXSB_TYPE_F32, vec, c};
                low_emit(low, rec, 5);
                value = c ? low_select_f32(low, low_cmp(low, MXSB_OP_EQ, lane, low_u32(low, c)), rec[1], value) : rec[1];
            }
            ids[i] = value;
        }
    }
    low_finish_lanes(low, intr->def.index, ids, comps);
    return low->fail ? -1 : 0;
}

static int cf_list_contains(struct exec_list *list, nir_cf_node *node)
{
    foreach_list_typed(nir_cf_node, cursor, node, list) {
        if (cursor == node)
            return 1;
    }
    return 0;
}

static int low_phi_merge(struct mx_low *low, nir_phi_instr *phi, nir_block *block)
{
    nir_phi_src *src;
    nir_def *defs[2];
    nir_block *preds[2];
    unsigned nsrc = 0;
    unsigned comps;
    unsigned i;
    unsigned ids[4];
    src = NULL;
    nir_foreach_phi_src(phi_src, phi) {
        if (nsrc < 2) {
            defs[nsrc] = phi_src->src.ssa;
            preds[nsrc] = phi_src->pred;
        }
        nsrc++;
    }
    comps = phi->def.num_components;
    if (nsrc != 2 || comps == 0 || comps > 4 || !defs[0] || !defs[1] || !preds[0] || !preds[1])
        return 0;
    if (defs[0]->index >= MX_LOW_DEFS || defs[1]->index >= MX_LOW_DEFS || !low->id_of[defs[0]->index] || !low->id_of[defs[1]->index])
        return 0;
    if (preds[0]->cf_node.parent && preds[0]->cf_node.parent == preds[1]->cf_node.parent &&
        preds[0]->cf_node.parent->type == nir_cf_node_if) {
        nir_if *nif = nir_cf_node_as_if(preds[0]->cf_node.parent);
        int then0 = cf_list_contains(&nif->then_list, &preds[0]->cf_node);
        unsigned cond;
        unsigned then_index = then0 ? 0u : 1u;
        unsigned else_index = then0 ? 1u : 0u;
        nir_def *cond_def = nif->condition.ssa;
        if (!cond_def)
            return 0;
        cond = low_component(low, cond_def, 0);
        if (!cond)
            return 0;
        for (i = 0; i < comps; i++) {
            unsigned on_true = low_component(low, defs[then_index], i);
            unsigned on_false = low_component(low, defs[else_index], i);
            if (phi->def.bit_size == 1)
                ids[i] = low_select(low, MXSB_TYPE_BOOL, cond, on_true, on_false);
            else
                ids[i] = low_select_f32(low, cond, on_true, on_false);
        }
        if (phi->def.bit_size == 1 || comps == 1)
            low_finish_lanes(low, phi->def.index, ids, comps);
        else {
            uint32_t built[8];
            built[0] = ((4u + comps) << 16) | MXSB_OP_CONSTRUCT;
            built[1] = low_id(low);
            built[2] = low_vec_type(comps);
            built[3] = comps;
            for (i = 0; i < comps; i++)
                built[4 + i] = ids[i];
            low_emit(low, built, 4 + comps);
            low_remember(low, phi->def.index, built[1], comps);
            low_keep_comps(low, phi->def.index, ids, comps);
        }
        return 1;
    }
    if (block) {
        nir_cf_node *prev = nir_cf_node_prev(&block->cf_node);
        if (prev && prev->type == nir_cf_node_loop) {
            int inside0 = 0;
            nir_cf_node *parent = &preds[0]->cf_node;
            while (parent) {
                if (parent == prev) {
                    inside0 = 1;
                    break;
                }
                parent = parent->parent;
            }
            low_remember(low, phi->def.index, low->id_of[defs[inside0 ? 0 : 1]->index], comps);
            if (comps > 1 && defs[inside0 ? 0 : 1]->index < MX_LOW_DEFS)
                low_keep_comps(low, phi->def.index, low->comp_id[defs[inside0 ? 0 : 1]->index], comps);
            return 1;
        }
    }
    (void)src;
    return 0;
}

static unsigned low_block_predicate(struct mx_low *low, nir_block *block)
{
    nir_cf_node *child = &block->cf_node;
    unsigned predicate = 0;
    if (low->predicate_block == block)
        return low->predicate;
    for (nir_cf_node *parent = child->parent; parent && !low->fail; child = parent, parent = parent->parent) {
        if (parent->type == nir_cf_node_loop) {
            low->fail = 1;
            return 0;
        }
        if (parent->type != nir_cf_node_if)
            continue;
        nir_if *nif = nir_cf_node_as_if(parent);
        unsigned cond = low_component(low, nif->condition.ssa, 0);
        if (cond && low->id_type[cond] != MXSB_TYPE_BOOL)
            cond = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL,
                               low_cmp(low, MXSB_OP_EQ, low_cast(low, cond, MXSB_TYPE_U32), low_u32(low, 0)));
        if (!cf_list_contains(&nif->then_list, child))
            cond = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, cond);
        predicate = predicate ? low_select(low, MXSB_TYPE_BOOL, predicate, cond, low_bool(low, 0)) : cond;
    }
    low->predicate_block = block;
    low->predicate = predicate;
    return predicate;
}

static unsigned low_guarded_index(struct mx_low *low, nir_block *block, unsigned index)
{
    unsigned predicate = low_block_predicate(low, block);
    if (!predicate)
        return index;
    return low_select(low, MXSB_TYPE_U32, predicate, index, low_u32(low, UINT32_MAX));
}

static unsigned low_word_index(struct mx_low *low, unsigned byte, unsigned extra_words)
{
    unsigned word = low_binop_ty(low, MXSB_OP_SHR, MXSB_TYPE_U32, low_cast(low, byte, MXSB_TYPE_U32), low_u32(low, 2));
    return extra_words ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, word, low_u32(low, extra_words)) : word;
}

static unsigned low_storage_binding(struct mx_low *low, const struct mx_descriptor_ref *ref, unsigned access)
{
    for (unsigned i = 0; i < low->storage_count; i++) {
        struct mxgpu_storage_binding *b = &low->storage[i];
        if (b->set == ref->set && b->binding == ref->binding && b->element == ref->element) {
            b->access |= access;
            return b->binding_id;
        }
    }
    if (low->storage_count >= MXGPU_SHADER_STORAGE_BUFFERS) {
        low->fail = 1;
        return 0;
    }
    struct mxgpu_storage_binding *b = &low->storage[low->storage_count];
    *b = (struct mxgpu_storage_binding){ref->set, ref->binding, ref->element,
                                        MXGPU_COMPUTE_UNIFORM_BINDING + 1 + low->storage_count,
                                        low->storage_count, access};
    low->storage_count++;
    return b->binding_id;
}

static int low_ssbo_address(struct mx_low *low, nir_deref_instr *deref, unsigned access,
                            unsigned *binding_id, unsigned *byte)
{
    uint64_t constant = 0;
    unsigned dynamic = 0;
    nir_deref_instr *cursor = deref;
    while (cursor && cursor->deref_type != nir_deref_type_cast) {
        nir_deref_instr *parent = nir_deref_instr_parent(cursor);
        if (!parent || !parent->type)
            return -1;
        if (glsl_type_is_matrix(parent->type) && glsl_matrix_type_is_row_major(parent->type))
            return -1;
        if (cursor->deref_type == nir_deref_type_struct) {
            int field = glsl_get_struct_field_offset(parent->type, cursor->strct.index);
            if (field < 0 || (field & 3))
                return -1;
            constant += (unsigned)field;
        } else if (cursor->deref_type == nir_deref_type_array || cursor->deref_type == nir_deref_type_ptr_as_array) {
            unsigned stride = nir_deref_instr_array_stride(cursor);
            nir_def *index = cursor->arr.index.ssa;
            if (!stride || (stride & 3) || !index)
                return -1;
            if (nir_src_is_const(cursor->arr.index)) {
                constant += (uint64_t)nir_src_as_uint(cursor->arr.index) * stride;
            } else {
                if (index->bit_size != 32 || index->index >= MX_LOW_DEFS || !low->id_of[index->index])
                    return -1;
                unsigned term = low_binop_ty(low, MXSB_OP_MUL, MXSB_TYPE_U32,
                                             low_cast(low, low_component(low, index, 0), MXSB_TYPE_U32),
                                             low_u32(low, stride));
                dynamic = dynamic ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, dynamic, term) : term;
            }
        } else {
            return -1;
        }
        if (constant > UINT32_MAX)
            return -1;
        cursor = parent;
    }
    if (!cursor || !(cursor->modes & nir_var_mem_ssbo) || !cursor->parent.ssa ||
        cursor->parent.ssa->index >= MX_LOW_DEFS || !low->descriptors[cursor->parent.ssa->index].valid)
        return -1;
    *binding_id = low_storage_binding(low, &low->descriptors[cursor->parent.ssa->index], access);
    *byte = dynamic ? (constant ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, dynamic, low_u32(low, (uint32_t)constant)) : dynamic)
                    : low_u32(low, (uint32_t)constant);
    return low->fail ? -1 : 0;
}

static bool low_atomic_opcodes(nir_atomic_op op, uint16_t *buffer, uint16_t *workgroup)
{
    *buffer = *workgroup = 0;
    switch (op) {
    case nir_atomic_op_iadd: *buffer = MXSB_OP_BUFFER_ATOMIC_ADD; *workgroup = MXSB_OP_WORKGROUP_ATOMIC_ADD; break;
    case nir_atomic_op_isub: *buffer = MXSB_OP_BUFFER_ATOMIC_SUBTRACT; break;
    case nir_atomic_op_umin: *buffer = MXSB_OP_BUFFER_ATOMIC_MINIMUM; break;
    case nir_atomic_op_umax: *buffer = MXSB_OP_BUFFER_ATOMIC_MAXIMUM; break;
    case nir_atomic_op_iand: *buffer = MXSB_OP_BUFFER_ATOMIC_AND; break;
    case nir_atomic_op_ior: *buffer = MXSB_OP_BUFFER_ATOMIC_OR; break;
    case nir_atomic_op_ixor: *buffer = MXSB_OP_BUFFER_ATOMIC_XOR; break;
    case nir_atomic_op_xchg: *buffer = MXSB_OP_BUFFER_ATOMIC_EXCHANGE; *workgroup = MXSB_OP_WORKGROUP_ATOMIC_EXCHANGE; break;
    case nir_atomic_op_cmpxchg: *buffer = MXSB_OP_BUFFER_ATOMIC_COMPARE_EXCHANGE; *workgroup = MXSB_OP_WORKGROUP_ATOMIC_COMPARE_EXCHANGE; break;
    default: return false;
    }
    return true;
}

static void low_compute_builtin(struct mx_low *low, nir_intrinsic_instr *intr, uint32_t builtin, bool vector)
{
    unsigned comps = intr->def.num_components, ids[3];
    if (intr->def.bit_size != 32 || !comps || comps > (vector ? 3u : 1u)) {
        low->fail = 1;
        return;
    }
    uint32_t rec[5] = {(4u << 16) | MXSB_OP_BUILTIN, low_id(low), vector ? MXSB_TYPE_U32X3 : MXSB_TYPE_U32, builtin};
    low_emit(low, rec, 4);
    if (!vector) {
        low_remember(low, intr->def.index, rec[1], 1);
        return;
    }
    for (unsigned lane = 0; lane < comps; lane++) {
        uint32_t extract[5] = {(5u << 16) | MXSB_OP_EXTRACT, low_id(low), MXSB_TYPE_U32, rec[1], lane};
        low_emit(low, extract, 5);
        ids[lane] = extract[1];
    }
    low_finish_lanes(low, intr->def.index, ids, comps);
}

static unsigned low_shared_index(struct mx_low *low, nir_intrinsic_instr *intr, unsigned offset_src)
{
    nir_def *offset = intr->src[offset_src].ssa;
    if (!offset || offset->bit_size != 32 || (nir_intrinsic_has_align_mul(intr) && nir_intrinsic_align(intr) < 4) ||
        (nir_intrinsic_base(intr) & 3))
        return 0;
    if (nir_src_is_const(intr->src[offset_src])) {
        uint64_t byte = (uint64_t)nir_src_as_uint(intr->src[offset_src]) + (unsigned)nir_intrinsic_base(intr);
        return byte > UINT32_MAX || (byte & 3) ? 0 : low_u32(low, (uint32_t)(byte / 4));
    }
    return low_word_index(low, low_component(low, offset, 0), (unsigned)nir_intrinsic_base(intr) / 4);
}

static int low_compute_intrinsic(struct mx_low *low, nir_intrinsic_instr *intr, nir_block *block)
{
    nir_intrinsic_op op = intr->intrinsic;
    bool has_dest = nir_intrinsic_infos[op].has_dest;
    if (has_dest && intr->def.index >= MX_LOW_DEFS)
        return low->fail = 1;
    switch (op) {
    case nir_intrinsic_load_global_invocation_id:
        low_compute_builtin(low, intr, MXSB_BUILTIN_GLOBAL_INVOCATION_ID, true);
        return 1;
    case nir_intrinsic_load_local_invocation_id:
        low_compute_builtin(low, intr, MXSB_BUILTIN_LOCAL_INVOCATION_ID, true);
        return 1;
    case nir_intrinsic_load_workgroup_id:
        low_compute_builtin(low, intr, MXSB_BUILTIN_WORKGROUP_ID, true);
        return 1;
    case nir_intrinsic_load_num_workgroups:
        low_compute_builtin(low, intr, MXSB_BUILTIN_NUM_WORKGROUPS, true);
        return 1;
    case nir_intrinsic_load_workgroup_size:
        low_compute_builtin(low, intr, MXSB_BUILTIN_WORKGROUP_SIZE, true);
        return 1;
    case nir_intrinsic_load_local_invocation_index:
        low_compute_builtin(low, intr, MXSB_BUILTIN_LOCAL_INVOCATION_INDEX, false);
        return 1;
    case nir_intrinsic_load_base_global_invocation_id:
    case nir_intrinsic_load_base_workgroup_id: {
        unsigned comps = intr->def.num_components, ids[3];
        if (intr->def.bit_size != 32 || !comps || comps > 3)
            return low->fail = 1;
        for (unsigned lane = 0; lane < comps; lane++)
            ids[lane] = low_u32(low, 0);
        low_finish_lanes(low, intr->def.index, ids, comps);
        return 1;
    }
    case nir_intrinsic_barrier: {
        mesa_scope execution = nir_intrinsic_execution_scope(intr);
        mesa_scope memory = nir_intrinsic_memory_scope(intr);
        nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
        uint32_t device = (1u << 16) | MXSB_OP_DEVICE_MEMORY_BARRIER;
        uint32_t control = (1u << 16) | MXSB_OP_CONTROL_BARRIER;
        uint32_t fence = (1u << 16) | MXSB_OP_MEMORY_BARRIER;
        if (memory != SCOPE_NONE && (modes & (nir_var_mem_ssbo | nir_var_mem_global | nir_var_image)))
            low_emit(low, &device, 1);
        if (execution != SCOPE_NONE)
            low_emit(low, &control, 1);
        else if (memory != SCOPE_NONE && (modes & nir_var_mem_shared))
            low_emit(low, &fence, 1);
        return 1;
    }
    case nir_intrinsic_load_shared:
    case nir_intrinsic_store_shared:
    case nir_intrinsic_shared_atomic:
    case nir_intrinsic_shared_atomic_swap: {
        bool store = op == nir_intrinsic_store_shared;
        unsigned value_bits = store ? intr->src[0].ssa->bit_size : intr->def.bit_size;
        unsigned comps = store ? intr->src[0].ssa->num_components : intr->def.num_components;
        unsigned index = low_shared_index(low, intr, store ? 1 : 0), ids[4];
        if (!index || value_bits != 32 || !comps || comps > 4 || !low->workgroup_bytes)
            return low->fail = 1;
        if (op == nir_intrinsic_load_shared) {
            for (unsigned lane = 0; lane < comps; lane++) {
                unsigned at = lane ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, index, low_u32(low, lane)) : index;
                uint32_t rec[5] = {(5u << 16) | MXSB_OP_WORKGROUP_LOAD, low_id(low), MXSB_TYPE_U32, 0, at};
                low_emit(low, rec, 5);
                ids[lane] = rec[1];
            }
            low_finish_lanes(low, intr->def.index, ids, comps);
        } else if (store) {
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned lane = 0; lane < comps; lane++) {
                if (!(mask & (1u << lane)))
                    continue;
                unsigned at = lane ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, index, low_u32(low, lane)) : index;
                uint32_t rec[5] = {(5u << 16) | MXSB_OP_WORKGROUP_STORE, MXSB_TYPE_U32, 0,
                                   low_guarded_index(low, block, at),
                                   low_cast(low, low_component(low, intr->src[0].ssa, lane), MXSB_TYPE_U32)};
                low_emit(low, rec, 5);
            }
        } else {
            uint16_t buffer_op, workgroup_op;
            bool swap = op == nir_intrinsic_shared_atomic_swap;
            if (comps != 1 || !low_atomic_opcodes(nir_intrinsic_atomic_op(intr), &buffer_op, &workgroup_op) || !workgroup_op)
                return low->fail = 1;
            uint32_t rec[7] = {((swap ? 7u : 6u) << 16) | workgroup_op, 0, MXSB_TYPE_U32, 0,
                               low_guarded_index(low, block, index),
                               low_cast(low, low_component(low, intr->src[1].ssa, 0), MXSB_TYPE_U32)};
            if (swap)
                rec[6] = low_cast(low, low_component(low, intr->src[2].ssa, 0), MXSB_TYPE_U32);
            rec[1] = low_id(low);
            low_emit(low, rec, swap ? 7 : 6);
            low_remember(low, intr->def.index, rec[1], 1);
        }
        return 1;
    }
    case nir_intrinsic_load_deref:
    case nir_intrinsic_store_deref:
    case nir_intrinsic_deref_atomic:
    case nir_intrinsic_deref_atomic_swap: {
        nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
        bool store = op == nir_intrinsic_store_deref, load = op == nir_intrinsic_load_deref;
        unsigned binding = 0, byte = 0, ids[4];
        if (!deref || !(deref->modes & nir_var_mem_ssbo))
            return 0;
        nir_def *value = store ? intr->src[1].ssa : &intr->def;
        unsigned comps = value->num_components;
        if (value->bit_size != 32 || !comps || comps > 4 ||
            low_ssbo_address(low, deref, load ? MXSB_ACCESS_READ : store ? MXSB_ACCESS_WRITE : MXSB_ACCESS_READ_WRITE,
                             &binding, &byte))
            return low->fail = 1;
        unsigned index = low_word_index(low, byte, 0);
        if (load) {
            for (unsigned lane = 0; lane < comps; lane++) {
                unsigned at = lane ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, index, low_u32(low, lane)) : index;
                uint32_t rec[5] = {(5u << 16) | MXSB_OP_BUFFER_LOAD, low_id(low), MXSB_TYPE_U32, binding, at};
                low_emit(low, rec, 5);
                ids[lane] = rec[1];
            }
            low_finish_lanes(low, intr->def.index, ids, comps);
        } else if (store) {
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned lane = 0; lane < comps; lane++) {
                if (!(mask & (1u << lane)))
                    continue;
                unsigned at = lane ? low_binop_ty(low, MXSB_OP_ADD, MXSB_TYPE_U32, index, low_u32(low, lane)) : index;
                uint32_t rec[4] = {(4u << 16) | MXSB_OP_BUFFER_STORE, binding, low_guarded_index(low, block, at),
                                   low_cast(low, low_component(low, value, lane), MXSB_TYPE_U32)};
                low_emit(low, rec, 4);
            }
        } else {
            uint16_t buffer_op, workgroup_op;
            bool swap = op == nir_intrinsic_deref_atomic_swap;
            if (comps != 1 || !low_atomic_opcodes(nir_intrinsic_atomic_op(intr), &buffer_op, &workgroup_op))
                return low->fail = 1;
            uint32_t rec[7] = {((swap ? 7u : 6u) << 16) | buffer_op, 0, MXSB_TYPE_U32, binding,
                               low_guarded_index(low, block, index),
                               low_cast(low, low_component(low, intr->src[1].ssa, 0), MXSB_TYPE_U32)};
            if (swap)
                rec[6] = low_cast(low, low_component(low, intr->src[2].ssa, 0), MXSB_TYPE_U32);
            rec[1] = low_id(low);
            low_emit(low, rec, swap ? 7 : 6);
            low_remember(low, intr->def.index, rec[1], 1);
        }
        return 1;
    }
    default:
        return 0;
    }
}

static int low_shader(const nir_shader *nir, int fragment, bool compute, struct mx_low *low)
{
    nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)nir);
    unsigned ret_id = 0;
    unsigned ret_lanes[4] = {0};
    bool partial_return = false;
    nir_block *resume = NULL;
    int stop = 0;
    nir_index_ssa_defs(impl);
    memset(low, 0, sizeof *low);
    low->next = 1;
    low->compute = compute;
    low->uniform_binding = compute ? MXGPU_COMPUTE_UNIFORM_BINDING : fragment ? 4 : 3;
    low->uniform_count = nir->num_uniforms;
    if (!fragment && !compute) {
        for (unsigned i = 0; i < MXGPU_SHADER_VERTEX_SLOTS; i++)
            low->vertex_input_locations[i] = UINT32_MAX;
        nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
            unsigned slots = glsl_count_attribute_slots(var->type, false);
            if (!slots || var->data.driver_location >= MXGPU_SHADER_VERTEX_SLOTS || slots > 32 - var->data.driver_location)
                return -1;
            unsigned end = var->data.driver_location + slots;
            for (unsigned i = 0; i < slots; i++)
                low->vertex_input_locations[var->data.driver_location + i] = var->data.location + i;
            if (low->vertex_attribute_count < end)
                low->vertex_attribute_count = end;
        }
    }
    if (!fragment && !compute) {
        nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
                if (instr->type != nir_instr_type_intrinsic)
                    continue;
                nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
                if (op == nir_intrinsic_load_vertex_id || op == nir_intrinsic_load_vertex_id_zero_base ||
                    op == nir_intrinsic_load_instance_id || op == nir_intrinsic_load_base_vertex ||
                    op == nir_intrinsic_load_first_vertex || op == nir_intrinsic_load_base_instance)
                    low->vertex_builtins = true;
            }
        }
        if (low->vertex_builtins) {
            if (low->vertex_attribute_count >= MXGPU_SHADER_VERTEX_SLOTS)
                return -1;
            low->vertex_builtin_slot = low->vertex_attribute_count++;
        }
    }
    unsigned ubo_sizes[MXGPU_UNIFORM_BUFFERS] = {0};
    unsigned push_size = 0;
    uint32_t push_words = 0;
    nir_foreach_block(block, impl) {
        nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
                continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_push_constant) {
                uint64_t size = (uint64_t)nir_intrinsic_base(intr) + nir_intrinsic_range(intr);
                if (intr->def.bit_size != 32 || !intr->def.num_components || intr->def.num_components > 4)
                    return -1;
                if (nir_src_is_const(intr->src[0])) {
                    uint64_t end = (uint64_t)nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[0]) + intr->def.num_components * 4;
                    if (!size || size > 128)
                        size = end;
                    if (size < end)
                        return -1;
                }
                if (!size || size > 128)
                    return -1;
                uint64_t start = nir_intrinsic_base(intr);
                uint64_t end = size;
                if (nir_src_is_const(intr->src[0])) {
                    start += nir_src_as_uint(intr->src[0]);
                    end = start + intr->def.num_components * 4;
                }
                if ((start & 3) || (end & 3) || end > 128 || end <= start)
                    return -1;
                for (unsigned word = start / 4; word < end / 4; word++)
                    push_words |= UINT32_C(1) << word;
                push_size = MAX2(push_size, size);
                continue;
            }
            if (intr->intrinsic != nir_intrinsic_load_ubo)
                continue;
            if (!nir_src_is_const(intr->src[0]) || intr->def.bit_size != 32 ||
                !intr->def.num_components || intr->def.num_components > 4)
                return -1;
            unsigned binding = nir_src_as_uint(intr->src[0]);
            uint64_t size = (uint64_t)nir_intrinsic_range_base(intr) + nir_intrinsic_range(intr);
            if (binding >= MXGPU_UNIFORM_BUFFERS - 1)
                return -1;
            if (nir_src_is_const(intr->src[1])) {
                uint64_t end = (uint64_t)nir_src_as_uint(intr->src[1]) + intr->def.num_components * 4;
                if (!size || size > 65536)
                    size = end;
                if (size < end)
                    return -1;
            }
            if (!size || size > 65536)
                return -1;
            if (ubo_sizes[binding] < size)
                ubo_sizes[binding] = size;
        }
    }
    for (unsigned binding = 0; binding < MXGPU_UNIFORM_BUFFERS - 1; binding++) {
        unsigned size = ubo_sizes[binding];
        if (!size)
            continue;
        unsigned slots = (size + 15) / 16;
        if (low->uniform_count > UINT32_MAX / 16 - slots || low->uniform_buffer_count >= MXGPU_UNIFORM_BUFFERS)
            return -1;
        struct mxgpu_uniform_buffer *buffer = &low->uniform_buffers[low->uniform_buffer_count++];
        *buffer = (struct mxgpu_uniform_buffer){.binding = binding, .offset = low->uniform_count * 16,
                                              .size = size, .constant_slot = binding + 1};
        low->uniform_count += slots;
    }
    if (push_size) {
        unsigned slots = (push_size + 15) / 16;
        if (low->uniform_count > UINT32_MAX / 16 - slots || low->uniform_buffer_count >= MXGPU_UNIFORM_BUFFERS)
            return -1;
        struct mxgpu_uniform_buffer *buffer = &low->uniform_buffers[low->uniform_buffer_count++];
        *buffer = (struct mxgpu_uniform_buffer){.offset = low->uniform_count * 16, .size = push_size, .push_constant = true,
                                              .push_constant_words = push_words};
        low->uniform_count += slots;
    }
    if (compute) {
        nir_foreach_block(block, impl) {
            for (nir_cf_node *parent = block->cf_node.parent; parent; parent = parent->parent)
                if (parent->type == nir_cf_node_loop)
                    return -1;
        }
        if (nir->info.shared_size) {
            if (nir->info.shared_size > UINT32_MAX - 3u)
                return -1;
            low->workgroup_bytes = (nir->info.shared_size + 3u) & ~3u;
            uint32_t rec[2] = {(2u << 16) | MXSB_OP_WORKGROUP_MEMORY, low->workgroup_bytes};
            low_emit(low, rec, 2);
        }
    }
    nir_foreach_block(block, impl) {
        int leave_block = 0;
        if (stop)
            break;
        if (resume) {
            if (block != resume)
                continue;
            resume = NULL;
        }
        nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref)
                continue;
            if (instr->type == nir_instr_type_load_const) {
                low_const(low, nir_instr_as_load_const(instr));
            } else if (instr->type == nir_instr_type_alu) {
                low_alu(low, nir_instr_as_alu(instr));
            } else if (instr->type == nir_instr_type_intrinsic) {
                nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                if (compute && low_compute_intrinsic(low, intr, block)) {
                } else if (intr->intrinsic == nir_intrinsic_load_ubo || intr->intrinsic == nir_intrinsic_load_push_constant) {
                    low->fail = low_gl_ubo_load(low, intr) != 0;
                } else if (intr->intrinsic == nir_intrinsic_load_vertex_id ||
                    intr->intrinsic == nir_intrinsic_load_vertex_id_zero_base ||
                    intr->intrinsic == nir_intrinsic_load_instance_id ||
                    intr->intrinsic == nir_intrinsic_load_base_vertex ||
                    intr->intrinsic == nir_intrinsic_load_first_vertex ||
                    intr->intrinsic == nir_intrinsic_load_base_instance) {
                    unsigned lane = intr->intrinsic == nir_intrinsic_load_instance_id ? 1 :
                                    intr->intrinsic == nir_intrinsic_load_base_instance ? 3 :
                                    (intr->intrinsic == nir_intrinsic_load_base_vertex ||
                                     intr->intrinsic == nir_intrinsic_load_first_vertex) ? 2 : 0;
                    if (fragment || intr->def.bit_size != 32 || intr->def.num_components != 1 ||
                        intr->def.index >= MX_LOW_DEFS || !low->vertex_builtins) {
                        low->fail = 1;
                    } else {
                        uint32_t rec[5] = {(5u << 16) | MXSB_OP_EXTRACT, low_id(low), MXSB_TYPE_F32,
                                          low_vertex_load(low, low->vertex_builtin_slot), lane};
                        low_emit(low, rec, 5);
                        unsigned value = low_cast(low, rec[1], MXSB_TYPE_U32);
                        if (intr->intrinsic == nir_intrinsic_load_vertex_id_zero_base) {
                            rec[1] = low_id(low);
                            rec[4] = 2;
                            low_emit(low, rec, 5);
                            value = low_binop_ty(low, MXSB_OP_SUB, MXSB_TYPE_U32, value,
                                                 low_cast(low, rec[1], MXSB_TYPE_U32));
                        }
                        low_remember(low, intr->def.index, value, 1);
                    }
                } else if (intr->intrinsic == nir_intrinsic_ddx || intr->intrinsic == nir_intrinsic_ddy) {
                    unsigned ids[4], comps = intr->def.num_components;
                    if (!fragment || intr->def.bit_size != 32 || !comps || comps > 4) {
                        low->fail = 1;
                    } else {
                        for (unsigned lane = 0; lane < comps; lane++)
                            ids[lane] = low_unop_ty(low, intr->intrinsic == nir_intrinsic_ddx ? MXSB_OP_DDX : MXSB_OP_DDY,
                                                    MXSB_TYPE_F32, low_cast(low, low_component(low, intr->src[0].ssa, lane), MXSB_TYPE_F32));
                        low_finish_lanes(low, intr->def.index, ids, comps);
                    }
                } else if (intr->intrinsic == nir_intrinsic_vulkan_resource_index ||
                    intr->intrinsic == nir_intrinsic_vulkan_resource_reindex ||
                    intr->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
                    struct mx_descriptor_ref ref = {0};
                    if (intr->def.index >= MX_LOW_DEFS) {
                        low->fail = 1;
                    } else if (intr->intrinsic == nir_intrinsic_vulkan_resource_index) {
                        if (!nir_src_is_const(intr->src[0]))
                            low->fail = 1;
                        else
                            ref = (struct mx_descriptor_ref){true, nir_intrinsic_desc_set(intr),
                                                            nir_intrinsic_binding(intr), nir_src_as_uint(intr->src[0])};
                    } else if (!intr->src[0].ssa || intr->src[0].ssa->index >= MX_LOW_DEFS ||
                               !low->descriptors[intr->src[0].ssa->index].valid) {
                        low->fail = 1;
                    } else {
                        ref = low->descriptors[intr->src[0].ssa->index];
                        if (intr->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
                            if (!nir_src_is_const(intr->src[1]) || nir_src_as_uint(intr->src[1]) > UINT32_MAX - ref.element)
                                low->fail = 1;
                            else
                                ref.element += nir_src_as_uint(intr->src[1]);
                        }
                    }
                    if (!low->fail)
                        low->descriptors[intr->def.index] = ref;
                } else if (intr->intrinsic == nir_intrinsic_load_deref) {
                    nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
                    nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
                    uint32_t rec[5];
                    unsigned id;
                    if (!var && deref && (deref->modes & nir_var_mem_ubo)) {
                        low->fail = low_ubo_load(low, intr, deref) != 0;
                    } else if (!var) {
                        low->fail = 1;
                    } else if (!fragment && var->data.mode == nir_var_shader_in) {
                        unsigned comps = intr->def.num_components;
                        unsigned base = 0;
                        unsigned location;
                        if (vertex_slot(deref, &location)) {
                            low->fail = 1;
                            continue;
                        }
                        unsigned loaded = low_vertex_load(low, location);
                        unsigned ids[4], lane;
                        uint32_t construct[8];
                        if (!comps || comps > 4 || base + comps > 4) {
                            low->fail = 1;
                            continue;
                        }
                        for (lane = 0; lane < comps; lane++) {
                            uint32_t extract[5] = {(5u << 16) | MXSB_OP_EXTRACT,
                                                  low_id(low), MXSB_TYPE_F32,
                                                  loaded, base + lane};
                            low_emit(low, extract, 5);
                            ids[lane] = extract[1];
                        }
                        id = ids[0];
                        if (comps > 1) {
                            id = low_id(low);
                            construct[0] = ((4u + comps) << 16) | MXSB_OP_CONSTRUCT;
                            construct[1] = id;
                            construct[2] = low_vec_type(comps);
                            construct[3] = comps;
                            for (lane = 0; lane < comps; lane++)
                                construct[4 + lane] = ids[lane];
                            low_emit(low, construct, 4 + comps);
                        }
                        low_remember(low, intr->def.index, id, comps);
                        low_keep_comps(low, intr->def.index, ids, comps);
                    } else if (var->data.mode == nir_var_uniform || var->data.mode == nir_var_mem_ubo) {
                        unsigned slot = 0;
                        unsigned comps = intr->def.num_components;
                        if (comps == 0 || comps > 4 || uniform_slot(deref, &slot) != 0)
                            low->fail = 1;
                        else {
                            unsigned vec = low_uniform_vec4(low, slot);
                            unsigned ids[4];
                            if (comps == 4) {
                                low_remember(low, intr->def.index, vec, comps);
                            } else {
                                for (unsigned lane = 0; lane < comps; lane++) {
                                    uint32_t extract[5] = {(5u << 16) | MXSB_OP_EXTRACT,
                                                          low_id(low), MXSB_TYPE_F32,
                                                          vec, lane};
                                    low_emit(low, extract, 5);
                                    ids[lane] = extract[1];
                                }
                                unsigned result = ids[0];
                                if (comps > 1) {
                                    uint32_t construct[8];
                                    result = low_id(low);
                                    construct[0] = ((4u + comps) << 16) | MXSB_OP_CONSTRUCT;
                                    construct[1] = result;
                                    construct[2] = low_vec_type(comps);
                                    construct[3] = comps;
                                    for (unsigned lane = 0; lane < comps; lane++)
                                        construct[4 + lane] = ids[lane];
                                    low_emit(low, construct, 4 + comps);
                                }
                                low_remember(low, intr->def.index, result, comps);
                                low_keep_comps(low, intr->def.index, ids, comps);
                            }
                        }
                    } else if (fragment && var->data.mode == nir_var_shader_in && var->data.location >= VARYING_SLOT_VAR0) {
                        unsigned comps = intr->def.num_components;
                        id = low_id(low);
                        rec[0] = (5u << 16) | MXSB_OP_STAGE_INPUT;
                        rec[1] = id;
                        rec[2] = comps == 1 ? MXSB_TYPE_F32 : low_vec_type(comps);
                        rec[3] = (uint32_t)(var->data.location - VARYING_SLOT_VAR0);
                        rec[4] = MXSB_INTERP_PERSPECTIVE;
                        low_emit(low, rec, 5);
                        if (comps == 0 || comps > 4)
                            low->fail = 1;
                        else
                            low_remember(low, intr->def.index, id, comps);
                    } else {
                        low->fail = 1;
                    }
                } else if (intr->intrinsic == nir_intrinsic_load_uniform) {
                    int base = nir_intrinsic_base(intr);
                    unsigned comps = intr->def.num_components;
                    unsigned bits = intr->def.bit_size;
                    if (base < 0 || comps == 0 || comps > 4 || (bits != 32 && bits != 1))
                        low->fail = 1;
                    else {
                        if (!nir_src_is_const(intr->src[0])) {
                            uint64_t end = (uint64_t)(unsigned)base + nir_intrinsic_range(intr);
                            if (end > UINT32_MAX)
                                low->fail = 1;
                            else if (low->uniform_count < end)
                                low->uniform_count = (unsigned)end;
                        }
                        unsigned index = low_uniform_index(low, base, &intr->src[0]);
                        unsigned vec;
                        unsigned ids[4];
                        unsigned lane;
                        if (!nir_src_is_const(intr->src[0])) {
                            static unsigned indirect_notes;
                            nir_def *def = intr->src[0].ssa;
                            if (indirect_notes < 4 && def && nir_def_instr(def)) {
                                nir_instr *parent = nir_def_instr(def);
                                indirect_notes++;
                                if (parent->type == nir_instr_type_alu) {
                                    nir_alu_instr *src_alu = nir_instr_as_alu(parent);
                                    fprintf(stderr, "mxgpu uniform index alu %s bits %u base %d\n",
                                            nir_op_infos[src_alu->op].name, def->bit_size, base);
                                } else if (parent->type == nir_instr_type_intrinsic) {
                                    nir_intrinsic_instr *src_intr = nir_instr_as_intrinsic(parent);
                                    fprintf(stderr, "mxgpu uniform index intrinsic %s bits %u base %d\n",
                                            nir_intrinsic_infos[src_intr->intrinsic].name, def->bit_size, base);
                                } else {
                                    fprintf(stderr, "mxgpu uniform index instr %u bits %u base %d\n",
                                            (unsigned)parent->type, def->bit_size, base);
                                }
                            }
                        }
                        if (!index)
                            low->fail = 1;
                        else {
                            vec = low_uniform_vec4_id(low, index);
                            if (bits == 32 && comps == 4) {
                                low_remember(low, intr->def.index, vec, 4);
                            } else {
                                unsigned zero = bits == 1 ? low_f32(low, 0.f) : 0;
                                for (lane = 0; lane < comps; lane++) {
                                    uint32_t rec[5];
                                    unsigned extracted = low_id(low);
                                    rec[0] = (5u << 16) | MXSB_OP_EXTRACT;
                                    rec[1] = extracted;
                                    rec[2] = MXSB_TYPE_F32;
                                    rec[3] = vec;
                                    rec[4] = lane;
                                    low_emit(low, rec, 5);
                                    if (bits == 1)
                                        ids[lane] = low_unop_ty(low, MXSB_OP_NOT, MXSB_TYPE_BOOL, low_cmp(low, MXSB_OP_EQ, extracted, zero));
                                    else
                                        ids[lane] = extracted;
                                }
                                if (bits == 1 || comps == 1)
                                    low_finish_lanes(low, intr->def.index, ids, comps);
                                else {
                                    uint32_t built[8];
                                    unsigned n;
                                    built[0] = ((4u + comps) << 16) | MXSB_OP_CONSTRUCT;
                                    built[1] = low_id(low);
                                    built[2] = low_vec_type(comps);
                                    built[3] = comps;
                                    for (n = 0; n < comps; n++)
                                        built[4 + n] = ids[n];
                                    low_emit(low, built, 4 + comps);
                                    low_remember(low, intr->def.index, built[1], comps);
                                    low_keep_comps(low, intr->def.index, ids, comps);
                                }
                            }
                        }
                    }
                } else if (intr->intrinsic == nir_intrinsic_store_deref) {
                    nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
                    nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
                    nir_def *value = intr->src[1].ssa;
                    uint32_t rec[3];
                    unsigned loc;
                    if (!var || !value || value->index >= MX_LOW_DEFS || !low->id_of[value->index]) {
                        low->fail = 1;
                    } else if (var->data.location == VARYING_SLOT_PSIZ) {
                        (void)value;
                    } else if (var->data.location == VARYING_SLOT_POS || var->data.location == FRAG_RESULT_DATA0 || var->data.location == FRAG_RESULT_COLOR) {
                        unsigned component = var->data.location_frac;
                        if (deref->deref_type == nir_deref_type_array &&
                            glsl_type_is_vector(nir_deref_instr_parent(deref)->type)) {
                            if (!nir_src_is_const(deref->arr.index)) {
                                low->fail = 1;
                                continue;
                            }
                            component += nir_src_as_uint(deref->arr.index);
                        }
                        unsigned mask = nir_intrinsic_write_mask(intr);
                        if (component + value->num_components > 4) {
                            low->fail = 1;
                            continue;
                        }
                        for (unsigned lane = 0; lane < value->num_components; lane++) {
                            if (mask & (1u << lane))
                                ret_lanes[component + lane] = low_component(low, value, lane);
                        }
                        partial_return = true;
                    } else if (var->data.location >= VARYING_SLOT_VAR0) {
                        loc = (unsigned)(var->data.location - VARYING_SLOT_VAR0);
                        rec[0] = (3u << 16) | MXSB_OP_STAGE_OUTPUT;
                        rec[1] = loc;
                        rec[2] = low->id_of[value->index];
                        low_emit(low, rec, 3);
                    } else {
                        low->fail = 1;
                    }
                } else {
                    low->fail = 1;
                }
            } else if (instr->type == nir_instr_type_tex) {
                nir_tex_instr *tex = nir_instr_as_tex(instr);
                int texture_src = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
                unsigned binding = tex->texture_index, set = 0, element = 0;
                if (texture_src >= 0) {
                    nir_deref_instr *deref = nir_src_as_deref(tex->src[texture_src].src);
                    nir_deref_instr *root = deref;
                    nir_variable *var;
                    while (root && root->deref_type != nir_deref_type_var)
                        root = nir_deref_instr_parent(root);
                    var = root ? root->var : NULL;
                    if (!var) {
                        low->fail = 1;
                        continue;
                    }
                    binding = var->data.binding;
                    set = var->data.descriptor_set;
                    while (deref && deref->deref_type != nir_deref_type_var) {
                        nir_deref_instr *parent = nir_deref_instr_parent(deref);
                        unsigned stride, index;
                        uint64_t offset;
                        if (deref->deref_type == nir_deref_type_cast) {
                            deref = parent;
                            continue;
                        }
                        if (deref->deref_type != nir_deref_type_array || !parent ||
                            !nir_src_is_const(deref->arr.index)) {
                            low->fail = 1;
                            break;
                        }
                        index = nir_src_as_uint(deref->arr.index);
                        if (!glsl_type_is_array(parent->type) || index >= glsl_get_length(parent->type)) {
                            low->fail = 1;
                            break;
                        }
                        stride = glsl_type_is_array(deref->type) ? glsl_get_aoa_size(deref->type) : 1;
                        offset = (uint64_t)index * stride + element;
                        if (offset > UINT32_MAX) {
                            low->fail = 1;
                            break;
                        }
                        element = (unsigned)offset;
                        deref = parent;
                    }
                    if (low->fail)
                        continue;
                }
                if (!fragment || (tex->sampler_dim != GLSL_SAMPLER_DIM_2D && tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE) || tex->is_array ||
                    tex->def.bit_size != 32 || (tex->def.num_components != 4u && !(tex->is_shadow && tex->def.num_components == 1u)) ||
                    (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl && tex->op != nir_texop_txd)) {
                    low->fail = 1;
                    continue;
                }
                unsigned texture_index;
                for (texture_index = 0; texture_index < low->texture_count; texture_index++) {
                    struct mxgpu_texture_binding *t = &low->textures[texture_index];
                    if (t->binding == binding && t->set == set && t->element == element)
                        break;
                }
                if (texture_index == low->texture_count) {
                    if (texture_index >= MXGPU_SHADER_TEXTURES) {
                        low->fail = 1;
                        continue;
                    }
                    low->textures[texture_index] = (struct mxgpu_texture_binding){
                        set, binding, element,
                        texture_index ? 4u + 2u * texture_index : 2u,
                        texture_index ? 3u + 2u * texture_index : 1u,
                        5u + 2u * texture_index, 4u + 2u * texture_index,
                        tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? MXSB_BINDING_TEXTURE_CUBE : MXSB_BINDING_TEXTURE_2D, tex->is_shadow
                    };
                    low->texture_count++;
                }
                if (!low->texture_live) {
                    low->texture_binding = binding;
                    low->texture_set = set;
                    low->texture_element = element;
                    low->texture_live = true;
                }
                int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
                nir_def *src = coord >= 0 ? tex->src[coord].src.ssa : NULL;
                uint32_t rec[MXGPU_SHADER_INSTRUCTION_WORDS] = {0};
                unsigned id;
                if (!src || src->num_components != (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3u : 2u) || src->index >= MX_LOW_DEFS || !low->id_of[src->index]) {
                    low->fail = 1;
                } else {
                    id = low_id(low);
                    rec[0] = (6u << 16) | MXSB_OP_TEXTURE_SAMPLE;
                    rec[1] = id;
                    rec[2] = MXSB_TYPE_F32X4;
                    rec[3] = low->textures[texture_index].texture_id;
                    rec[4] = low->id_of[src->index];
                    if (tex->is_shadow) {
                        int comparator = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
                        if (comparator < 0 || !tex->src[comparator].src.ssa || tex->src[comparator].src.ssa->num_components != 1) {
                            low->fail = 1;
                            continue;
                        }
                        unsigned lanes = src->num_components + 1;
                        uint32_t coord_words[8] = {((4u + lanes) << 16) | MXSB_OP_CONSTRUCT, low_id(low), low_vec_type(lanes), lanes};
                        for (unsigned lane = 0; lane < lanes - 1; lane++)
                            coord_words[4 + lane] = low_cast(low, low_component(low, src, lane), MXSB_TYPE_F32);
                        coord_words[3 + lanes] = low_cast(low, low_component(low, tex->src[comparator].src.ssa, 0), MXSB_TYPE_F32);
                        low_emit(low, coord_words, 4 + lanes);
                        rec[4] = coord_words[1];
                    }
                    rec[5] = 0x1111u;
                    unsigned words = 6;
                    int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
                    int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
                    if ((tex->op == nir_texop_txb) != (bias >= 0)) {
                        low->fail = 1;
                        continue;
                    }
                    if (tex->op == nir_texop_txb) {
                        if (!tex->src[bias].src.ssa || tex->src[bias].src.ssa->num_components != 1) {
                            low->fail = 1;
                            continue;
                        }
                        rec[0] = (14u << 16) | MXSB_OP_TEXTURE_SAMPLE_BOUND_OPERANDS;
                        rec[5] = rec[4];
                        rec[4] = low->textures[texture_index].sampler_id;
                        rec[6] = 1;
                        rec[7] = low_cast(low, low_component(low, tex->src[bias].src.ssa, 0), MXSB_TYPE_F32);
                        words = 14;
                    } else if (tex->op == nir_texop_txl) {
                        int level = nir_tex_instr_src_index(tex, nir_tex_src_lod);
                        if (level < 0 || !tex->src[level].src.ssa || tex->src[level].src.ssa->num_components != 1) {
                            low->fail = 1;
                            continue;
                        }
                        rec[0] = (7u << 16) | MXSB_OP_TEXTURE_SAMPLE_LOD;
                        rec[6] = low_cast(low, low_component(low, tex->src[level].src.ssa, 0), MXSB_TYPE_F32);
                        words = 7;
                    } else if (tex->op == nir_texop_txd) {
                        int dx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
                        int dy = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
                        if (dx < 0 || dy < 0 || !tex->src[dx].src.ssa || !tex->src[dy].src.ssa ||
                            tex->src[dx].src.ssa->num_components != src->num_components || tex->src[dy].src.ssa->num_components != src->num_components ||
                            tex->src[dx].src.ssa->index >= MX_LOW_DEFS || tex->src[dy].src.ssa->index >= MX_LOW_DEFS) {
                            low->fail = 1;
                            continue;
                        }
                        rec[0] = (8u << 16) | MXSB_OP_TEXTURE_SAMPLE_GRAD;
                        rec[6] = low->id_of[tex->src[dx].src.ssa->index];
                        rec[7] = low->id_of[tex->src[dy].src.ssa->index];
                        words = 8;
                        if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
                            rec[0] = (14u << 16) | MXSB_OP_TEXTURE_SAMPLE_BOUND_OPERANDS;
                            rec[8] = rec[7];
                            rec[7] = rec[6];
                            rec[6] = 3;
                            rec[5] = rec[4];
                            rec[4] = low->textures[texture_index].sampler_id;
                            words = 14;
                        }
                    }
                    if (offset >= 0) {
                        nir_src displacement = tex->src[offset].src;
                        if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D || !displacement.ssa ||
                            displacement.ssa->bit_size != 32 || displacement.ssa->num_components != 2 ||
                            !nir_src_is_const(displacement)) {
                            low->fail = 1;
                            continue;
                        }
                        const nir_const_value *components = nir_src_as_const_value(displacement);
                        if (components[0].i32 < -8 || components[0].i32 > 7 ||
                            components[1].i32 < -8 || components[1].i32 > 7) {
                            low->fail = 1;
                            continue;
                        }
                        if (words != 14) {
                            uint32_t operand0 = rec[6], operand1 = rec[7];
                            rec[0] = (14u << 16) | MXSB_OP_TEXTURE_SAMPLE_BOUND_OPERANDS;
                            rec[5] = rec[4];
                            rec[4] = low->textures[texture_index].sampler_id;
                            rec[6] = tex->op == nir_texop_txl ? 2 : tex->op == nir_texop_txd ? 3 : 0;
                            rec[7] = operand0;
                            rec[8] = operand1;
                            words = 14;
                        }
                        rec[10] = (uint32_t)components[0].i32;
                        rec[11] = (uint32_t)components[1].i32;
                    }
                    low_emit(low, rec, words);
                    if (tex->is_shadow && tex->def.num_components == 1) {
                        uint32_t extract[5] = {(5u << 16) | MXSB_OP_EXTRACT, low_id(low), MXSB_TYPE_F32, id, 0};
                        low_emit(low, extract, 5);
                        low_remember(low, tex->def.index, extract[1], 1);
                    } else {
                        low_remember(low, tex->def.index, id, 4);
                    }
                }
            } else if (instr->type == nir_instr_type_undef) {
                nir_undef_instr *und = nir_instr_as_undef(instr);
                unsigned n = und->def.num_components;
                unsigned ids[4];
                unsigned i;
                uint32_t rec[8];
                if (n == 0 || n > 4) {
                    low->fail = 1;
                } else {
                    for (i = 0; i < n; i++)
                        ids[i] = low_f32(low, 0.f);
                    if (n == 1) {
                        low_remember(low, und->def.index, ids[0], 1);
                    } else {
                        rec[0] = ((4u + n) << 16) | MXSB_OP_CONSTRUCT;
                        rec[1] = low_id(low);
                        rec[2] = low_vec_type(n);
                        rec[3] = n;
                        for (i = 0; i < n; i++)
                            rec[4 + i] = ids[i];
                        low_emit(low, rec, 4 + n);
                        low_remember(low, und->def.index, rec[1], n);
                        low_keep_comps(low, und->def.index, ids, n);
                    }
                }
            } else if (instr->type == nir_instr_type_phi) {
                nir_phi_instr *phi = nir_instr_as_phi(instr);
                nir_def *defined = NULL;
                int distinct = 0;
                nir_foreach_phi_src(src, phi) {
                    nir_def *def = src->src.ssa;
                    if (!def || def->index >= MX_LOW_DEFS || !low->id_of[def->index])
                        continue;
                    if (!defined || defined == def) {
                        defined = def;
                        if (distinct == 0)
                            distinct = 1;
                    } else {
                        distinct = 2;
                    }
                }
                if (distinct == 1) {
                    unsigned comps = low->comps[defined->index] ? low->comps[defined->index] : defined->num_components;
                    low_remember(low, phi->def.index, low->id_of[defined->index], comps);
                    if (comps > 1)
                        low_keep_comps(low, phi->def.index, low->comp_id[defined->index], comps);
                }
                else if (!low_phi_merge(low, phi, block))
                    low->fail = 1;
            } else if (instr->type == nir_instr_type_jump) {
                nir_jump_instr *jump = nir_instr_as_jump(instr);
                if (compute && block->cf_node.parent != &impl->cf_node) {
                    low->fail = 1;
                } else if (jump->type == nir_jump_return || jump->type == nir_jump_halt || jump->type == nir_jump_abort) {
                    stop = 1;
                    leave_block = 1;
                } else if (jump->type == nir_jump_break || jump->type == nir_jump_continue) {
                    nir_cf_node *parent = block->cf_node.parent;
                    nir_cf_node *next;
                    while (parent && parent->type != nir_cf_node_loop)
                        parent = parent->parent;
                    next = parent ? nir_cf_node_next(parent) : NULL;
                    if (!next || next->type != nir_cf_node_block)
                        low->fail = 1;
                    else {
                        resume = nir_cf_node_as_block(next);
                        leave_block = 1;
                    }
                } else {
                    low->fail = 1;
                }
            } else {
                low->fail = 1;
            }
            if (low->fail) {
                static unsigned notes;
                if (notes < 16) {
                    notes++;
                    if (instr->type == nir_instr_type_intrinsic) {
                        nir_intrinsic_instr *failed = nir_instr_as_intrinsic(instr);
                        nir_deref_instr *deref = failed->intrinsic == nir_intrinsic_load_deref || failed->intrinsic == nir_intrinsic_store_deref
                            ? nir_src_as_deref(failed->src[0]) : NULL;
                        nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
                        if (failed->intrinsic == nir_intrinsic_load_uniform) {
                            unsigned off = nir_src_is_const(failed->src[0]) ? nir_src_as_uint(failed->src[0]) : 0xffffffffu;
                            fprintf(stderr, "mxgpu lower fail frag %d count %u load_uniform base %d bits %u off %u comps %u\n",
                                    fragment, low->count, nir_intrinsic_base(failed),
                                    nir_intrinsic_infos[failed->intrinsic].has_dest ? failed->def.bit_size : 0,
                                    off, nir_intrinsic_infos[failed->intrinsic].has_dest ? failed->def.num_components : 0);
                        } else if (failed->intrinsic == nir_intrinsic_store_deref) {
                            nir_def *value = failed->src[1].ssa;
                            fprintf(stderr, "mxgpu output fail frag %d count %u location %d SSA %u comps %u mapped %u write_mask %u\n",
                                    fragment, low->count, var ? var->data.location : -1,
                                    value ? value->index : UINT32_MAX, value ? value->num_components : 0,
                                    value && value->index < MX_LOW_DEFS ? low->id_of[value->index] : 0,
                                    nir_intrinsic_write_mask(failed));
                        } else {
                            fprintf(stderr, "mxgpu lower fail frag %d count %u intrinsic %s mode %d location %d comps %u\n",
                                    fragment, low->count, nir_intrinsic_infos[failed->intrinsic].name,
                                    var ? (int)var->data.mode : -1, var ? var->data.location : -1,
                                    nir_intrinsic_infos[failed->intrinsic].has_dest ? failed->def.num_components : 0);
                        }
                    } else if (instr->type == nir_instr_type_alu) {
                        nir_alu_instr *alu = nir_instr_as_alu(instr);
                        fprintf(stderr, "mxgpu lower fail frag %d count %u alu %s bits %u comps %u\n",
                                fragment, low->count, nir_op_infos[alu->op].name, alu->def.bit_size, alu->def.num_components);
                    } else if (instr->type == nir_instr_type_tex) {
                        nir_tex_instr *tex = nir_instr_as_tex(instr);
                        fprintf(stderr, "mxgpu lower fail frag %d count %u tex op %u\n",
                                fragment, low->count, (unsigned)tex->op);
                    } else if (instr->type == nir_instr_type_phi) {
                        nir_phi_instr *phi = nir_instr_as_phi(instr);
                        unsigned nsrc = 0;
                        unsigned defined = 0;
                        nir_foreach_phi_src(src, phi) {
                            nir_def *def = src->src.ssa;
                            nsrc++;
                            if (def && def->index < MX_LOW_DEFS && low->id_of[def->index])
                                defined++;
                        }
                        fprintf(stderr, "mxgpu lower fail frag %d count %u phi sources %u defined %u comps %u\n",
                                fragment, low->count, nsrc, defined, phi->def.num_components);
                    } else if (instr->type == nir_instr_type_jump) {
                        nir_jump_instr *jump = nir_instr_as_jump(instr);
                        fprintf(stderr, "mxgpu lower fail frag %d count %u jump %u\n",
                                fragment, low->count, (unsigned)jump->type);
                    } else {
                        fprintf(stderr, "mxgpu lower fail frag %d count %u instr type %u\n",
                                fragment, low->count, (unsigned)instr->type);
                    }
                }
                if (getenv("MXGPU_DUMP_NIR")) {
                    fprintf(stderr, "mxgpu failing NIR fragment=%d\n", fragment);
                    nir_print_shader((nir_shader *)nir, stderr);
                }
                return -1;
            }
            if (leave_block)
                break;
        }
    }
    if (partial_return) {
        uint32_t rec[8] = {(8u << 16) | MXSB_OP_CONSTRUCT, low_id(low), MXSB_TYPE_F32X4, 4};
        for (unsigned lane = 0; lane < 4; lane++)
            rec[4 + lane] = ret_lanes[lane] ? low_cast(low, ret_lanes[lane], MXSB_TYPE_F32) : low_f32(low, 0.f);
        low_emit(low, rec, 8);
        ret_id = rec[1];
    }
    if (ret_id) {
        uint32_t rec[2];
        rec[0] = (2u << 16) | MXSB_OP_RETURN_VALUE;
        rec[1] = ret_id;
        low_emit(low, rec, 2);
    }
    if (compute) {
        uint32_t rec = (1u << 16) | MXSB_OP_RETURN_VOID;
        if (partial_return)
            return -1;
        low_emit(low, &rec, 1);
    }
    return low->fail || low->count == 0 ? -1 : 0;
}

int mxgpu_link_shaders_draw_samplers(const struct mxgpu_shader *vs, const struct mxgpu_shader *fs, unsigned vertex_count, bool bound_sampler, uint8_t *out, uint32_t cap, uint32_t *out_len, const uint32_t *sampler_compare)
{
    struct mxsb_writer writer;
    uint32_t words[MXGPU_LINK_MODULE_CAPACITY / 4u];
    const uint32_t *vptr[MX_LOW_INSNS];
    const uint32_t *fptr[MX_LOW_INSNS];
    uint32_t sampler_words[MX_LOW_INSNS][MXGPU_SHADER_INSTRUCTION_WORDS];
    const uint32_t no_color_words[3][8] = {
        {(4u << 16) | MXSB_OP_CONSTANT, 1, MXSB_TYPE_F32, 0},
        {(8u << 16) | MXSB_OP_CONSTRUCT, 2, MXSB_TYPE_F32X4, 4, 1, 1, 1, 1},
        {(2u << 16) | MXSB_OP_RETURN_VALUE, 2}
    };
    const uint32_t no_color_lens[3] = {4, 8, 2};
    unsigned i;
    unsigned attributes = vs ? vs->vertex_attribute_count : 0;
    unsigned vertex_capacity = vertex_count;
    if (!vertex_count || attributes > MXGPU_SHADER_VERTEX_SLOTS)
        return -1;
    if (vertex_count <= 65536u) {
        vertex_capacity = 1;
        while (vertex_capacity < vertex_count)
            vertex_capacity *= 2u;
    }
    if (vertex_capacity > UINT32_MAX / (attributes ? attributes : 1u))
        return -1;
    if (!vs || vs->count == 0 || vs->count > MX_LOW_INSNS ||
        (fs && (fs->count == 0 || fs->count > MX_LOW_INSNS)))
        return -1;
    if (mxsb_writer_init(&writer, words, MXGPU_LINK_MODULE_CAPACITY / 4u, MXSB_VERSION_MINOR) != MXSB_OK)
        return -1;
    if (mxsb_writer_binding(&writer, 1, 0, MXSB_BINDING_UNIFORM, MXSB_ACCESS_READ, MXSB_TYPE_F32X4, vertex_capacity * (attributes ? attributes : 1)) != MXSB_OK)
        return -1;
    if (mxsb_writer_binding(&writer, 2, 1, fs && fs->texture_count && fs->textures[0].binding_kind ? fs->textures[0].binding_kind : MXSB_BINDING_TEXTURE_2D, MXSB_ACCESS_READ, MXSB_TYPE_F32X4, 0) != MXSB_OK)
        return -1;
    if (vs->uses_uniforms && mxsb_writer_binding(&writer, 3, 2, MXSB_BINDING_UNIFORM,
                                                MXSB_ACCESS_READ, MXSB_TYPE_F32X4,
                                                vs->uniform_count) != MXSB_OK)
        return -1;
    if (fs && fs->uses_uniforms && mxsb_writer_binding(&writer, 4, 3, MXSB_BINDING_UNIFORM,
                                                MXSB_ACCESS_READ, MXSB_TYPE_F32X4,
                                                fs->uniform_count) != MXSB_OK)
        return -1;
    for (i = 0; fs && i < fs->texture_count; i++) {
        const struct mxgpu_texture_binding *t = &fs->textures[i];
        if (i && mxsb_writer_binding(&writer, t->texture_id, t->texture_slot, t->binding_kind ? t->binding_kind : MXSB_BINDING_TEXTURE_2D,
                                    MXSB_ACCESS_READ, MXSB_TYPE_F32X4, 0) != MXSB_OK)
            return -1;
        unsigned compare = sampler_compare ? sampler_compare[i] : 0;
        if (compare > 8 || t->shadow != (compare != 0) || (t->shadow && !bound_sampler))
            return -1;
        if (bound_sampler && mxsb_writer_binding(&writer, t->sampler_id, t->sampler_slot,
                                                MXSB_BINDING_SAMPLER, MXSB_ACCESS_READ, MXSB_TYPE_U32, compare + 1) != MXSB_OK)
            return -1;
    }
    if (mxsb_writer_entry(&writer, 1, MXSB_STAGE_VERTEX, 1) != MXSB_OK)
        return -1;
    if (mxsb_writer_entry(&writer, 2, MXSB_STAGE_FRAGMENT, 2) != MXSB_OK)
        return -1;
    for (i = 0; i < vs->count; i++)
        vptr[i] = vs->words[i];
    for (i = 0; fs && i < fs->count; i++) {
        fptr[i] = fs->words[i];
        unsigned opcode = fs->words[i][0] & 0xffffu;
        if (opcode == MXSB_OP_TEXTURE_SAMPLE_BOUND_OPERANDS && !bound_sampler)
            return -1;
        if (bound_sampler && (opcode == MXSB_OP_TEXTURE_SAMPLE || opcode == MXSB_OP_TEXTURE_SAMPLE_LOD || opcode == MXSB_OP_TEXTURE_SAMPLE_GRAD)) {
            if (fs->lens[i] != (opcode == MXSB_OP_TEXTURE_SAMPLE ? 6u : opcode == MXSB_OP_TEXTURE_SAMPLE_LOD ? 7u : 8u))
                return -1;
            memcpy(sampler_words[i], fs->words[i], sizeof sampler_words[i]);
            sampler_words[i][0] = (fs->lens[i] << 16) | (opcode == MXSB_OP_TEXTURE_SAMPLE ? MXSB_OP_TEXTURE_SAMPLE_BOUND : opcode == MXSB_OP_TEXTURE_SAMPLE_LOD ? MXSB_OP_TEXTURE_SAMPLE_BOUND_LOD : MXSB_OP_TEXTURE_SAMPLE_BOUND_GRAD);
            unsigned texture_index;
            for (texture_index = 0; texture_index < fs->texture_count; texture_index++)
                if (fs->textures[texture_index].texture_id == fs->words[i][3])
                    break;
            if (texture_index == fs->texture_count)
                return -1;
            sampler_words[i][4] = fs->textures[texture_index].sampler_id;
            sampler_words[i][5] = fs->words[i][4];
            fptr[i] = sampler_words[i];
        }
    }
    if (mxsb_writer_block(&writer, 1, 1, vptr, vs->lens, vs->count) != MXSB_OK)
        return -1;
    if (!fs)
        for (i = 0; i < 3; i++)
            fptr[i] = no_color_words[i];
    if (mxsb_writer_block(&writer, 2, 2, fptr, fs ? fs->lens : no_color_lens, fs ? fs->count : 3) != MXSB_OK)
        return -1;
    return mxsb_writer_finish(&writer, out, cap, out_len) == MXSB_OK ? 0 : -1;
}

int mxgpu_link_compute(const struct mxgpu_shader *cs, uint8_t *out, uint32_t cap, uint32_t *out_len)
{
    struct mxsb_writer writer;
    struct mxsb_limits limits;
    const uint32_t *records[MX_LOW_INSNS];
    uint32_t capacity = MXGPU_LINK_MODULE_CAPACITY / 4u;
    int result = -1;
    if (!cs || !cs->compute || !cs->count || cs->count > MX_LOW_INSNS || !out || !out_len ||
        cs->storage_count > MXGPU_SHADER_STORAGE_BUFFERS ||
        !cs->workgroup_size[0] || !cs->workgroup_size[1] || !cs->workgroup_size[2])
        return -1;
    uint32_t *words = malloc(capacity * sizeof *words);
    if (!words || mxsb_writer_init(&writer, words, capacity, MXSB_VERSION_MINOR) != MXSB_OK)
        goto done;
    for (unsigned i = 0; i < cs->storage_count; i++) {
        const struct mxgpu_storage_binding *b = &cs->storage[i];
        if (mxsb_writer_binding(&writer, b->binding_id, (uint16_t)b->slot, MXSB_BINDING_STORAGE, b->access,
                                MXSB_TYPE_U32, 1) != MXSB_OK)
            goto done;
    }
    if (cs->uses_uniforms &&
        (!cs->uniform_count || mxsb_writer_binding(&writer, MXGPU_COMPUTE_UNIFORM_BINDING, (uint16_t)cs->uniform_slot,
                                                   MXSB_BINDING_STORAGE, MXSB_ACCESS_READ, MXSB_TYPE_F32X4,
                                                   cs->uniform_count) != MXSB_OK))
        goto done;
    if (mxsb_writer_entry_workgroup(&writer, 1, MXSB_STAGE_COMPUTE, 1, cs->workgroup_size[0],
                                    cs->workgroup_size[1], cs->workgroup_size[2]) != MXSB_OK)
        goto done;
    for (unsigned i = 0; i < cs->count; i++)
        records[i] = cs->words[i];
    if (mxsb_writer_block(&writer, 1, 1, records, cs->lens, cs->count) != MXSB_OK ||
        mxsb_writer_finish(&writer, out, cap, out_len) != MXSB_OK ||
        mxsb_limits_default(&limits) != MXSB_OK || mxsb_verify(out, *out_len, &limits) != MXSB_OK)
        goto done;
    result = 0;
done:
    free(words);
    return result;
}

int mxgpu_link_shaders_draw(const struct mxgpu_shader *vs, const struct mxgpu_shader *fs, unsigned vertex_count, bool bound_sampler, uint8_t *out, uint32_t cap, uint32_t *out_len)
{
    return mxgpu_link_shaders_draw_samplers(vs, fs, vertex_count, bound_sampler, out, cap, out_len, NULL);
}

int mxgpu_link_shaders(const struct mxgpu_shader *vs, const struct mxgpu_shader *fs, uint8_t *out, uint32_t cap, uint32_t *out_len)
{
    return mxgpu_link_shaders_draw(vs, fs, 3, false, out, cap, out_len);
}

static bool shader_samples(const struct nir_shader *nir)
{
    nir_foreach_function_impl(impl, nir) {
        nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
                if (instr->type == nir_instr_type_tex)
                    return true;
            }
        }
    }
    return nir && nir->info.num_textures;
}

static void compute_lower(nir_shader *nir)
{
    nir_opt_peephole_select_options sel = {.limit = ~0u, .indirect_load_ok = true, .expensive_alu_ok = true};
    bool progress;
    unsigned rounds = 0;
    nir_lower_returns(nir);
    nir_lower_vars_to_explicit_types(nir, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
    nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_offset);
    nir_lower_system_values(nir);
    do {
        progress = false;
        progress |= nir_opt_copy_prop(nir);
        progress |= nir_opt_dce(nir);
        progress |= nir_opt_cse(nir);
        progress |= nir_opt_constant_folding(nir);
        progress |= nir_opt_remove_phis(nir);
        progress |= nir_opt_dead_cf(nir);
        progress |= nir_opt_loop_unroll(nir);
        progress |= nir_opt_peephole_select(nir, &sel);
    } while (progress && ++rounds < 32);
}

int mxgpu_compile_nir(struct nir_shader *nir, bool fragment, struct mxgpu_shader *shader)
{
    static const nir_shader_compiler_options compute_options = {.max_unroll_iterations = 64};
    if (!nir || !shader)
        return -1;
    memset(shader, 0, sizeof *shader);
    bool compute = nir->info.stage == MESA_SHADER_COMPUTE;
    if (compute && (fragment || nir->info.workgroup_size_variable))
        return -1;
    nir_shader_compiler_options default_options = {0};
    const nir_shader_compiler_options *options = nir->options;
    if (compute) {
        nir->options = &compute_options;
        compute_lower(nir);
    } else if (!nir->options) {
        nir->options = &default_options;
    }
    nir_lower_system_values(nir);
    if (!compute) {
        nir->options = options;
    }
        struct mx_low low;
        nir_opt_peephole_select_options sel;
        memset(&sel, 0, sizeof sel);
        sel.limit = ~0u;
        sel.indirect_load_ok = true;
        sel.expensive_alu_ok = true;
        sel.discard_ok = true;
        nir_opt_peephole_select(nir, &sel);
        nir_lower_continue_constructs(nir);
        if (nir->options)
            nir_opt_loop_unroll(nir);
        nir_opt_remove_phis(nir);
        nir_lower_undef_to_zero(nir, NULL);
        nir_opt_undef(nir);
        nir_opt_constant_folding(nir);
        nir_opt_dce(nir);
        int lowered = low_shader(nir, fragment, compute, &low);
        nir->options = options;
        if (lowered == 0 && low.count <= MX_LOW_INSNS) {
            unsigned i;
            shader->count = low.count;
            shader->vertex_attribute_count = low.vertex_attribute_count;
            shader->vertex_builtins = low.vertex_builtins;
            shader->vertex_builtin_slot = low.vertex_builtin_slot;
            memcpy(shader->vertex_input_locations, low.vertex_input_locations, sizeof shader->vertex_input_locations);
            shader->samples = fragment && shader_samples(nir);
            shader->texture_binding = low.texture_binding;
            shader->texture_set = low.texture_set;
            shader->texture_element = low.texture_element;
            shader->texture_count = low.texture_count;
            memcpy(shader->textures, low.textures, sizeof shader->textures);
            shader->uses_uniforms = low.uses_uniforms;
            shader->uniform_count = low.uniform_count;
            shader->uniform_buffer_count = low.uniform_buffer_count;
            memcpy(shader->uniform_buffers, low.uniform_buffers, sizeof shader->uniform_buffers);
            shader->compute = compute;
            if (compute) {
                for (i = 0; i < 3; i++)
                    shader->workgroup_size[i] = nir->info.workgroup_size[i];
                shader->workgroup_bytes = low.workgroup_bytes;
                shader->storage_count = low.storage_count;
                shader->uniform_slot = low.storage_count;
                memcpy(shader->storage, low.storage, sizeof shader->storage);
            }
            for (i = 0; i < low.count; i++) {
                memcpy(shader->words[i], low.words[i], sizeof low.words[i]);
                shader->lens[i] = low.lens[i];
            }
        } else {
            return -1;
        }
    return 0;
}
