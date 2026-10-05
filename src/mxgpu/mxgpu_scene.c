/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_scene.h"
#include "mxsb.h"

int mxgpu_scene_encode(uint8_t *out, uint32_t capacity, uint32_t *length,
                        uint32_t vertex_count)
{
    static const uint32_t vertex[][8] = {
        {(4u << 16) | MXSB_OP_CONSTANT, 101, MXSB_TYPE_F32, 0},
        {(4u << 16) | MXSB_OP_CONSTANT, 102, MXSB_TYPE_F32, 0x3f800000u},
        {(4u << 16) | MXSB_OP_BUILTIN, 103, MXSB_TYPE_U32, MXSB_BUILTIN_VERTEX_ID},
        {(5u << 16) | MXSB_OP_BUFFER_LOAD, 104, MXSB_TYPE_F32X4, 1, 103},
        {(5u << 16) | MXSB_OP_EXTRACT, 105, MXSB_TYPE_F32, 104, 0},
        {(5u << 16) | MXSB_OP_EXTRACT, 106, MXSB_TYPE_F32, 104, 1},
        {(5u << 16) | MXSB_OP_EXTRACT, 107, MXSB_TYPE_F32, 104, 2},
        {(5u << 16) | MXSB_OP_EXTRACT, 108, MXSB_TYPE_F32, 104, 3},
        {(6u << 16) | MXSB_OP_CONSTRUCT, 109, MXSB_TYPE_F32X2, 2, 107, 108},
        {(8u << 16) | MXSB_OP_CONSTRUCT, 110, MXSB_TYPE_F32X4, 4, 105, 106, 101, 102},
        {(3u << 16) | MXSB_OP_STAGE_OUTPUT, 0, 109},
        {(2u << 16) | MXSB_OP_RETURN_VALUE, 110},
    };
    static const uint32_t fragment[][6] = {
        {(5u << 16) | MXSB_OP_STAGE_INPUT, 201, MXSB_TYPE_F32X2, 0, MXSB_INTERP_PERSPECTIVE},
        {(6u << 16) | MXSB_OP_TEXTURE_SAMPLE, 202, MXSB_TYPE_F32X4, 2, 201, 0x1111u},
        {(2u << 16) | MXSB_OP_RETURN_VALUE, 202},
    };
    uint32_t words[160], record_lengths[12];
    const uint32_t *records[12];
    struct mxsb_writer writer;
    int status;
    if (!length)
        return MXSB_ERR_LENGTH;
    *length = 0;
    if (!out || !vertex_count || vertex_count > 65536u)
        return MXSB_ERR_VALUE;
#define ENCODE(call) do { status = (call); if (status != MXSB_OK) return status; } while (0)
    ENCODE(mxsb_writer_init(&writer, words, 160, MXSB_VERSION_MINOR));
    ENCODE(mxsb_writer_binding(&writer, 1, 0, MXSB_BINDING_UNIFORM,
                               MXSB_ACCESS_READ, MXSB_TYPE_F32X4, vertex_count));
    ENCODE(mxsb_writer_binding(&writer, 2, 1, MXSB_BINDING_TEXTURE_2D,
                               MXSB_ACCESS_READ, MXSB_TYPE_F32X4, 0));
    ENCODE(mxsb_writer_entry(&writer, 1, MXSB_STAGE_VERTEX, 1));
    ENCODE(mxsb_writer_entry(&writer, 2, MXSB_STAGE_FRAGMENT, 2));
    for (unsigned i = 0; i < 12; i++) {
        records[i] = vertex[i];
        record_lengths[i] = vertex[i][0] >> 16;
    }
    ENCODE(mxsb_writer_block(&writer, 1, 1, records, record_lengths, 12));
    for (unsigned i = 0; i < 3; i++) {
        records[i] = fragment[i];
        record_lengths[i] = fragment[i][0] >> 16;
    }
    ENCODE(mxsb_writer_block(&writer, 2, 2, records, record_lengths, 3));
    ENCODE(mxsb_writer_finish(&writer, out, capacity, length));
#undef ENCODE
    return MXSB_OK;
}
