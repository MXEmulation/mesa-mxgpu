/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_SCENE_H
#define MXGPU_SCENE_H

#include <stdint.h>

int mxgpu_scene_encode(uint8_t *out, uint32_t capacity, uint32_t *length,
                        uint32_t vertex_count);

#endif
