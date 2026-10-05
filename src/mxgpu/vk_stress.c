/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#define _POSIX_C_SOURCE 200809L
#include "mxgpu_driver.h"
#include "scene_common.h"
#include "vk_frame.h"

int main(void)
{
    unsigned char pixels[SCENE_W * SCENE_H * 4];
    unsigned first = 0;
    int frame;
    for (frame = 1; frame <= 121; frame++) {
        struct timespec a, b;
        unsigned hash;
        if (clock_gettime(CLOCK_MONOTONIC, &a))
            return 1;
        if (!mx_vk_frame(0, pixels))
            return 1;
        if (clock_gettime(CLOCK_MONOTONIC, &b))
            return 1;
        hash = scene_hash(pixels, sizeof pixels);
        if (frame == 1)
            first = hash;
        printf("frame %d hash %08x completion_ms %.3f submits %u\n", frame, hash, scene_ms(&a, &b), mxgpu_last_submits());
        if (!scene_pixel_ok(pixels) || hash != first || scene_ms(&a, &b) >= 100.0 || mxgpu_last_submits() != 1)
            return 1;
    }
    if (mxgpu_execute_scene(scene_vertices, 6, scene_texture, 2, 2,
                            pixels, SCENE_W, SCENE_H) != 0 || !scene_pixel_ok(pixels))
        return 1;
    if (mxgpu_debug_illegal_then_legal() != 0)
        return 1;
    if (!mx_vk_frame(0, pixels) || !scene_pixel_ok(pixels))
        return 1;
    printf("legal_after_illegal ok\n");
    mxgpu_device_close();
    return 0;
}
