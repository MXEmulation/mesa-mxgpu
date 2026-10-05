/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_driver.h"
#include "scene_common.h"
#include "vk_frame.h"

int main(void)
{
    unsigned char pixels[SCENE_W * SCENE_H * 4];
    printf("device MXGPU %s\n", MXGPU_DRIVER_VERSION);
    if (!mx_vk_frame(0, pixels))
        return 1;
    scene_print_pixels(pixels);
    printf("submits %u\n", mxgpu_last_submits());
    if (!scene_pixel_ok(pixels) || mxgpu_last_submits() != 1)
        return 1;
    if (!mx_vk_frame(1, pixels))
        return 1;
    for (unsigned i = 0; i < SCENE_W * SCENE_H; i++) {
        if (memcmp(pixels + i * 4, scene_texture, 4))
            return 1;
    }
    mxgpu_device_close();
    return 0;
}
