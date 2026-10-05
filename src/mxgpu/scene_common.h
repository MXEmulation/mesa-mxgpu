/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_SCENE_COMMON_H
#define MXGPU_SCENE_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SCENE_W 64
#define SCENE_H 64

static const float scene_vertices[] = {
    -1.f, -1.f, 0.f, 0.f,
    1.f, -1.f, 1.f, 0.f,
    -1.f, 1.f, 0.f, 1.f,
    -1.f, 1.f, 0.f, 1.f,
    1.f, -1.f, 1.f, 0.f,
    1.f, 1.f, 1.f, 1.f
};

static const unsigned char scene_texture[16] = {
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    255, 255, 255, 255
};

static unsigned scene_hash(const unsigned char *pixels, unsigned count)
{
    unsigned hash = 2166136261u;
    unsigned i;
    for (i = 0; i < count; i++) {
        hash ^= pixels[i];
        hash *= 16777619u;
    }
    return hash;
}

static int scene_pixel_ok(const unsigned char *pixels)
{
    for (unsigned y = 0; y < SCENE_H; y++) {
        for (unsigned x = 0; x < SCENE_W; x++) {
            unsigned texel = (y >= SCENE_H / 2 ? 2 : 0) + (x >= SCENE_W / 2 ? 1 : 0);
            const unsigned char *pixel = pixels + (y * SCENE_W + x) * 4;
            if (memcmp(pixel, scene_texture + texel * 4, 4))
                return 0;
        }
    }
    return 1;
}

static void scene_print_pixels(const unsigned char *pixels)
{
    const int xs[4] = {16, 48, 16, 48};
    const int ys[4] = {16, 16, 48, 48};
    int i;
    for (i = 0; i < 4; i++) {
        const unsigned char *p = pixels + (ys[i] * SCENE_W + xs[i]) * 4;
        printf("pixel %d,%d %02x%02x%02x%02x\n", xs[i], ys[i], p[0], p[1], p[2], p[3]);
    }
    printf("hash %08x\n", scene_hash(pixels, SCENE_W * SCENE_H * 4));
}

static double scene_ms(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) * 1000.0 + (double)(b->tv_nsec - a->tv_nsec) / 1000000.0;
}

#endif
