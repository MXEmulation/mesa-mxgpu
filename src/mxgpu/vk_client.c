/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#define _POSIX_C_SOURCE 200809L
#include "scene_common.h"
#include "vk_frame.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

static int fresh_frame(PFN_vkGetInstanceProcAddr get, int constant_uv, unsigned char *pixels)
{
    const char *counter = getenv("MXGPU_SUBMIT_FILE");
    if (!counter || !counter[0]) {
        fprintf(stderr, "MXGPU_SUBMIT_FILE must name the completion counter file\n");
        return 0;
    }
    if (unlink(counter) && errno != ENOENT)
        return 0;
    if (!mx_vk_frame_with_api(get, constant_uv, pixels))
        return 0;
    FILE *input = fopen(counter, "r");
    if (!input)
        return 0;
    unsigned submits;
    int valid = fscanf(input, "%u", &submits) == 1 && submits == 1;
    fclose(input);
    return valid;
}

static int solid_red(const unsigned char *pixels)
{
    for (unsigned i = 0; i < SCENE_W * SCENE_H; i++) {
        const unsigned char *p = pixels + i * 4u;
        if (p[0] != 255 || p[1] || p[2] || p[3] != 255)
            return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "stress") != 0))
        return 2;
    const char *path = getenv("MXGPU_ICD");
    if (!path || !path[0])
        return 2;
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    PFN_vkGetInstanceProcAddr get = (PFN_vkGetInstanceProcAddr)dlsym(library, "vkGetInstanceProcAddr");
    unsigned char pixels[SCENE_W * SCENE_H * 4];
    int status = 1;
    if (!get)
        goto cleanup;
    if (argc == 1) {
        if (!fresh_frame(get, 0, pixels) || !scene_pixel_ok(pixels))
            goto cleanup;
        scene_print_pixels(pixels);
        if (!fresh_frame(get, 1, pixels) || !solid_red(pixels))
            goto cleanup;
        puts("PASS: texture coordinates and constant texture coordinates");
    } else {
        unsigned first = 0;
        for (unsigned frame = 1; frame <= 121; frame++) {
            struct timespec before, after;
            if (clock_gettime(CLOCK_MONOTONIC, &before) ||
                !fresh_frame(get, 0, pixels) || !scene_pixel_ok(pixels) ||
                clock_gettime(CLOCK_MONOTONIC, &after))
                goto cleanup;
            unsigned hash = scene_hash(pixels, sizeof(pixels));
            if (frame == 1)
                first = hash;
            double elapsed = scene_ms(&before, &after);
            printf("frame %u hash %08x completion_ms %.3f\n", frame, hash, elapsed);
            if (hash != first || (frame > 1 && elapsed >= 100.0))
                goto cleanup;
        }
        if (!mx_vk_frame_after_rejected_device(get, pixels) || !scene_pixel_ok(pixels))
            goto cleanup;
        puts("PASS: pixel readback after rejected device extension");
    }
    status = 0;
cleanup:
    dlclose(library);
    return status;
}
