/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "scene_common.h"
#include "vk_frame.h"

#include <stdlib.h>

int main(void)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice *devices = NULL;
    uint32_t count = 0;
    int status = 1;
    if (vkCreateInstance(&(VkInstanceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0
        }
    }, NULL, &instance) != VK_SUCCESS)
        return 1;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) != VK_SUCCESS || !count || count > 256)
        goto cleanup;
    devices = calloc(count, sizeof(*devices));
    if (!devices || vkEnumeratePhysicalDevices(instance, &count, devices) != VK_SUCCESS)
        goto cleanup;
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        printf("device %s\n", properties.deviceName);
    }
    unsigned char pixels[SCENE_W * SCENE_H * 4];
    if (!mx_vk_frame_with_api(vkGetInstanceProcAddr, 0, pixels) || !scene_pixel_ok(pixels))
        goto cleanup;
    scene_print_pixels(pixels);
    puts("PASS: system Vulkan loader texture rendering");
    status = 0;
cleanup:
    free(devices);
    vkDestroyInstance(instance, NULL);
    return status;
}
