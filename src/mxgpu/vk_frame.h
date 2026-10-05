/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_VK_FRAME_H
#define MXGPU_VK_FRAME_H

#include <vulkan/vulkan.h>

int mx_vk_frame(int constant_uv, unsigned char *pixels);
int mx_vk_frame_with_api(PFN_vkGetInstanceProcAddr get, int constant_uv,
                         unsigned char *pixels);

int mx_vk_frame_after_rejected_device(PFN_vkGetInstanceProcAddr get,
                                       unsigned char *pixels);

#endif
