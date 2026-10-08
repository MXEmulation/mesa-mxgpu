/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "scene_common.h"
#include "vk_frame.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#define FEATURE(name) {#name, offsetof(VkPhysicalDeviceFeatures, name)}

static const struct {
    const char *name;
    size_t offset;
} webgpu_features[] = {
    FEATURE(robustBufferAccess), FEATURE(fullDrawIndexUint32), FEATURE(imageCubeArray),
    FEATURE(independentBlend), FEATURE(sampleRateShading), FEATURE(depthBiasClamp),
    FEATURE(fragmentStoresAndAtomics), FEATURE(shaderUniformBufferArrayDynamicIndexing),
    FEATURE(shaderSampledImageArrayDynamicIndexing), FEATURE(shaderStorageBufferArrayDynamicIndexing),
    FEATURE(shaderStorageImageArrayDynamicIndexing), FEATURE(textureCompressionBC),
    FEATURE(textureCompressionETC2), FEATURE(textureCompressionASTC_LDR),
};

static bool has_device_extension(VkPhysicalDevice device, const char *name)
{
    VkExtensionProperties extensions[64];
    uint32_t count = 64;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, extensions) < 0)
        return false;
    for (uint32_t i = 0; i < count; i++)
        if (!strcmp(extensions[i].extensionName, name))
            return true;
    return false;
}

static bool feature_structs_disabled(const void *chain)
{
    for (const VkBaseInStructure *next = chain; next; next = next->pNext) {
        size_t size = next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES ?
                      sizeof(VkPhysicalDeviceMultiviewFeatures) :
                      next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES ?
                      sizeof(VkPhysicalDevice16BitStorageFeatures) :
                      next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES ?
                      sizeof(VkPhysicalDeviceVariablePointersFeatures) :
                      next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES ?
                      sizeof(VkPhysicalDeviceProtectedMemoryFeatures) :
                      next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES ?
                      sizeof(VkPhysicalDeviceSamplerYcbcrConversionFeatures) :
                      sizeof(VkPhysicalDeviceShaderDrawParametersFeatures);
        const VkBool32 *flags = (const VkBool32 *)((const uint8_t *)next + sizeof *next);
        for (size_t i = 0; i < (size - sizeof *next) / sizeof(VkBool32); i++)
            if (flags[i] != VK_FALSE)
                return false;
    }
    return true;
}

static int report_vulkan11(VkInstance instance, VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties v1;
    VkPhysicalDeviceFeatures v1_features;
    VkPhysicalDeviceMemoryProperties v1_memory;
    VkQueueFamilyProperties v1_family;
    VkFormatProperties v1_format;
    VkImageFormatProperties v1_image;
    uint32_t count = 1;
    bool driver_extension = has_device_extension(device, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);
    VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = driver_extension ? &driver : NULL
    };
    VkPhysicalDeviceProtectedMemoryProperties protected_memory = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES, .pNext = &subgroup
    };
    VkPhysicalDevicePointClippingProperties clipping = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES, .pNext = &protected_memory
    };
    VkPhysicalDeviceMultiviewProperties multiview = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES, .pNext = &clipping
    };
    VkPhysicalDeviceMaintenance3Properties maintenance3 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, .pNext = &multiview
    };
    VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, .pNext = &maintenance3};
    VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
    VkPhysicalDeviceShaderDrawParametersFeatures draw_parameters = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES
    };
    VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, .pNext = &draw_parameters
    };
    VkPhysicalDeviceProtectedMemoryFeatures protected_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES, .pNext = &ycbcr
    };
    VkPhysicalDeviceVariablePointersFeatures pointers = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES, .pNext = &protected_features
    };
    VkPhysicalDeviceMultiviewFeatures multiview_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, .pNext = &pointers
    };
    VkPhysicalDevice16BitStorageFeatures storage16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, .pNext = &multiview_features
    };
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &storage16};
    memset((uint8_t *)&storage16 + sizeof(VkBaseOutStructure), 0xff,
           sizeof storage16 - sizeof(VkBaseOutStructure));
    vkGetPhysicalDeviceProperties(device, &v1);
    vkGetPhysicalDeviceFeatures(device, &v1_features);
    vkGetPhysicalDeviceProperties2(device, &properties);
    vkGetPhysicalDeviceFeatures2(device, &features);
    const VkPhysicalDeviceProperties *p = &properties.properties;
    const VkPhysicalDeviceLimits *l = &p->limits;
    printf("vulkan11 device %s api %u.%u.%u driver %s conformance %u.%u.%u.%u\n", p->deviceName,
           VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion),
           driver_extension ? driver.driverInfo : "(no VK_KHR_driver_properties)",
           driver.conformanceVersion.major, driver.conformanceVersion.minor,
           driver.conformanceVersion.subminor, driver.conformanceVersion.patch);
    if (p->apiVersion < VK_API_VERSION_1_1 || memcmp(p, &v1, sizeof v1) ||
        memcmp(&features.features, &v1_features, sizeof v1_features)) {
        puts("FAIL: Properties2 or Features2 disagree with the Vulkan 1.0 queries, or apiVersion is below 1.1");
        return 1;
    }
    for (size_t i = 0; i < sizeof webgpu_features / sizeof webgpu_features[0]; i++)
        printf("vulkan11 feature %s %u\n", webgpu_features[i].name,
               *(const VkBool32 *)((const uint8_t *)&features.features + webgpu_features[i].offset));
    printf("vulkan11 limits image2D %u sets %u perStage ubo %u ssbo %u sampled %u samplers %u storageImages %u "
           "uboRange %u ssboRange %u push %u workgroup %u %u %u invocations %u shared %u colorAttachments %u "
           "sampleCounts 0x%x maxAllocation %llu perSetDescriptors %u\n",
           l->maxImageDimension2D, l->maxBoundDescriptorSets, l->maxPerStageDescriptorUniformBuffers,
           l->maxPerStageDescriptorStorageBuffers, l->maxPerStageDescriptorSampledImages,
           l->maxPerStageDescriptorSamplers, l->maxPerStageDescriptorStorageImages, l->maxUniformBufferRange,
           l->maxStorageBufferRange, l->maxPushConstantsSize, l->maxComputeWorkGroupSize[0],
           l->maxComputeWorkGroupSize[1], l->maxComputeWorkGroupSize[2], l->maxComputeWorkGroupInvocations,
           l->maxComputeSharedMemorySize, l->maxColorAttachments, l->framebufferColorSampleCounts,
           (unsigned long long)maintenance3.maxMemoryAllocationSize, maintenance3.maxPerSetDescriptors);
    printf("vulkan11 subgroup size %u stages 0x%x operations 0x%x multiview views %u protectedNoFault %u\n",
           subgroup.subgroupSize, subgroup.supportedStages, subgroup.supportedOperations,
           multiview.maxMultiviewViewCount, protected_memory.protectedNoFault);
    bool uuid_set = false;
    for (unsigned i = 0; i < VK_UUID_SIZE; i++)
        uuid_set |= id.deviceUUID[i] || id.driverUUID[i];
    if (!uuid_set || !maintenance3.maxPerSetDescriptors || !maintenance3.maxMemoryAllocationSize ||
        !subgroup.subgroupSize || !feature_structs_disabled(&storage16) ||
        (driver_extension && (!driver.driverID || !driver.driverName[0]))) {
        puts("FAIL: a Vulkan 1.1 property or feature structure was not filled");
        return 1;
    }
    vkGetPhysicalDeviceMemoryProperties(device, &v1_memory);
    VkPhysicalDeviceMemoryProperties2 memory = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    vkGetPhysicalDeviceMemoryProperties2(device, &memory);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, &v1_family);
    VkQueueFamilyProperties2 family = {.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2};
    uint32_t count2 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(device, &count2, NULL);
    if (count2 == count) {
        count2 = 1;
        vkGetPhysicalDeviceQueueFamilyProperties2(device, &count2, &family);
    }
    vkGetPhysicalDeviceFormatProperties(device, VK_FORMAT_R8G8B8A8_UNORM, &v1_format);
    VkFormatProperties2 format = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
    vkGetPhysicalDeviceFormatProperties2(device, VK_FORMAT_R8G8B8A8_UNORM, &format);
    VkResult v1_image_result = vkGetPhysicalDeviceImageFormatProperties(device, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, 0, &v1_image);
    VkImageFormatProperties2 image = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    VkResult image_result = vkGetPhysicalDeviceImageFormatProperties2(device, &(VkPhysicalDeviceImageFormatInfo2){
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .format = VK_FORMAT_R8G8B8A8_UNORM,
        .type = VK_IMAGE_TYPE_2D, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT
    }, &image);
    VkPhysicalDeviceGroupProperties groups[8];
    uint32_t group_count = 8;
    bool grouped = false;
    for (uint32_t i = 0; i < 8; i++)
        groups[i] = (VkPhysicalDeviceGroupProperties){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES};
    if (vkEnumeratePhysicalDeviceGroups(instance, &group_count, groups) >= 0)
        for (uint32_t i = 0; i < group_count; i++)
            grouped |= groups[i].physicalDeviceCount == 1 && groups[i].physicalDevices[0] == device;
    if (memcmp(&memory.memoryProperties, &v1_memory, sizeof v1_memory) || count2 != count ||
        memcmp(&family.queueFamilyProperties, &v1_family, sizeof v1_family) ||
        memcmp(&format.formatProperties, &v1_format, sizeof v1_format) || image_result != v1_image_result ||
        (image_result == VK_SUCCESS && memcmp(&image.imageFormatProperties, &v1_image, sizeof v1_image)) || !grouped) {
        puts("FAIL: a Vulkan 1.1 physical-device query disagrees with its 1.0 form or the device is not in a group");
        return 1;
    }
    printf("vulkan11 queries agree: memory, queue family, format, image format, device group\n");
    return 0;
}

int main(void)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice *devices = NULL;
    uint32_t count = 0, version = 0;
    int status = 1, reported = 0;
    if (vkEnumerateInstanceVersion(&version) != VK_SUCCESS)
        return 1;
    printf("loader instance version %u.%u.%u\n", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
           VK_API_VERSION_PATCH(version));
    if (vkCreateInstance(&(VkInstanceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1
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
        if (!strcmp(properties.deviceName, "MXGPU")) {
            if (report_vulkan11(instance, devices[i]))
                goto cleanup;
            reported = 1;
        }
    }
    if (!reported) {
        puts("FAIL: no MXGPU device through the loader");
        goto cleanup;
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
