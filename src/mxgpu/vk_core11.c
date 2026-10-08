/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "core11-copy-comp.h"
#include "core11-alias-comp.h"
#include "core11-uniform-comp.h"
#include "core11-vert.h"
#include "core11-frag.h"

#define WORDS 64u
#define SENTINEL 0xb0000000u
#define IMAGE_SIZE 8u

struct context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t memory_type;
    VkCommandPool command_pool;
    VkDescriptorPool descriptor_pool;
    VkFence fence;
    bool robust;
};

struct memory {
    VkDeviceMemory handle;
    uint32_t *words;
};

struct compute {
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
};

static char g_failure[256];

static int fail(const char *message)
{
    snprintf(g_failure, sizeof g_failure, "%s", message);
    return -1;
}

static int fail_result(const char *what, VkResult result)
{
    snprintf(g_failure, sizeof g_failure, "%s returned %d", what, (int)result);
    return -1;
}

static int create_memory(struct context *ctx, VkDeviceSize bytes, struct memory *memory)
{
    void *mapped = NULL;
    if (vkAllocateMemory(ctx->device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = bytes,
            .memoryTypeIndex = ctx->memory_type
        }, NULL, &memory->handle) != VK_SUCCESS ||
        vkMapMemory(ctx->device, memory->handle, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
        return fail("memory allocation or mapping failed");
    if ((uintptr_t)mapped % 64u)
        return fail("mapped pointer is not aligned to minMemoryMapAlignment");
    memory->words = mapped;
    memset(mapped, 0, (size_t)bytes);
    return 0;
}

static int create_buffer(struct context *ctx, const struct memory *memory, VkDeviceSize offset, VkDeviceSize bytes,
                         VkBufferUsageFlags usage, VkBuffer *buffer)
{
    VkMemoryDedicatedRequirements dedicated = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, .requiresDedicatedAllocation = VK_TRUE
    };
    VkMemoryRequirements2 requirements = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    if (vkCreateBuffer(ctx->device, &(VkBufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = usage
        }, NULL, buffer) != VK_SUCCESS)
        return fail("vkCreateBuffer failed");
    vkGetBufferMemoryRequirements2(ctx->device, &(VkBufferMemoryRequirementsInfo2){
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2, .buffer = *buffer
    }, &requirements);
    if (requirements.memoryRequirements.size < bytes || offset % requirements.memoryRequirements.alignment ||
        !(requirements.memoryRequirements.memoryTypeBits & (1u << ctx->memory_type)) ||
        dedicated.requiresDedicatedAllocation || dedicated.prefersDedicatedAllocation)
        return fail("vkGetBufferMemoryRequirements2 reported unexpected requirements");
    VkResult result = vkBindBufferMemory2(ctx->device, 1, &(VkBindBufferMemoryInfo){
        .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO, .buffer = *buffer, .memory = memory->handle,
        .memoryOffset = offset
    });
    return result == VK_SUCCESS ? 0 : fail_result("vkBindBufferMemory2", result);
}

static int create_compute(struct context *ctx, const uint32_t *code, size_t size, const VkDescriptorType *types,
                          uint32_t count, struct compute *compute)
{
    VkDescriptorSetLayoutBinding bindings[4];
    VkShaderModule module;
    for (uint32_t i = 0; i < count; i++)
        bindings[i] = (VkDescriptorSetLayoutBinding){i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = count, .pBindings = bindings
    };
    VkDescriptorSetLayoutSupport support = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
    vkGetDescriptorSetLayoutSupport(ctx->device, &set_info, &support);
    if (!support.supported)
        return fail("vkGetDescriptorSetLayoutSupport refused a small layout");
    if (vkCreateDescriptorSetLayout(ctx->device, &set_info, NULL, &compute->set_layout) != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx->device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
            .pSetLayouts = &compute->set_layout, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, 0, 4}
        }, NULL, &compute->layout) != VK_SUCCESS ||
        vkCreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code
        }, NULL, &module) != VK_SUCCESS)
        return fail("compute layout or module creation failed");
    VkResult result = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"
        },
        .layout = compute->layout
    }, NULL, &compute->pipeline);
    vkDestroyShaderModule(ctx->device, module, NULL);
    return result == VK_SUCCESS ? 0 : fail_result("vkCreateComputePipelines", result);
}

static void destroy_compute(struct context *ctx, struct compute *compute)
{
    vkDestroyPipeline(ctx->device, compute->pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, compute->layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, compute->set_layout, NULL);
}

static int allocate_set(struct context *ctx, const struct compute *compute, VkDescriptorSet *set)
{
    return vkAllocateDescriptorSets(ctx->device, &(VkDescriptorSetAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = ctx->descriptor_pool,
        .descriptorSetCount = 1, .pSetLayouts = &compute->set_layout
    }, set) == VK_SUCCESS ? 0 : fail("vkAllocateDescriptorSets failed");
}

static void write_buffers(struct context *ctx, VkDescriptorSet set, const VkDescriptorType *types,
                          const VkDescriptorBufferInfo *infos, uint32_t count)
{
    VkWriteDescriptorSet writes[4];
    for (uint32_t i = 0; i < count; i++)
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i,
            .descriptorCount = 1, .descriptorType = types[i], .pBufferInfo = &infos[i]
        };
    vkUpdateDescriptorSets(ctx->device, count, writes, 0, NULL);
}

static VkResult begin(struct context *ctx, VkCommandBuffer *command)
{
    VkResult result = vkAllocateCommandBuffers(ctx->device, &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1
    }, command);
    if (result != VK_SUCCESS)
        return result;
    return vkBeginCommandBuffer(*command, &(VkCommandBufferBeginInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    });
}

static VkResult finish(struct context *ctx, VkCommandBuffer command)
{
    VkResult result = vkEndCommandBuffer(command);
    if (result == VK_SUCCESS)
        result = vkResetFences(ctx->device, 1, &ctx->fence);
    if (result == VK_SUCCESS)
        result = vkQueueSubmit(ctx->queue, 1, &(VkSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command
        }, ctx->fence);
    if (result == VK_SUCCESS)
        result = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command);
    return result;
}

static VkResult dispatch(struct context *ctx, const struct compute *compute, VkDescriptorSet set,
                         uint32_t count, uint32_t groups)
{
    VkCommandBuffer command;
    VkResult result = begin(ctx, &command);
    if (result != VK_SUCCESS)
        return result;
    vkCmdSetDeviceMask(command, 1);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, compute->pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, compute->layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(command, compute->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &count);
    vkCmdDispatchBase(command, 0, 0, 0, groups, 1, 1);
    return finish(ctx, command);
}

static int setup(struct context *ctx)
{
    uint32_t version = 0, count = 0;
    VkPhysicalDevice devices[16];
    if (vkEnumerateInstanceVersion(&version) != VK_SUCCESS || version < VK_API_VERSION_1_1)
        return fail("the loader does not offer Vulkan 1.1");
    if (vkCreateInstance(&(VkInstanceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &(VkApplicationInfo){
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1
            }
        }, NULL, &ctx->instance) != VK_SUCCESS)
        return fail("vkCreateInstance failed for Vulkan 1.1");
    count = 16;
    if (vkEnumeratePhysicalDevices(ctx->instance, &count, devices) < 0)
        return fail("vkEnumeratePhysicalDevices failed");
    for (uint32_t i = 0; i < count && !ctx->physical; i++) {
        VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        vkGetPhysicalDeviceProperties2(devices[i], &properties);
        if (!strcmp(properties.properties.deviceName, "MXGPU")) {
            if (properties.properties.apiVersion < VK_API_VERSION_1_1)
                return fail("MXGPU reports an apiVersion below 1.1");
            ctx->physical = devices[i];
        }
    }
    if (!ctx->physical)
        return fail("no MXGPU physical device");
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(ctx->physical, &features);
    if (!features.features.fullDrawIndexUint32)
        return fail("fullDrawIndexUint32 is not reported");
    ctx->robust = features.features.robustBufferAccess;
    VkPhysicalDeviceMemoryProperties2 memory = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    vkGetPhysicalDeviceMemoryProperties2(ctx->physical, &memory);
    ctx->memory_type = UINT32_MAX;
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memory.memoryProperties.memoryTypeCount && ctx->memory_type == UINT32_MAX; i++)
        if ((memory.memoryProperties.memoryTypes[i].propertyFlags & host) == host)
            ctx->memory_type = i;
    if (ctx->memory_type == UINT32_MAX)
        return fail("no host-visible coherent memory type");
    const VkDeviceQueueCreateInfo queue = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
        .pQueuePriorities = &(float){1.0f}
    };
    VkPhysicalDeviceMultiviewFeatures multiview = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, .multiview = VK_TRUE
    };
    VkResult result = vkCreateDevice(ctx->physical, &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &multiview, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue
    }, NULL, &ctx->device);
    if (result != VK_ERROR_FEATURE_NOT_PRESENT)
        return fail_result("vkCreateDevice with the unsupported multiview feature", result);
    VkPhysicalDeviceFeatures2 enabled = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enabled.features.fullDrawIndexUint32 = VK_TRUE;
    enabled.features.robustBufferAccess = ctx->robust;
    result = vkCreateDevice(ctx->physical, &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &enabled, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue
    }, NULL, &ctx->device);
    if (result != VK_SUCCESS)
        return fail_result("vkCreateDevice with VkPhysicalDeviceFeatures2", result);
    vkGetDeviceQueue2(ctx->device, &(VkDeviceQueueInfo2){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2, .queueFamilyIndex = 0, .queueIndex = 0
    }, &ctx->queue);
    if (!ctx->queue)
        return fail("vkGetDeviceQueue2 returned no queue");
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8}};
    if (vkCreateCommandPool(ctx->device, &(VkCommandPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0
        }, NULL, &ctx->command_pool) != VK_SUCCESS ||
        vkCreateDescriptorPool(ctx->device, &(VkDescriptorPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 16, .poolSizeCount = 2,
            .pPoolSizes = sizes
        }, NULL, &ctx->descriptor_pool) != VK_SUCCESS ||
        vkCreateFence(ctx->device, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
                      NULL, &ctx->fence) != VK_SUCCESS)
        return fail("device object creation failed");
    return 0;
}

static int run_template_copy(struct context *ctx, const struct compute *copy)
{
    struct { VkDescriptorBufferInfo source, destination; } data;
    struct memory memory;
    VkBuffer source, destination;
    VkDescriptorSet set;
    VkDescriptorUpdateTemplate template;
    const uint32_t count = 4 * WORDS;
    const VkDescriptorUpdateTemplateEntry entries[2] = {
        {0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, offsetof(__typeof__(data), source), sizeof(VkDescriptorBufferInfo)},
        {1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, offsetof(__typeof__(data), destination), sizeof(VkDescriptorBufferInfo)},
    };
    if (create_memory(ctx, 8 * count, &memory) ||
        create_buffer(ctx, &memory, 0, 4 * count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &source) ||
        create_buffer(ctx, &memory, 4 * count, 4 * count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &destination) ||
        allocate_set(ctx, copy, &set))
        return -1;
    if (vkCreateDescriptorUpdateTemplate(ctx->device, &(VkDescriptorUpdateTemplateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO, .descriptorUpdateEntryCount = 2,
            .pDescriptorUpdateEntries = entries, .templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
            .descriptorSetLayout = copy->set_layout
        }, NULL, &template) != VK_SUCCESS)
        return fail("vkCreateDescriptorUpdateTemplate failed");
    data.source = (VkDescriptorBufferInfo){source, 0, VK_WHOLE_SIZE};
    data.destination = (VkDescriptorBufferInfo){destination, 0, VK_WHOLE_SIZE};
    vkUpdateDescriptorSetWithTemplate(ctx->device, set, template, &data);
    vkDestroyDescriptorUpdateTemplate(ctx->device, template, NULL);
    for (uint32_t i = 0; i < count; i++)
        memory.words[i] = i * 7u + 3u;
    VkResult result = dispatch(ctx, copy, set, count, count / 64u);
    if (result != VK_SUCCESS)
        return fail_result("template copy dispatch", result);
    for (uint32_t i = 0; i < count; i++)
        if (memory.words[count + i] != i * 7u + 3u) {
            snprintf(g_failure, sizeof g_failure, "template copy element %u is 0x%x", i, memory.words[count + i]);
            return -1;
        }
    vkTrimCommandPool(ctx->device, ctx->command_pool, 0);
    vkDestroyBuffer(ctx->device, source, NULL);
    vkDestroyBuffer(ctx->device, destination, NULL);
    vkFreeMemory(ctx->device, memory.handle, NULL);
    printf("template copy: %u elements through vkUpdateDescriptorSetWithTemplate and vkCmdDispatchBase\n", count);
    return 0;
}

static int run_robust_loads(struct context *ctx, const struct compute *copy, int *robust)
{
    static const VkDescriptorType types[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    struct memory memory;
    VkBuffer first, second, output;
    VkDescriptorSet prime, probe;
    if (create_memory(ctx, 16 * WORDS, &memory) ||
        create_buffer(ctx, &memory, 0, 4 * WORDS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &first) ||
        create_buffer(ctx, &memory, 4 * WORDS, 4 * WORDS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &second) ||
        create_buffer(ctx, &memory, 8 * WORDS, 8 * WORDS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &output) ||
        allocate_set(ctx, copy, &prime) || allocate_set(ctx, copy, &probe))
        return -1;
    for (uint32_t i = 0; i < WORDS; i++) {
        memory.words[i] = i + 1u;
        memory.words[WORDS + i] = SENTINEL + i;
    }
    write_buffers(ctx, prime, types, (VkDescriptorBufferInfo[2]){{second, 0, VK_WHOLE_SIZE}, {output, 0, VK_WHOLE_SIZE}}, 2);
    write_buffers(ctx, probe, types, (VkDescriptorBufferInfo[2]){{first, 0, VK_WHOLE_SIZE}, {output, 0, VK_WHOLE_SIZE}}, 2);
    VkResult result = dispatch(ctx, copy, prime, WORDS, 1);
    if (result == VK_SUCCESS)
        result = dispatch(ctx, copy, probe, 2 * WORDS, 2);
    if (result != VK_SUCCESS)
        return fail_result("out-of-bounds storage load dispatch", result);
    *robust = 1;
    for (uint32_t i = 0; i < 2 * WORDS; i++) {
        uint32_t value = memory.words[2 * WORDS + i];
        if (i < WORDS ? value != i + 1u : value > WORDS) {
            printf("storage load %u returned 0x%x\n", i, value);
            *robust = 0;
            break;
        }
    }
    printf("storage loads past a %u-word binding: %s\n", WORDS, *robust ? "zero or in-buffer" : "leaked other memory");
    return 0;
}

static int run_robust_stores(struct context *ctx, const struct compute *alias, int *robust)
{
    static const VkDescriptorType types[3] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    struct memory memory;
    VkBuffer target, neighbour, result_buffer;
    VkDescriptorSet set;
    if (create_memory(ctx, 12 * WORDS, &memory) ||
        create_buffer(ctx, &memory, 0, 4 * WORDS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &target) ||
        create_buffer(ctx, &memory, 4 * WORDS, 4 * WORDS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &neighbour) ||
        create_buffer(ctx, &memory, 8 * WORDS, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &result_buffer) ||
        allocate_set(ctx, alias, &set))
        return -1;
    for (uint32_t i = 0; i < WORDS; i++)
        memory.words[WORDS + i] = SENTINEL + i;
    write_buffers(ctx, set, types, (VkDescriptorBufferInfo[3]){
        {target, 0, VK_WHOLE_SIZE}, {neighbour, 0, VK_WHOLE_SIZE}, {result_buffer, 0, VK_WHOLE_SIZE}}, 3);
    VkResult result = dispatch(ctx, alias, set, WORDS, 1);
    if (result != VK_SUCCESS)
        return fail_result("out-of-bounds storage store dispatch", result);
    *robust = memory.words[2 * WORDS] == SENTINEL && memory.words[2 * WORDS + 1] == SENTINEL + 1u;
    for (uint32_t i = 0; i < WORDS; i++)
        if (memory.words[WORDS + i] != SENTINEL + i)
            *robust = 0;
    printf("storage store and atomic past a %u-word binding: %s (neighbour read 0x%x 0x%x)\n", WORDS,
           *robust ? "dropped" : "reached other memory", memory.words[2 * WORDS], memory.words[2 * WORDS + 1]);
    return 0;
}

static int run_robust_short_range(struct context *ctx, const struct compute *copy, int *robust)
{
    static const VkDescriptorType types[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    struct memory memory;
    VkBuffer source, destination;
    VkDescriptorSet set;
    if (create_memory(ctx, 64, &memory) ||
        create_buffer(ctx, &memory, 0, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &source) ||
        create_buffer(ctx, &memory, 16, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &destination) ||
        allocate_set(ctx, copy, &set))
        return -1;
    for (uint32_t i = 0; i < 4; i++) {
        memory.words[i] = 21u + i;
        memory.words[4 + i] = SENTINEL + i;
    }
    write_buffers(ctx, set, types, (VkDescriptorBufferInfo[2]){{source, 0, 2}, {destination, 0, 2}}, 2);
    VkResult result = dispatch(ctx, copy, set, 4, 1);
    if (result != VK_SUCCESS)
        return fail_result("storage bindings of 2 bytes", result);
    *robust = 1;
    for (uint32_t i = 0; i < 4; i++)
        if (memory.words[4 + i] != SENTINEL + i)
            *robust = 0;
    printf("storage bindings of 2 bytes: %s\n", *robust ? "every word access contained" : "store reached memory");
    return 0;
}

static int run_robust_uniforms(struct context *ctx, const struct compute *uniform, int *robust)
{
    static const VkDescriptorType types[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    struct memory memory;
    VkBuffer block, neighbour, result_buffer;
    VkDescriptorSet set;
    if (create_memory(ctx, 256, &memory) ||
        create_buffer(ctx, &memory, 0, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &block) ||
        create_buffer(ctx, &memory, 16, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &neighbour) ||
        create_buffer(ctx, &memory, 64, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &result_buffer) ||
        allocate_set(ctx, uniform, &set))
        return -1;
    for (uint32_t i = 0; i < 4; i++) {
        memory.words[i] = 11u + i;
        memory.words[4 + i] = SENTINEL + i;
    }
    write_buffers(ctx, set, types, (VkDescriptorBufferInfo[2]){
        {block, 0, VK_WHOLE_SIZE}, {result_buffer, 0, VK_WHOLE_SIZE}}, 2);
    VkResult result = dispatch(ctx, uniform, set, 0, 1);
    if (result != VK_SUCCESS)
        return fail_result("uniform block larger than its 16-byte binding", result);
    const uint32_t *values = &memory.words[16];
    *robust = values[0] == 11u && values[1] == 14u &&
              (values[2] == 0 || (values[2] >= 11u && values[2] <= 14u)) &&
              (values[3] == 0 || (values[3] >= 11u && values[3] <= 14u));
    printf("uniform block past a 16-byte binding: %u %u %u %u (%s)\n", values[0], values[1], values[2], values[3],
           *robust ? "zero or in-buffer" : "leaked other memory");
    return 0;
}

static int run_robust_vertices(struct context *ctx, int *robust)
{
    struct memory memory;
    VkBuffer vertices, indices;
    VkImage image;
    VkImageView view;
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkShaderModule modules[2];
    VkCommandBuffer command;
    VkDescriptorSetLayout set_layout;
    VkDescriptorSet set;
    struct memory colors;
    VkBuffer color_buffer;
    const VkDeviceSize image_offset = 1024, color_bytes = 4096 * 16;
    if (create_memory(ctx, image_offset + IMAGE_SIZE * IMAGE_SIZE * 4, &memory) ||
        create_buffer(ctx, &memory, 0, 24, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertices) ||
        create_buffer(ctx, &memory, 64, 24, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &indices))
        return -1;
    const float triangle[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    const uint32_t index_values[6] = {0, 1, 2, 0xfffffff0u, 0xfffffff1u, 0xfffffff2u};
    memcpy(memory.words, triangle, sizeof triangle);
    memcpy(memory.words + 16, index_values, sizeof index_values);
    if (create_memory(ctx, color_bytes, &colors) ||
        create_buffer(ctx, &colors, 0, color_bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &color_buffer))
        return -1;
    for (uint32_t i = 0; i < 4096; i++) {
        float *value = (float *)colors.words + 4 * i;
        value[0] = i == 4095 ? 1.0f : 0.0f;
        value[1] = i == 4095 ? 0.0f : 1.0f;
        value[2] = 0.0f;
        value[3] = 1.0f;
    }
    if (vkCreateDescriptorSetLayout(ctx->device, &(VkDescriptorSetLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
            .pBindings = &(VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                                         VK_SHADER_STAGE_FRAGMENT_BIT, NULL}
        }, NULL, &set_layout) != VK_SUCCESS ||
        vkAllocateDescriptorSets(ctx->device, &(VkDescriptorSetAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = ctx->descriptor_pool,
            .descriptorSetCount = 1, .pSetLayouts = &set_layout
        }, &set) != VK_SUCCESS)
        return fail("uniform descriptor set creation failed");
    vkUpdateDescriptorSets(ctx->device, 1, &(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &(VkDescriptorBufferInfo){color_buffer, 0, VK_WHOLE_SIZE}
    }, 0, NULL);
    if (vkCreateImage(ctx->device, &(VkImageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {IMAGE_SIZE, IMAGE_SIZE, 1}, .mipLevels = 1,
            .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        }, NULL, &image) != VK_SUCCESS)
        return fail("vkCreateImage failed");
    VkMemoryRequirements2 requirements = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetImageMemoryRequirements2(ctx->device, &(VkImageMemoryRequirementsInfo2){
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = image
    }, &requirements);
    if (requirements.memoryRequirements.size > IMAGE_SIZE * IMAGE_SIZE * 4 ||
        vkBindImageMemory2(ctx->device, 1, &(VkBindImageMemoryInfo){
            .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, .image = image, .memory = memory.handle,
            .memoryOffset = image_offset
        }) != VK_SUCCESS)
        return fail("image memory binding failed");
    if (vkCreateImageView(ctx->device, &(VkImageViewCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        }, NULL, &view) != VK_SUCCESS ||
        vkCreateRenderPass(ctx->device, &(VkRenderPassCreateInfo){
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
            .pAttachments = &(VkAttachmentDescription){
                .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL
            },
            .subpassCount = 1,
            .pSubpasses = &(VkSubpassDescription){
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
                .pColorAttachments = &(VkAttachmentReference){0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
            }
        }, NULL, &pass) != VK_SUCCESS ||
        vkCreateFramebuffer(ctx->device, &(VkFramebufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass, .attachmentCount = 1,
            .pAttachments = &view, .width = IMAGE_SIZE, .height = IMAGE_SIZE, .layers = 1
        }, NULL, &framebuffer) != VK_SUCCESS ||
        vkCreatePipelineLayout(ctx->device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &set_layout
        }, NULL, &layout) != VK_SUCCESS ||
        vkCreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof core11_vertex, .pCode = core11_vertex
        }, NULL, &modules[0]) != VK_SUCCESS ||
        vkCreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof core11_fragment,
            .pCode = core11_fragment
        }, NULL, &modules[1]) != VK_SUCCESS)
        return fail("graphics object creation failed");
    const VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = modules[0], .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = modules[1], .pName = "main"},
    };
    VkResult result = vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages,
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){0, 8, VK_VERTEX_INPUT_RATE_VERTEX},
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &(VkVertexInputAttributeDescription){0, 0, VK_FORMAT_R32G32_SFLOAT, 0}
        },
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        },
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
            .pViewports = &(VkViewport){0, 0, IMAGE_SIZE, IMAGE_SIZE, 0, 1}, .scissorCount = 1,
            .pScissors = &(VkRect2D){{0, 0}, {IMAGE_SIZE, IMAGE_SIZE}}
        },
        .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f
        },
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
        },
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
            .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf}
        },
        .layout = layout, .renderPass = pass
    }, NULL, &pipeline);
    vkDestroyShaderModule(ctx->device, modules[0], NULL);
    vkDestroyShaderModule(ctx->device, modules[1], NULL);
    if (result != VK_SUCCESS)
        return fail_result("vkCreateGraphicsPipelines", result);
    result = begin(ctx, &command);
    if (result != VK_SUCCESS)
        return fail_result("graphics command buffer begin", result);
    vkCmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {IMAGE_SIZE, IMAGE_SIZE}}, .clearValueCount = 1,
        .pClearValues = &(VkClearValue){.color = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}}}
    }, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    vkCmdBindVertexBuffers(command, 0, 1, &vertices, (VkDeviceSize[]){0});
    vkCmdBindIndexBuffer(command, indices, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDraw(command, 6, 1, 0, 0);
    vkCmdDrawIndexed(command, 6, 1, 0, 0, 0);
    vkCmdEndRenderPass(command);
    result = finish(ctx, command);
    *robust = result == VK_SUCCESS;
    if (result != VK_SUCCESS)
        printf("draws fetching past the vertex buffer returned %d\n", (int)result);
    else {
        const uint8_t *pixels = (const uint8_t *)memory.words + image_offset;
        for (uint32_t i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
            if (pixels[4 * i] != 255 || pixels[4 * i + 1] || pixels[4 * i + 2] || pixels[4 * i + 3] != 255) {
                snprintf(g_failure, sizeof g_failure, "pixel %u is %02x%02x%02x%02x", i, pixels[4 * i],
                         pixels[4 * i + 1], pixels[4 * i + 2], pixels[4 * i + 3]);
                return -1;
            }
        printf("draw and 32-bit indexed draw past a 3-vertex buffer, colour from the last vec4 of a %u-byte "
               "uniform block: rendered\n", (unsigned)color_bytes);
    }
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, layout, NULL);
    vkDestroyFramebuffer(ctx->device, framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, pass, NULL);
    vkDestroyImageView(ctx->device, view, NULL);
    vkDestroyImage(ctx->device, image, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, set_layout, NULL);
    return 0;
}

int main(void)
{
    static const VkDescriptorType storage2[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    static const VkDescriptorType storage3[3] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    static const VkDescriptorType uniform2[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    struct context ctx;
    struct compute copy = {0}, alias = {0}, uniform = {0};
    int loads = 0, stores = 0, short_range = 0, uniforms = 0, vertices = 0;
    memset(&ctx, 0, sizeof ctx);
    int result = setup(&ctx);
    if (!result)
        result = create_compute(&ctx, core11_copy, sizeof core11_copy, storage2, 2, &copy);
    if (!result)
        result = create_compute(&ctx, core11_alias, sizeof core11_alias, storage3, 3, &alias);
    if (!result)
        result = create_compute(&ctx, core11_uniform, sizeof core11_uniform, uniform2, 2, &uniform);
    if (!result)
        result = run_template_copy(&ctx, &copy);
    if (!result)
        result = run_robust_loads(&ctx, &copy, &loads);
    if (!result)
        result = run_robust_stores(&ctx, &alias, &stores);
    if (!result)
        result = run_robust_short_range(&ctx, &copy, &short_range);
    if (!result)
        result = run_robust_uniforms(&ctx, &uniform, &uniforms);
    if (!result)
        result = run_robust_vertices(&ctx, &vertices);
    if (!result)
        printf("robustBufferAccess reported %d, observed loads %d stores %d short ranges %d uniforms %d vertices %d\n",
               ctx.robust, loads, stores, short_range, uniforms, vertices);
    if (!result && ctx.robust && !(loads && stores && short_range && uniforms && vertices))
        result = fail("robustBufferAccess is reported but an out-of-bounds access was not contained");
    if (ctx.device) {
        vkDeviceWaitIdle(ctx.device);
        if (copy.pipeline)
            destroy_compute(&ctx, &copy);
        if (alias.pipeline)
            destroy_compute(&ctx, &alias);
        if (uniform.pipeline)
            destroy_compute(&ctx, &uniform);
        vkDestroyFence(ctx.device, ctx.fence, NULL);
        vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, NULL);
        vkDestroyCommandPool(ctx.device, ctx.command_pool, NULL);
        vkDestroyDevice(ctx.device, NULL);
    }
    vkDestroyInstance(ctx.instance, NULL);
    if (result) {
        printf("vk_core11 FAIL: %s\n", g_failure);
        return 1;
    }
    printf("vk_core11 PASS\n");
    return 0;
}
