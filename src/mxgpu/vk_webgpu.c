/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "features-vert.h"
#include "webgpu-sixteen-frag.h"
#include "webgpu-gradient-frag.h"
#include "webgpu-probe-frag.h"
#include "webgpu-float-frag.h"
#include "webgpu-unorm-frag.h"

#define TEXTURE_COUNT 16u
#define PROBE_SIZE 8u

struct context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t memory_type;
    VkCommandPool pool;
    VkFence fence;
    VkPhysicalDeviceProperties properties;
    VkShaderModule vertex;
};

struct image {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    uint8_t *pixels;
};

struct layout {
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
};

struct target {
    struct image image;
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipeline pipeline;
};

static char g_failure[512];

static int fail(const char *message)
{
    snprintf(g_failure, sizeof g_failure, "%s", message);
    return -1;
}

static int check(VkResult result, const char *what)
{
    if (result == VK_SUCCESS)
        return 0;
    snprintf(g_failure, sizeof g_failure, "%s returned %d", what, (int)result);
    return -1;
}

static VkShaderModule module(struct context *ctx, const uint32_t *code, size_t size)
{
    VkShaderModule handle = VK_NULL_HANDLE;
    vkCreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code}, NULL, &handle);
    return handle;
}

static int make_image(struct context *ctx, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage,
                      struct image *out)
{
    VkMemoryRequirements requirements;
    void *pointer = NULL;
    if (check(vkCreateImage(ctx->device, &(VkImageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
            .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_LINEAR, .usage = usage}, NULL, &out->image), "vkCreateImage"))
        return -1;
    vkGetImageMemoryRequirements(ctx->device, out->image, &requirements);
    if (check(vkAllocateMemory(ctx->device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size,
            .memoryTypeIndex = ctx->memory_type}, NULL, &out->memory), "vkAllocateMemory") ||
        check(vkMapMemory(ctx->device, out->memory, 0, VK_WHOLE_SIZE, 0, &pointer), "vkMapMemory") ||
        check(vkBindImageMemory(ctx->device, out->image, out->memory, 0), "vkBindImageMemory"))
        return -1;
    out->pixels = pointer;
    return check(vkCreateImageView(ctx->device, &(VkImageViewCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = out->image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &out->view),
        "vkCreateImageView");
}

static void destroy_image(struct context *ctx, struct image *image)
{
    vkDestroyImageView(ctx->device, image->view, NULL);
    vkDestroyImage(ctx->device, image->image, NULL);
    vkFreeMemory(ctx->device, image->memory, NULL);
}

static int make_layout(struct context *ctx, uint32_t samplers, struct layout *out)
{
    VkDescriptorSetLayoutBinding bindings[TEXTURE_COUNT];
    memset(out, 0, sizeof *out);
    for (uint32_t i = 0; i < samplers; i++)
        bindings[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                     VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
    if (check(vkCreateDescriptorSetLayout(ctx->device, &(VkDescriptorSetLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = samplers,
            .pBindings = bindings}, NULL, &out->set_layout), "vkCreateDescriptorSetLayout") ||
        check(vkCreatePipelineLayout(ctx->device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = samplers ? 1 : 0,
            .pSetLayouts = &out->set_layout, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_VERTEX_BIT, 0, 4}},
            NULL, &out->pipeline_layout), "vkCreatePipelineLayout"))
        return -1;
    if (!samplers)
        return 0;
    return check(vkCreateDescriptorPool(ctx->device, &(VkDescriptorPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
            .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, samplers}},
            NULL, &out->pool), "vkCreateDescriptorPool") ||
        check(vkAllocateDescriptorSets(ctx->device, &(VkDescriptorSetAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = out->pool,
            .descriptorSetCount = 1, .pSetLayouts = &out->set_layout}, &out->set), "vkAllocateDescriptorSets");
}

static void destroy_layout(struct context *ctx, struct layout *layout)
{
    vkDestroyDescriptorPool(ctx->device, layout->pool, NULL);
    vkDestroyPipelineLayout(ctx->device, layout->pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, layout->set_layout, NULL);
}

static int make_target_blend(struct context *ctx, VkFormat format, uint32_t width, uint32_t height,
                             VkShaderModule fragment, VkPipelineLayout layout, bool additive, struct target *out)
{
    VkDynamicState dynamic[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    if (make_image(ctx, format, width, height, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   &out->image) ||
        check(vkCreateRenderPass(ctx->device, &(VkRenderPassCreateInfo){
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
            .pAttachments = &(VkAttachmentDescription){.format = format, .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL},
            .subpassCount = 1, .pSubpasses = &(VkSubpassDescription){
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
                .pColorAttachments = &(VkAttachmentReference){0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}},
            NULL, &out->pass), "vkCreateRenderPass") ||
        check(vkCreateFramebuffer(ctx->device, &(VkFramebufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = out->pass, .attachmentCount = 1,
            .pAttachments = &out->image.view, .width = width, .height = height, .layers = 1}, NULL,
            &out->framebuffer), "vkCreateFramebuffer"))
        return -1;
    const VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = ctx->vertex, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment, .pName = "main"},
    };
    return check(vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages,
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1},
        .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f},
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
            .pAttachments = &(VkPipelineColorBlendAttachmentState){.blendEnable = additive,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .colorBlendOp = VK_BLEND_OP_ADD, .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf}},
        .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2,
            .pDynamicStates = dynamic},
        .layout = layout, .renderPass = out->pass}, NULL, &out->pipeline), "vkCreateGraphicsPipelines");
}

static int make_target(struct context *ctx, VkFormat format, uint32_t width, uint32_t height,
                       VkShaderModule fragment, VkPipelineLayout layout, struct target *out)
{
    return make_target_blend(ctx, format, width, height, fragment, layout, false, out);
}

static void destroy_target(struct context *ctx, struct target *target)
{
    vkDestroyPipeline(ctx->device, target->pipeline, NULL);
    vkDestroyFramebuffer(ctx->device, target->framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, target->pass, NULL);
    destroy_image(ctx, &target->image);
}

static VkCommandBuffer begin(struct context *ctx)
{
    VkCommandBuffer command = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx->device, &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ctx->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &command);
    vkBeginCommandBuffer(command, &(VkCommandBufferBeginInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT});
    return command;
}

static int submit(struct context *ctx, VkCommandBuffer command, const char *what)
{
    VkResult result = vkEndCommandBuffer(command);
    if (result == VK_SUCCESS)
        result = vkResetFences(ctx->device, 1, &ctx->fence);
    if (result == VK_SUCCESS)
        result = vkQueueSubmit(ctx->queue, 1, &(VkSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command}, ctx->fence);
    if (result == VK_SUCCESS)
        result = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(ctx->device, ctx->pool, 1, &command);
    return check(result, what);
}

static void draw(VkCommandBuffer command, const struct target *target, VkPipelineLayout layout,
                 const VkDescriptorSet *set, uint32_t width, uint32_t height, VkClearValue clear)
{
    float depth = 0.5f;
    vkCmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = target->pass,
        .framebuffer = target->framebuffer, .renderArea = {{0, 0}, {width, height}}, .clearValueCount = 1,
        .pClearValues = &clear}, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, target->pipeline);
    if (set)
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, set, 0, NULL);
    vkCmdSetViewport(command, 0, 1, &(VkViewport){0, 0, (float)width, (float)height, 0, 1});
    vkCmdSetScissor(command, 0, 1, &(VkRect2D){{0, 0}, {width, height}});
    vkCmdPushConstants(command, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof depth, &depth);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
}

static int make_sampler(struct context *ctx, VkSampler *sampler)
{
    return check(vkCreateSampler(ctx->device, &(VkSamplerCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE}, NULL, sampler), "vkCreateSampler");
}

static int pixel_failure(const char *what, uint32_t x, uint32_t y, const uint8_t *pixel, const uint8_t *want)
{
    snprintf(g_failure, sizeof g_failure, "%s: pixel (%u,%u) is %u %u %u %u, expected %u %u %u %u", what, x, y,
             pixel[0], pixel[1], pixel[2], pixel[3], want[0], want[1], want[2], want[3]);
    return -1;
}

static int same_pixel(const uint8_t *pixel, const uint8_t *want)
{
    for (unsigned c = 0; c < 4; c++)
        if (abs(pixel[c] - want[c]) > 1)
            return 0;
    return 1;
}

static int test_limits(struct context *ctx)
{
    const VkPhysicalDeviceLimits *limits = &ctx->properties.limits;
    printf("maxImageDimension2D %u maxFramebufferWidth %u maxPerStageDescriptorSampledImages %u "
           "maxPerStageDescriptorSamplers %u maxComputeWorkGroupSize %u %u %u maxComputeWorkGroupInvocations %u "
           "maxStorageBufferRange %u\n", limits->maxImageDimension2D, limits->maxFramebufferWidth,
           limits->maxPerStageDescriptorSampledImages, limits->maxPerStageDescriptorSamplers,
           limits->maxComputeWorkGroupSize[0], limits->maxComputeWorkGroupSize[1], limits->maxComputeWorkGroupSize[2],
           limits->maxComputeWorkGroupInvocations, limits->maxStorageBufferRange);
    if (limits->maxImageDimension2D < 8192 || limits->maxFramebufferWidth < 8192 ||
        limits->maxFramebufferHeight < 8192 || limits->maxViewportDimensions[0] < 8192)
        return fail("2D image and framebuffer dimensions below 8192");
    if (limits->maxPerStageDescriptorSampledImages < TEXTURE_COUNT ||
        limits->maxPerStageDescriptorSamplers < TEXTURE_COUNT)
        return fail("fewer than 16 sampled images and samplers per stage");
    if (limits->maxComputeWorkGroupInvocations < 128 || limits->maxComputeWorkGroupSize[0] < 128 ||
        limits->maxComputeWorkGroupSize[1] < 128 || limits->maxComputeWorkGroupSize[2] < 64)
        return fail("compute work-group limits below the Vulkan minimums");
    return 0;
}

static int test_sixteen_textures(struct context *ctx)
{
    struct image textures[TEXTURE_COUNT];
    VkSampler samplers[TEXTURE_COUNT];
    VkDescriptorImageInfo infos[TEXTURE_COUNT];
    VkWriteDescriptorSet writes[TEXTURE_COUNT];
    struct layout layout;
    struct target target;
    VkShaderModule fragment = module(ctx, webgpu_sixteen, sizeof webgpu_sixteen);
    uint8_t texels[TEXTURE_COUNT][4];
    for (uint32_t i = 0; i < TEXTURE_COUNT; i++) {
        texels[i][0] = (uint8_t)(i * 16u + 8u);
        texels[i][1] = (uint8_t)(255u - i * 16u);
        texels[i][2] = (uint8_t)(i * 37u);
        texels[i][3] = (uint8_t)(255u - i);
        if (make_image(ctx, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT, &textures[i]) ||
            make_sampler(ctx, &samplers[i]))
            return -1;
        memcpy(textures[i].pixels, texels[i], 4);
        infos[i] = (VkDescriptorImageInfo){samplers[i], textures[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &infos[i]};
    }
    if (!fragment || make_layout(ctx, TEXTURE_COUNT, &layout) ||
        make_target(ctx, VK_FORMAT_R8G8B8A8_UNORM, TEXTURE_COUNT, 2, fragment, layout.pipeline_layout, &target))
        return fail("could not build the 16-texture pipeline");
    for (uint32_t i = 0; i < TEXTURE_COUNT; i++)
        writes[i].dstSet = layout.set;
    vkUpdateDescriptorSets(ctx->device, TEXTURE_COUNT, writes, 0, NULL);
    VkCommandBuffer command = begin(ctx);
    draw(command, &target, layout.pipeline_layout, &layout.set, TEXTURE_COUNT, 2,
         (VkClearValue){.color = {{0, 0, 0, 0}}});
    if (submit(ctx, command, "16-texture draw"))
        return -1;
    for (uint32_t y = 0; y < 2; y++)
        for (uint32_t x = 0; x < TEXTURE_COUNT; x++) {
            const uint8_t *pixel = target.image.pixels + (y * TEXTURE_COUNT + x) * 4;
            if (!same_pixel(pixel, texels[x]))
                return pixel_failure("16 combined image samplers in one draw", x, y, pixel, texels[x]);
        }
    destroy_target(ctx, &target);
    destroy_layout(ctx, &layout);
    vkDestroyShaderModule(ctx->device, fragment, NULL);
    for (uint32_t i = 0; i < TEXTURE_COUNT; i++) {
        vkDestroySampler(ctx->device, samplers[i], NULL);
        destroy_image(ctx, &textures[i]);
    }
    printf("16 textures with 16 samplers sampled in one draw: every column matches its texture\n");
    return 0;
}

static void gradient(uint32_t x, uint32_t y, uint8_t *out)
{
    out[0] = (uint8_t)(x % 256u);
    out[1] = (uint8_t)(y % 256u);
    out[2] = (uint8_t)(x / 256u * 8u);
    out[3] = (uint8_t)(y / 256u * 8u);
}

static int test_large_target(struct context *ctx, uint32_t size)
{
    struct layout plain, sampled;
    struct target large, probe;
    VkSampler sampler;
    VkShaderModule fill = module(ctx, webgpu_gradient, sizeof webgpu_gradient);
    VkShaderModule read = module(ctx, webgpu_probe, sizeof webgpu_probe);
    if (!fill || !read || make_layout(ctx, 0, &plain) || make_layout(ctx, 1, &sampled) || make_sampler(ctx, &sampler) ||
        make_target(ctx, VK_FORMAT_R8G8B8A8_UNORM, size, size, fill, plain.pipeline_layout, &large) ||
        make_target(ctx, VK_FORMAT_R8G8B8A8_UNORM, PROBE_SIZE, PROBE_SIZE, read, sampled.pipeline_layout, &probe))
        return fail("could not build the large-target pipelines");
    vkUpdateDescriptorSets(ctx->device, 1, &(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sampled.set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &(VkDescriptorImageInfo){sampler, large.image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        0, NULL);
    VkCommandBuffer command = begin(ctx);
    draw(command, &large, plain.pipeline_layout, NULL, size, size, (VkClearValue){.color = {{0, 0, 0, 0}}});
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &(VkMemoryBarrier){.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT}, 0, NULL, 0, NULL);
    draw(command, &probe, sampled.pipeline_layout, &sampled.set, PROBE_SIZE, PROBE_SIZE,
         (VkClearValue){.color = {{0, 0, 0, 0}}});
    if (submit(ctx, command, "large target render and sample"))
        return -1;
    const uint32_t corners[5][2] = {{0, 0}, {size - 1, 0}, {0, size - 1}, {size - 1, size - 1}, {size / 2 + 7, size / 3}};
    for (uint32_t i = 0; i < 5; i++) {
        uint8_t want[4];
        uint32_t x = corners[i][0], y = corners[i][1];
        gradient(x, y, want);
        const uint8_t *pixel = large.image.pixels + ((size_t)y * size + x) * 4;
        if (!same_pixel(pixel, want))
            return pixel_failure("rendered large target", x, y, pixel, want);
    }
    for (uint32_t y = 0; y < PROBE_SIZE; y++)
        for (uint32_t x = 0; x < PROBE_SIZE; x++) {
            uint8_t want[4];
            uint32_t sx = x * (size / PROBE_SIZE) + 300u, sy = y * (size / PROBE_SIZE) + 700u;
            gradient(sx < size ? sx : size - 1u, sy < size ? sy : size - 1u, want);
            const uint8_t *pixel = probe.image.pixels + (y * PROBE_SIZE + x) * 4;
            if (!same_pixel(pixel, want))
                return pixel_failure("large target sampled as a texture", x, y, pixel, want);
        }
    destroy_target(ctx, &probe);
    destroy_target(ctx, &large);
    vkDestroySampler(ctx->device, sampler, NULL);
    destroy_layout(ctx, &sampled);
    destroy_layout(ctx, &plain);
    vkDestroyShaderModule(ctx->device, fill, NULL);
    vkDestroyShaderModule(ctx->device, read, NULL);
    printf("%ux%u colour target rendered, read back and sampled: corners and 64 probes match\n", size, size);
    return 0;
}

static float half_value(uint16_t bits)
{
    int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    float magnitude = exponent ? ldexpf(1.0f + mantissa / 1024.0f, exponent - 15) : ldexpf(mantissa / 1024.0f, -14);
    return bits & 0x8000u ? -magnitude : magnitude;
}

static float small_float(uint32_t bits, unsigned mantissa_bits)
{
    uint32_t exponent = bits >> mantissa_bits, mantissa = bits & ((1u << mantissa_bits) - 1u);
    float scale = (float)(1u << mantissa_bits);
    return exponent ? ldexpf(1.0f + mantissa / scale, (int)exponent - 15) : ldexpf(mantissa / scale, -14);
}

static float srgb_value(uint8_t encoded)
{
    float c = encoded / 255.0f;
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

static void decode(VkFormat format, const uint8_t *texel, float *out)
{
    uint32_t word;
    memcpy(&word, texel, 4);
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    switch (format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
        for (unsigned c = 0; c < 3; c++)
            out[format == VK_FORMAT_B8G8R8A8_SRGB ? 2 - c : c] = srgb_value(texel[c]);
        out[3] = texel[3] / 255.0f;
        break;
    case VK_FORMAT_R8G8B8A8_SNORM:
        for (unsigned c = 0; c < 4; c++)
            out[c] = fmaxf((int8_t)texel[c] / 127.0f, -1.0f);
        break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        for (unsigned c = 0; c < 3; c++)
            out[c] = ((word >> (10 * c)) & 1023u) / 1023.0f;
        out[3] = (word >> 30) / 3.0f;
        break;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        out[0] = small_float(word & 2047u, 6);
        out[1] = small_float((word >> 11) & 2047u, 6);
        out[2] = small_float(word >> 22, 5);
        break;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        for (unsigned c = 0; c < 3; c++)
            out[c] = ldexpf((float)((word >> (9 * c)) & 511u), (int)(word >> 27) - 24);
        break;
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
        unsigned count = format == VK_FORMAT_R16_SFLOAT ? 1 : format == VK_FORMAT_R16G16_SFLOAT ? 2 : 4;
        for (unsigned c = 0; c < count; c++) {
            uint16_t bits;
            memcpy(&bits, texel + 2 * c, 2);
            out[c] = half_value(bits);
        }
        break;
    }
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        memcpy(out, texel, format == VK_FORMAT_R32_SFLOAT ? 4 : format == VK_FORMAT_R32G32_SFLOAT ? 8 : 16);
        break;
    default:
        break;
    }
}

static unsigned format_bytes(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R16_SFLOAT: return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 4;
    }
}

static int values_match(const float *got, const float *want, float tolerance)
{
    for (unsigned c = 0; c < 4; c++)
        if (!(fabsf(got[c] - want[c]) <= tolerance * fmaxf(1.0f, fabsf(want[c]))))
            return 0;
    return 1;
}

static int value_failure(const char *what, const char *format, const float *got, const float *want)
{
    snprintf(g_failure, sizeof g_failure, "%s %s: got %g %g %g %g, expected %g %g %g %g", what, format,
             got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    return -1;
}

struct format_case {
    VkFormat format;
    const char *name;
    bool unorm, blend;
    float want[4];
    float tolerance;
};

static int test_render_formats(struct context *ctx)
{
    const struct format_case cases[] = {
        {VK_FORMAT_R8G8B8A8_SRGB, "R8G8B8A8_SRGB", true, true, {0.75f, 0.5f, 0.25f, 1.0f}, 0.01f},
        {VK_FORMAT_B8G8R8A8_SRGB, "B8G8R8A8_SRGB", true, true, {0.75f, 0.5f, 0.25f, 1.0f}, 0.01f},
        {VK_FORMAT_A2B10G10R10_UNORM_PACK32, "A2B10G10R10_UNORM", true, false, {0.5f, 0.25f, 0.0f, 1.0f}, 0.002f},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT", false, false, {2.5f, -1000.5f, 0.0078125f, 65504.0f}, 0.0f},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT blended", false, true, {2.75f, -1000.25f, 0.2578125f, 65504.0f}, 0.001f},
        {VK_FORMAT_R16_SFLOAT, "R16_SFLOAT", false, false, {2.5f, 0.0f, 0.0f, 1.0f}, 0.0f},
        {VK_FORMAT_R16G16_SFLOAT, "R16G16_SFLOAT", false, false, {2.5f, -1000.5f, 0.0f, 1.0f}, 0.0f},
        {VK_FORMAT_R32_SFLOAT, "R32_SFLOAT", false, false, {2.5f, 0.0f, 0.0f, 1.0f}, 0.0f},
        {VK_FORMAT_R32G32_SFLOAT, "R32G32_SFLOAT", false, false, {2.5f, -1000.5f, 0.0f, 1.0f}, 0.0f},
        {VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32_SFLOAT", false, false, {2.5f, -1000.5f, 0.0078125f, 65504.0f}, 0.0f},
        {VK_FORMAT_B10G11R11_UFLOAT_PACK32, "B10G11R11_UFLOAT", false, false, {2.5f, 0.0f, 0.0078125f, 1.0f}, 0.0f},
        {VK_FORMAT_R8G8B8A8_SNORM, "R8G8B8A8_SNORM", false, false, {1.0f, -1.0f, 1.0f / 127.0f, 1.0f}, 0.0f},
    };
    struct layout layout;
    VkShaderModule unorm = module(ctx, webgpu_unorm, sizeof webgpu_unorm);
    VkShaderModule floats = module(ctx, webgpu_float, sizeof webgpu_float);
    if (!unorm || !floats || make_layout(ctx, 0, &layout))
        return fail("could not build the colour format pipelines");
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const struct format_case *c = &cases[i];
        VkFormatProperties properties;
        struct target target;
        vkGetPhysicalDeviceFormatProperties(ctx->physical, c->format, &properties);
        VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            (c->blend ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0);
        if ((properties.optimalTilingFeatures & needed) != needed) {
            snprintf(g_failure, sizeof g_failure, "%s lacks colour attachment, sampling or blending", c->name);
            return -1;
        }
        if (make_target_blend(ctx, c->format, 4, 4, c->unorm ? unorm : floats, layout.pipeline_layout, c->blend,
                              &target))
            return -1;
        VkClearValue clear = {.color = {{0.25f, 0.25f, 0.25f, c->unorm ? 1.0f : 0.0f}}};
        VkCommandBuffer command = begin(ctx);
        draw(command, &target, layout.pipeline_layout, NULL, 4, 4, clear);
        if (submit(ctx, command, c->name))
            return -1;
        for (uint32_t p = 0; p < 16; p++) {
            float got[4];
            decode(c->format, target.image.pixels + p * format_bytes(c->format), got);
            if (!values_match(got, c->want, c->tolerance))
                return value_failure("render target", c->name, got, c->want);
        }
        destroy_target(ctx, &target);
    }
    destroy_layout(ctx, &layout);
    vkDestroyShaderModule(ctx->device, unorm, NULL);
    vkDestroyShaderModule(ctx->device, floats, NULL);
    printf("render targets: sRGB with linear-space blending, 10-bit, 16-bit and 32-bit float, packed float and "
           "signed normalised formats store the expected values\n");
    return 0;
}

struct sample_case {
    VkFormat format;
    const char *name;
    uint8_t texel[16];
    float want[4];
    float tolerance;
};

static int test_sampled_formats(struct context *ctx)
{
    static const float half_values[4] = {-2.5f, 1000.5f, 0.0078125f, 65504.0f};
    struct sample_case cases[] = {
        {VK_FORMAT_R8G8B8A8_SRGB, "R8G8B8A8_SRGB", {188, 137, 0, 128}, {0}, 0.002f},
        {VK_FORMAT_B8G8R8A8_SRGB, "B8G8R8A8_SRGB", {0, 137, 188, 128}, {0}, 0.002f},
        {VK_FORMAT_R8G8B8A8_SNORM, "R8G8B8A8_SNORM", {127, 0x81, 64, 0}, {0}, 0.002f},
        {VK_FORMAT_A2B10G10R10_UNORM_PACK32, "A2B10G10R10_UNORM", {0}, {0}, 0.002f},
        {VK_FORMAT_B10G11R11_UFLOAT_PACK32, "B10G11R11_UFLOAT", {0}, {0}, 0.0f},
        {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, "E5B9G9R9_UFLOAT", {0}, {0}, 0.0f},
        {VK_FORMAT_R16_SFLOAT, "R16_SFLOAT", {0x00, 0xc1}, {0}, 0.0f},
        {VK_FORMAT_R16G16_SFLOAT, "R16G16_SFLOAT", {0x00, 0xc1, 0xd1, 0x63}, {0}, 0.0f},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT", {0x00, 0xc1, 0xd1, 0x63, 0x00, 0x20, 0xff, 0x7b}, {0}, 0.0f},
        {VK_FORMAT_R32_SFLOAT, "R32_SFLOAT", {0}, {0}, 0.0f},
        {VK_FORMAT_R32G32_SFLOAT, "R32G32_SFLOAT", {0}, {0}, 0.0f},
        {VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32_SFLOAT", {0}, {0}, 0.0f},
    };
    const float precise[4] = {16777215.0f, -0.333333343f, 1.0e-20f, 3.0e38f};
    uint32_t word = 256u | 512u << 10 | 768u << 20 | 2u << 30;
    memcpy(cases[3].texel, &word, 4);
    word = (16u << 6 | 16u) | (24u << 6 | 61u) << 11 | (8u << 5) << 22;
    memcpy(cases[4].texel, &word, 4);
    word = 2u | 500u << 9 | 1u << 18 | 25u << 27;
    memcpy(cases[5].texel, &word, 4);
    for (unsigned i = 9; i < 12; i++)
        memcpy(cases[i].texel, precise, sizeof precise);
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        decode(cases[i].format, cases[i].texel, cases[i].want);
        if (cases[i].format == VK_FORMAT_R16G16B16A16_SFLOAT && memcmp(cases[i].want, half_values, sizeof half_values))
            return fail("half-float reference encoding is wrong");
    }
    struct layout layout;
    VkSampler sampler;
    VkShaderModule probe = module(ctx, webgpu_probe, sizeof webgpu_probe);
    if (!probe || make_layout(ctx, 1, &layout) || make_sampler(ctx, &sampler))
        return fail("could not build the sampling pipeline");
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const struct sample_case *c = &cases[i];
        struct image texture;
        struct target target;
        if (make_image(ctx, c->format, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT, &texture) ||
            make_target(ctx, VK_FORMAT_R32G32B32A32_SFLOAT, PROBE_SIZE, PROBE_SIZE, probe, layout.pipeline_layout,
                        &target))
            return -1;
        memcpy(texture.pixels, c->texel, format_bytes(c->format));
        vkUpdateDescriptorSets(ctx->device, 1, &(VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = layout.set, .dstBinding = 0,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &(VkDescriptorImageInfo){sampler, texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
            0, NULL);
        VkCommandBuffer command = begin(ctx);
        draw(command, &target, layout.pipeline_layout, &layout.set, PROBE_SIZE, PROBE_SIZE,
             (VkClearValue){.color = {{0, 0, 0, 0}}});
        if (submit(ctx, command, c->name))
            return -1;
        for (uint32_t p = 0; p < PROBE_SIZE * PROBE_SIZE; p++) {
            const float *got = (const float *)target.image.pixels + p * 4;
            if (!values_match(got, c->want, c->tolerance))
                return value_failure("sampled", c->name, got, c->want);
        }
        destroy_target(ctx, &target);
        destroy_image(ctx, &texture);
    }
    struct image texture;
    struct target target;
    VkImageView swizzled;
    const float want[4] = {0.0078125f, 65504.0f, 1.0f, -2.5f};
    if (make_image(ctx, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT, &texture) ||
        make_target(ctx, VK_FORMAT_R32G32B32A32_SFLOAT, PROBE_SIZE, PROBE_SIZE, probe, layout.pipeline_layout, &target) ||
        check(vkCreateImageView(ctx->device, &(VkImageViewCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = texture.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R16G16B16A16_SFLOAT,
            .components = {VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A, VK_COMPONENT_SWIZZLE_ONE,
                           VK_COMPONENT_SWIZZLE_R},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &swizzled), "vkCreateImageView"))
        return -1;
    memcpy(texture.pixels, cases[8].texel, 8);
    vkUpdateDescriptorSets(ctx->device, 1, &(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = layout.set, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &(VkDescriptorImageInfo){sampler, swizzled, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}}, 0, NULL);
    VkCommandBuffer command = begin(ctx);
    draw(command, &target, layout.pipeline_layout, &layout.set, PROBE_SIZE, PROBE_SIZE,
         (VkClearValue){.color = {{0, 0, 0, 0}}});
    if (submit(ctx, command, "swizzled half-float view"))
        return -1;
    if (!values_match((const float *)target.image.pixels, want, 0.0f))
        return value_failure("sampled", "swizzled R16G16B16A16_SFLOAT view", (const float *)target.image.pixels, want);
    vkDestroyImageView(ctx->device, swizzled, NULL);
    destroy_target(ctx, &target);
    destroy_image(ctx, &texture);
    vkDestroySampler(ctx->device, sampler, NULL);
    destroy_layout(ctx, &layout);
    vkDestroyShaderModule(ctx->device, probe, NULL);
    printf("sampled textures: sRGB decode, signed normalised, 10-bit, packed and shared-exponent float, "
           "16-bit and 32-bit float formats and a swizzled half-float view return the expected values\n");
    return 0;
}

static int setup(struct context *ctx)
{
    uint32_t count = 16;
    VkPhysicalDevice devices[16];
    if (check(vkCreateInstance(&(VkInstanceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                                     .apiVersion = VK_API_VERSION_1_1}}, NULL, &ctx->instance),
            "vkCreateInstance"))
        return -1;
    if (vkEnumeratePhysicalDevices(ctx->instance, &count, devices) < 0)
        return fail("vkEnumeratePhysicalDevices failed");
    for (uint32_t i = 0; i < count && !ctx->physical; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &ctx->properties);
        if (!strcmp(ctx->properties.deviceName, "MXGPU"))
            ctx->physical = devices[i];
    }
    if (!ctx->physical)
        return fail("no MXGPU physical device");
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical, &memory);
    ctx->memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount && ctx->memory_type == UINT32_MAX; i++)
        if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            ctx->memory_type = i;
    if (check(vkCreateDevice(ctx->physical, &(VkDeviceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueCount = 1, .pQueuePriorities = &(float){1.0f}}}, NULL, &ctx->device), "vkCreateDevice"))
        return -1;
    vkGetDeviceQueue(ctx->device, 0, 0, &ctx->queue);
    ctx->vertex = module(ctx, features_vertex, sizeof features_vertex);
    return check(vkCreateCommandPool(ctx->device, &(VkCommandPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0}, NULL, &ctx->pool),
            "vkCreateCommandPool") ||
        check(vkCreateFence(ctx->device, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL,
                            &ctx->fence), "vkCreateFence");
}

int main(int argc, char **argv)
{
    struct context ctx;
    uint32_t size = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 8192u;
    memset(&ctx, 0, sizeof ctx);
    int result = size < PROBE_SIZE || size % PROBE_SIZE ? fail("target size must be a multiple of 8") : setup(&ctx);
    if (!result)
        result = test_limits(&ctx);
    if (!result)
        result = test_sixteen_textures(&ctx);
    if (!result)
        result = test_large_target(&ctx, size);
    if (!result)
        result = test_render_formats(&ctx);
    if (!result)
        result = test_sampled_formats(&ctx);
    if (ctx.device) {
        vkDeviceWaitIdle(ctx.device);
        vkDestroyShaderModule(ctx.device, ctx.vertex, NULL);
        vkDestroyFence(ctx.device, ctx.fence, NULL);
        vkDestroyCommandPool(ctx.device, ctx.pool, NULL);
        vkDestroyDevice(ctx.device, NULL);
    }
    if (ctx.instance)
        vkDestroyInstance(ctx.instance, NULL);
    if (result) {
        printf("vk_webgpu FAIL: %s\n", g_failure);
        return 1;
    }
    printf("vk_webgpu PASS\n");
    return 0;
}
