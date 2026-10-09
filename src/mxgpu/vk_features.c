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
#include "features-mrt-frag.h"
#include "features-sampler-frag.h"
#include "features-color-frag.h"

struct context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t memory_type;
    VkCommandPool pool;
    VkFence fence;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceProperties properties;
    VkShaderModule vertex, mrt, sampler, color;
    VkPipelineLayout layout;
};

struct image {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    uint8_t *pixels;
};

struct buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint8_t *bytes;
};

struct pipeline_desc {
    VkRenderPass pass;
    uint32_t subpass, color_count;
    VkShaderModule fragment;
    const VkPipelineColorBlendAttachmentState *blend;
    bool depth;
    VkCompareOp compare;
    bool bias, dynamic_bias;
    float bias_constant, bias_clamp;
    VkPipelineLayout layout;
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

static int allocate(struct context *ctx, VkMemoryRequirements requirements, VkDeviceMemory *memory, uint8_t **mapped)
{
    void *pointer = NULL;
    if (check(vkAllocateMemory(ctx->device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size,
            .memoryTypeIndex = ctx->memory_type}, NULL, memory), "vkAllocateMemory") ||
        check(vkMapMemory(ctx->device, *memory, 0, VK_WHOLE_SIZE, 0, &pointer), "vkMapMemory"))
        return -1;
    *mapped = pointer;
    memset(pointer, 0, (size_t)requirements.size);
    return 0;
}

static int make_buffer(struct context *ctx, VkDeviceSize size, VkBufferUsageFlags usage, struct buffer *out)
{
    VkMemoryRequirements requirements;
    if (check(vkCreateBuffer(ctx->device, &(VkBufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage}, NULL, &out->buffer),
            "vkCreateBuffer"))
        return -1;
    vkGetBufferMemoryRequirements(ctx->device, out->buffer, &requirements);
    if (allocate(ctx, requirements, &out->memory, &out->bytes))
        return -1;
    return check(vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0), "vkBindBufferMemory");
}

static int make_image(struct context *ctx, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage,
                      VkImageAspectFlags aspect, struct image *out)
{
    VkMemoryRequirements requirements;
    if (check(vkCreateImage(ctx->device, &(VkImageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
            .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_LINEAR, .usage = usage}, NULL, &out->image), "vkCreateImage"))
        return -1;
    vkGetImageMemoryRequirements(ctx->device, out->image, &requirements);
    if (allocate(ctx, requirements, &out->memory, &out->pixels) ||
        check(vkBindImageMemory(ctx->device, out->image, out->memory, 0), "vkBindImageMemory"))
        return -1;
    return check(vkCreateImageView(ctx->device, &(VkImageViewCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = out->image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format, .subresourceRange = {aspect, 0, 1, 0, 1}}, NULL, &out->view), "vkCreateImageView");
}

static void destroy_image(struct context *ctx, struct image *image)
{
    vkDestroyImageView(ctx->device, image->view, NULL);
    vkDestroyImage(ctx->device, image->image, NULL);
    vkFreeMemory(ctx->device, image->memory, NULL);
}

static void destroy_buffer(struct context *ctx, struct buffer *buffer)
{
    vkDestroyBuffer(ctx->device, buffer->buffer, NULL);
    vkFreeMemory(ctx->device, buffer->memory, NULL);
}

static VkShaderModule module(struct context *ctx, const uint32_t *code, size_t size)
{
    VkShaderModule handle = VK_NULL_HANDLE;
    vkCreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code}, NULL, &handle);
    return handle;
}

static int make_pipeline(struct context *ctx, const struct pipeline_desc *desc, VkPipeline *out)
{
    VkPipelineColorBlendAttachmentState opaque[8];
    VkDynamicState dynamic[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
    for (uint32_t i = 0; i < 8; i++)
        opaque[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
    const VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = ctx->vertex, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = desc->fragment, .pName = "main"},
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
            .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
            .depthBiasEnable = desc->bias, .depthBiasConstantFactor = desc->bias_constant,
            .depthBiasClamp = desc->bias_clamp},
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
        .pDepthStencilState = desc->depth ? &(VkPipelineDepthStencilStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE, .depthCompareOp = desc->compare} : NULL,
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = desc->color_count,
            .pAttachments = desc->blend ? desc->blend : opaque},
        .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = desc->dynamic_bias ? 3 : 2, .pDynamicStates = dynamic},
        .layout = desc->layout ? desc->layout : ctx->layout, .renderPass = desc->pass, .subpass = desc->subpass
    }, NULL, out), "vkCreateGraphicsPipelines");
}

static int make_pass(struct context *ctx, uint32_t color_count, const VkFormat *formats, VkFormat depth,
                     uint32_t subpass_count, VkRenderPass *out)
{
    VkAttachmentDescription attachments[9];
    VkAttachmentReference colors[8], depth_ref = {color_count, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpasses[2];
    uint32_t count = color_count + (depth != VK_FORMAT_UNDEFINED);
    for (uint32_t i = 0; i < count; i++)
        attachments[i] = (VkAttachmentDescription){
            .format = i < color_count ? formats[i] : depth, .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL};
    for (uint32_t i = 0; i < color_count; i++)
        colors[i] = (VkAttachmentReference){i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    if (subpass_count == 2) {
        subpasses[0] = (VkSubpassDescription){.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                              .colorAttachmentCount = 1, .pColorAttachments = &colors[0]};
        subpasses[1] = (VkSubpassDescription){.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                              .colorAttachmentCount = 1, .pColorAttachments = &colors[1]};
    } else {
        subpasses[0] = (VkSubpassDescription){.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = color_count, .pColorAttachments = colors,
            .pDepthStencilAttachment = depth != VK_FORMAT_UNDEFINED ? &depth_ref : NULL};
    }
    return check(vkCreateRenderPass(ctx->device, &(VkRenderPassCreateInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = count, .pAttachments = attachments,
        .subpassCount = subpass_count, .pSubpasses = subpasses,
        .dependencyCount = subpass_count == 2 ? 1 : 0,
        .pDependencies = &(VkSubpassDependency){0, 1, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0}}, NULL, out), "vkCreateRenderPass");
}

static int make_framebuffer(struct context *ctx, VkRenderPass pass, const VkImageView *views, uint32_t count,
                            uint32_t width, uint32_t height, VkFramebuffer *out)
{
    return check(vkCreateFramebuffer(ctx->device, &(VkFramebufferCreateInfo){
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass, .attachmentCount = count,
        .pAttachments = views, .width = width, .height = height, .layers = 1}, NULL, out), "vkCreateFramebuffer");
}

static VkCommandBuffer begin(struct context *ctx)
{
    VkCommandBuffer command = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx->device, &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ctx->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &command);
    vkBeginCommandBuffer(command, &(VkCommandBufferBeginInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT});
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

static void viewport(VkCommandBuffer command, uint32_t width, uint32_t height)
{
    vkCmdSetViewport(command, 0, 1, &(VkViewport){0, 0, (float)width, (float)height, 0, 1});
    vkCmdSetScissor(command, 0, 1, &(VkRect2D){{0, 0}, {width, height}});
}

static void push_depth(struct context *ctx, VkCommandBuffer command, float depth)
{
    vkCmdPushConstants(command, ctx->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof depth, &depth);
}

static int near_pixel(const uint8_t *pixel, int r, int g, int b, int a)
{
    return abs(pixel[0] - r) <= 1 && abs(pixel[1] - g) <= 1 && abs(pixel[2] - b) <= 1 && abs(pixel[3] - a) <= 1;
}

static int pixel_failure(const char *what, uint32_t x, uint32_t y, const uint8_t *pixel)
{
    snprintf(g_failure, sizeof g_failure, "%s: pixel (%u,%u) is %u %u %u %u", what, x, y,
             pixel[0], pixel[1], pixel[2], pixel[3]);
    return -1;
}

static int test_mrt(struct context *ctx)
{
    const VkFormat formats[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    const VkPipelineColorBlendAttachmentState blend[3] = {
        {.colorWriteMask = 0xf},
        {.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
         .colorBlendOp = VK_BLEND_OP_ADD, .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
         .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf},
        {.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
         .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
         .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
         .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf},
    };
    const VkClearValue clears[3] = {
        {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}}, {.color = {{0.0f, 0.25f, 0.0f, 0.0f}}}, {.color = {{1.0f, 1.0f, 1.0f, 1.0f}}}};
    struct image images[3];
    VkImageView views[3];
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipeline pipeline;
    for (uint32_t i = 0; i < 3; i++) {
        if (make_image(ctx, formats[i], 4, 4, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &images[i]))
            return -1;
        views[i] = images[i].view;
    }
    if (make_pass(ctx, 3, formats, VK_FORMAT_UNDEFINED, 1, &pass) ||
        make_framebuffer(ctx, pass, views, 3, 4, 4, &framebuffer) ||
        make_pipeline(ctx, &(struct pipeline_desc){.pass = pass, .color_count = 3, .fragment = ctx->mrt,
                                                   .blend = blend}, &pipeline))
        return -1;
    VkCommandBuffer command = begin(ctx);
    vkCmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {4, 4}}, .clearValueCount = 3, .pClearValues = clears}, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    viewport(command, 4, 4);
    push_depth(ctx, command, 0.5f);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdClearAttachments(command, 1, &(VkClearAttachment){VK_IMAGE_ASPECT_COLOR_BIT, 1,
                          {.color = {{1.0f, 1.0f, 0.0f, 1.0f}}}}, 1, &(VkClearRect){{{1, 1}, {2, 2}}, 0, 1});
    vkCmdEndRenderPass(command);
    if (submit(ctx, command, "MRT submission"))
        return -1;
    for (uint32_t y = 0; y < 4; y++)
        for (uint32_t x = 0; x < 4; x++) {
            const uint8_t *a = images[0].pixels + (y * 4 + x) * 4, *b = images[1].pixels + (y * 4 + x) * 4;
            const uint8_t *c = images[2].pixels + (y * 4 + x) * 4;
            bool cleared = x >= 1 && x < 3 && y >= 1 && y < 3;
            if (!near_pixel(a, 255, 0, 0, 255))
                return pixel_failure("attachment 0", x, y, a);
            if (cleared ? !near_pixel(b, 255, 255, 0, 255) : !near_pixel(b, 0, 191, 0, 255))
                return pixel_failure("attachment 1 (additive blend, vkCmdClearAttachments)", x, y, b);
            if (!near_pixel(c, 128, 128, 255, 128))
                return pixel_failure("attachment 2 (alpha blend)", x, y, c);
        }
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyFramebuffer(ctx->device, framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, pass, NULL);
    for (uint32_t i = 0; i < 3; i++)
        destroy_image(ctx, &images[i]);
    printf("3 colour attachments with independent blend states and vkCmdClearAttachments: rendered\n");
    return 0;
}

static int test_separate_sampler(struct context *ctx)
{
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    const uint8_t texels[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
    struct image texture, target;
    struct buffer staging;
    VkSampler sampler;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet set;
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipeline pipeline;
    if (make_image(ctx, format, 2, 2, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   VK_IMAGE_ASPECT_COLOR_BIT, &texture) ||
        make_image(ctx, format, 8, 8, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &target) ||
        make_buffer(ctx, sizeof texels, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging) ||
        check(vkCreateSampler(ctx->device, &(VkSamplerCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod = 0.25f}, NULL, &sampler), "vkCreateSampler"))
        return -1;
    memcpy(staging.bytes, texels, sizeof texels);
    const VkDescriptorSetLayoutBinding bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL}};
    if (check(vkCreateDescriptorSetLayout(ctx->device, &(VkDescriptorSetLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bindings},
            NULL, &set_layout), "vkCreateDescriptorSetLayout") ||
        check(vkCreatePipelineLayout(ctx->device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &set_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_VERTEX_BIT, 0, 4}},
            NULL, &layout), "vkCreatePipelineLayout") ||
        check(vkCreateDescriptorPool(ctx->device, &(VkDescriptorPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 2,
            .pPoolSizes = (VkDescriptorPoolSize[]){{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}}},
            NULL, &descriptor_pool), "vkCreateDescriptorPool") ||
        check(vkAllocateDescriptorSets(ctx->device, &(VkDescriptorSetAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1, .pSetLayouts = &set_layout}, &set), "vkAllocateDescriptorSets"))
        return -1;
    vkUpdateDescriptorSets(ctx->device, 2, (VkWriteDescriptorSet[]){
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .pImageInfo = &(VkDescriptorImageInfo){VK_NULL_HANDLE, texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
         .pImageInfo = &(VkDescriptorImageInfo){sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}}}, 0, NULL);
    if (make_pass(ctx, 1, &format, VK_FORMAT_UNDEFINED, 1, &pass) ||
        make_framebuffer(ctx, pass, &target.view, 1, 8, 8, &framebuffer) ||
        make_pipeline(ctx, &(struct pipeline_desc){.pass = pass, .color_count = 1, .fragment = ctx->sampler,
                                                   .layout = layout}, &pipeline))
        return -1;
    VkCommandBuffer command = begin(ctx);
    vkCmdCopyBufferToImage(command, staging.buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &(VkBufferImageCopy){.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                                .imageExtent = {2, 2, 1}});
    vkCmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {8, 8}}, .clearValueCount = 1,
        .pClearValues = &(VkClearValue){.color = {{0, 0, 0, 0}}}}, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    viewport(command, 8, 8);
    float depth = 0.5f;
    vkCmdPushConstants(command, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof depth, &depth);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
    if (submit(ctx, command, "separate sampler submission"))
        return -1;
    for (uint32_t y = 0; y < 8; y++)
        for (uint32_t x = 0; x < 8; x++) {
            const uint8_t *pixel = target.pixels + (y * 8 + x) * 4;
            const uint8_t *texel = texels + ((y / 4) * 2 + x / 4) * 4;
            if (!near_pixel(pixel, texel[0], texel[1], texel[2], texel[3]))
                return pixel_failure("separate sampled image and sampler with textureSize", x, y, pixel);
        }
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyFramebuffer(ctx->device, framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, pass, NULL);
    vkDestroyDescriptorPool(ctx->device, descriptor_pool, NULL);
    vkDestroyPipelineLayout(ctx->device, layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, set_layout, NULL);
    vkDestroySampler(ctx->device, sampler, NULL);
    destroy_buffer(ctx, &staging);
    destroy_image(ctx, &texture);
    destroy_image(ctx, &target);
    printf("SAMPLED_IMAGE and SAMPLER descriptors with textureSize: sampled 2x2 texels exactly\n");
    return 0;
}

static int depth_draw(struct context *ctx, bool dynamic, float constant, float clamp, float *out)
{
    const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    struct image color, depth;
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipeline pipeline;
    if (make_image(ctx, color_format, 4, 4, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &color) ||
        make_image(ctx, VK_FORMAT_D32_SFLOAT, 4, 4, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                   VK_IMAGE_ASPECT_DEPTH_BIT, &depth) ||
        make_pass(ctx, 1, &color_format, VK_FORMAT_D32_SFLOAT, 1, &pass) ||
        make_framebuffer(ctx, pass, (VkImageView[]){color.view, depth.view}, 2, 4, 4, &framebuffer) ||
        make_pipeline(ctx, &(struct pipeline_desc){.pass = pass, .color_count = 1, .fragment = ctx->color,
            .depth = true, .compare = VK_COMPARE_OP_LESS, .bias = true, .dynamic_bias = dynamic,
            .bias_constant = dynamic ? 0.0f : constant, .bias_clamp = dynamic ? 0.0f : clamp}, &pipeline))
        return -1;
    VkCommandBuffer command = begin(ctx);
    vkCmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {4, 4}}, .clearValueCount = 2,
        .pClearValues = (VkClearValue[]){{.color = {{0, 0, 0, 1}}}, {.depthStencil = {1.0f, 0}}}},
        VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    viewport(command, 4, 4);
    if (dynamic)
        vkCmdSetDepthBias(command, constant, clamp, 0.0f);
    push_depth(ctx, command, 0.5f);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
    int result = submit(ctx, command, "depth bias submission");
    if (!result)
        memcpy(out, depth.pixels + (1 * 4 + 1) * 4, sizeof *out);
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyFramebuffer(ctx->device, framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, pass, NULL);
    destroy_image(ctx, &color);
    destroy_image(ctx, &depth);
    return result;
}

static int test_depth_bias(struct context *ctx)
{
    float plain, biased, clamped;
    const float unit = ldexpf(1.0f, -24);
    if (depth_draw(ctx, false, 0.0f, 0.0f, &plain) || depth_draw(ctx, false, 1000.0f, 0.0f, &biased))
        return -1;
    if (fabsf(plain - 0.5f) > 1e-6f || fabsf(biased - (0.5f + 1000.0f * unit)) > 4.0f * unit) {
        snprintf(g_failure, sizeof g_failure, "depth bias 1000 gave depth %.9f (unbiased %.9f, expected %.9f)",
                 biased, plain, 0.5f + 1000.0f * unit);
        return -1;
    }
    printf("depthBiasConstantFactor 1000 on D32_SFLOAT at z 0.5: depth %.9f (expected %.9f)\n",
           biased, 0.5f + 1000.0f * unit);
    if (!ctx->features.depthBiasClamp)
        return 0;
    if (depth_draw(ctx, true, 1000.0f, 1e-5f, &clamped))
        return -1;
    if (fabsf(clamped - (0.5f + 1e-5f)) > 4.0f * unit) {
        snprintf(g_failure, sizeof g_failure, "depth bias clamp 1e-5 gave depth %.9f", clamped);
        return -1;
    }
    printf("vkCmdSetDepthBias 1000 clamped to 1e-5: depth %.9f\n", clamped);
    return 0;
}

static int colour_pass(struct context *ctx, uint32_t size, struct image *target, VkRenderPass *pass,
                       VkFramebuffer *framebuffer, VkPipeline *pipeline)
{
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    if (make_image(ctx, format, size, size, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, target) ||
        make_pass(ctx, 1, &format, VK_FORMAT_UNDEFINED, 1, pass) ||
        make_framebuffer(ctx, *pass, &target->view, 1, size, size, framebuffer))
        return -1;
    return pipeline ? make_pipeline(ctx, &(struct pipeline_desc){.pass = *pass, .color_count = 1,
                                                                 .fragment = ctx->color}, pipeline) : 0;
}

static void begin_pass(VkCommandBuffer command, VkRenderPass pass, VkFramebuffer framebuffer, uint32_t size,
                       VkSubpassContents contents)
{
    vkCmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {size, size}}, .clearValueCount = 2,
        .pClearValues = (VkClearValue[]){{.color = {{1, 0, 0, 1}}}, {.color = {{1, 0, 0, 1}}}}}, contents);
}

static int all_pixels(const struct image *image, uint32_t size, int r, int g, int b, int a, const char *what)
{
    for (uint32_t y = 0; y < size; y++)
        for (uint32_t x = 0; x < size; x++) {
            const uint8_t *pixel = image->pixels + (y * size + x) * 4;
            if (!near_pixel(pixel, r, g, b, a))
                return pixel_failure(what, x, y, pixel);
        }
    return 0;
}

static int test_indirect_and_queries(struct context *ctx)
{
    struct image target;
    struct buffer arguments, indices, results;
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipeline pipeline;
    VkQueryPool occlusion, timestamps;
    if (colour_pass(ctx, 4, &target, &pass, &framebuffer, &pipeline) ||
        make_buffer(ctx, 64, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &arguments) ||
        make_buffer(ctx, 16, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &indices) ||
        make_buffer(ctx, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &results) ||
        check(vkCreateQueryPool(ctx->device, &(VkQueryPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = 2},
            NULL, &occlusion), "vkCreateQueryPool occlusion") ||
        check(vkCreateQueryPool(ctx->device, &(VkQueryPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2},
            NULL, &timestamps), "vkCreateQueryPool timestamp"))
        return -1;
    const VkDrawIndirectCommand draws[2] = {{0, 1, 0, 0}, {3, 1, 0, 0}};
    const uint32_t index_values[3] = {0, 1, 2};
    memcpy(arguments.bytes, draws, sizeof draws);
    memcpy(arguments.bytes + 32, &(VkDrawIndexedIndirectCommand){3, 1, 0, 0, 0}, sizeof(VkDrawIndexedIndirectCommand));
    memcpy(indices.bytes, index_values, sizeof index_values);
    VkCommandBuffer command = begin(ctx);
    vkCmdResetQueryPool(command, occlusion, 0, 2);
    vkCmdResetQueryPool(command, timestamps, 0, 2);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 0);
    begin_pass(command, pass, framebuffer, 4, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    push_depth(ctx, command, 0.5f);
    vkCmdSetViewport(command, 0, 1, &(VkViewport){0, 0, 4, 4, 0, 1});
    vkCmdSetScissor(command, 0, 1, &(VkRect2D){{0, 0}, {4, 2}});
    vkCmdBeginQuery(command, occlusion, 0, 0);
    vkCmdDrawIndirect(command, arguments.buffer, 0, ctx->features.multiDrawIndirect ? 2 : 1,
                      sizeof(VkDrawIndirectCommand));
    if (!ctx->features.multiDrawIndirect)
        vkCmdDrawIndirect(command, arguments.buffer, sizeof(VkDrawIndirectCommand), 1, sizeof(VkDrawIndirectCommand));
    vkCmdEndQuery(command, occlusion, 0);
    vkCmdSetScissor(command, 0, 1, &(VkRect2D){{0, 2}, {4, 2}});
    vkCmdBeginQuery(command, occlusion, 1, 0);
    vkCmdBindIndexBuffer(command, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(command, arguments.buffer, 32, 1, sizeof(VkDrawIndexedIndirectCommand));
    vkCmdEndQuery(command, occlusion, 1);
    vkCmdEndRenderPass(command);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 1);
    vkCmdCopyQueryPoolResults(command, occlusion, 0, 2, results.buffer, 0, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (submit(ctx, command, "indirect draw and query submission") ||
        all_pixels(&target, 4, 0, 255, 0, 255, "vkCmdDrawIndirect and vkCmdDrawIndexedIndirect"))
        return -1;
    uint64_t samples[2], times[2], copied[4];
    if (check(vkGetQueryPoolResults(ctx->device, occlusion, 0, 2, sizeof samples, samples, 8,
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "occlusion results") ||
        check(vkGetQueryPoolResults(ctx->device, timestamps, 0, 2, sizeof times, times, 8,
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "timestamp results"))
        return -1;
    memcpy(copied, results.bytes, sizeof copied);
    if (samples[0] != 8 || samples[1] != 8 || copied[0] != 8 || copied[1] != 1 || copied[2] != 8 || copied[3] != 1) {
        snprintf(g_failure, sizeof g_failure, "occlusion counts %llu %llu, copied %llu/%llu %llu/%llu",
                 (unsigned long long)samples[0], (unsigned long long)samples[1], (unsigned long long)copied[0],
                 (unsigned long long)copied[1], (unsigned long long)copied[2], (unsigned long long)copied[3]);
        return -1;
    }
    if (times[1] < times[0]) {
        snprintf(g_failure, sizeof g_failure, "timestamps decrease: %llu then %llu",
                 (unsigned long long)times[0], (unsigned long long)times[1]);
        return -1;
    }
    printf("indirect draw (2 records), indexed indirect draw: rendered; occlusion samples %llu and %llu; "
           "timestamps %llu ns apart\n", (unsigned long long)samples[0], (unsigned long long)samples[1],
           (unsigned long long)(times[1] - times[0]));
    vkDestroyQueryPool(ctx->device, occlusion, NULL);
    vkDestroyQueryPool(ctx->device, timestamps, NULL);
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyFramebuffer(ctx->device, framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, pass, NULL);
    destroy_buffer(ctx, &arguments);
    destroy_buffer(ctx, &indices);
    destroy_buffer(ctx, &results);
    destroy_image(ctx, &target);
    return 0;
}

static int test_transfers(struct context *ctx)
{
    struct image source, destination, small, depth;
    struct buffer buffer;
    VkEvent host_event, device_event;
    const uint32_t update[4] = {11, 22, 33, 44};
    if (make_image(ctx, VK_FORMAT_R8G8B8A8_UNORM, 4, 4, VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &source) ||
        make_image(ctx, VK_FORMAT_R8G8B8A8_UNORM, 4, 4, VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &destination) ||
        make_image(ctx, VK_FORMAT_R8G8B8A8_UNORM, 2, 2, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   VK_IMAGE_ASPECT_COLOR_BIT, &small) ||
        make_image(ctx, VK_FORMAT_D32_SFLOAT_S8_UINT, 2, 2, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   VK_IMAGE_ASPECT_DEPTH_BIT, &depth) ||
        make_buffer(ctx, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer) ||
        check(vkCreateEvent(ctx->device, &(VkEventCreateInfo){.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO}, NULL,
                            &host_event), "vkCreateEvent") ||
        check(vkCreateEvent(ctx->device, &(VkEventCreateInfo){.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO}, NULL,
                            &device_event), "vkCreateEvent"))
        return -1;
    const VkImageSubresourceRange colour_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageSubresourceLayers layers = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    if (check(vkSetEvent(ctx->device, host_event), "vkSetEvent"))
        return -1;
    VkCommandBuffer command = begin(ctx);
    vkCmdWaitEvents(command, 1, &host_event, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, NULL, 0, NULL, 0, NULL);
    vkCmdFillBuffer(command, buffer.buffer, 0, VK_WHOLE_SIZE, 0xa5a5a5a5u);
    vkCmdUpdateBuffer(command, buffer.buffer, 16, sizeof update, update);
    vkCmdClearColorImage(command, source.image, VK_IMAGE_LAYOUT_GENERAL,
                         &(VkClearColorValue){{0.0f, 0.0f, 1.0f, 1.0f}}, 1, &colour_range);
    vkCmdClearColorImage(command, destination.image, VK_IMAGE_LAYOUT_GENERAL,
                         &(VkClearColorValue){{0.0f, 0.0f, 0.0f, 0.0f}}, 1, &colour_range);
    vkCmdCopyImage(command, source.image, VK_IMAGE_LAYOUT_GENERAL, destination.image, VK_IMAGE_LAYOUT_GENERAL, 1,
                   &(VkImageCopy){layers, {0, 0, 0}, layers, {2, 2, 0}, {2, 2, 1}});
    vkCmdBlitImage(command, destination.image, VK_IMAGE_LAYOUT_GENERAL, small.image, VK_IMAGE_LAYOUT_GENERAL, 1,
                   &(VkImageBlit){layers, {{0, 0, 0}, {4, 4, 1}}, layers, {{0, 0, 0}, {2, 2, 1}}}, VK_FILTER_LINEAR);
    vkCmdClearDepthStencilImage(command, depth.image, VK_IMAGE_LAYOUT_GENERAL, &(VkClearDepthStencilValue){0.25f, 7},
                                1, &(VkImageSubresourceRange){VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                                                              0, 1, 0, 1});
    vkCmdSetEvent(command, device_event, VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (submit(ctx, command, "transfer submission"))
        return -1;
    const uint32_t *words = (const uint32_t *)buffer.bytes;
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t expected = i >= 4 && i < 8 ? update[i - 4] : 0xa5a5a5a5u;
        if (words[i] != expected) {
            snprintf(g_failure, sizeof g_failure, "buffer word %u is 0x%x, expected 0x%x", i, words[i], expected);
            return -1;
        }
    }
    for (uint32_t y = 0; y < 4; y++)
        for (uint32_t x = 0; x < 4; x++) {
            const uint8_t *pixel = destination.pixels + (y * 4 + x) * 4;
            if (x >= 2 && y >= 2 ? !near_pixel(pixel, 0, 0, 255, 255) : !near_pixel(pixel, 0, 0, 0, 0))
                return pixel_failure("vkCmdClearColorImage and vkCmdCopyImage", x, y, pixel);
        }
    if (!near_pixel(small.pixels, 0, 0, 0, 0) || !near_pixel(small.pixels + 12, 0, 0, 255, 255))
        return pixel_failure("vkCmdBlitImage 4x4 to 2x2", 0, 0, small.pixels);
    float depth_value;
    memcpy(&depth_value, depth.pixels, sizeof depth_value);
    if (depth_value != 0.25f || depth.pixels[4] != 7) {
        snprintf(g_failure, sizeof g_failure, "vkCmdClearDepthStencilImage wrote depth %f stencil %u",
                 depth_value, depth.pixels[4]);
        return -1;
    }
    if (vkGetEventStatus(ctx->device, device_event) != VK_EVENT_SET ||
        vkResetEvent(ctx->device, device_event) != VK_SUCCESS ||
        vkGetEventStatus(ctx->device, device_event) != VK_EVENT_RESET)
        return fail("event set by vkCmdSetEvent was not observed or could not be reset");
    VkSubresourceLayout layout;
    VkDeviceSize committed = 0;
    vkGetImageSubresourceLayout(ctx->device, source.image, &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0},
                                &layout);
    vkGetDeviceMemoryCommitment(ctx->device, source.memory, &committed);
    VkExtent2D granularity;
    VkBufferView texel_view;
    vkGetRenderAreaGranularity(ctx->device, VK_NULL_HANDLE, &granularity);
    if (layout.rowPitch != 16 || layout.size != 64 || committed < 64 || granularity.width != 1 ||
        check(vkFlushMappedMemoryRanges(ctx->device, 1, &(VkMappedMemoryRange){
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = buffer.memory, .size = VK_WHOLE_SIZE}),
            "vkFlushMappedMemoryRanges") ||
        check(vkInvalidateMappedMemoryRanges(ctx->device, 1, &(VkMappedMemoryRange){
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = buffer.memory, .size = VK_WHOLE_SIZE}),
            "vkInvalidateMappedMemoryRanges") ||
        check(vkCreateBufferView(ctx->device, &(VkBufferViewCreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = buffer.buffer,
            .format = VK_FORMAT_R32_UINT, .range = VK_WHOLE_SIZE}, NULL, &texel_view), "vkCreateBufferView"))
        return g_failure[0] ? -1 : fail("subresource layout, memory commitment or render area granularity is wrong");
    vkDestroyBufferView(ctx->device, texel_view, NULL);
    vkDestroyEvent(ctx->device, host_event, NULL);
    vkDestroyEvent(ctx->device, device_event, NULL);
    destroy_buffer(ctx, &buffer);
    destroy_image(ctx, &source);
    destroy_image(ctx, &destination);
    destroy_image(ctx, &small);
    destroy_image(ctx, &depth);
    printf("events, fill, update, clear colour and depth-stencil images, copy and linear blit: verified\n");
    return 0;
}

static int test_subpasses(struct context *ctx)
{
    const VkFormat formats[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    struct image images[2];
    VkRenderPass pass;
    VkFramebuffer framebuffer;
    VkPipeline first, second;
    VkCommandBuffer secondary;
    for (uint32_t i = 0; i < 2; i++)
        if (make_image(ctx, formats[i], 4, 4, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                       &images[i]))
            return -1;
    if (make_pass(ctx, 2, formats, VK_FORMAT_UNDEFINED, 2, &pass) ||
        make_framebuffer(ctx, pass, (VkImageView[]){images[0].view, images[1].view}, 2, 4, 4, &framebuffer) ||
        make_pipeline(ctx, &(struct pipeline_desc){.pass = pass, .subpass = 0, .color_count = 1,
                                                   .fragment = ctx->color}, &first) ||
        make_pipeline(ctx, &(struct pipeline_desc){.pass = pass, .subpass = 1, .color_count = 1,
                                                   .fragment = ctx->color}, &second) ||
        check(vkAllocateCommandBuffers(ctx->device, &(VkCommandBufferAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ctx->pool,
            .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = 1}, &secondary),
            "vkAllocateCommandBuffers secondary"))
        return -1;
    vkBeginCommandBuffer(secondary, &(VkCommandBufferBeginInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        .pInheritanceInfo = &(VkCommandBufferInheritanceInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, .renderPass = pass, .subpass = 1}});
    vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, second);
    viewport(secondary, 4, 4);
    push_depth(ctx, secondary, 0.5f);
    vkCmdDraw(secondary, 3, 1, 0, 0);
    if (check(vkEndCommandBuffer(secondary), "secondary vkEndCommandBuffer"))
        return -1;
    VkCommandBuffer command = begin(ctx);
    begin_pass(command, pass, framebuffer, 4, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, first);
    vkCmdSetViewport(command, 0, 1, &(VkViewport){0, 0, 4, 4, 0, 1});
    vkCmdSetScissor(command, 0, 1, &(VkRect2D){{0, 0}, {2, 4}});
    push_depth(ctx, command, 0.5f);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdNextSubpass(command, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    vkCmdExecuteCommands(command, 1, &secondary);
    vkCmdEndRenderPass(command);
    if (submit(ctx, command, "subpass submission"))
        return -1;
    for (uint32_t y = 0; y < 4; y++)
        for (uint32_t x = 0; x < 4; x++) {
            const uint8_t *a = images[0].pixels + (y * 4 + x) * 4, *b = images[1].pixels + (y * 4 + x) * 4;
            if (x < 2 ? !near_pixel(a, 0, 255, 0, 255) : !near_pixel(a, 255, 0, 0, 255))
                return pixel_failure("subpass 0 with scissor", x, y, a);
            if (!near_pixel(b, 0, 255, 0, 255))
                return pixel_failure("subpass 1 from a secondary command buffer", x, y, b);
        }
    vkFreeCommandBuffers(ctx->device, ctx->pool, 1, &secondary);
    vkDestroyPipeline(ctx->device, first, NULL);
    vkDestroyPipeline(ctx->device, second, NULL);
    vkDestroyFramebuffer(ctx->device, framebuffer, NULL);
    vkDestroyRenderPass(ctx->device, pass, NULL);
    for (uint32_t i = 0; i < 2; i++)
        destroy_image(ctx, &images[i]);
    printf("two subpasses, vkCmdNextSubpass and vkCmdExecuteCommands with an inherited render pass: rendered\n");
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
    vkGetPhysicalDeviceFeatures(ctx->physical, &ctx->features);
    const VkPhysicalDeviceLimits *limits = &ctx->properties.limits;
    printf("maxColorAttachments %u independentBlend %u depthBiasClamp %u multiDrawIndirect %u "
           "maxPerStageDescriptorSampledImages %u maxPerStageDescriptorSamplers %u timestampPeriod %.1f\n",
           limits->maxColorAttachments, ctx->features.independentBlend, ctx->features.depthBiasClamp,
           ctx->features.multiDrawIndirect, limits->maxPerStageDescriptorSampledImages,
           limits->maxPerStageDescriptorSamplers, limits->timestampPeriod);
    if (limits->maxColorAttachments < 3 || !ctx->features.independentBlend)
        return fail("three colour attachments with independent blending are not reported");
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical, &memory);
    ctx->memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount && ctx->memory_type == UINT32_MAX; i++)
        if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            ctx->memory_type = i;
    VkPhysicalDeviceFeatures enabled = {
        .independentBlend = VK_TRUE, .depthBiasClamp = ctx->features.depthBiasClamp,
        .multiDrawIndirect = ctx->features.multiDrawIndirect};
    if (check(vkCreateDevice(ctx->physical, &(VkDeviceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueCount = 1, .pQueuePriorities = &(float){1.0f}}, .pEnabledFeatures = &enabled}, NULL,
            &ctx->device), "vkCreateDevice"))
        return -1;
    vkGetDeviceQueue(ctx->device, 0, 0, &ctx->queue);
    ctx->vertex = module(ctx, features_vertex, sizeof features_vertex);
    ctx->mrt = module(ctx, features_mrt, sizeof features_mrt);
    ctx->sampler = module(ctx, features_sampler, sizeof features_sampler);
    ctx->color = module(ctx, features_color, sizeof features_color);
    return check(vkCreateCommandPool(ctx->device, &(VkCommandPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0}, NULL, &ctx->pool),
            "vkCreateCommandPool") ||
        check(vkCreateFence(ctx->device, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL,
                            &ctx->fence), "vkCreateFence") ||
        check(vkCreatePipelineLayout(ctx->device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_VERTEX_BIT, 0, 4}}, NULL, &ctx->layout),
            "vkCreatePipelineLayout");
}

int main(void)
{
    struct context ctx;
    memset(&ctx, 0, sizeof ctx);
    int result = setup(&ctx);
    if (!result)
        result = test_mrt(&ctx);
    if (!result)
        result = test_separate_sampler(&ctx);
    if (!result)
        result = test_depth_bias(&ctx);
    if (!result)
        result = test_indirect_and_queries(&ctx);
    if (!result)
        result = test_transfers(&ctx);
    if (!result)
        result = test_subpasses(&ctx);
    if (ctx.device) {
        vkDeviceWaitIdle(ctx.device);
        vkDestroyShaderModule(ctx.device, ctx.vertex, NULL);
        vkDestroyShaderModule(ctx.device, ctx.mrt, NULL);
        vkDestroyShaderModule(ctx.device, ctx.sampler, NULL);
        vkDestroyShaderModule(ctx.device, ctx.color, NULL);
        vkDestroyPipelineLayout(ctx.device, ctx.layout, NULL);
        vkDestroyFence(ctx.device, ctx.fence, NULL);
        vkDestroyCommandPool(ctx.device, ctx.pool, NULL);
        vkDestroyDevice(ctx.device, NULL);
    }
    if (ctx.instance)
        vkDestroyInstance(ctx.instance, NULL);
    if (result) {
        printf("vk_features FAIL: %s\n", g_failure);
        return 1;
    }
    printf("vk_features PASS\n");
    return 0;
}
