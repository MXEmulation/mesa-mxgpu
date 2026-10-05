/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "scene_common.h"
#include "vk_frame.h"
#include "mxgpu_wire.h"
#include "scene-vert.h"
#include "scene-frag.h"
#include "scene-zero-frag.h"

#include <stdlib.h>

#define INSTANCE_FUNCTIONS(X) \
    X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
    X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) \
    X(CreateDevice) X(GetDeviceProcAddr)
#define DEVICE_FUNCTIONS(X) \
    X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer) \
    X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
    X(MapMemory) X(UnmapMemory) X(CreateImage) X(DestroyImage) \
    X(GetImageMemoryRequirements) X(BindImageMemory) X(CreateImageView) X(DestroyImageView) \
    X(CreateSampler) X(DestroySampler) X(CreateShaderModule) X(DestroyShaderModule) \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreateDescriptorPool) \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateRenderPass) X(DestroyRenderPass) \
    X(CreateGraphicsPipelines) X(DestroyPipeline) X(CreateFramebuffer) X(DestroyFramebuffer) \
    X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) \
    X(CmdPipelineBarrier) X(CmdCopyBufferToImage) X(CmdBeginRenderPass) X(CmdBindPipeline) \
    X(CmdBindVertexBuffers) X(CmdBindDescriptorSets) X(CmdDraw) X(CmdEndRenderPass) \
    X(CmdCopyImageToBuffer) X(EndCommandBuffer) X(CreateFence) X(DestroyFence) \
    X(QueueSubmit) X(WaitForFences) X(DeviceWaitIdle)

struct scene_api {
#define DECLARE_FUNCTION(name) PFN_vk##name name;
    INSTANCE_FUNCTIONS(DECLARE_FUNCTION)
    DEVICE_FUNCTIONS(DECLARE_FUNCTION)
#undef DECLARE_FUNCTION
};

static int memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                        uint32_t bits, VkMemoryPropertyFlags required, uint32_t *index)
{
    for (uint32_t i = 0; i < properties->memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (properties->memoryTypes[i].propertyFlags & required) == required) {
            *index = i;
            return 1;
        }
    }
    return 0;
}

static int frame_with_api(PFN_vkGetInstanceProcAddr get, int constant_uv,
                           int rejected_device, unsigned char *pixels)
{
    struct scene_api api = {0};
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDevice *devices = NULL;
    VkQueueFamilyProperties *families = NULL;
    VkBuffer buffers[3] = {0};
    VkDeviceMemory buffer_memory[3] = {0};
    VkImage images[2] = {0};
    VkDeviceMemory image_memory[2] = {0};
    VkImageView views[2] = {0};
    VkShaderModule shaders[2] = {0};
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory;
    void *mapped = NULL;
    VkDeviceMemory mapped_memory = VK_NULL_HANDLE;
    uint32_t count = 0, family = UINT32_MAX;
    int submitted = 0, success = 0;
    if (!get || !pixels)
        return 0;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)get(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create)
        return 0;
#define CHECK(call) do { \
    VkResult result = (call); \
    if (result != VK_SUCCESS) { \
        fprintf(stderr, "%s returned %d\n", #call, result); \
        goto cleanup; \
    } \
} while (0)
    CHECK(create(&(VkInstanceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .apiVersion = VK_API_VERSION_1_0,
            .pApplicationName = "MXGPU scene validation"
        }
    }, NULL, &instance));
#define LOAD_INSTANCE(name) do { \
    api.name = (PFN_vk##name)get(instance, "vk" #name); \
    if (!api.name) { fprintf(stderr, "Missing vk%s\n", #name); goto cleanup; } \
} while (0);
    INSTANCE_FUNCTIONS(LOAD_INSTANCE)
#undef LOAD_INSTANCE
    CHECK(api.EnumeratePhysicalDevices(instance, &count, NULL));
    if (!count || count > 256)
        goto cleanup;
    devices = calloc(count, sizeof(*devices));
    if (!devices)
        goto cleanup;
    CHECK(api.EnumeratePhysicalDevices(instance, &count, devices));
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties properties;
        api.GetPhysicalDeviceProperties(devices[i], &properties);
        if (properties.vendorID == MX_PCI_VENDOR_ID &&
            strcmp(properties.deviceName, "MXGPU") == 0) {
            physical = devices[i];
            break;
        }
    }
    if (!physical)
        goto cleanup;
    api.GetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    if (!count || count > 256)
        goto cleanup;
    families = calloc(count, sizeof(*families));
    if (!families)
        goto cleanup;
    api.GetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            family = i;
            break;
        }
    }
    if (family == UINT32_MAX)
        goto cleanup;
    float priority = 1.0f;
    if (rejected_device) {
        const char *unsupported = "VK_MX_scene_validation_unavailable";
        VkDevice rejected = VK_NULL_HANDLE;
        VkResult result = api.CreateDevice(physical, &(VkDeviceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family, .queueCount = 1,
                .pQueuePriorities = &priority
            },
            .enabledExtensionCount = 1, .ppEnabledExtensionNames = &unsupported
        }, NULL, &rejected);
        if (result != VK_ERROR_EXTENSION_NOT_PRESENT || rejected != VK_NULL_HANDLE) {
            if (rejected) {
                PFN_vkDestroyDevice destroy = (PFN_vkDestroyDevice)
                    api.GetDeviceProcAddr(rejected, "vkDestroyDevice");
                if (destroy)
                    destroy(rejected, NULL);
            }
            fprintf(stderr, "Unsupported device extension returned %d\n", result);
            goto cleanup;
        }
    }
    CHECK(api.CreateDevice(physical, &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &priority
        }
    }, NULL, &device));
#define LOAD_DEVICE(name) do { \
    api.name = (PFN_vk##name)api.GetDeviceProcAddr(device, "vk" #name); \
    if (!api.name) { fprintf(stderr, "Missing vk%s\n", #name); goto cleanup; } \
} while (0);
    DEVICE_FUNCTIONS(LOAD_DEVICE)
#undef LOAD_DEVICE
    api.GetDeviceQueue(device, family, 0, &queue);
    api.GetPhysicalDeviceMemoryProperties(physical, &memory);
    if (!queue)
        goto cleanup;
    const VkDeviceSize buffer_sizes[3] = {
        sizeof(scene_vertices), sizeof(scene_texture), SCENE_W * SCENE_H * 4u
    };
    const VkBufferUsageFlags buffer_usage[3] = {
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT
    };
    for (unsigned i = 0; i < 3; i++) {
        CHECK(api.CreateBuffer(device, &(VkBufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = buffer_sizes[i], .usage = buffer_usage[i]
        }, NULL, &buffers[i]));
        VkMemoryRequirements requirements;
        api.GetBufferMemoryRequirements(device, buffers[i], &requirements);
        uint32_t type;
        if (!memory_type(&memory, requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &type))
            goto cleanup;
        CHECK(api.AllocateMemory(device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size, .memoryTypeIndex = type
        }, NULL, &buffer_memory[i]));
        CHECK(api.BindBufferMemory(device, buffers[i], buffer_memory[i], 0));
        if (i < 2) {
            CHECK(api.MapMemory(device, buffer_memory[i], 0, buffer_sizes[i], 0, &mapped));
            mapped_memory = buffer_memory[i];
            memcpy(mapped, i == 0 ? (const void *)scene_vertices : (const void *)scene_texture,
                   buffer_sizes[i]);
            api.UnmapMemory(device, mapped_memory);
            mapped_memory = VK_NULL_HANDLE;
        }
    }
    for (unsigned i = 0; i < 2; i++) {
        CHECK(api.CreateImage(device, &(VkImageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = i == 0 ? (VkExtent3D){SCENE_W, SCENE_H, 1} : (VkExtent3D){2, 2, 1},
            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = i == 0 ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT :
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        }, NULL, &images[i]));
        VkMemoryRequirements requirements;
        api.GetImageMemoryRequirements(device, images[i], &requirements);
        uint32_t type;
        if (!memory_type(&memory, requirements.memoryTypeBits, 0, &type))
            goto cleanup;
        CHECK(api.AllocateMemory(device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size, .memoryTypeIndex = type
        }, NULL, &image_memory[i]));
        CHECK(api.BindImageMemory(device, images[i], image_memory[i], 0));
        CHECK(api.CreateImageView(device, &(VkImageViewCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        }, NULL, &views[i]));
    }
    CHECK(api.CreateSampler(device, &(VkSamplerCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0
    }, NULL, &sampler));
    const uint32_t *fragment = constant_uv ? scene_zero_fragment : scene_fragment;
    size_t fragment_bytes = constant_uv ? sizeof(scene_zero_fragment) : sizeof(scene_fragment);
    CHECK(api.CreateShaderModule(device, &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(scene_vertex), .pCode = scene_vertex
    }, NULL, &shaders[0]));
    CHECK(api.CreateShaderModule(device, &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = fragment_bytes, .pCode = fragment
    }, NULL, &shaders[1]));
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
    };
    CHECK(api.CreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding
    }, NULL, &descriptor_layout));
    CHECK(api.CreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
        .poolSizeCount = 1, .pPoolSizes = &(VkDescriptorPoolSize){
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1
        }
    }, NULL, &descriptor_pool));
    CHECK(api.AllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_layout
    }, &descriptor));
    VkDescriptorImageInfo sampled = {sampler, views[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    api.UpdateDescriptorSets(device, 1, &(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &sampled
    }, 0, NULL);
    CHECK(api.CreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &descriptor_layout
    }, NULL, &pipeline_layout));
    VkAttachmentDescription attachment = {
        .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    };
    VkAttachmentReference color = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1, .pColorAttachments = &color
    };
    VkSubpassDependency dependency = {
        .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
    };
    CHECK(api.CreateRenderPass(device, &(VkRenderPassCreateInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &attachment,
        .subpassCount = 1, .pSubpasses = &subpass,
        .dependencyCount = 1, .pDependencies = &dependency
    }, NULL, &render_pass));
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = shaders[0], .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = shaders[1], .pName = "main"}
    };
    VkVertexInputBindingDescription vertex_binding = {0, 4 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vertex_attribute = {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0};
    VkViewport viewport = {0, 0, SCENE_W, SCENE_H, 0, 1};
    VkRect2D scissor = {{0, 0}, {SCENE_W, SCENE_H}};
    VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = 15};
    CHECK(api.CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages,
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vertex_binding,
            .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = &vertex_attribute
        },
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        },
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor
        },
        .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f
        },
        .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
        },
        .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &blend
        },
        .layout = pipeline_layout, .renderPass = render_pass
    }, NULL, &pipeline));
    CHECK(api.CreateFramebuffer(device, &(VkFramebufferCreateInfo){
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = render_pass, .attachmentCount = 1, .pAttachments = &views[0],
        .width = SCENE_W, .height = SCENE_H, .layers = 1
    }, NULL, &framebuffer));
    CHECK(api.CreateCommandPool(device, &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family
    }, NULL, &command_pool));
    CHECK(api.AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    }, &command));
    CHECK(api.BeginCommandBuffer(command, &(VkCommandBufferBeginInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    }));
    VkImageMemoryBarrier texture_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = images[1], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    api.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &texture_barrier);
    VkBufferImageCopy texture_copy = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {2, 2, 1}
    };
    api.CmdCopyBufferToImage(command, buffers[1], images[1], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             1, &texture_copy);
    texture_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    texture_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    texture_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    texture_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    api.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &texture_barrier);
    VkClearValue clear = {.color = {{0, 0, 0, 1}}};
    api.CmdBeginRenderPass(command, &(VkRenderPassBeginInfo){
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass, .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {SCENE_W, SCENE_H}}, .clearValueCount = 1, .pClearValues = &clear
    }, VK_SUBPASS_CONTENTS_INLINE);
    api.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize offset = 0;
    api.CmdBindVertexBuffers(command, 0, 1, &buffers[0], &offset);
    api.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                              0, 1, &descriptor, 0, NULL);
    api.CmdDraw(command, 6, 1, 0, 0);
    api.CmdEndRenderPass(command);
    VkBufferImageCopy readback = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {SCENE_W, SCENE_H, 1}
    };
    api.CmdCopyImageToBuffer(command, images[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             buffers[2], 1, &readback);
    VkBufferMemoryBarrier host_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffers[2], .size = VK_WHOLE_SIZE
    };
    api.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &host_barrier, 0, NULL);
    CHECK(api.EndCommandBuffer(command));
    CHECK(api.CreateFence(device, &(VkFenceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    }, NULL, &fence));
    CHECK(api.QueueSubmit(queue, 1, &(VkSubmitInfo){
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &command
    }, fence));
    submitted = 1;
    CHECK(api.WaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull));
    submitted = 0;
    CHECK(api.MapMemory(device, buffer_memory[2], 0, buffer_sizes[2], 0, &mapped));
    mapped_memory = buffer_memory[2];
    memcpy(pixels, mapped, buffer_sizes[2]);
    success = 1;
cleanup:
    if (submitted && api.DeviceWaitIdle)
        api.DeviceWaitIdle(device);
    if (mapped_memory && api.UnmapMemory)
        api.UnmapMemory(device, mapped_memory);
    if (fence && api.DestroyFence) api.DestroyFence(device, fence, NULL);
    if (command_pool && api.DestroyCommandPool) api.DestroyCommandPool(device, command_pool, NULL);
    if (framebuffer && api.DestroyFramebuffer) api.DestroyFramebuffer(device, framebuffer, NULL);
    if (pipeline && api.DestroyPipeline) api.DestroyPipeline(device, pipeline, NULL);
    if (render_pass && api.DestroyRenderPass) api.DestroyRenderPass(device, render_pass, NULL);
    if (pipeline_layout && api.DestroyPipelineLayout) api.DestroyPipelineLayout(device, pipeline_layout, NULL);
    if (descriptor_pool && api.DestroyDescriptorPool) api.DestroyDescriptorPool(device, descriptor_pool, NULL);
    if (descriptor_layout && api.DestroyDescriptorSetLayout) api.DestroyDescriptorSetLayout(device, descriptor_layout, NULL);
    for (unsigned i = 0; i < 2; i++) {
        if (shaders[i] && api.DestroyShaderModule) api.DestroyShaderModule(device, shaders[i], NULL);
        if (views[i] && api.DestroyImageView) api.DestroyImageView(device, views[i], NULL);
        if (images[i] && api.DestroyImage) api.DestroyImage(device, images[i], NULL);
        if (image_memory[i] && api.FreeMemory) api.FreeMemory(device, image_memory[i], NULL);
    }
    if (sampler && api.DestroySampler) api.DestroySampler(device, sampler, NULL);
    for (unsigned i = 0; i < 3; i++) {
        if (buffers[i] && api.DestroyBuffer) api.DestroyBuffer(device, buffers[i], NULL);
        if (buffer_memory[i] && api.FreeMemory) api.FreeMemory(device, buffer_memory[i], NULL);
    }
    if (device && api.DestroyDevice) api.DestroyDevice(device, NULL);
    if (instance && api.DestroyInstance) api.DestroyInstance(instance, NULL);
    free(families);
    free(devices);
#undef CHECK
    return success;
}

int mx_vk_frame_with_api(PFN_vkGetInstanceProcAddr get, int constant_uv,
                         unsigned char *pixels)
{
    return frame_with_api(get, constant_uv, 0, pixels);
}

int mx_vk_frame_after_rejected_device(PFN_vkGetInstanceProcAddr get,
                                       unsigned char *pixels)
{
    return frame_with_api(get, 0, 1, pixels);
}

int mx_vk_frame(int constant_uv, unsigned char *pixels)
{
    return mx_vk_frame_with_api(vkGetInstanceProcAddr, constant_uv, pixels);
}
