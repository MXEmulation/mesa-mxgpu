/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_driver.h"
#include "mxgpu_compiler.h"
#include "nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "vulkan/util/vk_format.h"
#include "mx_le.h"
#include "mxsb.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_icd.h>

#define MX_EXPORT __attribute__((visibility("default")))
#define MXGPU_VK_PUSH_CONSTANT_BYTES 128u
#define MXGPU_VK_HEAP_BYTES (64ull << 20)
#define MXGPU_VK_STAGES 3u
#define MXGPU_VK_COMPUTE_STAGE 2u
#define MXGPU_VK_STORAGE_OFFSET_ALIGNMENT 16u
#define MXGPU_VK_API_VERSION VK_API_VERSION_1_1
#define MXGPU_VK_MAX_IMAGE_DIMENSION 2048u
#define MXGPU_VK_UNIFORM_BUFFER_RANGE 65536u
#define MXGPU_VK_STAGE_UNIFORM_BUFFERS (MXGPU_UNIFORM_BUFFERS - 1u)
#define MXGPU_VK_INTERSTAGE_COMPONENTS 128u
#define MXGPU_VK_MAP_ALIGNMENT 64u
#define MXGPU_VK_MAX_BOUND_SETS 8u
#define MXGPU_VK_MAX_PER_SET_DESCRIPTORS 1024u
#define MXGPU_VK_DRIVER_ID ((VkDriverId)0x4d584750)
static const uint8_t mx_pipeline_cache_uuid[VK_UUID_SIZE] = {
    'M', 'X', 'G', 'P', 'U', '-', 'S', 'P', 'I', 'R', 'V', '-', '0', '0', '0', '1'
};
static const uint8_t mx_device_uuid[VK_UUID_SIZE] = {
    'M', 'X', 'G', 'P', 'U', '-', 'D', 'E', 'V', 'I', 'C', 'E',
    MX_PCI_VENDOR_ID >> 8, MX_PCI_VENDOR_ID & 0xff, MXGPU_PCI_DEVICE_ID >> 8, MXGPU_PCI_DEVICE_ID & 0xff
};
static const uint8_t mx_driver_uuid[VK_UUID_SIZE] = {
    'M', 'X', 'G', 'P', 'U', '-', 'V', 'U', 'L', 'K', 'A', 'N', '-', '0', '0', '1'
};

struct mx_pipeline_layout;

struct mx_mem {
    VkDeviceSize size;
    void *ptr;
    int mapped;
    uint32_t storage;
};

struct mx_buf {
    VkDeviceSize size;
    struct mx_mem *mem;
    VkDeviceSize offset;
};

struct mx_img {
    uint32_t width;
    uint32_t height;
    struct mx_mem *mem;
    VkDeviceSize offset;
    VkDeviceSize size;
    uint32_t depth, levels, layers, samples;
    VkFormat format;
};

struct mx_view {
    struct mx_img *image;
    VkComponentMapping components;
    uint32_t width;
    uint32_t height;
    VkDeviceSize offset;
    VkFormat format;
    VkImageAspectFlags aspects;
};

struct mx_shader {
    int spirv;
    int samples;
    uint8_t *bytes;
    uint32_t len;
};

struct mx_uniform_stage {
    bool uses_uniforms;
    unsigned uniform_count, uniform_buffer_count;
    struct mxgpu_uniform_buffer uniform_buffers[MXGPU_UNIFORM_BUFFERS];
};

struct mx_pipe {
    struct mx_pipeline_layout *layout;
    struct mxgpu_shader *vertex_shader, *fragment_shader;
    struct mxgpu_native_render_state native_state;
    VkViewport viewport;
    VkRect2D scissor;
    bool viewport_set, scissor_set;
    bool dynamic_viewport, dynamic_scissor, dynamic_blend;
    struct mxgpu_depth_stencil_state depth_stencil;
    bool depth_enabled;
    uint32_t stencil_reference;
    bool dynamic_stencil_compare, dynamic_stencil_write, dynamic_stencil_reference;
    uint32_t stencil_compare[2], stencil_write[2], stencil_ref[2];
    bool vertex_layout;
    VkVertexInputBindingDescription vertex_bindings[32];
    uint32_t vertex_binding_mask;
    VkVertexInputAttributeDescription vertex_attributes[32];
    uint32_t vertex_attribute_count;
    uint32_t vertex_input_locations[MXGPU_SHADER_VERTEX_SLOTS];
    bool vertex_builtins;
    uint32_t vertex_builtin_slot;
    uint32_t vertex_attribute_mask;
    VkPrimitiveTopology topology;
    bool primitive_restart;
    int samples;
    int uses_texture;
    int reflected_texture;
    uint32_t texture_binding;
    uint32_t texture_set;
    uint32_t texture_element;
    struct mx_uniform_stage uniform_stages[MXGPU_VK_STAGES];
    uint8_t *bytes;
    uint32_t len;
    bool compute;
    uint32_t compute_pipeline;
    uint32_t uniform_storage;
    uint32_t uniform_slot;
    uint32_t storage_count;
    struct mxgpu_storage_binding storage[MXGPU_SHADER_STORAGE_BUFFERS];
};

struct mx_fb {
    struct mx_view *color;
    struct mx_view *depth;
    uint32_t width, height;
};

struct mx_renderpass {
    uint32_t color_attachment;
    VkAttachmentLoadOp load;
    uint32_t depth_attachment;
    VkFormat color_format, depth_format;
    VkAttachmentLoadOp depth_load, stencil_load;
};

struct mx_sampler {
    VkSamplerCreateInfo info;
};

struct mx_descriptor_value {
    VkDescriptorImageInfo image;
    VkDescriptorBufferInfo buffer;
    VkBufferView texel;
};

struct mx_descriptor {
    uint32_t binding, element;
    VkDescriptorType type;
    VkShaderStageFlags stages;
    VkSampler immutable_sampler;
    struct mx_descriptor_value value;
};

struct mx_ds_layout {
    uint32_t count;
    struct mx_descriptor *descriptors;
};

struct mx_pipeline_layout {
    atomic_uint references;
    uint32_t set_count, range_count;
    struct mx_ds_layout *sets;
    VkPushConstantRange *ranges;
    VkShaderStageFlags coverage[MXGPU_VK_PUSH_CONSTANT_BYTES / 4];
};

static void retain_layout(struct mx_pipeline_layout *layout)
{
    if (layout)
        atomic_fetch_add_explicit(&layout->references, 1, memory_order_relaxed);
}

static void release_layout(struct mx_pipeline_layout *layout)
{
    if (!layout || atomic_fetch_sub_explicit(&layout->references, 1, memory_order_acq_rel) != 1)
        return;
    for (uint32_t i = 0; i < layout->set_count; i++)
        free(layout->sets[i].descriptors);
    free(layout->sets);
    free(layout->ranges);
    free(layout);
}

static bool push_layout_compatible(const struct mx_pipeline_layout *a,
                                    const struct mx_pipeline_layout *b)
{
    if (a == b)
        return true;
    if (!a || !b || a->range_count != b->range_count)
        return false;
    for (uint32_t i = 0; i < a->range_count; i++)
        if (a->ranges[i].offset != b->ranges[i].offset ||
            a->ranges[i].size != b->ranges[i].size ||
            a->ranges[i].stageFlags != b->ranges[i].stageFlags)
            return false;
    return true;
}

struct mx_ds_pool;

struct mx_set {
    struct mx_view *image;
    struct mx_set *next;
    struct mx_ds_pool *pool;
    uint32_t descriptor_count;
    struct mx_descriptor *descriptors;
};

struct mx_ds_pool {
    struct mx_set *sets;
    uint32_t count;
    uint32_t max_sets;
};

struct mx_pool;
struct mx_draw;

struct mx_bound_offsets {
    uint32_t *values;
    uint32_t count;
};

struct mx_cmd {
    VK_LOADER_DATA loader_data;
    struct mx_pool *pool;
    struct mx_cmd *next;
    struct mx_pipe *pipe;
    struct mx_buf *vbo;
    VkDeviceSize voff;
    struct mx_buf *vertex_buffers[32];
    VkDeviceSize vertex_offsets[32];
    bool packed_vertices;
    struct mx_fb *fb;
    struct mx_set *set;
    struct mx_set **bound_sets;
    struct mx_bound_offsets *bound_offsets;
    uint32_t bound_set_count;
    uint32_t vertex_count;
    uint32_t first_vertex;
    uint32_t instance_count;
    uint32_t first_instance;
    struct mx_draw *draws;
    struct mx_draw *last_draw;
    VkResult record_result;
    uint8_t push_constants[MXGPU_VK_STAGES][MXGPU_VK_PUSH_CONSTANT_BYTES];
    struct mx_pipeline_layout *push_layouts[MXGPU_VK_STAGES][MXGPU_VK_PUSH_CONSTANT_BYTES / 4];
    struct mx_pipe *compute_pipe;
    struct mx_set **compute_sets;
    struct mx_bound_offsets *compute_offsets;
    uint32_t compute_set_count;
    int draw;
    int open;
    int clear;
    VkClearColorValue clear_color;
    VkRect2D clear_area;
    VkImageAspectFlags clear_depth_aspects;
    VkClearDepthStencilValue clear_depth;
    struct mx_buf *ibo;
    VkDeviceSize ioff;
    VkIndexType index_type;
    uint32_t first_index;
    int32_t vertex_bias;
    int indexed;
    VkViewport viewport;
    VkRect2D scissor;
    float blend_constants[4];
    bool viewport_set, scissor_set, blend_set;
    uint32_t stencil_compare[2], stencil_write[2], stencil_ref[2];
    unsigned stencil_compare_set, stencil_write_set, stencil_ref_set;
};

struct mx_draw {
    struct mx_cmd state;
    int operation;
    struct mx_img *copy_image;
    VkBufferImageCopy image_region;
    struct mx_buf *copy_src, *copy_dst;
    VkBufferCopy copy_region;
    uint32_t dispatch[3];
    struct mx_buf *indirect;
    VkDeviceSize indirect_offset;
    struct mx_draw *next;
};

static void release_push_layouts(struct mx_cmd *cmd)
{
    for (unsigned stage = 0; stage < MXGPU_VK_STAGES; stage++)
        for (unsigned word = 0; word < MXGPU_VK_PUSH_CONSTANT_BYTES / 4; word++)
            release_layout(cmd->push_layouts[stage][word]);
}

static void free_bound_offsets(struct mx_bound_offsets *offsets, uint32_t count)
{
    if (!offsets)
        return;
    for (uint32_t i = 0; i < count; i++)
        free(offsets[i].values);
    free(offsets);
}

static void free_bound_state(struct mx_cmd *cmd)
{
    free_bound_offsets(cmd->bound_offsets, cmd->bound_set_count);
    free(cmd->bound_sets);
    free_bound_offsets(cmd->compute_offsets, cmd->compute_set_count);
    free(cmd->compute_sets);
    release_push_layouts(cmd);
}

static void free_draws(struct mx_cmd *cmd)
{
    while (cmd->draws) {
        struct mx_draw *draw = cmd->draws;
        cmd->draws = draw->next;
        free_bound_offsets(draw->state.bound_offsets, draw->state.bound_set_count);
        free(draw->state.bound_sets);
        release_push_layouts(&draw->state);
        free(draw);
    }
    cmd->last_draw = NULL;
}

struct mx_pool {
    struct mx_cmd *commands;
};

struct mx_fence {
    atomic_int signaled;
};

struct mx_physical_device {
    VK_LOADER_DATA loader_data;
    struct mx_instance *instance;
};

struct mx_instance {
    VK_LOADER_DATA loader_data;
    struct mx_physical_device physical;
    bool surface_enabled, wayland_enabled;
};

struct mx_device;

struct mx_semaphore {
    struct mx_device *device;
    atomic_int signaled;
};

struct mx_queue {
    VK_LOADER_DATA loader_data;
    struct mx_device *device;
};

struct mx_device {
    VK_LOADER_DATA loader_data;
    atomic_int lost;
    struct mx_queue queue;
    bool swapchain_enabled;
};

static bool instance_extension_supported(const char *name)
{
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    return name && (!strcmp(name, VK_KHR_SURFACE_EXTENSION_NAME) ||
                    !strcmp(name, VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME));
#else
    (void)name;
    return false;
#endif
}

static VkResult create_instance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc, VkInstance *out)
{
    (void)alloc;
    struct mx_instance *instance;
    *out = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < info->enabledExtensionCount; i++)
        if (!instance_extension_supported(info->ppEnabledExtensionNames[i]))
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    instance = calloc(1, sizeof *instance);
    if (!instance)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    set_loader_magic_value(instance);
    set_loader_magic_value(&instance->physical);
    instance->physical.instance = instance;
    for (uint32_t i = 0; i < info->enabledExtensionCount; i++) {
        instance->surface_enabled |= !strcmp(info->ppEnabledExtensionNames[i], VK_KHR_SURFACE_EXTENSION_NAME);
        instance->wayland_enabled |= !strcmp(info->ppEnabledExtensionNames[i], "VK_KHR_wayland_surface");
    }
    if (instance->wayland_enabled && !instance->surface_enabled) {
        free(instance);
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    *out = (VkInstance)instance;
    return VK_SUCCESS;
}

static void destroy_instance(VkInstance instance, const VkAllocationCallbacks *alloc)
{
    (void)alloc;
    free(instance);
}

static VkResult enum_devices(VkInstance instance, uint32_t *count, VkPhysicalDevice *devices)
{
    struct mx_instance *owner = (struct mx_instance *)instance;
    if (!count)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!mxgpu_device_available()) {
        *count = 0;
        return VK_SUCCESS;
    }
    if (!devices) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count < 1) {
        *count = 0;
        return VK_INCOMPLETE;
    }
    devices[0] = (VkPhysicalDevice)&owner->physical;
    *count = 1;
    return VK_SUCCESS;
}

static bool vk_compute_supported(void)
{
    return mxgpu_device_open() == 0 && mxgpu_compute_available();
}

static uint32_t vk_stage_textures(void)
{
    uint32_t textures = MXGPU_SHADER_TEXTURES < MXGPU_TEXTURE_INPUTS ? MXGPU_SHADER_TEXTURES : MXGPU_TEXTURE_INPUTS;
    return mxgpu_native_sampler_available(-1) ? textures : 1;
}

static void device_props(VkPhysicalDevice gpu, VkPhysicalDeviceProperties *props)
{
    VkPhysicalDeviceLimits *limits = &props->limits;
    uint32_t textures = vk_stage_textures();
    bool compute = vk_compute_supported();
    (void)gpu;
    memset(props, 0, sizeof *props);
    props->apiVersion = MXGPU_VK_API_VERSION;
    props->driverVersion = VK_MAKE_VERSION(MXGPU_DRIVER_VERSION_MAJOR, MXGPU_DRIVER_VERSION_MINOR,
                                           MXGPU_DRIVER_VERSION_PATCH);
    props->vendorID = MX_PCI_VENDOR_ID;
    props->deviceID = MXGPU_PCI_DEVICE_ID;
    memcpy(props->pipelineCacheUUID, mx_pipeline_cache_uuid, VK_UUID_SIZE);
    props->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    memcpy(props->deviceName, "MXGPU", 6);
    limits->maxImageDimension2D = MXGPU_VK_MAX_IMAGE_DIMENSION;
    limits->maxImageArrayLayers = 1;
    limits->maxUniformBufferRange = MXGPU_VK_UNIFORM_BUFFER_RANGE;
    limits->maxPushConstantsSize = MXGPU_VK_PUSH_CONSTANT_BYTES;
    limits->maxMemoryAllocationCount = 4096;
    limits->maxSamplerAllocationCount = 4000;
    limits->bufferImageGranularity = 1;
    limits->maxBoundDescriptorSets = MXGPU_VK_MAX_BOUND_SETS;
    limits->maxPerStageDescriptorSamplers = textures;
    limits->maxPerStageDescriptorUniformBuffers = MXGPU_VK_STAGE_UNIFORM_BUFFERS;
    limits->maxPerStageDescriptorSampledImages = textures;
    limits->maxPerStageResources = MXGPU_VK_STAGE_UNIFORM_BUFFERS + textures + 1;
    limits->maxDescriptorSetSamplers = textures;
    limits->maxDescriptorSetUniformBuffers = 2 * MXGPU_VK_STAGE_UNIFORM_BUFFERS;
    limits->maxDescriptorSetUniformBuffersDynamic = 2 * MXGPU_VK_STAGE_UNIFORM_BUFFERS;
    limits->maxDescriptorSetSampledImages = textures;
    limits->maxVertexInputAttributes = 32;
    limits->maxVertexInputBindings = 32;
    limits->maxVertexInputAttributeOffset = UINT32_MAX;
    limits->maxVertexInputBindingStride = UINT32_MAX;
    limits->maxVertexOutputComponents = MXGPU_VK_INTERSTAGE_COMPONENTS;
    limits->maxFragmentInputComponents = MXGPU_VK_INTERSTAGE_COMPONENTS;
    limits->maxFragmentOutputAttachments = 1;
    limits->maxFragmentCombinedOutputResources = 1;
    limits->subPixelPrecisionBits = 4;
    limits->subTexelPrecisionBits = 4;
    limits->mipmapPrecisionBits = 4;
    limits->maxDrawIndexedIndexValue = UINT32_MAX;
    limits->maxDrawIndirectCount = 1;
    limits->maxSamplerLodBias = 2.0f;
    limits->maxSamplerAnisotropy = 1.0f;
    limits->maxViewports = 1;
    limits->maxViewportDimensions[0] = limits->maxViewportDimensions[1] = MXGPU_VK_MAX_IMAGE_DIMENSION;
    limits->viewportBoundsRange[0] = -2.0f * MXGPU_VK_MAX_IMAGE_DIMENSION;
    limits->viewportBoundsRange[1] = 2.0f * MXGPU_VK_MAX_IMAGE_DIMENSION - 1.0f;
    limits->minMemoryMapAlignment = MXGPU_VK_MAP_ALIGNMENT;
    limits->minTexelBufferOffsetAlignment = 256;
    limits->minUniformBufferOffsetAlignment = MXGPU_VK_STORAGE_OFFSET_ALIGNMENT;
    limits->minStorageBufferOffsetAlignment = MXGPU_VK_STORAGE_OFFSET_ALIGNMENT;
    limits->maxFramebufferWidth = MXGPU_VK_MAX_IMAGE_DIMENSION;
    limits->maxFramebufferHeight = MXGPU_VK_MAX_IMAGE_DIMENSION;
    limits->maxFramebufferLayers = 1;
    limits->framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->maxColorAttachments = 1;
    limits->sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    limits->maxSampleMaskWords = 1;
    limits->discreteQueuePriorities = 2;
    limits->pointSizeRange[0] = limits->pointSizeRange[1] = 1.0f;
    limits->lineWidthRange[0] = limits->lineWidthRange[1] = 1.0f;
    limits->optimalBufferCopyOffsetAlignment = 16;
    limits->optimalBufferCopyRowPitchAlignment = 4;
    limits->nonCoherentAtomSize = MXGPU_VK_MAP_ALIGNMENT;
    if (compute) {
        limits->maxComputeSharedMemorySize = 16384;
        limits->maxComputeWorkGroupCount[0] = 65535;
        limits->maxComputeWorkGroupCount[1] = 65535;
        limits->maxComputeWorkGroupCount[2] = 65535;
        limits->maxComputeWorkGroupInvocations = 128;
        limits->maxComputeWorkGroupSize[0] = 128;
        limits->maxComputeWorkGroupSize[1] = 128;
        limits->maxComputeWorkGroupSize[2] = 64;
        limits->maxStorageBufferRange = (uint32_t)MXGPU_VK_HEAP_BYTES;
        limits->maxPerStageDescriptorStorageBuffers = MXGPU_SHADER_STORAGE_BUFFERS;
        limits->maxDescriptorSetStorageBuffers = MXGPU_SHADER_STORAGE_BUFFERS;
        limits->maxDescriptorSetStorageBuffersDynamic = MXGPU_SHADER_STORAGE_BUFFERS;
        if (MXGPU_SHADER_STORAGE_BUFFERS + MXGPU_VK_STAGE_UNIFORM_BUFFERS > limits->maxPerStageResources)
            limits->maxPerStageResources = MXGPU_SHADER_STORAGE_BUFFERS + MXGPU_VK_STAGE_UNIFORM_BUFFERS;
    }
}

static void queue_props(VkPhysicalDevice gpu, uint32_t *count, VkQueueFamilyProperties *props)
{
    (void)gpu;
    if (!props) {
        *count = 1;
        return;
    }
    if (!*count)
        return;
    memset(props, 0, sizeof *props);
    props[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | (vk_compute_supported() ? VK_QUEUE_COMPUTE_BIT : 0);
    props[0].queueCount = 1;
    props[0].minImageTransferGranularity = (VkExtent3D){1, 1, 1};
    *count = 1;
}

static VkResult device_create_info_supported(VkPhysicalDevice physical, const VkDeviceCreateInfo *info);

static VkResult create_device(VkPhysicalDevice gpu, const VkDeviceCreateInfo *info, const VkAllocationCallbacks *alloc, VkDevice *out)
{
    struct mx_device *device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    bool swapchain = false;
    for (uint32_t i = 0; i < info->enabledExtensionCount; i++) {
        if (!strcmp(info->ppEnabledExtensionNames[i], VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME))
            continue;
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
        if (!strcmp(info->ppEnabledExtensionNames[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            swapchain = true;
        else
#endif
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    VkResult supported = device_create_info_supported(gpu, info);
    if (supported != VK_SUCCESS)
        return supported;
    if (swapchain && (!gpu || !((struct mx_physical_device *)gpu)->instance ||
                      !((struct mx_physical_device *)gpu)->instance->surface_enabled))
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    device = calloc(1, sizeof *device);
    if (!device)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    set_loader_magic_value(device);
    atomic_init(&device->lost, 0);
    set_loader_magic_value(&device->queue);
    device->queue.device = device;
    device->swapchain_enabled = swapchain;
    *out = (VkDevice)device;
    return VK_SUCCESS;
}

static void destroy_device(VkDevice device, const VkAllocationCallbacks *alloc)
{
    (void)alloc;
    free(device);
}

static void get_queue(VkDevice device, uint32_t family, uint32_t index, VkQueue *queue)
{
    struct mx_device *owner = (struct mx_device *)device;
    *queue = family == 0 && index == 0 ? (VkQueue)&owner->queue : VK_NULL_HANDLE;
}

static VkResult create_buffer(VkDevice device, const VkBufferCreateInfo *info, const VkAllocationCallbacks *alloc, VkBuffer *out)
{
    struct mx_buf *buf = calloc(1, sizeof *buf);
    (void)device;
    (void)alloc;
    if (!buf)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    buf->size = info->size;
    *out = (VkBuffer)buf;
    return VK_SUCCESS;
}

static void buffer_reqs(VkDevice device, VkBuffer buffer, VkMemoryRequirements *reqs)
{
    struct mx_buf *buf = (struct mx_buf *)buffer;
    (void)device;
    memset(reqs, 0, sizeof *reqs);
    reqs->size = buf->size;
    reqs->alignment = 16;
    reqs->memoryTypeBits = 1;
}

static VkResult alloc_mem(VkDevice device, const VkMemoryAllocateInfo *info, const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
    struct mx_mem *mem;
    *out = VK_NULL_HANDLE;
    if (!info->allocationSize || info->allocationSize > SIZE_MAX || info->memoryTypeIndex != 0)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    mem = calloc(1, sizeof *mem);
    (void)device;
    (void)alloc;
    if (!mem)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    mem->size = info->allocationSize;
    if (mem->size > SIZE_MAX - (MXGPU_VK_MAP_ALIGNMENT - 1)) {
        free(mem);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    size_t bytes = ((size_t)mem->size + MXGPU_VK_MAP_ALIGNMENT - 1) & ~(size_t)(MXGPU_VK_MAP_ALIGNMENT - 1);
    mem->ptr = aligned_alloc(MXGPU_VK_MAP_ALIGNMENT, bytes);
    if (mem->ptr)
        memset(mem->ptr, 0, bytes);
    if (!mem->ptr) {
        free(mem);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *out = (VkDeviceMemory)mem;
    return VK_SUCCESS;
}

static VkResult bind_buffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset)
{
    struct mx_buf *buf = (struct mx_buf *)buffer;
    (void)device;
    struct mx_mem *mem = (struct mx_mem *)memory;
    if (!mem || offset % 16u || offset > mem->size || buf->size > mem->size - offset)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    buf->mem = mem;
    buf->offset = offset;
    return VK_SUCCESS;
}

static VkResult map_mem(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void **data)
{
    struct mx_mem *mem = (struct mx_mem *)memory;
    (void)device;
    (void)flags;
    *data = NULL;
    if (!mem || mem->mapped || offset >= mem->size || !size ||
        (size != VK_WHOLE_SIZE && size > mem->size - offset))
        return VK_ERROR_MEMORY_MAP_FAILED;
    mem->mapped = 1;
    *data = (unsigned char *)mem->ptr + offset;
    return VK_SUCCESS;
}

static void unmap_mem(VkDevice device, VkDeviceMemory memory)
{
    struct mx_mem *mem = (struct mx_mem *)memory;
    (void)device;
    if (mem)
        mem->mapped = 0;
}

static void free_mem(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *alloc)
{
    struct mx_mem *mem = (struct mx_mem *)memory;
    (void)device;
    (void)alloc;
    if (!mem)
        return;
    if (mem->storage)
        mxgpu_storage_buffer_destroy(mem->storage);
    free(mem->ptr);
    free(mem);
}

static void destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(buffer);
}

static uint32_t vk_depth_format(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_D32_SFLOAT: return MXGPU_FMT_DEPTH32_FLOAT;
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return MXGPU_FMT_DEPTH32_FLOAT_STENCIL8;
    case VK_FORMAT_D24_UNORM_S8_UINT: return MXGPU_FMT_DEPTH24_UNORM_STENCIL8;
    default: return 0;
    }
}

static enum pipe_format vk_color_format(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8B8_UNORM:
    case VK_FORMAT_B8G8R8_UNORM:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM: return vk_format_to_pipe_format(format);
    default: return PIPE_FORMAT_NONE;
    }
}

static uint8_t vk_color_write_mask(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: return 3;
    case VK_FORMAT_R8G8B8_UNORM:
    case VK_FORMAT_B8G8R8_UNORM: return 7;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM: return 15;
    default: return 0;
    }
}

static unsigned vk_format_bytes(VkFormat format)
{
    enum pipe_format color = vk_color_format(format);
    if (color != PIPE_FORMAT_NONE)
        return util_format_get_blocksize(color);
    switch (format) {
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D24_UNORM_S8_UINT: return 4;
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return 8;
    default: return 0;
    }
}

static enum pipe_format vk_vertex_format(VkFormat format)
{
    enum pipe_format mapped = vk_format_to_pipe_format(format);
    if (mapped == PIPE_FORMAT_NONE)
        return PIPE_FORMAT_NONE;
    const struct util_format_description *desc = util_format_description(mapped);
    const struct util_format_unpack_description *unpack = util_format_unpack_description(mapped);
    if (!desc || !unpack || !unpack->unpack_rgba || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
        desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB || desc->block.width != 1 ||
        desc->block.height != 1 || desc->block.depth != 1 || !desc->block.bits || desc->block.bits > 128)
        return PIPE_FORMAT_NONE;
    for (unsigned i = 0; i < 4; i++) {
        const struct util_format_channel_description *channel = &desc->channel[i];
        if (channel->type != UTIL_FORMAT_TYPE_VOID &&
            ((channel->type != UTIL_FORMAT_TYPE_UNSIGNED && channel->type != UTIL_FORMAT_TYPE_SIGNED &&
              channel->type != UTIL_FORMAT_TYPE_FLOAT) || channel->size > 32))
            return PIPE_FORMAT_NONE;
    }
    return mapped;
}

static VkImageAspectFlags mx_vk_format_aspects(VkFormat format)
{
    if (vk_color_format(format) != PIPE_FORMAT_NONE)
        return VK_IMAGE_ASPECT_COLOR_BIT;
    switch (format) {
    case VK_FORMAT_D32_SFLOAT: return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default: return 0;
    }
}

static VkResult create_image(VkDevice device, const VkImageCreateInfo *info, const VkAllocationCallbacks *alloc, VkImage *out)
{
    struct mx_img *img;
    VkDeviceSize total = 0;
    uint32_t width = info->extent.width, height = info->extent.height;
    uint32_t depth = info->extent.depth, level;
    unsigned pixel_bytes = vk_format_bytes(info->format);
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (!pixel_bytes)
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    if (!width || !height || !depth || !info->mipLevels || !info->arrayLayers || !info->samples)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (level = 0; level < info->mipLevels; level++) {
        VkDeviceSize bytes = (VkDeviceSize)width * height;
        if (bytes > UINT64_MAX / depth)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        bytes *= depth;
        if (bytes > UINT64_MAX / pixel_bytes)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        bytes *= pixel_bytes;
        if (bytes > UINT64_MAX / info->arrayLayers)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        bytes *= info->arrayLayers;
        if (bytes > UINT64_MAX / info->samples)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        bytes *= info->samples;
        if (bytes > UINT64_MAX - total)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        total += bytes;
        if (level + 1 < info->mipLevels && width == 1 && height == 1 && depth == 1)
            return VK_ERROR_INITIALIZATION_FAILED;
        width = width > 1 ? width / 2 : 1;
        height = height > 1 ? height / 2 : 1;
        depth = depth > 1 ? depth / 2 : 1;
    }
    if (total > SIZE_MAX)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    img = calloc(1, sizeof *img);
    if (!img)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    img->width = info->extent.width;
    img->height = info->extent.height;
    img->size = total;
    img->depth = info->extent.depth;
    img->levels = info->mipLevels;
    img->layers = info->arrayLayers;
    img->samples = info->samples;
    img->format = info->format;
    *out = (VkImage)img;
    return VK_SUCCESS;
}

static void image_reqs(VkDevice device, VkImage image, VkMemoryRequirements *reqs)
{
    struct mx_img *img = (struct mx_img *)image;
    (void)device;
    memset(reqs, 0, sizeof *reqs);
    reqs->size = img->size;
    reqs->alignment = 16;
    reqs->memoryTypeBits = 1;
}

static VkResult bind_image(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset)
{
    struct mx_img *img = (struct mx_img *)image;
    (void)device;
    struct mx_mem *mem = (struct mx_mem *)memory;
    VkDeviceSize bytes;
    if (!img->width || !img->height)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    bytes = img->size;
    if (!mem || offset % 16u || offset > mem->size || bytes > mem->size - offset)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    img->mem = mem;
    img->offset = offset;
    return VK_SUCCESS;
}

static void destroy_image(VkDevice device, VkImage image, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(image);
}

static VkResult create_view(VkDevice device, const VkImageViewCreateInfo *info, const VkAllocationCallbacks *alloc, VkImageView *out)
{
    struct mx_img *img = (struct mx_img *)info->image;
    struct mx_view *view;
    const VkImageSubresourceRange *range = &info->subresourceRange;
    VkDeviceSize offset = 0;
    uint32_t width, height, depth, level;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (!img || info->format != img->format || !range->aspectMask ||
        (range->aspectMask & ~mx_vk_format_aspects(img->format)) ||
        range->baseMipLevel >= img->levels || range->baseArrayLayer >= img->layers ||
        !range->levelCount || !range->layerCount ||
        (range->levelCount != VK_REMAINING_MIP_LEVELS && range->levelCount > img->levels - range->baseMipLevel) ||
        (range->layerCount != VK_REMAINING_ARRAY_LAYERS && range->layerCount > img->layers - range->baseArrayLayer))
        return VK_ERROR_INITIALIZATION_FAILED;
    if ((unsigned)info->components.r > VK_COMPONENT_SWIZZLE_A ||
        (unsigned)info->components.g > VK_COMPONENT_SWIZZLE_A ||
        (unsigned)info->components.b > VK_COMPONENT_SWIZZLE_A ||
        (unsigned)info->components.a > VK_COMPONENT_SWIZZLE_A)
        return VK_ERROR_INITIALIZATION_FAILED;
    width = img->width;
    height = img->height;
    depth = img->depth;
    for (level = 0; level < range->baseMipLevel; level++) {
        offset += (VkDeviceSize)width * height * depth * vk_format_bytes(img->format) * img->samples * img->layers;
        width = width > 1 ? width / 2 : 1;
        height = height > 1 ? height / 2 : 1;
        depth = depth > 1 ? depth / 2 : 1;
    }
    offset += (VkDeviceSize)width * height * depth * vk_format_bytes(img->format) * img->samples * range->baseArrayLayer;
    view = calloc(1, sizeof *view);
    if (!view)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    view->image = img;
    view->components = info->components;
    view->width = width;
    view->height = height;
    view->offset = offset;
    view->format = info->format;
    view->aspects = range->aspectMask;
    *out = (VkImageView)view;
    return VK_SUCCESS;
}

static void destroy_view(VkDevice device, VkImageView view, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(view);
}

static VkResult create_sampler(VkDevice device, const VkSamplerCreateInfo *info, const VkAllocationCallbacks *alloc, VkSampler *out)
{
    struct mx_sampler *sampler;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    sampler = calloc(1, sizeof *sampler);
    if (!sampler)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    sampler->info = *info;
    sampler->info.pNext = NULL;
    *out = (VkSampler)sampler;
    return VK_SUCCESS;
}

static void destroy_sampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(sampler);
}

static VkResult create_shader(VkDevice device, const VkShaderModuleCreateInfo *info, const VkAllocationCallbacks *alloc, VkShaderModule *out)
{
    struct mx_shader *shader;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (!info->pCode || info->codeSize < 4 || info->codeSize % 4 || info->codeSize > UINT32_MAX)
        return VK_ERROR_INVALID_SHADER_NV;
    shader = calloc(1, sizeof *shader);
    if (!shader)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (info->pCode && info->codeSize >= 4) {
        const uint32_t *words = (const uint32_t *)info->pCode;
        if (words[0] == 0x07230203u) {
            if (info->codeSize < 20 || info->codeSize % 4 || info->codeSize > UINT32_MAX) {
                free(shader);
                return VK_ERROR_INVALID_SHADER_NV;
            }
            shader->bytes = malloc(info->codeSize);
            if (!shader->bytes) {
                free(shader);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            memcpy(shader->bytes, info->pCode, info->codeSize);
            shader->len = (uint32_t)info->codeSize;
            shader->spirv = 1;
        } else if (words[0] == MXSB_MAGIC) {
            struct mxsb_limits limits;
            mxsb_limits_default(&limits);
            if (mxsb_verify((const uint8_t *)info->pCode, (uint32_t)info->codeSize, &limits) != MXSB_OK) {
                free(shader);
                return VK_ERROR_INVALID_SHADER_NV;
            }
            shader->len = (uint32_t)info->codeSize;
            shader->bytes = malloc(shader->len);
            if (!shader->bytes) {
                free(shader);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            memcpy(shader->bytes, info->pCode, shader->len);
            shader->samples = 1;
        } else {
            free(shader);
            return VK_ERROR_INVALID_SHADER_NV;
        }
    }
    *out = (VkShaderModule)shader;
    return VK_SUCCESS;
}

static void destroy_shader(VkDevice device, VkShaderModule shader, const VkAllocationCallbacks *alloc)
{
    struct mx_shader *sh = (struct mx_shader *)shader;
    (void)device;
    (void)alloc;
    if (sh)
        free(sh->bytes);
    free(sh);
}

static int push_range_order(const void *left, const void *right)
{
    const VkPushConstantRange *a = left, *b = right;
    if (a->offset != b->offset)
        return a->offset < b->offset ? -1 : 1;
    if (a->size != b->size)
        return a->size < b->size ? -1 : 1;
    return a->stageFlags < b->stageFlags ? -1 : a->stageFlags > b->stageFlags;
}

static VkResult create_layout(VkDevice device, const VkPipelineLayoutCreateInfo *info, const VkAllocationCallbacks *alloc, VkPipelineLayout *out)
{
    struct mx_pipeline_layout *layout;
    VkShaderStageFlags stages = 0;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (!info || (info->setLayoutCount && !info->pSetLayouts) ||
        (info->pushConstantRangeCount && !info->pPushConstantRanges))
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < info->pushConstantRangeCount; i++) {
        const VkPushConstantRange *range = &info->pPushConstantRanges[i];
        if (!range->stageFlags || (stages & range->stageFlags) ||
            !range->size || range->offset % 4 || range->size % 4 ||
            range->offset > MXGPU_VK_PUSH_CONSTANT_BYTES ||
            range->size > MXGPU_VK_PUSH_CONSTANT_BYTES - range->offset)
            return VK_ERROR_INITIALIZATION_FAILED;
        stages |= range->stageFlags;
    }
    layout = calloc(1, sizeof *layout);
    if (!layout)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    atomic_init(&layout->references, 1);
    if (info->setLayoutCount) {
        layout->sets = calloc(info->setLayoutCount, sizeof *layout->sets);
        if (!layout->sets)
            goto allocation_failure;
        layout->set_count = info->setLayoutCount;
        for (uint32_t i = 0; i < layout->set_count; i++) {
            const struct mx_ds_layout *set = (const struct mx_ds_layout *)info->pSetLayouts[i];
            if (!set) {
                release_layout(layout);
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            layout->sets[i].count = set->count;
            if (!set->count)
                continue;
            layout->sets[i].descriptors = malloc((size_t)set->count * sizeof *set->descriptors);
            if (!layout->sets[i].descriptors)
                goto allocation_failure;
            memcpy(layout->sets[i].descriptors, set->descriptors,
                   (size_t)set->count * sizeof *set->descriptors);
        }
    }
    if (info->pushConstantRangeCount) {
        layout->ranges = calloc(info->pushConstantRangeCount, sizeof *layout->ranges);
        if (!layout->ranges)
            goto allocation_failure;
        layout->range_count = info->pushConstantRangeCount;
        memcpy(layout->ranges, info->pPushConstantRanges,
               (size_t)layout->range_count * sizeof *layout->ranges);
        qsort(layout->ranges, layout->range_count, sizeof *layout->ranges, push_range_order);
        for (uint32_t i = 0; i < layout->range_count; i++) {
            const VkPushConstantRange *range = &layout->ranges[i];
            for (uint32_t word = range->offset / 4; word < (range->offset + range->size) / 4; word++)
                layout->coverage[word] |= range->stageFlags;
        }
    }
    *out = (VkPipelineLayout)layout;
    return VK_SUCCESS;
allocation_failure:
    release_layout(layout);
    return VK_ERROR_OUT_OF_HOST_MEMORY;
}

static void destroy_layout(VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    release_layout((struct mx_pipeline_layout *)layout);
}

static VkResult create_renderpass(VkDevice device, const VkRenderPassCreateInfo *info, const VkAllocationCallbacks *alloc, VkRenderPass *out)
{
    struct mx_renderpass *pass;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (info->subpassCount != 1 || !info->pSubpasses ||
        (info->attachmentCount && !info->pAttachments))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    const VkSubpassDescription *subpass = info->pSubpasses;
    if (subpass->pipelineBindPoint != VK_PIPELINE_BIND_POINT_GRAPHICS ||
        subpass->colorAttachmentCount > 1 || subpass->inputAttachmentCount ||
        subpass->pResolveAttachments ||
        (subpass->colorAttachmentCount && !subpass->pColorAttachments))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    pass = calloc(1, sizeof *pass);
    if (!pass)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    pass->color_attachment = VK_ATTACHMENT_UNUSED;
    pass->depth_attachment = VK_ATTACHMENT_UNUSED;
    if (subpass->colorAttachmentCount) {
        uint32_t attachment = subpass->pColorAttachments[0].attachment;
        if (attachment != VK_ATTACHMENT_UNUSED) {
            if (attachment >= info->attachmentCount ||
                vk_color_format(info->pAttachments[attachment].format) == PIPE_FORMAT_NONE ||
                info->pAttachments[attachment].samples != VK_SAMPLE_COUNT_1_BIT)
                goto unsupported;
            pass->color_attachment = attachment;
            pass->load = info->pAttachments[attachment].loadOp;
            pass->color_format = info->pAttachments[attachment].format;
        }
    }
    if (subpass->pDepthStencilAttachment &&
        subpass->pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED) {
        uint32_t attachment = subpass->pDepthStencilAttachment->attachment;
        if (attachment >= info->attachmentCount ||
            !vk_depth_format(info->pAttachments[attachment].format) ||
            info->pAttachments[attachment].samples != VK_SAMPLE_COUNT_1_BIT)
            goto unsupported;
        pass->depth_attachment = attachment;
        pass->depth_format = info->pAttachments[attachment].format;
        pass->depth_load = info->pAttachments[attachment].loadOp;
        pass->stencil_load = info->pAttachments[attachment].stencilLoadOp;
    }
    *out = (VkRenderPass)pass;
    return VK_SUCCESS;
unsupported:
    free(pass);
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

static void destroy_renderpass(VkDevice device, VkRenderPass pass, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(pass);
}

static int compile_spirv_stage(const VkPipelineShaderStageCreateInfo *stage, struct mxgpu_shader *compiled)
{
    struct mx_shader *shader = (struct mx_shader *)stage->module;
    struct spirv_to_nir_options options = {
        .environment = NIR_SPIRV_VULKAN,
        .skip_os_break_in_debug_build = true,
    };
    static const nir_shader_compiler_options nir_options = {0};
    struct nir_spirv_specialization spec = {0};
    nir_shader *nir;
    int result;
    uint32_t i;
    mesa_shader_stage mesa_stage = stage->stage == VK_SHADER_STAGE_VERTEX_BIT ? MESA_SHADER_VERTEX :
                                   stage->stage == VK_SHADER_STAGE_COMPUTE_BIT ? MESA_SHADER_COMPUTE : MESA_SHADER_FRAGMENT;
    if (!shader || !shader->spirv || !stage->pName)
        return -1;
    if (stage->pSpecializationInfo) {
        const VkSpecializationInfo *info = stage->pSpecializationInfo;
        if (info->mapEntryCount && (!info->pMapEntries || !info->pData))
            return -1;
        spec.num_entries = info->mapEntryCount;
        spec.entries = calloc(spec.num_entries, sizeof *spec.entries);
        if (spec.num_entries && !spec.entries)
            return -1;
        for (i = 0; i < spec.num_entries; i++) {
            const VkSpecializationMapEntry *entry = &info->pMapEntries[i];
            if (entry->offset > info->dataSize || entry->size > info->dataSize - entry->offset || entry->size > UINT32_MAX) {
                free(spec.entries);
                return -1;
            }
            spec.entries[i].id = entry->constantID;
            spec.entries[i].size = (uint32_t)entry->size;
            spec.entries[i].data = (uint8_t *)info->pData + entry->offset;
        }
    }
    glsl_type_singleton_init_or_ref();
    nir = spirv_to_nir((const uint32_t *)shader->bytes, shader->len / 4u, &spec,
                       mesa_stage, stage->pName, &options, &nir_options);
    free(spec.entries);
    if (!nir) {
        glsl_type_singleton_decref();
        return -1;
    }
    nir_split_var_copies(nir);
    nir_lower_var_copies(nir);
    nir_lower_vars_to_ssa(nir);
    nir_opt_copy_prop_vars(nir);
    nir_opt_dce(nir);
    nir_lower_explicit_io(nir, nir_var_mem_push_const, nir_address_format_32bit_offset);
    nir_opt_constant_folding(nir);
    nir_opt_dce(nir);
    result = mxgpu_compile_nir(nir, mesa_stage == MESA_SHADER_FRAGMENT, compiled);
    ralloc_free(nir);
    glsl_type_singleton_decref();
    return result;
}


struct mx_cache_entry {
    struct mx_cache_entry *next;
    uint8_t *key;
    uint32_t key_size;
    struct mxgpu_shader compiled;
};

struct mx_pipeline_cache {
    VkDevice device;
    VkAllocationCallbacks allocator;
    bool custom_allocator;
    pthread_mutex_t mutex;
    struct mx_cache_entry *entries;
};

static void *cache_alloc(struct mx_pipeline_cache *cache, size_t size, size_t alignment)
{
    if (cache->custom_allocator)
        return cache->allocator.pfnAllocation(cache->allocator.pUserData, size, alignment,
                                              VK_SYSTEM_ALLOCATION_SCOPE_CACHE);
    return malloc(size);
}

static void cache_free(struct mx_pipeline_cache *cache, void *ptr)
{
    if (!ptr)
        return;
    if (cache->custom_allocator)
        cache->allocator.pfnFree(cache->allocator.pUserData, ptr);
    else
        free(ptr);
}

static uint32_t cache_read_u32(const uint8_t *data)
{
    return (uint32_t)data[0] | (uint32_t)data[1] << 8 |
           (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static void cache_write_u32(uint8_t *data, uint32_t value)
{
    for (unsigned i = 0; i < 4; i++)
        data[i] = (uint8_t)(value >> (8 * i));
}

static struct mx_cache_entry *cache_find(struct mx_pipeline_cache *cache,
                                        const uint8_t *key, uint32_t size)
{
    for (struct mx_cache_entry *entry = cache->entries; entry; entry = entry->next)
        if (entry->key_size == size && !memcmp(entry->key, key, size))
            return entry;
    return NULL;
}

static VkResult cache_insert(struct mx_pipeline_cache *cache, const uint8_t *key,
                              uint32_t size, const struct mxgpu_shader *compiled)
{
    if (cache_find(cache, key, size))
        return VK_SUCCESS;
    struct mx_cache_entry *entry = cache_alloc(cache, sizeof *entry, _Alignof(struct mx_cache_entry));
    if (!entry)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    entry->key = cache_alloc(cache, size, _Alignof(uint32_t));
    if (!entry->key) {
        cache_free(cache, entry);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    memcpy(entry->key, key, size);
    entry->key_size = size;
    entry->compiled = *compiled;
    entry->next = NULL;
    struct mx_cache_entry **tail = &cache->entries;
    while (*tail)
        tail = &(*tail)->next;
    *tail = entry;
    return VK_SUCCESS;
}

static uint8_t *cache_stage_key(const VkPipelineShaderStageCreateInfo *stage, uint32_t *size)
{
    struct mx_shader *shader = (struct mx_shader *)stage->module;
    const VkSpecializationInfo *spec = stage->pSpecializationInfo;
    size_t name_size = stage->pName ? strlen(stage->pName) + 1 : 0;
    uint32_t count = spec ? spec->mapEntryCount : 0;
    size_t data_size = spec ? spec->dataSize : 0;
    if (!shader || !shader->spirv || !shader->bytes || !name_size || name_size > UINT32_MAX ||
        data_size > UINT32_MAX || (data_size && !spec->pData) || (count && (!spec->pMapEntries || !spec->pData)))
        return NULL;
    uint64_t total = 24ull + shader->len + name_size + 12ull * count + data_size;
    if (total > UINT32_MAX)
        return NULL;
    for (unsigned i = 0; i < count; i++)
        if (spec->pMapEntries[i].offset > data_size ||
            spec->pMapEntries[i].size > data_size - spec->pMapEntries[i].offset ||
            spec->pMapEntries[i].size > UINT32_MAX)
            return NULL;
    uint8_t *key = malloc((size_t)total);
    if (!key)
        return NULL;
    cache_write_u32(key, stage->stage);
    cache_write_u32(key + 4, stage->flags);
    cache_write_u32(key + 8, shader->len);
    cache_write_u32(key + 12, (uint32_t)name_size);
    cache_write_u32(key + 16, count);
    cache_write_u32(key + 20, (uint32_t)data_size);
    uint8_t *cursor = key + 24;
    memcpy(cursor, shader->bytes, shader->len);
    cursor += shader->len;
    memcpy(cursor, stage->pName, name_size);
    cursor += name_size;
    for (unsigned i = 0; i < count; i++) {
        cache_write_u32(cursor, spec->pMapEntries[i].constantID);
        cache_write_u32(cursor + 4, spec->pMapEntries[i].offset);
        cache_write_u32(cursor + 8, (uint32_t)spec->pMapEntries[i].size);
        cursor += 12;
    }
    if (data_size)
        memcpy(cursor, spec->pData, data_size);
    *size = (uint32_t)total;
    return key;
}

static int cached_spirv_stage(struct mx_pipeline_cache *cache,
                              const VkPipelineShaderStageCreateInfo *stage,
                              struct mxgpu_shader *compiled)
{
    uint32_t key_size;
    uint8_t *key = cache ? cache_stage_key(stage, &key_size) : NULL;
    if (key) {
        pthread_mutex_lock(&cache->mutex);
        struct mx_cache_entry *entry = cache_find(cache, key, key_size);
        if (entry)
            *compiled = entry->compiled;
        pthread_mutex_unlock(&cache->mutex);
        if (entry) {
            free(key);
            return 0;
        }
    }
    int result = compile_spirv_stage(stage, compiled);
    if (!result && key) {
        pthread_mutex_lock(&cache->mutex);
        cache_insert(cache, key, key_size, compiled);
        pthread_mutex_unlock(&cache->mutex);
    }
    free(key);
    return result;
}

static void destroy_pipeline_cache(VkDevice device, VkPipelineCache handle,
                                    const VkAllocationCallbacks *alloc)
{
    struct mx_pipeline_cache *cache = (struct mx_pipeline_cache *)handle;
    (void)device;
    (void)alloc;
    if (!cache)
        return;
    while (cache->entries) {
        struct mx_cache_entry *entry = cache->entries;
        cache->entries = entry->next;
        cache_free(cache, entry->key);
        cache_free(cache, entry);
    }
    pthread_mutex_destroy(&cache->mutex);
    bool custom = cache->custom_allocator;
    VkAllocationCallbacks allocator = cache->allocator;
    if (custom)
        allocator.pfnFree(allocator.pUserData, cache);
    else
        free(cache);
}

static VkResult cache_import_entry(struct mx_pipeline_cache *cache,
                                   const uint8_t *key, uint32_t size)
{
    if (size < 24)
        return VK_ERROR_INVALID_SHADER_NV;
    uint32_t stage_kind = cache_read_u32(key), flags = cache_read_u32(key + 4);
    uint32_t shader_size = cache_read_u32(key + 8), name_size = cache_read_u32(key + 12);
    uint32_t count = cache_read_u32(key + 16), data_size = cache_read_u32(key + 20);
    uint64_t required = 24ull + shader_size + name_size + 12ull * count + data_size;
    if (required != size || !shader_size || (shader_size & 3) || !name_size ||
        (stage_kind != VK_SHADER_STAGE_VERTEX_BIT && stage_kind != VK_SHADER_STAGE_FRAGMENT_BIT &&
         stage_kind != VK_SHADER_STAGE_COMPUTE_BIT))
        return VK_ERROR_INVALID_SHADER_NV;
    const uint8_t *name = key + 24 + shader_size;
    if (name[name_size - 1] || memchr(name, 0, name_size - 1))
        return VK_ERROR_INVALID_SHADER_NV;
    struct mx_shader shader = {.spirv = 1, .len = shader_size};
    shader.bytes = malloc(shader_size);
    VkSpecializationMapEntry *maps = count ? calloc(count, sizeof *maps) : NULL;
    struct mxgpu_shader *compiled = malloc(sizeof *compiled);
    if (!shader.bytes || (count && !maps) || !compiled) {
        free(shader.bytes); free(maps); free(compiled);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    memcpy(shader.bytes, key + 24, shader_size);
    const uint8_t *cursor = name + name_size;
    VkResult result = VK_ERROR_INVALID_SHADER_NV;
    for (unsigned i = 0; i < count; i++, cursor += 12) {
        maps[i].constantID = cache_read_u32(cursor);
        maps[i].offset = cache_read_u32(cursor + 4);
        maps[i].size = cache_read_u32(cursor + 8);
        if (maps[i].offset > data_size || maps[i].size > data_size - maps[i].offset)
            goto done;
    }
    VkSpecializationInfo spec = {count, maps, data_size, cursor};
    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .flags = flags, .stage = stage_kind, .module = (VkShaderModule)&shader,
        .pName = (const char *)name, .pSpecializationInfo = count || data_size ? &spec : NULL
    };
    memset(compiled, 0, sizeof *compiled);
    if (!compile_spirv_stage(&stage, compiled))
        result = cache_insert(cache, key, size, compiled);
done:
    free(shader.bytes); free(maps); free(compiled);
    return result;
}

static VkResult create_pipeline_cache(VkDevice device, const VkPipelineCacheCreateInfo *info,
                                      const VkAllocationCallbacks *alloc, VkPipelineCache *out)
{
    *out = VK_NULL_HANDLE;
    struct mx_pipeline_cache *cache = alloc ? alloc->pfnAllocation(alloc->pUserData,
        sizeof *cache, _Alignof(struct mx_pipeline_cache), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) : malloc(sizeof *cache);
    if (!cache)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    memset(cache, 0, sizeof *cache);
    cache->device = device;
    cache->custom_allocator = alloc != NULL;
    if (alloc)
        cache->allocator = *alloc;
    if (pthread_mutex_init(&cache->mutex, NULL)) {
        if (alloc) alloc->pfnFree(alloc->pUserData, cache); else free(cache);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const uint8_t *data = info->pInitialData;
    size_t size = info->initialDataSize;
    if (size >= 40 && data && cache_read_u32(data) == 32 &&
        cache_read_u32(data + 4) == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
        cache_read_u32(data + 8) == MX_PCI_VENDOR_ID && cache_read_u32(data + 12) == MXGPU_PCI_DEVICE_ID &&
        !memcmp(data + 16, mx_pipeline_cache_uuid, VK_UUID_SIZE) && cache_read_u32(data + 32) == 1) {
        uint32_t count = cache_read_u32(data + 36);
        size_t cursor = 40;
        bool valid = true;
        for (uint32_t i = 0; i < count; i++) {
            if (cursor > size || size - cursor < 4) { valid = false; break; }
            uint32_t bytes = cache_read_u32(data + cursor);
            cursor += 4;
            if (bytes > size - cursor) { valid = false; break; }
            VkResult result = cache_import_entry(cache, data + cursor, bytes);
            if (result == VK_ERROR_OUT_OF_HOST_MEMORY) {
                destroy_pipeline_cache(device, (VkPipelineCache)cache, alloc);
                return result;
            }
            if (result != VK_SUCCESS) { valid = false; break; }
            cursor += bytes;
        }
        if (!valid || cursor != size) {
            while (cache->entries) {
                struct mx_cache_entry *entry = cache->entries;
                cache->entries = entry->next;
                cache_free(cache, entry->key); cache_free(cache, entry);
            }
        }
    }
    *out = (VkPipelineCache)cache;
    return VK_SUCCESS;
}

static VkResult get_pipeline_cache_data(VkDevice device, VkPipelineCache handle,
                                        size_t *size, void *data)
{
    struct mx_pipeline_cache *cache = (struct mx_pipeline_cache *)handle;
    (void)device;
    pthread_mutex_lock(&cache->mutex);
    size_t required = 40;
    for (struct mx_cache_entry *entry = cache->entries; entry; entry = entry->next) {
        if (required > SIZE_MAX - 4 || entry->key_size > SIZE_MAX - required - 4) {
            pthread_mutex_unlock(&cache->mutex);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        required += 4 + entry->key_size;
    }
    if (!data) {
        *size = required;
        pthread_mutex_unlock(&cache->mutex);
        return VK_SUCCESS;
    }
    size_t capacity = *size, used = 0;
    uint8_t *output = data;
    uint32_t written = 0;
    if (capacity >= 32) {
        cache_write_u32(output, 32);
        cache_write_u32(output + 4, VK_PIPELINE_CACHE_HEADER_VERSION_ONE);
        cache_write_u32(output + 8, MX_PCI_VENDOR_ID);
        cache_write_u32(output + 12, MXGPU_PCI_DEVICE_ID);
        memcpy(output + 16, mx_pipeline_cache_uuid, VK_UUID_SIZE);
        used = 32;
    }
    if (capacity >= 40) {
        cache_write_u32(output + 32, 1);
        used = 40;
        for (struct mx_cache_entry *entry = cache->entries; entry; entry = entry->next) {
            if ((uint64_t)entry->key_size + 4 > capacity - used)
                break;
            cache_write_u32(output + used, entry->key_size);
            memcpy(output + used + 4, entry->key, entry->key_size);
            used += 4 + entry->key_size;
            written++;
        }
        cache_write_u32(output + 36, written);
    }
    *size = used;
    pthread_mutex_unlock(&cache->mutex);
    return used == required ? VK_SUCCESS : VK_INCOMPLETE;
}

static VkResult merge_pipeline_caches(VkDevice device, VkPipelineCache destination,
                                      uint32_t count, const VkPipelineCache *sources)
{
    struct mx_pipeline_cache *dst = (struct mx_pipeline_cache *)destination;
    (void)device;
    for (uint32_t i = 0; i < count; i++) {
        struct mx_pipeline_cache *src = (struct mx_pipeline_cache *)sources[i];
        if (src == dst)
            continue;
        struct mx_pipeline_cache *first = (uintptr_t)dst < (uintptr_t)src ? dst : src;
        struct mx_pipeline_cache *second = first == dst ? src : dst;
        pthread_mutex_lock(&first->mutex);
        pthread_mutex_lock(&second->mutex);
        VkResult result = VK_SUCCESS;
        for (struct mx_cache_entry *entry = src->entries; entry && result == VK_SUCCESS; entry = entry->next)
            result = cache_insert(dst, entry->key, entry->key_size, &entry->compiled);
        pthread_mutex_unlock(&second->mutex);
        pthread_mutex_unlock(&first->mutex);
        if (result != VK_SUCCESS)
            return result;
    }
    return VK_SUCCESS;
}

static uint32_t vk_float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static uint8_t vk_native_blend_factor(VkBlendFactor factor)
{
    switch (factor) {
    case VK_BLEND_FACTOR_ZERO: return MXGPU_BLEND_ZERO;
    case VK_BLEND_FACTOR_ONE: return MXGPU_BLEND_ONE;
    case VK_BLEND_FACTOR_SRC_COLOR: return MXGPU_BLEND_SRC_COLOR;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return MXGPU_BLEND_INV_SRC_COLOR;
    case VK_BLEND_FACTOR_DST_COLOR: return MXGPU_BLEND_DST_COLOR;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return MXGPU_BLEND_INV_DST_COLOR;
    case VK_BLEND_FACTOR_SRC_ALPHA: return MXGPU_BLEND_SRC_ALPHA;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return MXGPU_BLEND_INV_SRC_ALPHA;
    case VK_BLEND_FACTOR_DST_ALPHA: return MXGPU_BLEND_DST_ALPHA;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MXGPU_BLEND_INV_DST_ALPHA;
    case VK_BLEND_FACTOR_CONSTANT_COLOR: return MXGPU_BLEND_CONSTANT_COLOR;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return MXGPU_BLEND_INV_CONSTANT_COLOR;
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return MXGPU_BLEND_CONSTANT_ALPHA;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return MXGPU_BLEND_INV_CONSTANT_ALPHA;
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return MXGPU_BLEND_SRC_ALPHA_SATURATE;
    default: return 0;
    }
}

static uint8_t vk_native_blend_op(VkBlendOp op)
{
    switch (op) {
    case VK_BLEND_OP_ADD: return MXGPU_BLEND_ADD;
    case VK_BLEND_OP_SUBTRACT: return MXGPU_BLEND_SUBTRACT;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return MXGPU_BLEND_REVERSE_SUBTRACT;
    case VK_BLEND_OP_MIN: return MXGPU_BLEND_MIN;
    case VK_BLEND_OP_MAX: return MXGPU_BLEND_MAX;
    default: return 0;
    }
}

static uint8_t vk_native_compare(VkCompareOp op)
{
    switch (op) {
    case VK_COMPARE_OP_NEVER: return MXGPU_COMPARE_NEVER;
    case VK_COMPARE_OP_LESS: return MXGPU_COMPARE_LESS;
    case VK_COMPARE_OP_EQUAL: return MXGPU_COMPARE_EQUAL;
    case VK_COMPARE_OP_LESS_OR_EQUAL: return MXGPU_COMPARE_LESS_EQUAL;
    case VK_COMPARE_OP_GREATER: return MXGPU_COMPARE_GREATER;
    case VK_COMPARE_OP_NOT_EQUAL: return MXGPU_COMPARE_NOT_EQUAL;
    case VK_COMPARE_OP_GREATER_OR_EQUAL: return MXGPU_COMPARE_GREATER_EQUAL;
    case VK_COMPARE_OP_ALWAYS: return MXGPU_COMPARE_ALWAYS;
    default: return 0;
    }
}

static uint8_t vk_native_stencil_op(VkStencilOp op)
{
    switch (op) {
    case VK_STENCIL_OP_KEEP: return MXGPU_STENCIL_KEEP;
    case VK_STENCIL_OP_ZERO: return MXGPU_STENCIL_ZERO;
    case VK_STENCIL_OP_REPLACE: return MXGPU_STENCIL_REPLACE;
    case VK_STENCIL_OP_INCREMENT_AND_CLAMP: return MXGPU_STENCIL_INCREMENT_CLAMP;
    case VK_STENCIL_OP_DECREMENT_AND_CLAMP: return MXGPU_STENCIL_DECREMENT_CLAMP;
    case VK_STENCIL_OP_INVERT: return MXGPU_STENCIL_INVERT;
    case VK_STENCIL_OP_INCREMENT_AND_WRAP: return MXGPU_STENCIL_INCREMENT_WRAP;
    case VK_STENCIL_OP_DECREMENT_AND_WRAP: return MXGPU_STENCIL_DECREMENT_WRAP;
    default: return 0;
    }
}

static bool vk_pipeline_depth_state(struct mx_pipe *pipe, const VkPipelineDepthStencilStateCreateInfo *info)
{
    struct mxgpu_depth_stencil_state *state = &pipe->depth_stencil;
    memset(state, 0, sizeof *state);
    pipe->depth_enabled = false;
    pipe->stencil_reference = 0;
    state->state_id = 1;
    state->depth_compare = MXGPU_COMPARE_ALWAYS;
    state->front = state->back = (struct mxgpu_stencil_face){
        MXGPU_STENCIL_KEEP, MXGPU_STENCIL_KEEP, MXGPU_STENCIL_KEEP, MXGPU_COMPARE_ALWAYS};
    if (!info)
        return true;
    if (info->depthBoundsTestEnable)
        return false;
    state->depth_test_enable = info->depthTestEnable;
    state->depth_write_enable = info->depthTestEnable && info->depthWriteEnable;
    state->stencil_enable = info->stencilTestEnable;
    if (state->depth_test_enable) {
        state->depth_compare = vk_native_compare(info->depthCompareOp);
        if (!state->depth_compare)
            return false;
    }
    if (state->stencil_enable) {
        const VkStencilOpState *faces[2] = {&info->front, &info->back};
        struct mxgpu_stencil_face *targets[2] = {&state->front, &state->back};
        if ((!pipe->dynamic_stencil_compare && (info->front.compareMask & 255u) != (info->back.compareMask & 255u)) ||
            (!pipe->dynamic_stencil_write && (info->front.writeMask & 255u) != (info->back.writeMask & 255u)) ||
            (!pipe->dynamic_stencil_reference && (info->front.reference & 255u) != (info->back.reference & 255u)))
            return false;
        state->stencil_read_mask = info->front.compareMask;
        state->stencil_write_mask = info->front.writeMask;
        pipe->stencil_reference = info->front.reference & 255u;
        for (unsigned i = 0; i < 2; i++) {
            pipe->stencil_compare[i] = faces[i]->compareMask & 255u;
            pipe->stencil_write[i] = faces[i]->writeMask & 255u;
            pipe->stencil_ref[i] = faces[i]->reference & 255u;
            targets[i]->fail = vk_native_stencil_op(faces[i]->failOp);
            targets[i]->depth_fail = vk_native_stencil_op(faces[i]->depthFailOp);
            targets[i]->pass = vk_native_stencil_op(faces[i]->passOp);
            targets[i]->compare = vk_native_compare(faces[i]->compareOp);
            if (!targets[i]->fail || !targets[i]->depth_fail || !targets[i]->pass || !targets[i]->compare)
                return false;
        }
    }
    pipe->depth_enabled = state->depth_test_enable || state->stencil_enable;
    return true;
}

static bool vk_pipeline_render_state(struct mx_pipe *pipe, const VkGraphicsPipelineCreateInfo *info)
{
    struct mxgpu_native_render_state *state = &pipe->native_state;
    struct mxgpu_blend_target *target = &state->blend.targets[0];
    state->blend.state_id = state->rasterizer.state_id = 1;
    state->blend.target_count = 1;
    target->write_mask = 15;
    target->src_color = target->src_alpha = MXGPU_BLEND_ONE;
    target->dst_color = target->dst_alpha = MXGPU_BLEND_ZERO;
    target->color_op = target->alpha_op = MXGPU_BLEND_ADD;
    state->rasterizer.fill_mode = MXGPU_FILL_SOLID;
    state->rasterizer.cull_mode = MXGPU_CULL_NONE;
    state->rasterizer.front_face = MXGPU_FRONT_COUNTERCLOCKWISE;
    state->rasterizer.depth_clip_enable = state->rasterizer.scissor_enable = 1;
    if (info->pDynamicState) {
        if (info->pDynamicState->dynamicStateCount && !info->pDynamicState->pDynamicStates)
            return false;
        for (uint32_t i = 0; i < info->pDynamicState->dynamicStateCount; i++) {
            switch (info->pDynamicState->pDynamicStates[i]) {
            case VK_DYNAMIC_STATE_VIEWPORT: pipe->dynamic_viewport = true; break;
            case VK_DYNAMIC_STATE_SCISSOR: pipe->dynamic_scissor = true; break;
            case VK_DYNAMIC_STATE_BLEND_CONSTANTS: pipe->dynamic_blend = true; break;
            case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK: pipe->dynamic_stencil_compare = true; break;
            case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK: pipe->dynamic_stencil_write = true; break;
            case VK_DYNAMIC_STATE_STENCIL_REFERENCE: pipe->dynamic_stencil_reference = true; break;
            default: return false;
            }
        }
    }
    if (info->pColorBlendState) {
        const VkPipelineColorBlendStateCreateInfo *blend = info->pColorBlendState;
        if (blend->logicOpEnable || blend->attachmentCount > 1 ||
            (blend->attachmentCount && !blend->pAttachments))
            return false;
        const VkPipelineColorBlendAttachmentState *b = blend->pAttachments;
        target->enable = b && blend->attachmentCount && b->blendEnable;
        target->write_mask = blend->attachmentCount ? b->colorWriteMask : 0;
        if (target->write_mask & ~15u) return false;
        if (target->enable) {
            target->src_color = vk_native_blend_factor(b->srcColorBlendFactor);
            target->dst_color = vk_native_blend_factor(b->dstColorBlendFactor);
            target->color_op = vk_native_blend_op(b->colorBlendOp);
            target->src_alpha = vk_native_blend_factor(b->srcAlphaBlendFactor);
            target->dst_alpha = vk_native_blend_factor(b->dstAlphaBlendFactor);
            target->alpha_op = vk_native_blend_op(b->alphaBlendOp);
            if (!target->src_color || !target->dst_color || !target->color_op ||
                !target->src_alpha || !target->dst_alpha || !target->alpha_op)
                return false;
        }
        for (unsigned i = 0; i < 4; i++) {
            state->blend_factor[i] = vk_float_bits(blend->blendConstants[i]);
            if (!mxgpu_f32_finite(state->blend_factor[i])) return false;
        }
    }
    if (info->pRasterizationState) {
        const VkPipelineRasterizationStateCreateInfo *r = info->pRasterizationState;
        if (r->rasterizerDiscardEnable || r->depthBiasEnable || r->lineWidth != 1.f ||
            r->polygonMode != VK_POLYGON_MODE_FILL ||
            (r->cullMode != VK_CULL_MODE_NONE && r->cullMode != VK_CULL_MODE_FRONT_BIT &&
             r->cullMode != VK_CULL_MODE_BACK_BIT) ||
            (r->frontFace != VK_FRONT_FACE_CLOCKWISE && r->frontFace != VK_FRONT_FACE_COUNTER_CLOCKWISE))
            return false;
        state->rasterizer.cull_mode = r->cullMode == VK_CULL_MODE_FRONT_BIT ? MXGPU_CULL_FRONT :
                                     r->cullMode == VK_CULL_MODE_BACK_BIT ? MXGPU_CULL_BACK : MXGPU_CULL_NONE;
        state->rasterizer.front_face = r->frontFace == VK_FRONT_FACE_CLOCKWISE ?
                                      MXGPU_FRONT_CLOCKWISE : MXGPU_FRONT_COUNTERCLOCKWISE;
        state->rasterizer.depth_clip_enable = !r->depthClampEnable;
    }
    if (info->pMultisampleState && (info->pMultisampleState->rasterizationSamples != VK_SAMPLE_COUNT_1_BIT ||
        info->pMultisampleState->sampleShadingEnable || info->pMultisampleState->alphaToCoverageEnable ||
        info->pMultisampleState->alphaToOneEnable ||
        (info->pMultisampleState->pSampleMask && !(info->pMultisampleState->pSampleMask[0] & 1))))
        return false;
    if (!vk_pipeline_depth_state(pipe, info->pDepthStencilState))
        return false;
    if (info->pViewportState) {
        const VkPipelineViewportStateCreateInfo *v = info->pViewportState;
        if (v->viewportCount != 1 || v->scissorCount != 1 ||
            (!pipe->dynamic_viewport && !v->pViewports) || (!pipe->dynamic_scissor && !v->pScissors))
            return false;
        if (!pipe->dynamic_viewport) { pipe->viewport = v->pViewports[0]; pipe->viewport_set = true; }
        if (!pipe->dynamic_scissor) { pipe->scissor = v->pScissors[0]; pipe->scissor_set = true; }
    }
    return true;
}

static VkResult create_pipelines(VkDevice device, VkPipelineCache cache, uint32_t count, const VkGraphicsPipelineCreateInfo *info, const VkAllocationCallbacks *alloc, VkPipeline *out)
{
    VkResult result = VK_SUCCESS;
    uint32_t index;
    (void)device;
    (void)cache;
    (void)alloc;
    for (index = 0; index < count; index++)
        out[index] = VK_NULL_HANDLE;
    for (index = 0; index < count; index++) {
        struct mx_pipe *pipe = calloc(1, sizeof *pipe);
        uint32_t stage;
        struct mxgpu_shader vs = {0}, fs = {0};
        int have_vs = 0, have_fs = 0, spirv_failed = 0;
        struct mx_renderpass *pass = (struct mx_renderpass *)info[index].renderPass;
        bool depth_only = pass && pass->color_attachment == VK_ATTACHMENT_UNUSED &&
                          pass->depth_attachment != VK_ATTACHMENT_UNUSED;
        if (!pipe)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        if (!vk_pipeline_render_state(pipe, &info[index]))
            goto unsupported_vertex;
        if (depth_only)
            pipe->native_state.blend.targets[0].write_mask = 0;
        pipe->topology = info[index].pInputAssemblyState ? info[index].pInputAssemblyState->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipe->primitive_restart = info[index].pInputAssemblyState && info[index].pInputAssemblyState->primitiveRestartEnable;
        const VkPipelineVertexInputStateCreateInfo *input = info[index].pVertexInputState;
        if (input) {
            pipe->vertex_layout = true;
            if ((input->vertexBindingDescriptionCount && !input->pVertexBindingDescriptions) ||
                (input->vertexAttributeDescriptionCount && !input->pVertexAttributeDescriptions))
                goto unsupported_vertex;
            for (unsigned i = 0; i < input->vertexBindingDescriptionCount; i++) {
                const VkVertexInputBindingDescription *binding = &input->pVertexBindingDescriptions[i];
                if (binding->binding >= 32 ||
                    (binding->inputRate != VK_VERTEX_INPUT_RATE_VERTEX && binding->inputRate != VK_VERTEX_INPUT_RATE_INSTANCE) ||
                    (pipe->vertex_binding_mask & (1u << binding->binding)))
                    goto unsupported_vertex;
                pipe->vertex_bindings[binding->binding] = *binding;
                pipe->vertex_binding_mask |= 1u << binding->binding;
            }
            for (unsigned i = 0; i < input->vertexAttributeDescriptionCount; i++) {
                const VkVertexInputAttributeDescription *attribute = &input->pVertexAttributeDescriptions[i];
                if (attribute->location >= 32 || attribute->binding >= 32 ||
                    !(pipe->vertex_binding_mask & (1u << attribute->binding)) ||
                    (pipe->vertex_attribute_mask & (1u << attribute->location)) ||
                    vk_vertex_format(attribute->format) == PIPE_FORMAT_NONE)
                    goto unsupported_vertex;
                pipe->vertex_attributes[attribute->location] = *attribute;
                pipe->vertex_attribute_mask |= 1u << attribute->location;
            }
        }
        if (pipe->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
            pipe->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
            pipe->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)
            goto unsupported_vertex;
        for (stage = 0; stage < info[index].stageCount; stage++) {
            struct mx_shader *shader = (struct mx_shader *)info[index].pStages[stage].module;
            const VkPipelineShaderStageCreateInfo *stage_info = &info[index].pStages[stage];
            if (shader && shader->spirv) {
                if (stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT) {
                    have_vs = cached_spirv_stage((struct mx_pipeline_cache *)cache, stage_info, &vs) == 0;
                    spirv_failed |= !have_vs;
                } else if (stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                    have_fs = cached_spirv_stage((struct mx_pipeline_cache *)cache, stage_info, &fs) == 0;
                    spirv_failed |= !have_fs;
                } else {
                    spirv_failed = 1;
                }
            }
            if (shader && shader->samples && shader->bytes && shader->len && !pipe->bytes) {
                pipe->bytes = malloc(shader->len);
                if (!pipe->bytes) {
                    free(pipe);
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
                memcpy(pipe->bytes, shader->bytes, shader->len);
                pipe->len = shader->len;
                pipe->samples = 1;
                pipe->uses_texture = 1;
            }
        }
        if (have_vs && (have_fs || depth_only) && !spirv_failed) {
            const struct mxgpu_shader *compiled_stages[2] = {&vs, &fs};
            pipe->vertex_attribute_count = vs.vertex_attribute_count;
            if (pipe->vertex_attribute_count > MXGPU_SHADER_VERTEX_SLOTS)
                goto unsupported_compiled;
            memcpy(pipe->vertex_input_locations, vs.vertex_input_locations, sizeof pipe->vertex_input_locations);
            pipe->vertex_builtins = vs.vertex_builtins;
            pipe->vertex_builtin_slot = vs.vertex_builtin_slot;
            for (unsigned uniform_stage = 0; uniform_stage < 2; uniform_stage++) {
                struct mx_uniform_stage *uniforms = &pipe->uniform_stages[uniform_stage];
                const struct mxgpu_shader *compiled = compiled_stages[uniform_stage];
                uniforms->uses_uniforms = compiled->uses_uniforms;
                uniforms->uniform_count = compiled->uniform_count;
                uniforms->uniform_buffer_count = compiled->uniform_buffer_count;
                memcpy(uniforms->uniform_buffers, compiled->uniform_buffers, sizeof uniforms->uniform_buffers);
            }
            pipe->uses_texture = fs.samples;
            pipe->reflected_texture = have_fs;
            pipe->texture_binding = fs.texture_binding;
            pipe->texture_set = fs.texture_set;
            pipe->texture_element = fs.texture_element;
            free(pipe->bytes);
            pipe->bytes = NULL;
            pipe->len = 0;
            pipe->samples = have_vs && (have_fs || depth_only);
        }
        if (!pipe->samples || spirv_failed || (!have_vs && have_fs) ||
            (have_vs && !have_fs && !depth_only)) {
            free(pipe->bytes);
            free(pipe);
            result = VK_ERROR_INVALID_SHADER_NV;
            if (info[index].flags & VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT)
                return result;
            continue;
        }
        if (have_vs) {
            pipe->vertex_shader = malloc(sizeof vs);
            if (have_fs)
                pipe->fragment_shader = malloc(sizeof fs);
            if (!pipe->vertex_shader || (have_fs && !pipe->fragment_shader)) {
                free(pipe->vertex_shader); free(pipe->fragment_shader); free(pipe->bytes); free(pipe);
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            *pipe->vertex_shader = vs;
            if (have_fs)
                *pipe->fragment_shader = fs;
        }
        pipe->layout = (struct mx_pipeline_layout *)info[index].layout;
        retain_layout(pipe->layout);
        out[index] = (VkPipeline)pipe;
        continue;
unsupported_compiled:
        free(pipe->bytes);
unsupported_vertex:
        free(pipe);
        result = VK_ERROR_FEATURE_NOT_PRESENT;
        if (info[index].flags & VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT)
            return result;
    }
    return result;
}

static VkResult create_compute_pipeline(struct mx_pipeline_cache *cache, const VkComputePipelineCreateInfo *info,
                                        struct mx_pipe **out)
{
    struct mx_shader *module = (struct mx_shader *)info->stage.module;
    struct mxgpu_shader *compiled = NULL;
    uint8_t *bytes = NULL;
    uint32_t len = 0;
    VkResult result = VK_ERROR_INVALID_SHADER_NV;
    *out = NULL;
    if (!vk_compute_supported())
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (info->stage.stage != VK_SHADER_STAGE_COMPUTE_BIT || !module || !module->spirv || !info->layout)
        return VK_ERROR_INVALID_SHADER_NV;
    struct mx_pipe *pipe = calloc(1, sizeof *pipe);
    compiled = malloc(sizeof *compiled);
    bytes = malloc(MXGPU_LINK_MODULE_CAPACITY);
    if (!pipe || !compiled || !bytes) {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto fail;
    }
    if (cached_spirv_stage(cache, &info->stage, compiled) || !compiled->compute ||
        mxgpu_link_compute(compiled, bytes, MXGPU_LINK_MODULE_CAPACITY, &len))
        goto fail;
    struct mx_uniform_stage *uniforms = &pipe->uniform_stages[MXGPU_VK_COMPUTE_STAGE];
    uniforms->uses_uniforms = compiled->uses_uniforms;
    uniforms->uniform_count = compiled->uniform_count;
    uniforms->uniform_buffer_count = compiled->uniform_buffer_count;
    memcpy(uniforms->uniform_buffers, compiled->uniform_buffers, sizeof uniforms->uniform_buffers);
    pipe->storage_count = compiled->storage_count;
    memcpy(pipe->storage, compiled->storage, sizeof pipe->storage);
    pipe->uniform_slot = compiled->uniform_slot;
    pipe->compute = true;
    result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (uniforms->uses_uniforms &&
        (!uniforms->uniform_count || uniforms->uniform_count > UINT32_MAX / 16 ||
         !(pipe->uniform_storage = mxgpu_storage_buffer_create(uniforms->uniform_count * 16))))
        goto fail;
    pipe->compute_pipeline = mxgpu_compute_pipeline_create(bytes, len, 1);
    if (!pipe->compute_pipeline)
        goto fail;
    pipe->layout = (struct mx_pipeline_layout *)info->layout;
    retain_layout(pipe->layout);
    free(compiled);
    free(bytes);
    *out = pipe;
    return VK_SUCCESS;
fail:
    if (pipe && pipe->uniform_storage)
        mxgpu_storage_buffer_destroy(pipe->uniform_storage);
    free(pipe);
    free(compiled);
    free(bytes);
    return result;
}

static VkResult create_compute_pipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                         const VkComputePipelineCreateInfo *info,
                                         const VkAllocationCallbacks *alloc, VkPipeline *out)
{
    VkResult result = VK_SUCCESS;
    (void)device;
    (void)alloc;
    for (uint32_t index = 0; index < count; index++)
        out[index] = VK_NULL_HANDLE;
    for (uint32_t index = 0; index < count; index++) {
        struct mx_pipe *pipe;
        VkResult created = create_compute_pipeline((struct mx_pipeline_cache *)cache, &info[index], &pipe);
        if (created == VK_SUCCESS) {
            out[index] = (VkPipeline)pipe;
            continue;
        }
        result = created;
        if (created == VK_ERROR_OUT_OF_HOST_MEMORY ||
            (info[index].flags & VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT))
            return result;
    }
    return result;
}

static void destroy_pipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *alloc)
{
    struct mx_pipe *pipe = (struct mx_pipe *)pipeline;
    (void)device;
    (void)alloc;
    if (pipe) {
        if (pipe->compute_pipeline)
            mxgpu_compute_pipeline_destroy(pipe->compute_pipeline);
        if (pipe->uniform_storage)
            mxgpu_storage_buffer_destroy(pipe->uniform_storage);
        free(pipe->vertex_shader);
        free(pipe->fragment_shader);
        free(pipe->bytes);
        release_layout(pipe->layout);
    }
    free(pipe);
}

static VkResult create_fb(VkDevice device, const VkFramebufferCreateInfo *info, const VkAllocationCallbacks *alloc, VkFramebuffer *out)
{
    struct mx_fb *fb = calloc(1, sizeof *fb);
    struct mx_renderpass *pass = (struct mx_renderpass *)info->renderPass;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (!fb)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (!pass || !info->width || !info->height || info->layers != 1 ||
        (info->attachmentCount && !info->pAttachments))
        goto invalid;
    if (pass && pass->color_attachment < info->attachmentCount && info->pAttachments)
        fb->color = (struct mx_view *)info->pAttachments[pass->color_attachment];
    if (pass->depth_attachment < info->attachmentCount && info->pAttachments)
        fb->depth = (struct mx_view *)info->pAttachments[pass->depth_attachment];
    if ((pass->color_attachment != VK_ATTACHMENT_UNUSED &&
         (!fb->color || fb->color->format != pass->color_format ||
          fb->color->width < info->width || fb->color->height < info->height)) ||
        (pass->depth_attachment != VK_ATTACHMENT_UNUSED &&
         (!fb->depth || fb->depth->format != pass->depth_format ||
          fb->depth->width < info->width || fb->depth->height < info->height)))
        goto invalid;
    fb->width = info->width;
    fb->height = info->height;
    *out = (VkFramebuffer)fb;
    return VK_SUCCESS;
invalid:
    free(fb);
    return VK_ERROR_INITIALIZATION_FAILED;
}

static void destroy_fb(VkDevice device, VkFramebuffer fb, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(fb);
}

static VkResult create_pool(VkDevice device, const VkCommandPoolCreateInfo *info, const VkAllocationCallbacks *alloc, VkCommandPool *out)
{
    (void)device;
    (void)info;
    (void)alloc;
    *out = (VkCommandPool)calloc(1, sizeof(struct mx_pool));
    return *out ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
}

static void destroy_pool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks *alloc)
{
    struct mx_pool *owner = (struct mx_pool *)pool;
    (void)device;
    (void)alloc;
    if (!owner)
        return;
    while (owner->commands) {
        struct mx_cmd *cmd = owner->commands;
        owner->commands = cmd->next;
        free_draws(cmd);
        free_bound_state(cmd);
        free(cmd);
    }
    free(owner);
}

static void free_cmds(VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer *commands)
{
    struct mx_pool *owner = (struct mx_pool *)pool;
    uint32_t i;
    (void)device;
    for (i = 0; i < count; i++) {
        struct mx_cmd *cmd = (struct mx_cmd *)commands[i];
        struct mx_cmd **link = &owner->commands;
        while (*link && *link != cmd)
            link = &(*link)->next;
        if (*link) {
            *link = cmd->next;
            free_draws(cmd);
            free_bound_state(cmd);
            free(cmd);
        }
    }
}

static VkResult alloc_cmds(VkDevice device, const VkCommandBufferAllocateInfo *info, VkCommandBuffer *out)
{
    struct mx_pool *owner = (struct mx_pool *)info->commandPool;
    uint32_t i;
    for (i = 0; i < info->commandBufferCount; i++)
        out[i] = VK_NULL_HANDLE;
    for (i = 0; i < info->commandBufferCount; i++) {
        struct mx_cmd *cmd = calloc(1, sizeof *cmd);
        if (!cmd) {
            free_cmds(device, info->commandPool, i, out);
            for (i = 0; i < info->commandBufferCount; i++)
                out[i] = VK_NULL_HANDLE;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        set_loader_magic_value(cmd);
        cmd->pool = owner;
        cmd->next = owner->commands;
        owner->commands = cmd;
        out[i] = (VkCommandBuffer)cmd;
    }
    return VK_SUCCESS;
}

static VkResult reset_cmd(VkCommandBuffer command, VkCommandBufferResetFlags flags)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    VK_LOADER_DATA loader_data = cmd->loader_data;
    struct mx_pool *pool = cmd->pool;
    struct mx_cmd *next = cmd->next;
    (void)flags;
    free_draws(cmd);
    free_bound_state(cmd);
    memset(cmd, 0, sizeof *cmd);
    cmd->loader_data = loader_data;
    cmd->pool = pool;
    cmd->next = next;
    return VK_SUCCESS;
}

static VkResult reset_pool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags)
{
    struct mx_cmd *cmd;
    (void)device;
    (void)flags;
    for (cmd = ((struct mx_pool *)pool)->commands; cmd; cmd = cmd->next)
        reset_cmd((VkCommandBuffer)cmd, 0);
    return VK_SUCCESS;
}

static VkResult begin_cmd(VkCommandBuffer command, const VkCommandBufferBeginInfo *info)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    (void)info;
    reset_cmd(command, 0);
    cmd->open = 1;
    return VK_SUCCESS;
}

static void cmd_begin_rp(VkCommandBuffer command, const VkRenderPassBeginInfo *info, VkSubpassContents contents)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    struct mx_renderpass *pass = (struct mx_renderpass *)info->renderPass;
    (void)contents;
    cmd->fb = (struct mx_fb *)info->framebuffer;
    bool color_clear = pass && pass->color_attachment != VK_ATTACHMENT_UNUSED && pass->load == VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkImageAspectFlags depth_clear = 0;
    if (pass && pass->depth_attachment != VK_ATTACHMENT_UNUSED) {
        if (pass->depth_load == VK_ATTACHMENT_LOAD_OP_CLEAR) depth_clear |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (pass->stencil_load == VK_ATTACHMENT_LOAD_OP_CLEAR)
            depth_clear |= mx_vk_format_aspects(pass->depth_format) & VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (!cmd->open || !pass || !cmd->fb ||
        ((color_clear || depth_clear) && !info->pClearValues) ||
        (color_clear && pass->color_attachment >= info->clearValueCount) ||
        (depth_clear && pass->depth_attachment >= info->clearValueCount)) {
        cmd->record_result = VK_ERROR_DEVICE_LOST;
        return;
    }
    if ((color_clear || depth_clear) && cmd->record_result == VK_SUCCESS) {
        struct mx_draw *clear = calloc(1, sizeof *clear);
        if (!clear) {
            cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
            return;
        }
        clear->state.fb = cmd->fb;
        clear->state.clear = color_clear;
        if (color_clear)
            clear->state.clear_color = info->pClearValues[pass->color_attachment].color;
        clear->state.clear_depth_aspects = depth_clear;
        if (depth_clear)
            clear->state.clear_depth = info->pClearValues[pass->depth_attachment].depthStencil;
        clear->state.clear_area = info->renderArea;
        if (cmd->last_draw)
            cmd->last_draw->next = clear;
        else
            cmd->draws = clear;
        cmd->last_draw = clear;
    }
}

static void cmd_end_rp(VkCommandBuffer command)
{
    (void)command;
}

static void cmd_bind_pipe(VkCommandBuffer command, VkPipelineBindPoint point, VkPipeline pipeline)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    struct mx_pipe *pipe = (struct mx_pipe *)pipeline;
    if (point == VK_PIPELINE_BIND_POINT_COMPUTE) {
        if (pipe && !pipe->compute)
            cmd->record_result = VK_ERROR_DEVICE_LOST;
        cmd->compute_pipe = pipe;
        return;
    }
    if (pipe && pipe->compute)
        cmd->record_result = VK_ERROR_DEVICE_LOST;
    cmd->pipe = pipe;
}

static void cmd_set_viewport(VkCommandBuffer command, uint32_t first, uint32_t count, const VkViewport *viewports)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    if (!cmd->open || first || count != 1 || !viewports) { cmd->record_result = VK_ERROR_DEVICE_LOST; return; }
    cmd->viewport = viewports[0];
    cmd->viewport_set = true;
}

static void cmd_set_scissor(VkCommandBuffer command, uint32_t first, uint32_t count, const VkRect2D *scissors)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    if (!cmd->open || first || count != 1 || !scissors) { cmd->record_result = VK_ERROR_DEVICE_LOST; return; }
    cmd->scissor = scissors[0];
    cmd->scissor_set = true;
}

static void cmd_set_blend_constants(VkCommandBuffer command, const float constants[4])
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    if (!cmd->open || !constants) { cmd->record_result = VK_ERROR_DEVICE_LOST; return; }
    memcpy(cmd->blend_constants, constants, sizeof cmd->blend_constants);
    cmd->blend_set = true;
}

static bool vk_set_stencil_values(struct mx_cmd *cmd, VkStencilFaceFlags faces, uint32_t value,
                                  uint32_t values[2], unsigned *set)
{
    if (!cmd->open || !faces || (faces & ~VK_STENCIL_FACE_FRONT_AND_BACK)) {
        cmd->record_result = VK_ERROR_DEVICE_LOST;
        return false;
    }
    if (faces & VK_STENCIL_FACE_FRONT_BIT) { values[0] = value & 255u; *set |= 1; }
    if (faces & VK_STENCIL_FACE_BACK_BIT) { values[1] = value & 255u; *set |= 2; }
    return true;
}

static void cmd_set_stencil_compare(VkCommandBuffer command, VkStencilFaceFlags faces, uint32_t value)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    vk_set_stencil_values(cmd, faces, value, cmd->stencil_compare, &cmd->stencil_compare_set);
}

static void cmd_set_stencil_write(VkCommandBuffer command, VkStencilFaceFlags faces, uint32_t value)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    vk_set_stencil_values(cmd, faces, value, cmd->stencil_write, &cmd->stencil_write_set);
}

static void cmd_set_stencil_reference(VkCommandBuffer command, VkStencilFaceFlags faces, uint32_t value)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    vk_set_stencil_values(cmd, faces, value, cmd->stencil_ref, &cmd->stencil_ref_set);
}

static bool vk_draw_render_state(struct mx_cmd *cmd, uint32_t width, uint32_t height,
                                  struct mxgpu_native_render_state *state)
{
    struct mx_pipe *pipe = cmd->pipe;
    *state = pipe->native_state;
    state->depth_enabled = pipe->depth_enabled;
    state->depth_stencil = pipe->depth_stencil;
    state->stencil_reference = pipe->stencil_reference;
    if (state->depth_stencil.stencil_enable) {
        if (pipe->dynamic_stencil_compare) {
            if (cmd->stencil_compare_set != 3 || cmd->stencil_compare[0] != cmd->stencil_compare[1]) return false;
            state->depth_stencil.stencil_read_mask = cmd->stencil_compare[0];
        }
        if (pipe->dynamic_stencil_write) {
            if (cmd->stencil_write_set != 3 || cmd->stencil_write[0] != cmd->stencil_write[1]) return false;
            state->depth_stencil.stencil_write_mask = cmd->stencil_write[0];
        }
        if (pipe->dynamic_stencil_reference) {
            if (cmd->stencil_ref_set != 3 || cmd->stencil_ref[0] != cmd->stencil_ref[1]) return false;
            state->stencil_reference = cmd->stencil_ref[0];
        }
    }
    VkViewport viewport = pipe->viewport_set ? pipe->viewport : (VkViewport){0, 0, width, height, 0, 1};
    VkRect2D scissor = pipe->scissor_set ? pipe->scissor : (VkRect2D){{0, 0}, {width, height}};
    if (pipe->dynamic_viewport) { if (!cmd->viewport_set) return false; viewport = cmd->viewport; }
    if (pipe->dynamic_scissor) { if (!cmd->scissor_set) return false; scissor = cmd->scissor; }
    if (pipe->dynamic_blend) {
        if (!cmd->blend_set) return false;
        for (unsigned i = 0; i < 4; i++) state->blend_factor[i] = vk_float_bits(cmd->blend_constants[i]);
    }
    state->viewport.x = vk_float_bits(viewport.x);
    state->viewport.y = vk_float_bits(viewport.y + viewport.height);
    state->viewport.width = vk_float_bits(viewport.width);
    state->viewport.height = vk_float_bits(-viewport.height);
    state->viewport.min_depth = vk_float_bits(viewport.minDepth);
    state->viewport.max_depth = vk_float_bits(viewport.maxDepth);
    const uint32_t *values = &state->viewport.x;
    for (unsigned i = 0; i < 6; i++) if (!mxgpu_f32_finite(values[i])) return false;
    for (unsigned i = 0; i < 4; i++) if (!mxgpu_f32_finite(state->blend_factor[i])) return false;
    if (viewport.width <= 0 || viewport.height == 0 || viewport.minDepth < 0 || viewport.minDepth > 1 ||
        viewport.maxDepth < 0 || viewport.maxDepth > 1 || scissor.offset.x < 0 || scissor.offset.y < 0)
        return false;
    uint64_t right = (uint64_t)scissor.offset.x + scissor.extent.width;
    uint64_t bottom = (uint64_t)scissor.offset.y + scissor.extent.height;
    state->scissor.left = (uint32_t)scissor.offset.x < width ? scissor.offset.x : width;
    state->scissor.top = (uint32_t)scissor.offset.y < height ? scissor.offset.y : height;
    state->scissor.right = right < width ? right : width;
    state->scissor.bottom = bottom < height ? bottom : height;
    return true;
}

static void cmd_bind_vbos(VkCommandBuffer command, uint32_t first, uint32_t count, const VkBuffer *buffers, const VkDeviceSize *offsets)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    if (first > 32 || count > 32 - first || (count && (!buffers || !offsets))) {
        cmd->record_result = VK_ERROR_DEVICE_LOST;
        return;
    }
    for (unsigned i = 0; i < count; i++) {
        cmd->vertex_buffers[first + i] = (struct mx_buf *)buffers[i];
        cmd->vertex_offsets[first + i] = offsets[i];
    }
    if (!first && count) {
        cmd->vbo = (struct mx_buf *)buffers[0];
        cmd->voff = offsets[0];
    }
}

static void cmd_push_constants(VkCommandBuffer command, VkPipelineLayout handle,
                                VkShaderStageFlags stages, uint32_t offset,
                                uint32_t size, const void *values)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    struct mx_pipeline_layout *layout = (struct mx_pipeline_layout *)handle;
    if (cmd->record_result != VK_SUCCESS)
        return;
    if (!layout || !stages || !values || !size || offset % 4 || size % 4 ||
        offset > MXGPU_VK_PUSH_CONSTANT_BYTES || size > MXGPU_VK_PUSH_CONSTANT_BYTES - offset) {
        cmd->record_result = VK_ERROR_DEVICE_LOST;
        return;
    }
    for (unsigned word = offset / 4; word < (offset + size) / 4; word++)
        if ((layout->coverage[word] & stages) != stages) {
            cmd->record_result = VK_ERROR_DEVICE_LOST;
            return;
        }
    const VkShaderStageFlagBits stage_bits[MXGPU_VK_STAGES] = {
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT};
    for (unsigned stage = 0; stage < MXGPU_VK_STAGES; stage++) {
        if (!(stages & stage_bits[stage]))
            continue;
        memcpy(cmd->push_constants[stage] + offset, values, size);
        for (unsigned word = offset / 4; word < (offset + size) / 4; word++) {
            retain_layout(layout);
            release_layout(cmd->push_layouts[stage][word]);
            cmd->push_layouts[stage][word] = layout;
        }
    }
}

static void cmd_bind_sets(VkCommandBuffer command, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count, const VkDescriptorSet *sets, uint32_t dynamic_count, const uint32_t *dynamic)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    struct mx_set **bound;
    struct mx_bound_offsets *offsets;
    const struct mx_pipeline_layout *owner = (const struct mx_pipeline_layout *)layout;
    uint32_t i, needed, used = 0;
    bool compute = point == VK_PIPELINE_BIND_POINT_COMPUTE;
    struct mx_set ***bound_field = compute ? &cmd->compute_sets : &cmd->bound_sets;
    struct mx_bound_offsets **offsets_field = compute ? &cmd->compute_offsets : &cmd->bound_offsets;
    uint32_t *count_field = compute ? &cmd->compute_set_count : &cmd->bound_set_count;
    if (!count || !sets || cmd->record_result != VK_SUCCESS)
        return;
    if (owner) {
        if (first > owner->set_count || count > owner->set_count - first) {
            cmd->record_result = VK_ERROR_DEVICE_LOST;
            return;
        }
        for (uint32_t i = 0; i < count; i++) {
            const struct mx_set *set = (const struct mx_set *)sets[i];
            const struct mx_ds_layout *expected = &owner->sets[first + i];
            if (!set || set->descriptor_count != expected->count) {
                cmd->record_result = VK_ERROR_DEVICE_LOST;
                return;
            }
            for (uint32_t j = 0; j < expected->count; j++) {
                const struct mx_descriptor *a = &set->descriptors[j], *b = &expected->descriptors[j];
                if (a->binding != b->binding || a->element != b->element ||
                    a->type != b->type || a->stages != b->stages) {
                    cmd->record_result = VK_ERROR_DEVICE_LOST;
                    return;
                }
            }
        }
    }
    if (count > UINT32_MAX - first ||
        (uint64_t)first + count > SIZE_MAX / sizeof *offsets ||
        (uint64_t)first + count > SIZE_MAX / sizeof *bound) {
        cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return;
    }
    needed = first + count > *count_field ? first + count : *count_field;
    bound = calloc(needed, sizeof *bound);
    offsets = calloc(needed, sizeof *offsets);
    if (!bound || !offsets) {
        free(bound);
        free(offsets);
        cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return;
    }
    if (*count_field) {
        memcpy(bound, *bound_field, (size_t)*count_field * sizeof *bound);
        if (*offsets_field)
            memcpy(offsets, *offsets_field, (size_t)*count_field * sizeof *offsets);
    }
    for (i = first; i < first + count; i++)
        offsets[i] = (struct mx_bound_offsets){0};
    for (i = 0; i < count; i++) {
        struct mx_set *set = (struct mx_set *)sets[i];
        bound[first + i] = set;
        if (!set)
            continue;
        for (uint32_t j = 0; j < set->descriptor_count; j++) {
            VkDescriptorType type = set->descriptors[j].type;
            if (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC && type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                continue;
            if (!dynamic || used >= dynamic_count) {
                cmd->record_result = VK_ERROR_DEVICE_LOST;
                goto fail;
            }
            if (!offsets[first + i].values) {
                offsets[first + i].values = calloc(set->descriptor_count, sizeof(uint32_t));
                if (!offsets[first + i].values) {
                    cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
                    goto fail;
                }
                offsets[first + i].count = set->descriptor_count;
            }
            offsets[first + i].values[j] = dynamic[used++];
        }
    }
    if (used != dynamic_count) {
        cmd->record_result = VK_ERROR_DEVICE_LOST;
        goto fail;
    }
    for (i = first; *offsets_field && i < first + count && i < *count_field; i++)
        free((*offsets_field)[i].values);
    free(*offsets_field);
    free(*bound_field);
    *bound_field = bound;
    *offsets_field = offsets;
    *count_field = needed;
    if (!compute)
        cmd->set = cmd->bound_sets[0];
    return;
fail:
    for (i = first; i < first + count; i++)
        free(offsets[i].values);
    free(offsets);
    free(bound);
}

static struct mx_draw *record_snapshot(struct mx_cmd *cmd, struct mx_set **sets,
                                       const struct mx_bound_offsets *offsets, uint32_t set_count)
{
    struct mx_draw *draw = calloc(1, sizeof *draw);
    if (!draw) {
        cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return NULL;
    }
    draw->state = *cmd;
    draw->state.bound_sets = NULL;
    draw->state.bound_offsets = NULL;
    draw->state.bound_set_count = set_count;
    draw->state.compute_sets = NULL;
    draw->state.compute_offsets = NULL;
    draw->state.compute_set_count = 0;
    if (set_count) {
        draw->state.bound_sets = malloc((size_t)set_count * sizeof *sets);
        if (!draw->state.bound_sets) {
            free(draw);
            cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
            return NULL;
        }
        memcpy(draw->state.bound_sets, sets, (size_t)set_count * sizeof *sets);
    }
    if (offsets) {
        draw->state.bound_offsets = calloc(set_count, sizeof *offsets);
        if (!draw->state.bound_offsets)
            goto snapshot_fail;
        for (uint32_t i = 0; i < set_count; i++) {
            uint32_t count = offsets[i].count;
            if (!count)
                continue;
            draw->state.bound_offsets[i].values = malloc((size_t)count * sizeof(uint32_t));
            if (!draw->state.bound_offsets[i].values)
                goto snapshot_fail;
            draw->state.bound_offsets[i].count = count;
            memcpy(draw->state.bound_offsets[i].values, offsets[i].values, (size_t)count * sizeof(uint32_t));
        }
    }
    draw->state.draws = NULL;
    draw->state.last_draw = NULL;
    for (unsigned stage = 0; stage < MXGPU_VK_STAGES; stage++)
        for (unsigned word = 0; word < MXGPU_VK_PUSH_CONSTANT_BYTES / 4; word++)
            retain_layout(draw->state.push_layouts[stage][word]);
    if (cmd->last_draw)
        cmd->last_draw->next = draw;
    else
        cmd->draws = draw;
    cmd->last_draw = draw;
    return draw;
snapshot_fail:
    free_bound_offsets(draw->state.bound_offsets, set_count);
    free(draw->state.bound_sets);
    free(draw);
    cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    return NULL;
}

static void cmd_draw(VkCommandBuffer command, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    if (!vertex_count || !instance_count || cmd->record_result != VK_SUCCESS)
        return;
    cmd->vertex_count = vertex_count;
    cmd->indexed = 0;
    cmd->first_vertex = first_vertex;
    cmd->instance_count = instance_count;
    cmd->first_instance = first_instance;
    cmd->draw = 1;
    record_snapshot(cmd, cmd->bound_sets, cmd->bound_offsets, cmd->bound_set_count);
}

static void record_dispatch(struct mx_cmd *cmd, const uint32_t groups[3], struct mx_buf *indirect,
                            VkDeviceSize indirect_offset)
{
    if (cmd->record_result != VK_SUCCESS)
        return;
    if (!cmd->open || !cmd->compute_pipe || !cmd->compute_pipe->compute ||
        (indirect && (!indirect->mem || indirect_offset % 4 || indirect_offset > indirect->size ||
                      indirect->size - indirect_offset < 3 * sizeof(uint32_t)))) {
        cmd->record_result = VK_ERROR_DEVICE_LOST;
        return;
    }
    if (!indirect && (!groups[0] || !groups[1] || !groups[2]))
        return;
    struct mx_pipe *graphics = cmd->pipe;
    cmd->pipe = cmd->compute_pipe;
    struct mx_draw *dispatch = record_snapshot(cmd, cmd->compute_sets, cmd->compute_offsets, cmd->compute_set_count);
    cmd->pipe = graphics;
    if (!dispatch)
        return;
    dispatch->operation = 5;
    dispatch->state.draw = 0;
    dispatch->state.set = cmd->compute_set_count ? cmd->compute_sets[0] : NULL;
    if (!indirect)
        memcpy(dispatch->dispatch, groups, sizeof dispatch->dispatch);
    dispatch->indirect = indirect;
    dispatch->indirect_offset = indirect_offset;
}

static void cmd_dispatch(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z)
{
    const uint32_t groups[3] = {x, y, z};
    record_dispatch((struct mx_cmd *)command, groups, NULL, 0);
}

static void cmd_dispatch_base(VkCommandBuffer command, uint32_t base_x, uint32_t base_y, uint32_t base_z,
                              uint32_t x, uint32_t y, uint32_t z)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    const uint32_t groups[3] = {x, y, z};
    if (base_x || base_y || base_z) {
        if (cmd->record_result == VK_SUCCESS)
            cmd->record_result = VK_ERROR_DEVICE_LOST;
        return;
    }
    record_dispatch(cmd, groups, NULL, 0);
}

static void cmd_dispatch_indirect(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    const uint32_t groups[3] = {0};
    if (!buffer) {
        if (cmd->record_result == VK_SUCCESS)
            cmd->record_result = VK_ERROR_DEVICE_LOST;
        return;
    }
    record_dispatch(cmd, groups, (struct mx_buf *)buffer, offset);
}

static void cmd_copy_buffer(VkCommandBuffer command, VkBuffer source, VkBuffer destination,
                            uint32_t count, const VkBufferCopy *regions)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    uint32_t i;
    if (cmd->record_result != VK_SUCCESS)
        return;
    for (i = 0; i < count; i++) {
        struct mx_draw *operation = calloc(1, sizeof *operation);
        if (!operation) {
            cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
            return;
        }
        operation->operation = 1;
        operation->copy_src = (struct mx_buf *)source;
        operation->copy_dst = (struct mx_buf *)destination;
        operation->copy_region = regions[i];
        if (cmd->last_draw)
            cmd->last_draw->next = operation;
        else
            cmd->draws = operation;
        cmd->last_draw = operation;
    }
}

static VkResult perform_buffer_copy(const struct mx_draw *operation)
{
    struct mx_buf *source = operation->copy_src, *destination = operation->copy_dst;
    const VkBufferCopy *region = &operation->copy_region;
    if (!source || !destination || !source->mem || !destination->mem ||
        !source->mem->ptr || !destination->mem->ptr ||
        region->srcOffset > source->size || region->size > source->size - region->srcOffset ||
        region->dstOffset > destination->size || region->size > destination->size - region->dstOffset ||
        source->offset > source->mem->size || source->size > source->mem->size - source->offset ||
        destination->offset > destination->mem->size || destination->size > destination->mem->size - destination->offset)
        return VK_ERROR_DEVICE_LOST;
    memmove((uint8_t *)destination->mem->ptr + destination->offset + region->dstOffset,
            (const uint8_t *)source->mem->ptr + source->offset + region->srcOffset,
            (size_t)region->size);
    return VK_SUCCESS;
}

static void record_buffer_image_copy(VkCommandBuffer command, VkBuffer buffer, VkImage image,
                                     uint32_t count, const VkBufferImageCopy *regions, int direction)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    uint32_t i;
    if (cmd->record_result != VK_SUCCESS)
        return;
    for (i = 0; i < count; i++) {
        struct mx_draw *operation = calloc(1, sizeof *operation);
        if (!operation) {
            cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
            return;
        }
        operation->operation = direction;
        operation->copy_src = (struct mx_buf *)buffer;
        operation->copy_image = (struct mx_img *)image;
        operation->image_region = regions[i];
        if (cmd->last_draw)
            cmd->last_draw->next = operation;
        else
            cmd->draws = operation;
        cmd->last_draw = operation;
    }
}

static void cmd_copy_buffer_to_image(VkCommandBuffer command, VkBuffer buffer, VkImage image,
                                     VkImageLayout layout, uint32_t count, const VkBufferImageCopy *regions)
{
    (void)layout;
    record_buffer_image_copy(command, buffer, image, count, regions, 2);
}

static void cmd_copy_image_to_buffer(VkCommandBuffer command, VkImage image, VkImageLayout layout,
                                     VkBuffer buffer, uint32_t count, const VkBufferImageCopy *regions)
{
    (void)layout;
    record_buffer_image_copy(command, buffer, image, count, regions, 3);
}

static void cmd_pipeline_barrier(VkCommandBuffer command,
                                 VkPipelineStageFlags src_stage,
                                 VkPipelineStageFlags dst_stage,
                                 VkDependencyFlags flags,
                                 uint32_t memory_count, const VkMemoryBarrier *memory,
                                 uint32_t buffer_count, const VkBufferMemoryBarrier *buffers,
                                 uint32_t image_count, const VkImageMemoryBarrier *images)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    struct mx_draw *operation;
    if (cmd->record_result != VK_SUCCESS)
        return;
    if (!cmd->open || !src_stage || !dst_stage ||
        (flags & ~VK_DEPENDENCY_BY_REGION_BIT) ||
        (memory_count && !memory) || (buffer_count && !buffers) ||
        (image_count && !images))
        goto invalid;
    for (uint32_t i = 0; i < memory_count; i++)
        if (memory[i].sType != VK_STRUCTURE_TYPE_MEMORY_BARRIER || memory[i].pNext)
            goto invalid;
    for (uint32_t i = 0; i < buffer_count; i++) {
        const VkBufferMemoryBarrier *barrier = &buffers[i];
        struct mx_buf *buffer = (struct mx_buf *)barrier->buffer;
        if (barrier->sType != VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER || barrier->pNext ||
            !buffer || !buffer->mem || barrier->offset >= buffer->size ||
            !barrier->size || (barrier->size != VK_WHOLE_SIZE &&
                              barrier->size > buffer->size - barrier->offset) ||
            !((barrier->srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
               barrier->dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) ||
              (barrier->srcQueueFamilyIndex == 0 && barrier->dstQueueFamilyIndex == 0)))
            goto invalid;
    }
    for (uint32_t i = 0; i < image_count; i++) {
        const VkImageMemoryBarrier *barrier = &images[i];
        struct mx_img *image = (struct mx_img *)barrier->image;
        const VkImageSubresourceRange *range = &barrier->subresourceRange;
        if (barrier->sType != VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER || barrier->pNext ||
            !image || !image->mem || !range->aspectMask ||
            (range->aspectMask & ~mx_vk_format_aspects(image->format)) ||
            range->baseMipLevel >= image->levels || range->baseArrayLayer >= image->layers ||
            !range->levelCount || !range->layerCount ||
            (range->levelCount != VK_REMAINING_MIP_LEVELS &&
             range->levelCount > image->levels - range->baseMipLevel) ||
            (range->layerCount != VK_REMAINING_ARRAY_LAYERS &&
             range->layerCount > image->layers - range->baseArrayLayer) ||
            barrier->newLayout == VK_IMAGE_LAYOUT_UNDEFINED ||
            barrier->newLayout == VK_IMAGE_LAYOUT_PREINITIALIZED ||
            !((barrier->srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
               barrier->dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) ||
              (barrier->srcQueueFamilyIndex == 0 && barrier->dstQueueFamilyIndex == 0)))
            goto invalid;
    }
    operation = calloc(1, sizeof *operation);
    if (!operation) {
        cmd->record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return;
    }
    operation->operation = 4;
    if (cmd->last_draw)
        cmd->last_draw->next = operation;
    else
        cmd->draws = operation;
    cmd->last_draw = operation;
    return;
invalid:
    cmd->record_result = VK_ERROR_DEVICE_LOST;
}

static VkResult perform_pipeline_barrier(void)
{
    atomic_thread_fence(memory_order_seq_cst);
    return VK_SUCCESS;
}

static VkResult perform_buffer_image_copy(const struct mx_draw *operation)
{
    struct mx_buf *buffer = operation->copy_src;
    struct mx_img *image = operation->copy_image;
    const VkBufferImageCopy *region = &operation->image_region;
    const VkImageSubresourceLayers *subresource = &region->imageSubresource;
    VkDeviceSize level_offset = 0, image_slice, image_layer, row_pitch, slice_pitch, last_byte, slices;
    uint32_t width, height, depth, level, layer, z, y, row_length, image_height;
    unsigned image_bpp = image ? vk_format_bytes(image->format) : 0;
    unsigned buffer_bpp = vk_color_format(image ? image->format : VK_FORMAT_UNDEFINED) != PIPE_FORMAT_NONE ? image_bpp : 4;
    unsigned component_offset = 0;
    VkImageAspectFlags aspect = subresource->aspectMask;
    if (!image_bpp || !aspect || (aspect & (aspect - 1)) ||
        (aspect & ~mx_vk_format_aspects(image->format)))
        return VK_ERROR_DEVICE_LOST;
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
        buffer_bpp = 1;
        component_offset = image->format == VK_FORMAT_D24_UNORM_S8_UINT ? 3 : 4;
    }
    if (!buffer || !image || !buffer->mem || !image->mem || !buffer->mem->ptr || !image->mem->ptr ||
        image->samples != 1 || subresource->mipLevel >= image->levels ||
        !subresource->layerCount || subresource->baseArrayLayer >= image->layers ||
        subresource->layerCount > image->layers - subresource->baseArrayLayer ||
        region->imageOffset.x < 0 || region->imageOffset.y < 0 || region->imageOffset.z < 0 ||
        !region->imageExtent.width || !region->imageExtent.height || !region->imageExtent.depth ||
        buffer->offset > buffer->mem->size || buffer->size > buffer->mem->size - buffer->offset ||
        image->offset > image->mem->size || image->size > image->mem->size - image->offset)
        return VK_ERROR_DEVICE_LOST;
    width = image->width;
    height = image->height;
    depth = image->depth;
    for (level = 0; level < subresource->mipLevel; level++) {
        level_offset += (VkDeviceSize)width * height * depth * image_bpp * image->layers;
        width = width > 1 ? width / 2 : 1;
        height = height > 1 ? height / 2 : 1;
        depth = depth > 1 ? depth / 2 : 1;
    }
    if ((uint32_t)region->imageOffset.x > width || region->imageExtent.width > width - region->imageOffset.x ||
        (uint32_t)region->imageOffset.y > height || region->imageExtent.height > height - region->imageOffset.y ||
        (uint32_t)region->imageOffset.z > depth || region->imageExtent.depth > depth - region->imageOffset.z)
        return VK_ERROR_DEVICE_LOST;
    row_length = region->bufferRowLength ? region->bufferRowLength : region->imageExtent.width;
    image_height = region->bufferImageHeight ? region->bufferImageHeight : region->imageExtent.height;
    if (row_length < region->imageExtent.width || image_height < region->imageExtent.height)
        return VK_ERROR_DEVICE_LOST;
    row_pitch = (VkDeviceSize)row_length * buffer_bpp;
    if (row_pitch > UINT64_MAX / image_height)
        return VK_ERROR_DEVICE_LOST;
    slice_pitch = row_pitch * image_height;
    slices = (VkDeviceSize)subresource->layerCount * region->imageExtent.depth;
    last_byte = (VkDeviceSize)(region->imageExtent.height - 1) * row_pitch + (VkDeviceSize)region->imageExtent.width * buffer_bpp;
    if (slices - 1 > (UINT64_MAX - last_byte) / slice_pitch)
        return VK_ERROR_DEVICE_LOST;
    last_byte += (slices - 1) * slice_pitch;
    if (region->bufferOffset > buffer->size || last_byte > buffer->size - region->bufferOffset)
        return VK_ERROR_DEVICE_LOST;
    image_slice = (VkDeviceSize)width * height * image_bpp;
    image_layer = image_slice * depth;
    for (layer = 0; layer < subresource->layerCount; layer++) {
        for (z = 0; z < region->imageExtent.depth; z++) {
            VkDeviceSize image_base = image->offset + level_offset +
                (subresource->baseArrayLayer + layer) * image_layer +
                (region->imageOffset.z + z) * image_slice;
            VkDeviceSize buffer_base = buffer->offset + region->bufferOffset +
                ((VkDeviceSize)layer * region->imageExtent.depth + z) * slice_pitch;
            for (y = 0; y < region->imageExtent.height; y++) {
                uint8_t *image_row = (uint8_t *)image->mem->ptr + image_base +
                    ((VkDeviceSize)(region->imageOffset.y + y) * width + region->imageOffset.x) * image_bpp;
                uint8_t *buffer_row = (uint8_t *)buffer->mem->ptr + buffer_base + y * row_pitch;
                if (image_bpp == buffer_bpp && image->format != VK_FORMAT_D24_UNORM_S8_UINT) {
                    if (operation->operation == 2)
                        memcpy(image_row, buffer_row, (size_t)region->imageExtent.width * buffer_bpp);
                    else
                        memcpy(buffer_row, image_row, (size_t)region->imageExtent.width * buffer_bpp);
                } else {
                    for (uint32_t x = 0; x < region->imageExtent.width; x++) {
                        uint8_t *pixel = image_row + (size_t)x * image_bpp;
                        uint8_t *element = buffer_row + (size_t)x * buffer_bpp;
                        if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT && image->format == VK_FORMAT_D24_UNORM_S8_UINT) {
                            if (operation->operation == 2)
                                mx_w32(pixel, 0, (mx_r32(pixel, 0) & 0xff000000u) | (mx_r32(element, 0) & 0xffffffu));
                            else
                                mx_w32(element, 0, mx_r32(pixel, 0) & 0xffffffu);
                        } else if (operation->operation == 2) {
                            memcpy(pixel + component_offset, element, buffer_bpp);
                        } else {
                            memcpy(element, pixel + component_offset, buffer_bpp);
                        }
                        if (operation->operation == 2 && image_bpp == 8)
                            memset(pixel + 5, 0, 3);
                    }
                }
            }
        }
    }
    return VK_SUCCESS;
}

static void cmd_bind_index(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset, VkIndexType type)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    cmd->ibo = (struct mx_buf *)buffer;
    cmd->ioff = offset;
    cmd->index_type = type;
}

static void cmd_draw_indexed(VkCommandBuffer command, uint32_t count, uint32_t instances,
                             uint32_t first, int32_t bias, uint32_t first_instance)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    struct mx_draw *previous = cmd->last_draw;
    cmd_draw(command, count, instances, 0, first_instance);
    if (cmd->last_draw != previous) {
        cmd->last_draw->state.indexed = 1;
        cmd->last_draw->state.first_index = first;
        cmd->last_draw->state.vertex_bias = bias;
    }
}

static VkResult collect_uniforms(struct mx_cmd *cmd, unsigned stage, uint8_t **data, uint32_t *size)
{
    const struct mx_uniform_stage *shader = &cmd->pipe->uniform_stages[stage];
    *data = NULL;
    *size = 0;
    if (!shader->uses_uniforms)
        return VK_SUCCESS;
    if (!shader->uniform_buffer_count || shader->uniform_count > UINT32_MAX / 16)
        return VK_ERROR_DEVICE_LOST;
    *size = shader->uniform_count * 16;
    *data = calloc(1, *size);
    if (!*data)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    for (unsigned i = 0; i < shader->uniform_buffer_count; i++) {
        const struct mxgpu_uniform_buffer *ref = &shader->uniform_buffers[i];
        if (ref->push_constant) {
            VkShaderStageFlags stage_bit = stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT :
                                           stage == MXGPU_VK_COMPUTE_STAGE ? VK_SHADER_STAGE_COMPUTE_BIT :
                                           VK_SHADER_STAGE_FRAGMENT_BIT;
            if (!cmd->pipe->layout || ref->size > MXGPU_VK_PUSH_CONSTANT_BYTES ||
                ref->offset > *size || ref->size > *size - ref->offset)
                goto invalid;
            for (unsigned word = 0; word < (ref->size + 3) / 4; word++) {
                if (!(ref->push_constant_words & (1u << word)))
                    continue;
                const struct mx_pipeline_layout *written = cmd->push_layouts[stage][word];
                if (written && (!(cmd->pipe->layout->coverage[word] & stage_bit) ||
                                !push_layout_compatible(cmd->pipe->layout, written)))
                    goto invalid;
            }
            memcpy(*data + ref->offset, cmd->push_constants[stage], ref->size);
            continue;
        }
        struct mx_set *set = ref->set < cmd->bound_set_count ? cmd->bound_sets[ref->set] :
                             ref->set == 0 ? cmd->set : NULL;
        struct mx_descriptor *descriptor = NULL;
        unsigned descriptor_slot = 0;
        if (set)
            for (unsigned j = 0; j < set->descriptor_count; j++)
                if (set->descriptors[j].binding == ref->binding && set->descriptors[j].element == ref->element)
                    descriptor = &set->descriptors[j], descriptor_slot = j;
        struct mx_buf *buffer = descriptor ? (struct mx_buf *)descriptor->value.buffer.buffer : NULL;
        VkDeviceSize offset, range, memory_offset;
        if (!descriptor ||
            (descriptor->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
             descriptor->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
            !buffer || !buffer->mem || !buffer->mem->ptr)
            goto invalid;
        offset = descriptor->value.buffer.offset;
        range = descriptor->value.buffer.range;
        if (offset > buffer->size)
            goto invalid;
        if (range == VK_WHOLE_SIZE)
            range = buffer->size - offset;
        if (descriptor->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
            if (ref->set >= cmd->bound_set_count || !cmd->bound_offsets ||
                descriptor_slot >= cmd->bound_offsets[ref->set].count)
                goto invalid;
            uint32_t dynamic_offset = cmd->bound_offsets[ref->set].values[descriptor_slot];
            if (dynamic_offset > buffer->size - offset)
                goto invalid;
            offset += dynamic_offset;
        }
        if (range > buffer->size - offset ||
            buffer->offset > buffer->mem->size || offset > buffer->mem->size - buffer->offset)
            goto invalid;
        VkDeviceSize bytes = ref->size <= range ? ref->size : range & ~(VkDeviceSize)3;
        memory_offset = buffer->offset + offset;
        if (bytes > buffer->mem->size - memory_offset ||
            ref->offset > *size || ref->size > *size - ref->offset)
            goto invalid;
        memcpy(*data + ref->offset, (uint8_t *)buffer->mem->ptr + memory_offset, (size_t)bytes);
    }
    return VK_SUCCESS;
invalid:
    free(*data);
    *data = NULL;
    *size = 0;
    return VK_ERROR_DEVICE_LOST;
}

static VkResult storage_descriptor_range(struct mx_cmd *cmd, const struct mxgpu_storage_binding *ref,
                                         struct mx_mem **memory, VkDeviceSize *start, VkDeviceSize *bytes)
{
    struct mx_set *set = ref->set < cmd->bound_set_count ? cmd->bound_sets[ref->set] : NULL;
    struct mx_descriptor *descriptor = NULL;
    unsigned slot = 0;
    for (unsigned j = 0; set && j < set->descriptor_count; j++)
        if (set->descriptors[j].binding == ref->binding && set->descriptors[j].element == ref->element)
            descriptor = &set->descriptors[j], slot = j;
    struct mx_buf *buffer = descriptor ? (struct mx_buf *)descriptor->value.buffer.buffer : NULL;
    if (!descriptor || !buffer || !buffer->mem || !buffer->mem->ptr ||
        (descriptor->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
         descriptor->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC))
        return VK_ERROR_DEVICE_LOST;
    VkDeviceSize offset = descriptor->value.buffer.offset, range = descriptor->value.buffer.range;
    if (offset > buffer->size)
        return VK_ERROR_DEVICE_LOST;
    if (descriptor->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
        if (!cmd->bound_offsets || slot >= cmd->bound_offsets[ref->set].count ||
            cmd->bound_offsets[ref->set].values[slot] > buffer->size - offset)
            return VK_ERROR_DEVICE_LOST;
        offset += cmd->bound_offsets[ref->set].values[slot];
    }
    if (range == VK_WHOLE_SIZE)
        range = buffer->size - offset;
    if (range > buffer->size - offset || !range || buffer->offset > buffer->mem->size ||
        offset > buffer->mem->size - buffer->offset || range > buffer->mem->size - buffer->offset - offset ||
        (buffer->offset + offset) % 4 || buffer->mem->size > UINT32_MAX)
        return VK_ERROR_DEVICE_LOST;
    *memory = buffer->mem;
    *start = buffer->offset + offset;
    *bytes = range & ~(VkDeviceSize)3;
    return VK_SUCCESS;
}

static VkResult perform_dispatch(struct mx_draw *operation)
{
    struct mx_cmd *cmd = &operation->state;
    struct mx_pipe *pipe = cmd->pipe;
    struct mxgpu_compute_binding bindings[MXGPU_SHADER_STORAGE_BUFFERS + 1];
    struct mx_mem *memories[MXGPU_SHADER_STORAGE_BUFFERS];
    uint32_t scratch[MXGPU_SHADER_STORAGE_BUFFERS] = {0};
    static const uint8_t scratch_zero[4];
    uint32_t groups[3], count = 0;
    uint8_t *uniforms = NULL;
    uint32_t uniform_bytes = 0;
    VkResult result;
    if (!pipe || !pipe->compute || pipe->storage_count > MXGPU_SHADER_STORAGE_BUFFERS)
        return VK_ERROR_DEVICE_LOST;
    memcpy(groups, operation->dispatch, sizeof groups);
    if (operation->indirect) {
        struct mx_buf *buffer = operation->indirect;
        if (!buffer->mem || !buffer->mem->ptr || buffer->offset > buffer->mem->size ||
            operation->indirect_offset > buffer->mem->size - buffer->offset ||
            buffer->mem->size - buffer->offset - operation->indirect_offset < sizeof groups)
            return VK_ERROR_DEVICE_LOST;
        memcpy(groups, (const uint8_t *)buffer->mem->ptr + buffer->offset + operation->indirect_offset, sizeof groups);
    }
    if (!groups[0] || !groups[1] || !groups[2])
        return VK_SUCCESS;
    for (uint32_t i = 0; i < pipe->storage_count; i++) {
        const struct mxgpu_storage_binding *ref = &pipe->storage[i];
        VkDeviceSize start, bytes;
        uint16_t access = ref->access == MXSB_ACCESS_READ ? MXGPU_BIND_ACCESS_READ :
                          ref->access == MXSB_ACCESS_WRITE ? MXGPU_BIND_ACCESS_WRITE : MXGPU_BIND_ACCESS_READ_WRITE;
        result = storage_descriptor_range(cmd, ref, &memories[i], &start, &bytes);
        if (result != VK_SUCCESS)
            goto done;
        if (!bytes) {
            memories[i] = NULL;
            result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            if (!(scratch[i] = mxgpu_storage_buffer_create(sizeof scratch_zero)))
                goto done;
            result = VK_ERROR_DEVICE_LOST;
            if (mxgpu_storage_buffer_upload(scratch[i], 0, scratch_zero, sizeof scratch_zero))
                goto done;
            bindings[count++] = (struct mxgpu_compute_binding){(uint16_t)ref->slot, access, scratch[i], 0,
                                                               sizeof scratch_zero};
            continue;
        }
        struct mx_mem *mem = memories[i];
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        if (!mem->storage && !(mem->storage = mxgpu_storage_buffer_create((uint32_t)mem->size)))
            goto done;
        result = VK_ERROR_DEVICE_LOST;
        if (mxgpu_storage_buffer_upload(mem->storage, (uint32_t)start, (const uint8_t *)mem->ptr + start, (uint32_t)bytes))
            goto done;
        bindings[count++] = (struct mxgpu_compute_binding){(uint16_t)ref->slot, access, mem->storage, start, bytes};
    }
    result = collect_uniforms(cmd, MXGPU_VK_COMPUTE_STAGE, &uniforms, &uniform_bytes);
    if (result != VK_SUCCESS)
        goto done;
    if (uniform_bytes) {
        int uploaded = pipe->uniform_storage &&
                       !mxgpu_storage_buffer_upload(pipe->uniform_storage, 0, uniforms, uniform_bytes);
        free(uniforms);
        result = VK_ERROR_DEVICE_LOST;
        if (!uploaded)
            goto done;
        bindings[count++] = (struct mxgpu_compute_binding){
            (uint16_t)pipe->uniform_slot, MXGPU_BIND_ACCESS_READ, pipe->uniform_storage, 0, uniform_bytes};
    }
    result = VK_ERROR_DEVICE_LOST;
    if (mxgpu_compute_dispatch(pipe->compute_pipeline, MXGPU_DISPATCH_THREADGROUPS, groups, bindings, count))
        goto done;
    for (uint32_t i = 0; i < pipe->storage_count; i++) {
        if (!(pipe->storage[i].access & MXSB_ACCESS_WRITE) || !memories[i])
            continue;
        uint8_t *target = (uint8_t *)memories[i]->ptr + bindings[i].offset;
        if (mxgpu_storage_buffer_read(memories[i]->storage, (uint32_t)bindings[i].offset, target,
                                      (uint32_t)bindings[i].size))
            goto done;
    }
    result = VK_SUCCESS;
done:
    for (uint32_t i = 0; i < pipe->storage_count; i++)
        if (scratch[i])
            mxgpu_storage_buffer_destroy(scratch[i]);
    return result;
}

static VkResult stage_view_texture(const struct mx_view *view, const uint8_t *source,
                                   VkDeviceSize size, const uint8_t **data, uint8_t **storage)
{
    VkComponentSwizzle components[4] = {view->components.r, view->components.g,
                                        view->components.b, view->components.a};
    unsigned channels[4];
    int identity = 1;
    *data = source;
    *storage = NULL;
    enum pipe_format format = vk_color_format(view->format);
    unsigned bpp = vk_format_bytes(view->format);
    if (format == PIPE_FORMAT_NONE || !bpp || size > SIZE_MAX || size % bpp ||
        size / bpp > SIZE_MAX / 4u)
        return VK_ERROR_DEVICE_LOST;
    for (unsigned c = 0; c < 4; c++) {
        VkComponentSwizzle component = components[c];
        if ((unsigned)component > VK_COMPONENT_SWIZZLE_A)
            return VK_ERROR_DEVICE_LOST;
        channels[c] = component == VK_COMPONENT_SWIZZLE_IDENTITY ? c :
                      component >= VK_COMPONENT_SWIZZLE_R ? component - VK_COMPONENT_SWIZZLE_R :
                      component == VK_COMPONENT_SWIZZLE_ZERO ? 4 : 5;
        identity &= channels[c] == c;
    }
    if (identity && view->format == VK_FORMAT_R8G8B8A8_UNORM)
        return VK_SUCCESS;
    *storage = malloc((size_t)(size / bpp) * 4u);
    if (!*storage)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    size_t pixels = (size_t)(size / bpp), converted = 0;
    while (converted < pixels) {
        unsigned count = pixels - converted > UINT32_MAX ? UINT32_MAX : (unsigned)(pixels - converted);
        util_format_unpack_rgba_8unorm_rect(format, *storage + converted * 4u, 0,
            source + converted * bpp, 0, count, 1);
        converted += count;
    }
    for (size_t pixel = 0; !identity && pixel < pixels; pixel++) {
        uint8_t rgba[4];
        memcpy(rgba, *storage + pixel * 4u, sizeof rgba);
        for (unsigned c = 0; c < 4; c++) {
            unsigned channel = channels[c];
            (*storage)[pixel * 4u + c] = channel < 4 ? rgba[channel] :
                                        channel == 4 ? 0 : 255;
        }
    }
    *data = *storage;
    return VK_SUCCESS;
}

static uint8_t vk_sampler_address(VkSamplerAddressMode address)
{
    switch (address) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT: return MXGPU_ADDRESS_REPEAT;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return MXGPU_ADDRESS_MIRRORED_REPEAT;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return MXGPU_ADDRESS_CLAMP_TO_EDGE;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return MXGPU_ADDRESS_CLAMP_TO_BORDER;
    default: return 0;
    }
}

static bool vk_native_sampler(const struct mx_sampler *sampler, struct mxgpu_sampler_state *state)
{
    if (!sampler) return false;
    const VkSamplerCreateInfo *s = &sampler->info;
    memset(state, 0, sizeof *state);
    if (s->compareEnable || s->unnormalizedCoordinates || s->anisotropyEnable ||
        (s->minFilter != VK_FILTER_NEAREST && s->minFilter != VK_FILTER_LINEAR) ||
        (s->magFilter != VK_FILTER_NEAREST && s->magFilter != VK_FILTER_LINEAR) ||
        (s->mipmapMode != VK_SAMPLER_MIPMAP_MODE_NEAREST && s->mipmapMode != VK_SAMPLER_MIPMAP_MODE_LINEAR))
        return false;
    state->sampler_id = 1;
    state->min_filter = s->minFilter == VK_FILTER_LINEAR ? MXGPU_FILTER_LINEAR : MXGPU_FILTER_NEAREST;
    state->mag_filter = s->magFilter == VK_FILTER_LINEAR ? MXGPU_FILTER_LINEAR : MXGPU_FILTER_NEAREST;
    state->mip_filter = s->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? MXGPU_FILTER_LINEAR : MXGPU_FILTER_NEAREST;
    state->address_u = vk_sampler_address(s->addressModeU);
    state->address_v = vk_sampler_address(s->addressModeV);
    state->address_w = vk_sampler_address(s->addressModeW);
    state->max_anisotropy = 1;
    state->mip_lod_bias = vk_float_bits(s->mipLodBias);
    state->min_lod = vk_float_bits(s->minLod);
    state->max_lod = vk_float_bits(s->maxLod);
    if (!state->address_u || !state->address_v || !state->address_w ||
        !mxgpu_f32_finite(state->mip_lod_bias) || !mxgpu_f32_finite(state->min_lod) ||
        !mxgpu_f32_finite(state->max_lod)) return false;
    switch (s->borderColor) {
    case VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK: break;
    case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK: state->border_color[3] = vk_float_bits(1.f); break;
    case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
        for (unsigned i = 0; i < 4; i++) state->border_color[i] = vk_float_bits(1.f);
        break;
    default: return false;
    }
    return true;
}

static void vk_free_texture_inputs(uint8_t *storage[MXGPU_TEXTURE_INPUTS])
{
    for (unsigned i = 0; i < MXGPU_TEXTURE_INPUTS; i++) free(storage[i]);
}

static VkResult vk_collect_texture_inputs(struct mx_cmd *cmd,
                                           struct mxgpu_texture_input inputs[MXGPU_TEXTURE_INPUTS],
                                           uint8_t *storage[MXGPU_TEXTURE_INPUTS], uint32_t *count)
{
    const struct mxgpu_shader *shader = cmd->pipe->fragment_shader;
    *count = shader ? shader->texture_count : 0;
    if (*count > MXGPU_TEXTURE_INPUTS) return VK_ERROR_DEVICE_LOST;
    for (unsigned i = 0; i < *count; i++) {
        const struct mxgpu_texture_binding *binding = &shader->textures[i];
        struct mx_set *set = NULL;
        if (cmd->bound_set_count) {
            if (binding->set >= cmd->bound_set_count) goto invalid;
            set = cmd->bound_sets[binding->set];
        } else if (!binding->set) set = cmd->set;
        if (!set) goto invalid;
        const struct mx_descriptor *descriptor = NULL;
        for (unsigned j = 0; j < set->descriptor_count; j++)
            if (set->descriptors[j].binding == binding->binding && set->descriptors[j].element == binding->element)
                descriptor = &set->descriptors[j];
        if (!descriptor || descriptor->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) goto invalid;
        struct mx_view *view = (struct mx_view *)descriptor->value.image.imageView;
        struct mx_img *image = view ? view->image : NULL;
        if (!view || !image || !image->mem || !image->mem->ptr ||
            vk_color_format(view->format) == PIPE_FORMAT_NONE || !view->width || !view->height ||
            view->width > INT32_MAX || view->height > INT32_MAX ||
            image->offset > image->mem->size || view->offset > image->mem->size - image->offset)
            goto invalid;
        VkDeviceSize offset = image->offset + view->offset;
        VkDeviceSize bytes = (VkDeviceSize)view->width * view->height * vk_format_bytes(view->format);
        if (bytes > image->mem->size - offset ||
            !vk_native_sampler((struct mx_sampler *)descriptor->value.image.sampler, &inputs[i].sampler))
            goto invalid;
        inputs[i].width = view->width;
        inputs[i].height = view->height;
        inputs[i].texture_slot = binding->texture_slot;
        inputs[i].sampler_slot = binding->sampler_slot;
        VkResult result = stage_view_texture(view, (const uint8_t *)image->mem->ptr + offset,
                                             bytes, &inputs[i].pixels, &storage[i]);
        if (result != VK_SUCCESS) {
            vk_free_texture_inputs(storage);
            memset(storage, 0, sizeof(uint8_t *) * MXGPU_TEXTURE_INPUTS);
            return result;
        }
    }
    return VK_SUCCESS;
invalid:
    vk_free_texture_inputs(storage);
    memset(storage, 0, sizeof(uint8_t *) * MXGPU_TEXTURE_INPUTS);
    return VK_ERROR_DEVICE_LOST;
}

static VkResult vk_view_backing(struct mx_view *view, uint8_t **data)
{
    struct mx_img *image = view ? view->image : NULL;
    unsigned bpp = view ? vk_format_bytes(view->format) : 0;
    if (!image || !image->mem || !image->mem->ptr || !bpp || !view->width || !view->height ||
        (VkDeviceSize)view->width * view->height > SIZE_MAX / bpp ||
        image->offset > image->mem->size || view->offset > image->mem->size - image->offset)
        return VK_ERROR_DEVICE_LOST;
    VkDeviceSize offset = image->offset + view->offset;
    VkDeviceSize bytes = (VkDeviceSize)view->width * view->height * bpp;
    if (offset > SIZE_MAX || bytes > image->mem->size - offset)
        return VK_ERROR_DEVICE_LOST;
    *data = (uint8_t *)image->mem->ptr + offset;
    return VK_SUCCESS;
}

static VkResult vk_stage_attachment(struct mx_view *view, uint32_t width, uint32_t height, uint8_t **pixels)
{
    uint8_t *source;
    VkResult result = vk_view_backing(view, &source);
    *pixels = NULL;
    if (result != VK_SUCCESS)
        return result;
    unsigned bpp = vk_format_bytes(view->format);
    enum pipe_format color_format = vk_color_format(view->format);
    unsigned staged_bpp = color_format != PIPE_FORMAT_NONE ? 4 : bpp;
    if (!width || !height || width > view->width || height > view->height ||
        (VkDeviceSize)width * height > SIZE_MAX / staged_bpp)
        return VK_ERROR_DEVICE_LOST;
    *pixels = malloc((size_t)width * height * staged_bpp);
    if (!*pixels)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    for (uint32_t y = 0; y < height; y++) {
        if (color_format != PIPE_FORMAT_NONE)
            util_format_unpack_rgba_8unorm_rect(color_format, *pixels + (size_t)y * width * 4u, 0,
                source + (size_t)y * view->width * bpp, 0, width, 1);
        else
            memcpy(*pixels + (size_t)y * width * bpp, source + (size_t)y * view->width * bpp, (size_t)width * bpp);
    }
    return VK_SUCCESS;
}

static void vk_commit_attachment(struct mx_view *view, uint32_t width, uint32_t height, const uint8_t *pixels)
{
    uint8_t *destination;
    if (vk_view_backing(view, &destination) != VK_SUCCESS)
        return;
    unsigned bpp = vk_format_bytes(view->format);
    enum pipe_format color_format = vk_color_format(view->format);
    for (uint32_t y = 0; y < height; y++) {
        if (color_format != PIPE_FORMAT_NONE)
            util_format_pack_description(color_format)->pack_rgba_8unorm(
                destination + (size_t)y * view->width * bpp, 0, pixels + (size_t)y * width * 4u, 0, width, 1);
        else
            memcpy(destination + (size_t)y * view->width * bpp, pixels + (size_t)y * width * bpp, (size_t)width * bpp);
    }
}

static VkResult vk_clear_attachments(struct mx_cmd *cmd)
{
    struct mx_view *color_view = cmd->fb ? cmd->fb->color : NULL;
    struct mx_view *depth_view = cmd->fb ? cmd->fb->depth : NULL;
    uint8_t *color_data = NULL, *depth_data = NULL;
    if (!cmd->fb ||
        (cmd->clear && (!color_view || vk_color_format(color_view->format) == PIPE_FORMAT_NONE ||
                       vk_view_backing(color_view, &color_data) != VK_SUCCESS)) ||
        (cmd->clear_depth_aspects && (!depth_view ||
          (cmd->clear_depth_aspects & ~mx_vk_format_aspects(depth_view->format)) ||
          vk_view_backing(depth_view, &depth_data) != VK_SUCCESS)))
        return VK_ERROR_DEVICE_LOST;
    uint32_t depth_bits = 0;
    if (cmd->clear_depth_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
        float depth = cmd->clear_depth.depth;
        if (!mxgpu_f32_finite(vk_float_bits(depth)) || depth < 0 || depth > 1)
            return VK_ERROR_DEVICE_LOST;
        depth_bits = depth_view->format == VK_FORMAT_D24_UNORM_S8_UINT ?
            (uint32_t)((double)depth * 16777215.0 + 0.5) : vk_float_bits(depth);
    }
    uint8_t rgba[4];
    for (unsigned c = 0; c < 4; c++) {
        float value = cmd->clear_color.float32[c];
        rgba[c] = !(value > 0.f) ? 0 : value >= 1.f ? 255 : (uint8_t)(value * 255.f + 0.5f);
    }
    uint8_t packed_color[4];
    unsigned color_bpp = color_data ? vk_format_bytes(color_view->format) : 0;
    if (color_data)
        util_format_pack_description(vk_color_format(color_view->format))->pack_rgba_8unorm(packed_color, 0, rgba, 0, 1, 1);
    int64_t left = cmd->clear_area.offset.x, top = cmd->clear_area.offset.y;
    int64_t right = left + cmd->clear_area.extent.width, bottom = top + cmd->clear_area.extent.height;
    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > cmd->fb->width ? cmd->fb->width : right;
    bottom = bottom > cmd->fb->height ? cmd->fb->height : bottom;
    for (int64_t y = top; y < bottom; y++) {
        for (int64_t x = left; x < right; x++) {
            if (color_data)
                memcpy(color_data + ((size_t)y * color_view->width + x) * color_bpp, packed_color, color_bpp);
            if (depth_data) {
                unsigned bpp = vk_format_bytes(depth_view->format);
                uint8_t *pixel = depth_data + ((size_t)y * depth_view->width + x) * bpp;
                if (cmd->clear_depth_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
                    if (depth_view->format == VK_FORMAT_D24_UNORM_S8_UINT)
                        mx_w32(pixel, 0, (mx_r32(pixel, 0) & 0xff000000u) | depth_bits);
                    else
                        mx_w32(pixel, 0, depth_bits);
                }
                if (cmd->clear_depth_aspects & VK_IMAGE_ASPECT_STENCIL_BIT)
                    pixel[depth_view->format == VK_FORMAT_D24_UNORM_S8_UINT ? 3 : 4] = cmd->clear_depth.stencil;
                if (bpp == 8)
                    memset(pixel + 5, 0, 3);
            }
        }
    }
    return VK_SUCCESS;
}

static VkResult perform_draw(struct mx_cmd *cmd);

static int fetch_vertex_attribute(struct mx_cmd *cmd, uint32_t binding, uint32_t stride,
                                  uint32_t attribute_offset, uint64_t vertex, unsigned bytes, void *destination)
{
    struct mx_buf *buffer = cmd->vertex_buffers[binding];
    VkDeviceSize base = cmd->vertex_offsets[binding];
    if (!buffer && !binding) {
        buffer = cmd->vbo;
        base = cmd->voff;
    }
    if (!buffer || !buffer->mem || !buffer->mem->ptr || buffer->offset > buffer->mem->size ||
        buffer->size > buffer->mem->size - buffer->offset)
        return -1;
    if (stride && vertex > (UINT64_MAX - attribute_offset) / stride)
        return 1;
    VkDeviceSize offset = vertex * stride + attribute_offset;
    if (base > UINT64_MAX - offset)
        return 1;
    offset += base;
    if (offset > buffer->size || bytes > buffer->size - offset)
        return 1;
    memcpy(destination, (uint8_t *)buffer->mem->ptr + buffer->offset + offset, bytes);
    return 0;
}

static int gather_vertex(struct mx_cmd *cmd, uint64_t vertex, uint64_t instance, float *out)
{
    struct mx_pipe *pipe = cmd->pipe;
    uint32_t slots = pipe->vertex_attribute_count ? pipe->vertex_attribute_count : 1;
    if (pipe->vertex_shader && !pipe->vertex_attribute_count) {
        memset(out, 0, 16);
        return 0;
    }
    if (pipe->vertex_builtins && (pipe->vertex_builtin_slot >= slots ||
        vertex > UINT32_MAX || instance < cmd->first_instance || instance - cmd->first_instance > UINT32_MAX))
        return -1;
    if (!pipe->vertex_layout) {
        if (pipe->vertex_builtins)
            return -1;
        int fetched = fetch_vertex_attribute(cmd, 0, slots * 16, 0, vertex, slots * 16, out);
        if (fetched > 0)
            memset(out, 0, (size_t)slots * 16);
        return fetched < 0 ? -1 : 0;
    }
    for (uint32_t slot = 0; slot < slots; slot++) {
        float *value = out + slot * 4;
        value[0] = value[1] = value[2] = 0.f;
        value[3] = 1.f;
        if (pipe->vertex_builtins && slot == pipe->vertex_builtin_slot) {
            uint32_t builtins[4] = {(uint32_t)vertex, (uint32_t)(instance - cmd->first_instance),
                cmd->indexed ? (uint32_t)cmd->vertex_bias : cmd->first_vertex, cmd->first_instance};
            memcpy(value, builtins, sizeof builtins);
            continue;
        }
        uint32_t location = pipe->vertex_input_locations[slot];
        if (location == UINT32_MAX) continue;
        if (location < VERT_ATTRIB_GENERIC0 || location - VERT_ATTRIB_GENERIC0 >= 32) return -1;
        location -= VERT_ATTRIB_GENERIC0;
        if (!(pipe->vertex_attribute_mask & (1u << location))) return -1;
        const VkVertexInputAttributeDescription *attribute = &pipe->vertex_attributes[location];
        enum pipe_format format = vk_vertex_format(attribute->format);
        if (format == PIPE_FORMAT_NONE)
            return -1;
        unsigned bytes = util_format_get_blocksize(format);
        union { uint8_t bytes[16]; uint32_t words[4]; } packed;
        unsigned stride = pipe->vertex_bindings[attribute->binding].stride;
        uint64_t element = pipe->vertex_bindings[attribute->binding].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? instance : vertex;
        int fetched = fetch_vertex_attribute(cmd, attribute->binding, stride, attribute->offset, element, bytes, packed.bytes);
        if (fetched < 0)
            return -1;
        if (fetched > 0)
            memset(packed.bytes, 0, sizeof packed.bytes);
        util_format_unpack_rgba(format, value, packed.bytes, 1);
    }
    return 0;
}

static VkResult perform_vertex_draw(struct mx_cmd *cmd)
{
    uint32_t count = cmd->vertex_count, index_bytes = 0;
    const uint8_t *index_data = NULL;
    uint32_t *indices;
    float *vertices;
    VkResult result = VK_ERROR_DEVICE_LOST;
    uint32_t instances = cmd->instance_count ? cmd->instance_count : 1;
    uint64_t max_vertices = (uint64_t)count * instances;
    if (!cmd->pipe || (uint64_t)cmd->first_instance + instances - 1 > UINT32_MAX ||
        max_vertices > UINT32_MAX / 3)
        return VK_ERROR_DEVICE_LOST;
    uint32_t stride_bytes = (cmd->pipe->vertex_attribute_count ? cmd->pipe->vertex_attribute_count : 1) * 16;
    max_vertices *= 3;
    if (max_vertices > UINT32_MAX || max_vertices > SIZE_MAX / stride_bytes || count > SIZE_MAX / sizeof *indices)
        return VK_ERROR_DEVICE_LOST;
    if (cmd->indexed) {
        struct mx_buf *buffer = cmd->ibo;
        index_bytes = cmd->index_type == VK_INDEX_TYPE_UINT16 ? 2 : cmd->index_type == VK_INDEX_TYPE_UINT32 ? 4 : 0;
        VkDeviceSize offset = (VkDeviceSize)cmd->first_index * index_bytes;
        VkDeviceSize bytes = (VkDeviceSize)count * index_bytes;
        if (!index_bytes || !buffer || !buffer->mem || !buffer->mem->ptr || cmd->ioff > UINT64_MAX - offset)
            return VK_ERROR_DEVICE_LOST;
        offset += cmd->ioff;
        if (offset > buffer->size || bytes > buffer->size - offset ||
            buffer->offset > buffer->mem->size || offset > buffer->mem->size - buffer->offset ||
            bytes > buffer->mem->size - buffer->offset - offset)
            return VK_ERROR_DEVICE_LOST;
        index_data = (uint8_t *)buffer->mem->ptr + buffer->offset + offset;
    }
    indices = malloc((size_t)count * sizeof *indices);
    vertices = malloc((size_t)max_vertices * stride_bytes);
    if (!indices || !vertices) {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto done;
    }
    for (unsigned i = 0; i < count; i++) {
        if (index_bytes == 2) {
            uint16_t value;
            memcpy(&value, index_data + (size_t)i * 2, 2);
            indices[i] = value;
        } else if (index_bytes == 4) {
            memcpy(&indices[i], index_data + (size_t)i * 4, 4);
        } else {
            if (i > UINT32_MAX - cmd->first_vertex)
                goto done;
            indices[i] = i + cmd->first_vertex;
        }
    }
    uint32_t output = 0;
    for (uint32_t instance = 0; instance < instances; instance++) {
        uint32_t begin = 0;
        while (begin < count) {
            uint32_t end = begin;
            uint32_t restart = index_bytes == 2 ? UINT16_MAX : UINT32_MAX;
            while (end < count && !(cmd->indexed && cmd->pipe->primitive_restart && indices[end] == restart))
                end++;
            uint32_t length = end - begin;
            unsigned triangles = cmd->pipe->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ? length / 3 : length > 2 ? length - 2 : 0;
            for (unsigned triangle = 0; triangle < triangles; triangle++) {
                unsigned source[3];
                if (cmd->pipe->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
                    source[0] = begin + triangle * 3;
                    source[1] = source[0] + 1;
                    source[2] = source[0] + 2;
                } else if (cmd->pipe->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) {
                    source[0] = begin + triangle + (triangle & 1);
                    source[1] = begin + triangle + !(triangle & 1);
                    source[2] = begin + triangle + 2;
                } else {
                    source[0] = begin;
                    source[1] = begin + triangle + 1;
                    source[2] = begin + triangle + 2;
                }
                for (unsigned lane = 0; lane < 3; lane++) {
                    int64_t vertex = (int64_t)indices[source[lane]] + (cmd->indexed ? cmd->vertex_bias : 0);
                    if (gather_vertex(cmd, (uint64_t)(uint32_t)vertex, (uint64_t)cmd->first_instance + instance,
                                      vertices + (size_t)output * (stride_bytes / 4)))
                        goto done;
                    output++;
                }
            }
            begin = end + (end < count);
        }
    }
    if (!output) {
        result = VK_SUCCESS;
        goto done;
    }
    struct mx_mem memory = {.ptr = vertices, .size = (VkDeviceSize)output * stride_bytes};
    struct mx_buf buffer = {.mem = &memory, .size = memory.size};
    struct mx_cmd direct = *cmd;
    direct.vbo = &buffer;
    direct.voff = 0;
    direct.first_vertex = 0;
    direct.vertex_count = output;
    direct.instance_count = 1;
    direct.first_instance = 0;
    direct.indexed = 0;
    direct.packed_vertices = true;
    result = perform_draw(&direct);
done:
    free(indices);
    free(vertices);
    return result;
}

static VkResult perform_draw(struct mx_cmd *cmd)
{
    uint8_t white[4] = {255, 255, 255, 255};
    struct mx_mem fallback_mem = {.size = 4, .ptr = white};
    struct mx_img fallback_img = {.width = 1, .height = 1, .mem = &fallback_mem, .size = 4, .format = VK_FORMAT_R8G8B8A8_UNORM};
    struct mx_view fallback_view = {.image = &fallback_img, .width = 1, .height = 1, .format = VK_FORMAT_R8G8B8A8_UNORM};
    unsigned char *color;
    struct mx_img *tex;
    struct mx_img *dst;
    struct mx_view *tv, *dv;
    VkDeviceSize tex_offset, dst_offset;
    const float *verts;
    VkDeviceSize vertex_bytes, texture_bytes, color_bytes, vertex_offset;
    VkDeviceSize binding_offset;
    if (cmd->draw && !cmd->clear && !cmd->clear_depth_aspects && cmd->vertex_count && !cmd->packed_vertices)
        return perform_vertex_draw(cmd);
    if (cmd->clear || cmd->clear_depth_aspects)
        return vk_clear_attachments(cmd);
    if (!cmd->draw || !cmd->vertex_count)
        return VK_SUCCESS;
    if (!cmd->pipe || (!cmd->pipe->bytes && !cmd->pipe->vertex_shader) || !cmd->vbo || !cmd->vbo->mem ||
        !cmd->fb || (!cmd->fb->color && !cmd->fb->depth))
        return VK_ERROR_DEVICE_LOST;
    if (cmd->pipe->uses_texture) {
        uint32_t i;
        struct mx_set *set = cmd->set;
        if (cmd->pipe->reflected_texture && cmd->bound_set_count) {
            if (cmd->pipe->texture_set >= cmd->bound_set_count)
                return VK_ERROR_DEVICE_LOST;
            set = cmd->bound_sets[cmd->pipe->texture_set];
        } else if (cmd->pipe->reflected_texture && cmd->pipe->texture_set) {
            return VK_ERROR_DEVICE_LOST;
        }
        if (!set)
            return VK_ERROR_DEVICE_LOST;
        tv = set->image;
        if (cmd->pipe->reflected_texture) {
            tv = NULL;
            for (i = 0; i < set->descriptor_count; i++) {
                struct mx_descriptor *descriptor = &set->descriptors[i];
                if (descriptor->binding == cmd->pipe->texture_binding &&
                    descriptor->element == cmd->pipe->texture_element)
                    tv = (struct mx_view *)descriptor->value.image.imageView;
            }
        }
        if (!tv)
            return VK_ERROR_DEVICE_LOST;
    } else {
        tv = &fallback_view;
    }
    dv = cmd->fb->color;
    tex = tv->image;
    dst = dv ? dv->image : NULL;
    uint32_t render_width = cmd->fb->width, render_height = cmd->fb->height;
    if (!tex || !tex->mem || (dv && (!dst || !dst->mem)) || !tv->width ||
        !tv->height || !render_width || !render_height ||
        tv->width > INT32_MAX || tv->height > INT32_MAX ||
        render_width > INT32_MAX || render_height > INT32_MAX ||
        cmd->vertex_count > INT32_MAX)
        return VK_ERROR_DEVICE_LOST;
    if (tv->offset > UINT64_MAX - tex->offset || (dv && dv->offset > UINT64_MAX - dst->offset))
        return VK_ERROR_DEVICE_LOST;
    tex_offset = tex->offset + tv->offset;
    dst_offset = dv ? dst->offset + dv->offset : 0;
    uint32_t stride_bytes = (cmd->pipe->vertex_attribute_count ? cmd->pipe->vertex_attribute_count : 1) * 16;
    vertex_bytes = (VkDeviceSize)cmd->vertex_count * stride_bytes;
    binding_offset = (VkDeviceSize)cmd->first_vertex * stride_bytes;
    if (cmd->voff > UINT64_MAX - binding_offset)
        return VK_ERROR_DEVICE_LOST;
    binding_offset += cmd->voff;
    unsigned texture_bpp = vk_format_bytes(tv->format);
    unsigned color_bpp = dv ? vk_format_bytes(dv->format) : 4;
    if (vk_color_format(tv->format) == PIPE_FORMAT_NONE ||
        (dv && vk_color_format(dv->format) == PIPE_FORMAT_NONE))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    texture_bytes = (VkDeviceSize)tv->width * tv->height * texture_bpp;
    color_bytes = (VkDeviceSize)render_width * render_height * 4u;
    if (binding_offset > cmd->vbo->size || vertex_bytes > cmd->vbo->size - binding_offset ||
        cmd->vbo->offset > cmd->vbo->mem->size ||
        binding_offset > cmd->vbo->mem->size - cmd->vbo->offset)
        return VK_ERROR_DEVICE_LOST;
    vertex_offset = cmd->vbo->offset + binding_offset;
    if (vertex_bytes > cmd->vbo->mem->size - vertex_offset ||
        tex_offset > tex->mem->size || texture_bytes > tex->mem->size - tex_offset ||
        (dv && (dst_offset > dst->mem->size ||
          (VkDeviceSize)render_width * render_height * color_bpp > dst->mem->size - dst_offset || !dst->mem->ptr)) ||
        color_bytes > SIZE_MAX || !cmd->vbo->mem->ptr || !tex->mem->ptr)
        return VK_ERROR_DEVICE_LOST;
    if (dv) {
        VkResult color_result = vk_stage_attachment(dv, render_width, render_height, &color);
        if (color_result != VK_SUCCESS)
            return color_result;
    } else {
        color = calloc(1, (size_t)color_bytes);
        if (!color)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    verts = (const float *)((unsigned char *)cmd->vbo->mem->ptr + vertex_offset);
    uint8_t *uniform_data[2] = {NULL, NULL};
    uint32_t uniform_size[2] = {0, 0};
    VkResult uniform_result = collect_uniforms(cmd, 0, &uniform_data[0], &uniform_size[0]);
    if (uniform_result == VK_SUCCESS)
        uniform_result = collect_uniforms(cmd, 1, &uniform_data[1], &uniform_size[1]);
    if (uniform_result != VK_SUCCESS) {
        free(uniform_data[0]);
        free(uniform_data[1]);
        free(color);
        return uniform_result;
    }
    const uint8_t *texels;
    uint8_t *mapped_texels;
    VkResult mapping_result = stage_view_texture(tv, (const uint8_t *)tex->mem->ptr + tex_offset,
                                                 texture_bytes, &texels, &mapped_texels);
    if (mapping_result != VK_SUCCESS) {
        free(uniform_data[0]);
        free(uniform_data[1]);
        free(color);
        return mapping_result;
    }
    struct mxgpu_native_render_state native_state;
    bool native = mxgpu_native_render_available(-1);
    if (!vk_draw_render_state(cmd, render_width, render_height, &native_state)) {
        free(mapped_texels); free(uniform_data[0]); free(uniform_data[1]); free(color);
        return VK_ERROR_DEVICE_LOST;
    }
    if (native_state.scissor.left >= native_state.scissor.right || native_state.scissor.top >= native_state.scissor.bottom) {
        free(mapped_texels); free(uniform_data[0]); free(uniform_data[1]); free(color);
        return VK_SUCCESS;
    }
    native_state.depth_enabled = cmd->fb->depth && cmd->pipe->depth_enabled;
    if (!dv) native_state.blend.targets[0].write_mask = 0;
    else native_state.blend.targets[0].write_mask &= vk_color_write_mask(dv->format);
    if (!native && (native_state.depth_enabled || native_state.blend.targets[0].enable || native_state.blend.targets[0].write_mask != 15 ||
        native_state.rasterizer.cull_mode != MXGPU_CULL_NONE || !native_state.rasterizer.depth_clip_enable ||
        native_state.viewport.x || native_state.viewport.y != vk_float_bits((float)render_height) ||
        native_state.viewport.width != vk_float_bits((float)render_width) ||
        native_state.viewport.height != vk_float_bits(-(float)render_height) ||
        native_state.viewport.min_depth || native_state.viewport.max_depth != vk_float_bits(1.f) ||
        native_state.scissor.left || native_state.scissor.top ||
        native_state.scissor.right != render_width || native_state.scissor.bottom != render_height)) {
        free(mapped_texels); free(uniform_data[0]); free(uniform_data[1]); free(color);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    struct mxgpu_texture_input texture_inputs[MXGPU_TEXTURE_INPUTS] = {0};
    uint8_t *texture_storage[MXGPU_TEXTURE_INPUTS] = {0};
    uint32_t texture_count = 0;
    VkResult texture_result = native ? vk_collect_texture_inputs(cmd, texture_inputs, texture_storage, &texture_count) :
        (cmd->pipe->fragment_shader && cmd->pipe->fragment_shader->texture_count > 1 ?
         VK_ERROR_FEATURE_NOT_PRESENT : VK_SUCCESS);
    if (texture_result != VK_SUCCESS || (texture_count && (!native || !mxgpu_native_sampler_available(-1)))) {
        vk_free_texture_inputs(texture_storage);
        free(mapped_texels); free(uniform_data[0]); free(uniform_data[1]); free(color);
        return texture_result != VK_SUCCESS ? texture_result : VK_ERROR_FEATURE_NOT_PRESENT;
    }
    native_state.sampler_enabled = texture_count != 0;
    uint8_t *draw_module = NULL;
    const uint8_t *module = cmd->pipe->bytes;
    uint32_t module_len = cmd->pipe->len;
    if (cmd->pipe->vertex_shader) {
        draw_module = malloc(MXGPU_LINK_MODULE_CAPACITY);
        uint32_t sampler_compare[MXGPU_TEXTURE_INPUTS] = {0};
        for (uint32_t i = 0; i < texture_count; i++)
            sampler_compare[i] = texture_inputs[i].sampler.compare;
        if (!draw_module || mxgpu_link_shaders_draw_samplers(cmd->pipe->vertex_shader, cmd->pipe->fragment_shader,
                cmd->vertex_count, texture_count != 0, draw_module, MXGPU_LINK_MODULE_CAPACITY, &module_len,
                sampler_compare) != 0) {
            VkResult failure = draw_module ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
            vk_free_texture_inputs(texture_storage);
            free(draw_module); free(mapped_texels); free(uniform_data[0]); free(uniform_data[1]); free(color);
            return failure;
        }
        module = draw_module;
    }
    uint8_t *depth_pixels = NULL;
    if (native_state.depth_enabled) {
        struct mx_view *depth_view = cmd->fb->depth;
        uint32_t depth_format = vk_depth_format(depth_view->format);
        VkResult depth_result = !native || !mxgpu_native_depth_available(-1, depth_format) ?
            VK_ERROR_FEATURE_NOT_PRESENT : vk_stage_attachment(depth_view, render_width, render_height, &depth_pixels);
        if (depth_result != VK_SUCCESS) {
            free(draw_module); vk_free_texture_inputs(texture_storage);
            free(mapped_texels); free(uniform_data[0]); free(uniform_data[1]); free(color);
            return depth_result;
        }
        native_state.depth = (struct mxgpu_depth_input){depth_pixels, depth_format, render_width, render_height};
        if (depth_view->format == VK_FORMAT_D32_SFLOAT)
            native_state.depth_stencil.stencil_enable = 0;
    }
    int execute_result = mxgpu_execute_module_transaction_native_resources(-1, module, module_len, verts,
                             (int)cmd->vertex_count,
                             texels,
                             (int)tv->width, (int)tv->height,
                             color, color,
                             (int)render_width, (int)render_height,
                             uniform_data[0], uniform_size[0], uniform_data[1], uniform_size[1], NULL, native ? &native_state : NULL, stride_bytes, texture_inputs, texture_count);
    free(draw_module);
    vk_free_texture_inputs(texture_storage);
    free(mapped_texels);
    free(uniform_data[0]);
    free(uniform_data[1]);
    if (execute_result != 0) {
        free(depth_pixels);
        free(color);
        return VK_ERROR_DEVICE_LOST;
    }
    if (dv) vk_commit_attachment(dv, render_width, render_height, color);
    if (depth_pixels) vk_commit_attachment(cmd->fb->depth, render_width, render_height, depth_pixels);
    free(depth_pixels);
    free(color);
    return VK_SUCCESS;
}

static VkResult end_cmd(VkCommandBuffer command)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    cmd->open = 0;
    return cmd->record_result;
}

static VkResult create_fence(VkDevice device, const VkFenceCreateInfo *info, const VkAllocationCallbacks *alloc, VkFence *out)
{
    struct mx_fence *fence = calloc(1, sizeof *fence);
    (void)device;
    (void)alloc;
    if (!fence)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    atomic_init(&fence->signaled, (info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0);
    *out = (VkFence)fence;
    return VK_SUCCESS;
}

static void destroy_fence(VkDevice device, VkFence fence, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(fence);
}

static VkResult reset_fences(VkDevice device, uint32_t count, const VkFence *fences)
{
    struct mx_device *owner = (struct mx_device *)device;
    uint32_t i;
    if (owner && atomic_load_explicit(&owner->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    for (i = 0; i < count; i++)
        ((struct mx_fence *)fences[i])->signaled = 0;
    return VK_SUCCESS;
}

static VkResult create_semaphore(VkDevice device, const VkSemaphoreCreateInfo *info,
                                 const VkAllocationCallbacks *alloc, VkSemaphore *out)
{
    struct mx_semaphore *semaphore;
    struct mx_device *owner = (struct mx_device *)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (owner && atomic_load_explicit(&owner->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    if (!info || info->flags)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (const VkBaseInStructure *next = info->pNext; next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO &&
            ((const VkSemaphoreTypeCreateInfo *)next)->semaphoreType != VK_SEMAPHORE_TYPE_BINARY)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    semaphore = calloc(1, sizeof *semaphore);
    if (!semaphore)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    semaphore->device = owner;
    atomic_init(&semaphore->signaled, 0);
    *out = (VkSemaphore)semaphore;
    return VK_SUCCESS;
}

static void destroy_semaphore(VkDevice device, VkSemaphore semaphore,
                              const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(semaphore);
}

static VkResult semaphore_transition(struct mx_device *device, VkSemaphore handle, int signal)
{
    struct mx_semaphore *semaphore = (struct mx_semaphore *)handle;
    int expected = !signal;
    if (!semaphore || semaphore->device != device ||
        !atomic_compare_exchange_strong_explicit(&semaphore->signaled, &expected, signal,
                                                 memory_order_acq_rel, memory_order_acquire)) {
        if (device)
            atomic_store_explicit(&device->lost, 1, memory_order_release);
        return VK_ERROR_DEVICE_LOST;
    }
    return VK_SUCCESS;
}

static VkResult queue_submit(VkQueue queue, uint32_t count, const VkSubmitInfo *submits, VkFence fence)
{
    struct mx_device *device = queue ? ((struct mx_queue *)queue)->device : NULL;
    uint32_t i, c;
    if (device && atomic_load_explicit(&device->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    for (i = 0; i < count; i++) {
        for (c = 0; c < submits[i].commandBufferCount; c++) {
            struct mx_cmd *cmd = (struct mx_cmd *)submits[i].pCommandBuffers[c];
            if (!cmd || cmd->open) {
                if (device)
                    atomic_store_explicit(&device->lost, 1, memory_order_release);
                return VK_ERROR_DEVICE_LOST;
            }
            if (cmd->record_result != VK_SUCCESS)
                return cmd->record_result;
        }
    }
    for (i = 0; i < count; i++) {
        for (c = 0; c < submits[i].waitSemaphoreCount; c++) {
            VkResult result = semaphore_transition(device, submits[i].pWaitSemaphores[c], 0);
            if (result != VK_SUCCESS)
                return result;
        }
        for (c = 0; c < submits[i].commandBufferCount; c++) {
            struct mx_cmd *cmd = (struct mx_cmd *)submits[i].pCommandBuffers[c];
            struct mx_draw *draw;
            for (draw = cmd->draws; draw; draw = draw->next) {
                VkResult result = draw->operation == 5 ? perform_dispatch(draw) :
                                  draw->operation == 4 ? perform_pipeline_barrier() :
                                  draw->operation == 1 ? perform_buffer_copy(draw) :
                                  draw->operation >= 2 ? perform_buffer_image_copy(draw) : perform_draw(&draw->state);
                if (result != VK_SUCCESS) {
                    if (device && result == VK_ERROR_DEVICE_LOST)
                        atomic_store_explicit(&device->lost, 1, memory_order_release);
                    return result;
                }
            }
        }
        for (c = 0; c < submits[i].signalSemaphoreCount; c++) {
            VkResult result = semaphore_transition(device, submits[i].pSignalSemaphores[c], 1);
            if (result != VK_SUCCESS)
                return result;
        }
    }
    if (fence)
        ((struct mx_fence *)fence)->signaled = 1;
    return VK_SUCCESS;
}

static VkResult wait_fences(VkDevice device, uint32_t count, const VkFence *fences, VkBool32 wait_all, uint64_t timeout)
{
    struct mx_device *owner = (struct mx_device *)device;
    struct timespec start, now;
    if (owner && atomic_load_explicit(&owner->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    if (!count)
        return VK_SUCCESS;
    if (clock_gettime(CLOCK_MONOTONIC, &start))
        return VK_ERROR_UNKNOWN;
    for (;;) {
        uint32_t i, signaled = 0;
        uint64_t elapsed, remaining, pause_ns;
        struct timespec pause;
        if (owner && atomic_load_explicit(&owner->lost, memory_order_acquire))
            return VK_ERROR_DEVICE_LOST;
        for (i = 0; i < count; i++)
            signaled += atomic_load_explicit(&((struct mx_fence *)fences[i])->signaled,
                                             memory_order_acquire) != 0;
        if (wait_all ? signaled == count : signaled != 0)
            return VK_SUCCESS;
        if (!timeout)
            return VK_TIMEOUT;
        if (clock_gettime(CLOCK_MONOTONIC, &now))
            return VK_ERROR_UNKNOWN;
        elapsed = (uint64_t)(now.tv_sec - start.tv_sec) * 1000000000ull;
        if (now.tv_nsec >= start.tv_nsec)
            elapsed += (uint64_t)(now.tv_nsec - start.tv_nsec);
        else
            elapsed -= (uint64_t)(start.tv_nsec - now.tv_nsec);
        if (timeout != UINT64_MAX && elapsed >= timeout)
            return VK_TIMEOUT;
        remaining = timeout == UINT64_MAX ? UINT64_MAX : timeout - elapsed;
        pause_ns = remaining < 1000000ull ? remaining : 1000000ull;
        pause.tv_sec = 0;
        pause.tv_nsec = (long)pause_ns;
        nanosleep(&pause, NULL);
    }
}

static VkResult fence_status(VkDevice device, VkFence fence)
{
    struct mx_device *owner = (struct mx_device *)device;
    if (owner && atomic_load_explicit(&owner->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    return atomic_load_explicit(&((struct mx_fence *)fence)->signaled,
                                memory_order_acquire) ? VK_SUCCESS : VK_NOT_READY;
}

static VkResult queue_wait(VkQueue queue)
{
    struct mx_device *device = queue ? ((struct mx_queue *)queue)->device : NULL;
    return device && atomic_load_explicit(&device->lost, memory_order_acquire) ?
        VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

static VkResult device_wait(VkDevice device)
{
    struct mx_device *owner = (struct mx_device *)device;
    return owner && atomic_load_explicit(&owner->lost, memory_order_acquire) ?
        VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

static int descriptor_order(const void *left, const void *right)
{
    const struct mx_descriptor *a = left, *b = right;
    if (a->binding != b->binding)
        return a->binding < b->binding ? -1 : 1;
    return a->element < b->element ? -1 : a->element > b->element;
}

static VkResult create_dsl(VkDevice device, const VkDescriptorSetLayoutCreateInfo *info, const VkAllocationCallbacks *alloc, VkDescriptorSetLayout *out)
{
    struct mx_ds_layout *layout;
    uint64_t count = 0;
    uint32_t i, j, at = 0;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    for (i = 0; i < info->bindingCount; i++)
        count += info->pBindings[i].descriptorCount;
    if (count > UINT32_MAX || count > SIZE_MAX / sizeof(struct mx_descriptor))
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    layout = calloc(1, sizeof *layout);
    if (!layout)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    layout->count = (uint32_t)count;
    if (count) {
        layout->descriptors = calloc((size_t)count, sizeof *layout->descriptors);
        if (!layout->descriptors) {
            free(layout);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    for (i = 0; i < info->bindingCount; i++) {
        const VkDescriptorSetLayoutBinding *binding = &info->pBindings[i];
        for (j = 0; j < binding->descriptorCount; j++) {
            struct mx_descriptor *descriptor = &layout->descriptors[at++];
            descriptor->binding = binding->binding;
            descriptor->element = j;
            descriptor->type = binding->descriptorType;
            descriptor->stages = binding->stageFlags;
            if (binding->pImmutableSamplers)
                descriptor->immutable_sampler = descriptor->value.image.sampler = binding->pImmutableSamplers[j];
        }
    }
    if (count)
        qsort(layout->descriptors, (size_t)count, sizeof *layout->descriptors, descriptor_order);
    *out = (VkDescriptorSetLayout)layout;
    return VK_SUCCESS;
}

static void destroy_dsl(VkDevice device, VkDescriptorSetLayout handle, const VkAllocationCallbacks *alloc)
{
    struct mx_ds_layout *layout = (struct mx_ds_layout *)handle;
    (void)device;
    (void)alloc;
    if (layout)
        free(layout->descriptors);
    free(layout);
}

static VkResult create_pool_ds(VkDevice device, const VkDescriptorPoolCreateInfo *info, const VkAllocationCallbacks *alloc, VkDescriptorPool *out)
{
    struct mx_ds_pool *pool;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    pool = calloc(1, sizeof *pool);
    if (!pool)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    pool->max_sets = info->maxSets;
    *out = (VkDescriptorPool)pool;
    return VK_SUCCESS;
}

static VkResult reset_pool_ds(VkDevice device, VkDescriptorPool handle, VkDescriptorPoolResetFlags flags)
{
    struct mx_ds_pool *pool = (struct mx_ds_pool *)handle;
    (void)device;
    (void)flags;
    while (pool->sets) {
        struct mx_set *set = pool->sets;
        pool->sets = set->next;
        free(set->descriptors);
        free(set);
    }
    pool->count = 0;
    return VK_SUCCESS;
}

static VkResult free_sets(VkDevice device, VkDescriptorPool handle, uint32_t count, const VkDescriptorSet *sets)
{
    struct mx_ds_pool *pool = (struct mx_ds_pool *)handle;
    uint32_t i;
    (void)device;
    for (i = 0; i < count; i++) {
        struct mx_set **link = &pool->sets;
        while (*link && *link != (struct mx_set *)sets[i])
            link = &(*link)->next;
        if (*link) {
            struct mx_set *set = *link;
            *link = set->next;
            free(set->descriptors);
            free(set);
            pool->count--;
        }
    }
    return VK_SUCCESS;
}

static void destroy_pool_ds(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    if (!pool)
        return;
    reset_pool_ds(device, pool, 0);
    free(pool);
}

static VkResult alloc_sets(VkDevice device, const VkDescriptorSetAllocateInfo *info, VkDescriptorSet *out)
{
    struct mx_ds_pool *pool = (struct mx_ds_pool *)info->descriptorPool;
    uint32_t i;
    for (i = 0; i < info->descriptorSetCount; i++)
        out[i] = VK_NULL_HANDLE;
    if (info->descriptorSetCount > pool->max_sets - pool->count)
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    for (i = 0; i < info->descriptorSetCount; i++) {
        struct mx_ds_layout *layout = (struct mx_ds_layout *)info->pSetLayouts[i];
        struct mx_set *set = calloc(1, sizeof *set);
        if (set && layout->count) {
            set->descriptors = malloc((size_t)layout->count * sizeof *set->descriptors);
            if (!set->descriptors) {
                free(set);
                set = NULL;
            } else {
                memcpy(set->descriptors, layout->descriptors, (size_t)layout->count * sizeof *set->descriptors);
                set->descriptor_count = layout->count;
            }
        }
        if (!set) {
            free_sets(device, info->descriptorPool, i, out);
            for (i = 0; i < info->descriptorSetCount; i++)
                out[i] = VK_NULL_HANDLE;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        set->pool = pool;
        set->next = pool->sets;
        pool->sets = set;
        pool->count++;
        out[i] = (VkDescriptorSet)set;
    }
    return VK_SUCCESS;
}

static uint32_t descriptor_index(struct mx_set *set, uint32_t binding, uint32_t element)
{
    uint32_t i;
    for (i = 0; i < set->descriptor_count; i++)
        if (set->descriptors[i].binding == binding && set->descriptors[i].element == element)
            return i;
    return UINT32_MAX;
}

static void write_descriptor(struct mx_set *set, struct mx_descriptor *descriptor, const VkDescriptorImageInfo *image,
                             const VkDescriptorBufferInfo *buffer, const VkBufferView *texel)
{
    if (image) {
        descriptor->value.image = *image;
        if (descriptor->immutable_sampler)
            descriptor->value.image.sampler = descriptor->immutable_sampler;
        if (descriptor->value.image.imageView)
            set->image = (struct mx_view *)descriptor->value.image.imageView;
    } else if (buffer) {
        descriptor->value.buffer = *buffer;
    } else if (texel) {
        descriptor->value.texel = *texel;
    }
}

static void update_sets(VkDevice device, uint32_t count, const VkWriteDescriptorSet *writes, uint32_t copy_count, const VkCopyDescriptorSet *copies)
{
    uint32_t i, j;
    (void)device;
    for (i = 0; i < count; i++) {
        const VkWriteDescriptorSet *write = &writes[i];
        struct mx_set *set = (struct mx_set *)write->dstSet;
        uint32_t at = descriptor_index(set, write->dstBinding, write->dstArrayElement);
        if (at == UINT32_MAX || write->descriptorCount > set->descriptor_count - at)
            continue;
        for (j = 0; j < write->descriptorCount; j++) {
            struct mx_descriptor *descriptor = &set->descriptors[at + j];
            if (descriptor->type != write->descriptorType)
                break;
            write_descriptor(set, descriptor, write->pImageInfo ? &write->pImageInfo[j] : NULL,
                             write->pBufferInfo ? &write->pBufferInfo[j] : NULL,
                             write->pTexelBufferView ? &write->pTexelBufferView[j] : NULL);
        }
    }
    for (i = 0; i < copy_count; i++) {
        const VkCopyDescriptorSet *copy = &copies[i];
        struct mx_set *src = (struct mx_set *)copy->srcSet, *dst = (struct mx_set *)copy->dstSet;
        uint32_t from = descriptor_index(src, copy->srcBinding, copy->srcArrayElement);
        uint32_t to = descriptor_index(dst, copy->dstBinding, copy->dstArrayElement);
        if (from == UINT32_MAX || to == UINT32_MAX ||
            copy->descriptorCount > src->descriptor_count - from || copy->descriptorCount > dst->descriptor_count - to)
            continue;
        for (j = 0; j < copy->descriptorCount; j++) {
            struct mx_descriptor *d = &dst->descriptors[to + j], *s = &src->descriptors[from + j];
            if (d->type != s->type)
                break;
            d->value = s->value;
            if (d->immutable_sampler)
                d->value.image.sampler = d->immutable_sampler;
            if (d->value.image.imageView)
                dst->image = (struct mx_view *)d->value.image.imageView;
        }
    }
}

struct mx_update_template {
    uint32_t count;
    VkDescriptorUpdateTemplateEntry *entries;
};

static VkResult create_update_template(VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *info,
                                       const VkAllocationCallbacks *alloc, VkDescriptorUpdateTemplate *out)
{
    struct mx_update_template *template;
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    template = calloc(1, sizeof *template);
    if (!template)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (info->descriptorUpdateEntryCount) {
        template->entries = malloc((size_t)info->descriptorUpdateEntryCount * sizeof *template->entries);
        if (!template->entries) {
            free(template);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        memcpy(template->entries, info->pDescriptorUpdateEntries,
               (size_t)info->descriptorUpdateEntryCount * sizeof *template->entries);
        template->count = info->descriptorUpdateEntryCount;
    }
    *out = (VkDescriptorUpdateTemplate)template;
    return VK_SUCCESS;
}

static void destroy_update_template(VkDevice device, VkDescriptorUpdateTemplate handle,
                                    const VkAllocationCallbacks *alloc)
{
    struct mx_update_template *template = (struct mx_update_template *)handle;
    (void)device;
    (void)alloc;
    if (template)
        free(template->entries);
    free(template);
}

static void update_set_with_template(VkDevice device, VkDescriptorSet handle, VkDescriptorUpdateTemplate tmpl,
                                     const void *data)
{
    struct mx_set *set = (struct mx_set *)handle;
    const struct mx_update_template *template = (const struct mx_update_template *)tmpl;
    (void)device;
    for (uint32_t i = 0; set && template && i < template->count; i++) {
        const VkDescriptorUpdateTemplateEntry *entry = &template->entries[i];
        uint32_t at = descriptor_index(set, entry->dstBinding, entry->dstArrayElement);
        if (at == UINT32_MAX || entry->descriptorCount > set->descriptor_count - at)
            continue;
        for (uint32_t j = 0; j < entry->descriptorCount; j++) {
            struct mx_descriptor *descriptor = &set->descriptors[at + j];
            const uint8_t *source = (const uint8_t *)data + entry->offset + (size_t)j * entry->stride;
            if (descriptor->type != entry->descriptorType)
                break;
            switch (entry->descriptorType) {
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
                VkDescriptorImageInfo image;
                memcpy(&image, source, sizeof image);
                write_descriptor(set, descriptor, &image, NULL, NULL);
                break;
            }
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
                VkDescriptorBufferInfo buffer;
                memcpy(&buffer, source, sizeof buffer);
                write_descriptor(set, descriptor, NULL, &buffer, NULL);
                break;
            }
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
                VkBufferView texel;
                memcpy(&texel, source, sizeof texel);
                write_descriptor(set, descriptor, NULL, NULL, &texel);
                break;
            }
            default:
                break;
            }
        }
    }
}

struct mx_ycbcr_conversion {
    VkSamplerYcbcrConversionCreateInfo info;
};

static VkResult create_ycbcr_conversion(VkDevice device, const VkSamplerYcbcrConversionCreateInfo *info,
                                        const VkAllocationCallbacks *alloc, VkSamplerYcbcrConversion *out)
{
    struct mx_ycbcr_conversion *conversion = calloc(1, sizeof *conversion);
    (void)device;
    (void)alloc;
    *out = VK_NULL_HANDLE;
    if (!conversion)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    conversion->info = *info;
    conversion->info.pNext = NULL;
    *out = (VkSamplerYcbcrConversion)conversion;
    return VK_SUCCESS;
}

static void destroy_ycbcr_conversion(VkDevice device, VkSamplerYcbcrConversion conversion,
                                     const VkAllocationCallbacks *alloc)
{
    (void)device;
    (void)alloc;
    free(conversion);
}

static void layout_support(VkDevice device, const VkDescriptorSetLayoutCreateInfo *info,
                           VkDescriptorSetLayoutSupport *support)
{
    uint64_t count = 0;
    (void)device;
    for (uint32_t i = 0; i < info->bindingCount; i++)
        count += info->pBindings[i].descriptorCount;
    support->supported = count <= UINT32_MAX && count <= SIZE_MAX / sizeof(struct mx_descriptor);
    for (VkBaseOutStructure *next = support->pNext; next; next = next->pNext)
        if (next->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT)
            ((VkDescriptorSetVariableDescriptorCountLayoutSupport *)next)->maxVariableDescriptorCount = 0;
}

static void get_queue2(VkDevice device, const VkDeviceQueueInfo2 *info, VkQueue *queue)
{
    if (info->flags) {
        *queue = VK_NULL_HANDLE;
        return;
    }
    get_queue(device, info->queueFamilyIndex, info->queueIndex, queue);
}

static void dedicated_requirements(void *chain)
{
    for (VkBaseOutStructure *next = chain; next; next = next->pNext)
        if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
            VkMemoryDedicatedRequirements *dedicated = (VkMemoryDedicatedRequirements *)next;
            dedicated->prefersDedicatedAllocation = VK_FALSE;
            dedicated->requiresDedicatedAllocation = VK_FALSE;
        }
}

static void buffer_reqs2(VkDevice device, const VkBufferMemoryRequirementsInfo2 *info, VkMemoryRequirements2 *reqs)
{
    buffer_reqs(device, info->buffer, &reqs->memoryRequirements);
    dedicated_requirements(reqs->pNext);
}

static void image_reqs2(VkDevice device, const VkImageMemoryRequirementsInfo2 *info, VkMemoryRequirements2 *reqs)
{
    image_reqs(device, info->image, &reqs->memoryRequirements);
    dedicated_requirements(reqs->pNext);
}

static void image_sparse_reqs(VkDevice device, VkImage image, uint32_t *count,
                              VkSparseImageMemoryRequirements *reqs)
{
    (void)device;
    (void)image;
    (void)reqs;
    *count = 0;
}

static void image_sparse_reqs2(VkDevice device, const VkImageSparseMemoryRequirementsInfo2 *info, uint32_t *count,
                               VkSparseImageMemoryRequirements2 *reqs)
{
    (void)device;
    (void)info;
    (void)reqs;
    *count = 0;
}

static VkResult bind_buffer2(VkDevice device, uint32_t count, const VkBindBufferMemoryInfo *infos)
{
    for (uint32_t i = 0; i < count; i++) {
        VkResult result = bind_buffer(device, infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
        if (result != VK_SUCCESS)
            return result;
    }
    return VK_SUCCESS;
}

static VkResult bind_image2(VkDevice device, uint32_t count, const VkBindImageMemoryInfo *infos)
{
    for (uint32_t i = 0; i < count; i++) {
        VkResult result = bind_image(device, infos[i].image, infos[i].memory, infos[i].memoryOffset);
        if (result != VK_SUCCESS)
            return result;
    }
    return VK_SUCCESS;
}

static void trim_pool(VkDevice device, VkCommandPool pool, VkCommandPoolTrimFlags flags)
{
    (void)device;
    (void)pool;
    (void)flags;
}

static void peer_memory_features(VkDevice device, uint32_t heap, uint32_t local, uint32_t remote,
                                 VkPeerMemoryFeatureFlags *features)
{
    (void)device;
    (void)heap;
    (void)local;
    (void)remote;
    *features = 0;
}

static void cmd_set_device_mask(VkCommandBuffer command, uint32_t mask)
{
    struct mx_cmd *cmd = (struct mx_cmd *)command;
    if (mask != 1 && cmd->record_result == VK_SUCCESS)
        cmd->record_result = VK_ERROR_DEVICE_LOST;
}

static void mem_props(VkPhysicalDevice gpu, VkPhysicalDeviceMemoryProperties *props)
{
    (void)gpu;
    memset(props, 0, sizeof *props);
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    props->memoryHeapCount = 1;
    props->memoryHeaps[0].size = MXGPU_VK_HEAP_BYTES;
    props->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#include "mxgpu_vk_wsi.c"
#endif

static PFN_vkVoidFunction device_proc(const char *name)
{
    if (!name)
        return NULL;
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    PFN_vkVoidFunction wsi = wsi_device_proc(name);
    if (wsi)
        return wsi;
#endif
    if (strcmp(name, "vkGetDeviceQueue") == 0) return (PFN_vkVoidFunction)get_queue;
    if (strcmp(name, "vkDestroyDevice") == 0) return (PFN_vkVoidFunction)destroy_device;
    if (strcmp(name, "vkCreateBuffer") == 0) return (PFN_vkVoidFunction)create_buffer;
    if (strcmp(name, "vkDestroyBuffer") == 0) return (PFN_vkVoidFunction)destroy_buffer;
    if (strcmp(name, "vkGetBufferMemoryRequirements") == 0) return (PFN_vkVoidFunction)buffer_reqs;
    if (strcmp(name, "vkAllocateMemory") == 0) return (PFN_vkVoidFunction)alloc_mem;
    if (strcmp(name, "vkFreeMemory") == 0) return (PFN_vkVoidFunction)free_mem;
    if (strcmp(name, "vkBindBufferMemory") == 0) return (PFN_vkVoidFunction)bind_buffer;
    if (strcmp(name, "vkMapMemory") == 0) return (PFN_vkVoidFunction)map_mem;
    if (strcmp(name, "vkUnmapMemory") == 0) return (PFN_vkVoidFunction)unmap_mem;
    if (strcmp(name, "vkCreateImage") == 0) return (PFN_vkVoidFunction)create_image;
    if (strcmp(name, "vkDestroyImage") == 0) return (PFN_vkVoidFunction)destroy_image;
    if (strcmp(name, "vkGetImageMemoryRequirements") == 0) return (PFN_vkVoidFunction)image_reqs;
    if (strcmp(name, "vkBindImageMemory") == 0) return (PFN_vkVoidFunction)bind_image;
    if (strcmp(name, "vkCreateImageView") == 0) return (PFN_vkVoidFunction)create_view;
    if (strcmp(name, "vkDestroyImageView") == 0) return (PFN_vkVoidFunction)destroy_view;
    if (strcmp(name, "vkCreateSampler") == 0) return (PFN_vkVoidFunction)create_sampler;
    if (strcmp(name, "vkDestroySampler") == 0) return (PFN_vkVoidFunction)destroy_sampler;
    if (strcmp(name, "vkCreateShaderModule") == 0) return (PFN_vkVoidFunction)create_shader;
    if (strcmp(name, "vkDestroyShaderModule") == 0) return (PFN_vkVoidFunction)destroy_shader;
    if (strcmp(name, "vkCreatePipelineLayout") == 0) return (PFN_vkVoidFunction)create_layout;
    if (strcmp(name, "vkDestroyPipelineLayout") == 0) return (PFN_vkVoidFunction)destroy_layout;
    if (strcmp(name, "vkCreateRenderPass") == 0) return (PFN_vkVoidFunction)create_renderpass;
    if (strcmp(name, "vkDestroyRenderPass") == 0) return (PFN_vkVoidFunction)destroy_renderpass;
    if (strcmp(name, "vkCreatePipelineCache") == 0) return (PFN_vkVoidFunction)create_pipeline_cache;
    if (strcmp(name, "vkDestroyPipelineCache") == 0) return (PFN_vkVoidFunction)destroy_pipeline_cache;
    if (strcmp(name, "vkGetPipelineCacheData") == 0) return (PFN_vkVoidFunction)get_pipeline_cache_data;
    if (strcmp(name, "vkMergePipelineCaches") == 0) return (PFN_vkVoidFunction)merge_pipeline_caches;
    if (strcmp(name, "vkCreateGraphicsPipelines") == 0) return (PFN_vkVoidFunction)create_pipelines;
    if (strcmp(name, "vkCreateComputePipelines") == 0) return (PFN_vkVoidFunction)create_compute_pipelines;
    if (strcmp(name, "vkCmdDispatch") == 0) return (PFN_vkVoidFunction)cmd_dispatch;
    if (strcmp(name, "vkCmdDispatchBase") == 0) return (PFN_vkVoidFunction)cmd_dispatch_base;
    if (strcmp(name, "vkCmdDispatchIndirect") == 0) return (PFN_vkVoidFunction)cmd_dispatch_indirect;
    if (strcmp(name, "vkDestroyPipeline") == 0) return (PFN_vkVoidFunction)destroy_pipeline;
    if (strcmp(name, "vkCreateFramebuffer") == 0) return (PFN_vkVoidFunction)create_fb;
    if (strcmp(name, "vkDestroyFramebuffer") == 0) return (PFN_vkVoidFunction)destroy_fb;
    if (strcmp(name, "vkCreateCommandPool") == 0) return (PFN_vkVoidFunction)create_pool;
    if (strcmp(name, "vkDestroyCommandPool") == 0) return (PFN_vkVoidFunction)destroy_pool;
    if (strcmp(name, "vkFreeCommandBuffers") == 0) return (PFN_vkVoidFunction)free_cmds;
    if (strcmp(name, "vkResetCommandBuffer") == 0) return (PFN_vkVoidFunction)reset_cmd;
    if (strcmp(name, "vkResetCommandPool") == 0) return (PFN_vkVoidFunction)reset_pool;
    if (strcmp(name, "vkAllocateCommandBuffers") == 0) return (PFN_vkVoidFunction)alloc_cmds;
    if (strcmp(name, "vkBeginCommandBuffer") == 0) return (PFN_vkVoidFunction)begin_cmd;
    if (strcmp(name, "vkCmdBeginRenderPass") == 0) return (PFN_vkVoidFunction)cmd_begin_rp;
    if (strcmp(name, "vkCmdEndRenderPass") == 0) return (PFN_vkVoidFunction)cmd_end_rp;
    if (strcmp(name, "vkCmdBindPipeline") == 0) return (PFN_vkVoidFunction)cmd_bind_pipe;
    if (strcmp(name, "vkCmdBindVertexBuffers") == 0) return (PFN_vkVoidFunction)cmd_bind_vbos;
    if (strcmp(name, "vkCmdBindDescriptorSets") == 0) return (PFN_vkVoidFunction)cmd_bind_sets;
    if (strcmp(name, "vkCmdPushConstants") == 0) return (PFN_vkVoidFunction)cmd_push_constants;
    if (strcmp(name, "vkCmdCopyBufferToImage") == 0) return (PFN_vkVoidFunction)cmd_copy_buffer_to_image;
    if (strcmp(name, "vkCmdSetViewport") == 0) return (PFN_vkVoidFunction)cmd_set_viewport;
    if (strcmp(name, "vkCmdSetScissor") == 0) return (PFN_vkVoidFunction)cmd_set_scissor;
    if (strcmp(name, "vkCmdSetBlendConstants") == 0) return (PFN_vkVoidFunction)cmd_set_blend_constants;
    if (strcmp(name, "vkCmdSetStencilCompareMask") == 0) return (PFN_vkVoidFunction)cmd_set_stencil_compare;
    if (strcmp(name, "vkCmdSetStencilWriteMask") == 0) return (PFN_vkVoidFunction)cmd_set_stencil_write;
    if (strcmp(name, "vkCmdSetStencilReference") == 0) return (PFN_vkVoidFunction)cmd_set_stencil_reference;
    if (strcmp(name, "vkCmdPipelineBarrier") == 0) return (PFN_vkVoidFunction)cmd_pipeline_barrier;
    if (strcmp(name, "vkCmdCopyImageToBuffer") == 0) return (PFN_vkVoidFunction)cmd_copy_image_to_buffer;
    if (strcmp(name, "vkCmdCopyBuffer") == 0) return (PFN_vkVoidFunction)cmd_copy_buffer;
    if (strcmp(name, "vkCmdDraw") == 0) return (PFN_vkVoidFunction)cmd_draw;
    if (strcmp(name, "vkCmdDrawIndexed") == 0) return (PFN_vkVoidFunction)cmd_draw_indexed;
    if (strcmp(name, "vkCmdBindIndexBuffer") == 0) return (PFN_vkVoidFunction)cmd_bind_index;
    if (strcmp(name, "vkEndCommandBuffer") == 0) return (PFN_vkVoidFunction)end_cmd;
    if (strcmp(name, "vkCreateFence") == 0) return (PFN_vkVoidFunction)create_fence;
    if (strcmp(name, "vkDestroyFence") == 0) return (PFN_vkVoidFunction)destroy_fence;
    if (strcmp(name, "vkResetFences") == 0) return (PFN_vkVoidFunction)reset_fences;
    if (strcmp(name, "vkCreateSemaphore") == 0) return (PFN_vkVoidFunction)create_semaphore;
    if (strcmp(name, "vkDestroySemaphore") == 0) return (PFN_vkVoidFunction)destroy_semaphore;
    if (strcmp(name, "vkQueueSubmit") == 0) return (PFN_vkVoidFunction)queue_submit;
    if (strcmp(name, "vkWaitForFences") == 0) return (PFN_vkVoidFunction)wait_fences;
    if (strcmp(name, "vkGetFenceStatus") == 0) return (PFN_vkVoidFunction)fence_status;
    if (strcmp(name, "vkQueueWaitIdle") == 0) return (PFN_vkVoidFunction)queue_wait;
    if (strcmp(name, "vkDeviceWaitIdle") == 0) return (PFN_vkVoidFunction)device_wait;
    if (strcmp(name, "vkCreateDescriptorSetLayout") == 0) return (PFN_vkVoidFunction)create_dsl;
    if (strcmp(name, "vkDestroyDescriptorSetLayout") == 0) return (PFN_vkVoidFunction)destroy_dsl;
    if (strcmp(name, "vkCreateDescriptorPool") == 0) return (PFN_vkVoidFunction)create_pool_ds;
    if (strcmp(name, "vkDestroyDescriptorPool") == 0) return (PFN_vkVoidFunction)destroy_pool_ds;
    if (strcmp(name, "vkAllocateDescriptorSets") == 0) return (PFN_vkVoidFunction)alloc_sets;
    if (strcmp(name, "vkFreeDescriptorSets") == 0) return (PFN_vkVoidFunction)free_sets;
    if (strcmp(name, "vkResetDescriptorPool") == 0) return (PFN_vkVoidFunction)reset_pool_ds;
    if (strcmp(name, "vkUpdateDescriptorSets") == 0) return (PFN_vkVoidFunction)update_sets;
    if (strcmp(name, "vkGetDeviceQueue2") == 0) return (PFN_vkVoidFunction)get_queue2;
    if (strcmp(name, "vkGetBufferMemoryRequirements2") == 0) return (PFN_vkVoidFunction)buffer_reqs2;
    if (strcmp(name, "vkGetImageMemoryRequirements2") == 0) return (PFN_vkVoidFunction)image_reqs2;
    if (strcmp(name, "vkGetImageSparseMemoryRequirements") == 0) return (PFN_vkVoidFunction)image_sparse_reqs;
    if (strcmp(name, "vkGetImageSparseMemoryRequirements2") == 0) return (PFN_vkVoidFunction)image_sparse_reqs2;
    if (strcmp(name, "vkBindBufferMemory2") == 0) return (PFN_vkVoidFunction)bind_buffer2;
    if (strcmp(name, "vkBindImageMemory2") == 0) return (PFN_vkVoidFunction)bind_image2;
    if (strcmp(name, "vkTrimCommandPool") == 0) return (PFN_vkVoidFunction)trim_pool;
    if (strcmp(name, "vkGetDescriptorSetLayoutSupport") == 0) return (PFN_vkVoidFunction)layout_support;
    if (strcmp(name, "vkCmdSetDeviceMask") == 0) return (PFN_vkVoidFunction)cmd_set_device_mask;
    if (strcmp(name, "vkGetDeviceGroupPeerMemoryFeatures") == 0) return (PFN_vkVoidFunction)peer_memory_features;
    if (strcmp(name, "vkCreateDescriptorUpdateTemplate") == 0) return (PFN_vkVoidFunction)create_update_template;
    if (strcmp(name, "vkDestroyDescriptorUpdateTemplate") == 0) return (PFN_vkVoidFunction)destroy_update_template;
    if (strcmp(name, "vkUpdateDescriptorSetWithTemplate") == 0) return (PFN_vkVoidFunction)update_set_with_template;
    if (strcmp(name, "vkCreateSamplerYcbcrConversion") == 0) return (PFN_vkVoidFunction)create_ycbcr_conversion;
    if (strcmp(name, "vkDestroySamplerYcbcrConversion") == 0) return (PFN_vkVoidFunction)destroy_ycbcr_conversion;
    return NULL;
}

static void device_features(VkPhysicalDevice physical, VkPhysicalDeviceFeatures *features)
{
    (void)physical;
    memset(features, 0, sizeof *features);
    features->robustBufferAccess = VK_TRUE;
    features->fullDrawIndexUint32 = VK_TRUE;
}

static void format_properties(VkPhysicalDevice physical, VkFormat format, VkFormatProperties *properties)
{
    (void)physical;
    memset(properties, 0, sizeof *properties);
    if (vk_color_format(format) != PIPE_FORMAT_NONE) {
        properties->linearTilingFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if (mxgpu_native_render_available(-1))
            properties->linearTilingFeatures |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
        if (mxgpu_native_sampler_available(-1))
            properties->linearTilingFeatures |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        properties->optimalTilingFeatures = properties->linearTilingFeatures;
    }
    uint32_t depth_format = vk_depth_format(format);
    if (depth_format && mxgpu_native_depth_available(-1, depth_format)) {
        properties->linearTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        properties->optimalTilingFeatures = properties->linearTilingFeatures;
    }
    if (vk_vertex_format(format) != PIPE_FORMAT_NONE)
        properties->bufferFeatures = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
}

static VkResult image_format_properties(VkPhysicalDevice physical, VkFormat format, VkImageType type,
                                       VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
                                       VkImageFormatProperties *properties)
{
    (void)physical;
    memset(properties, 0, sizeof *properties);
    uint32_t depth_format = vk_depth_format(format);
    VkImageUsageFlags supported_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vk_color_format(format) != PIPE_FORMAT_NONE)
        supported_usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    else if (depth_format && mxgpu_native_depth_available(-1, depth_format))
        supported_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    else
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    if (type != VK_IMAGE_TYPE_2D ||
        (tiling != VK_IMAGE_TILING_LINEAR && tiling != VK_IMAGE_TILING_OPTIMAL) || flags ||
        (usage & ~supported_usage))
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    properties->maxExtent = (VkExtent3D){MXGPU_VK_MAX_IMAGE_DIMENSION, MXGPU_VK_MAX_IMAGE_DIMENSION, 1};
    properties->maxMipLevels = 12;
    properties->maxArrayLayers = 1;
    properties->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
    properties->maxResourceSize = (VkDeviceSize)MXGPU_VK_MAX_IMAGE_DIMENSION * MXGPU_VK_MAX_IMAGE_DIMENSION *
                                  vk_format_bytes(format) * 2;
    return VK_SUCCESS;
}

static void sparse_format_properties(VkPhysicalDevice physical, VkFormat format, VkImageType type,
                                     VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageTiling tiling,
                                     uint32_t *count, VkSparseImageFormatProperties *properties)
{
    (void)physical;
    (void)format;
    (void)type;
    (void)samples;
    (void)usage;
    (void)tiling;
    (void)properties;
    *count = 0;
}

static size_t feature_struct_size(VkStructureType type)
{
    switch (type) {
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: return sizeof(VkPhysicalDeviceVulkan11Features);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: return sizeof(VkPhysicalDevice16BitStorageFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES: return sizeof(VkPhysicalDeviceMultiviewFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES: return sizeof(VkPhysicalDeviceVariablePointersFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES: return sizeof(VkPhysicalDeviceProtectedMemoryFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES:
        return sizeof(VkPhysicalDeviceSamplerYcbcrConversionFeatures);
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES:
        return sizeof(VkPhysicalDeviceShaderDrawParametersFeatures);
    default: return 0;
    }
}

static void device_features2(VkPhysicalDevice physical, VkPhysicalDeviceFeatures2 *features)
{
    device_features(physical, &features->features);
    for (VkBaseOutStructure *next = features->pNext; next; next = next->pNext) {
        size_t size = feature_struct_size(next->sType);
        if (size)
            memset((uint8_t *)next + sizeof *next, 0, size - sizeof *next);
    }
}

static bool features_supported(VkPhysicalDevice physical, const VkPhysicalDeviceFeatures *requested)
{
    VkPhysicalDeviceFeatures supported;
    const VkBool32 *want = (const VkBool32 *)requested, *have = (const VkBool32 *)&supported;
    device_features(physical, &supported);
    for (size_t i = 0; i < sizeof supported / sizeof(VkBool32); i++)
        if (want[i] && !have[i])
            return false;
    return true;
}

static VkResult device_create_info_supported(VkPhysicalDevice physical, const VkDeviceCreateInfo *info)
{
    if (info->pEnabledFeatures && !features_supported(physical, info->pEnabledFeatures))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    for (uint32_t i = 0; i < info->queueCreateInfoCount; i++)
        if (info->pQueueCreateInfos[i].flags & VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    for (const VkBaseInStructure *next = info->pNext; next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            if (!features_supported(physical, &((const VkPhysicalDeviceFeatures2 *)next)->features))
                return VK_ERROR_FEATURE_NOT_PRESENT;
            continue;
        }
        if (next->sType == VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO) {
            const VkDeviceGroupDeviceCreateInfo *group = (const VkDeviceGroupDeviceCreateInfo *)next;
            if (group->physicalDeviceCount > 1 ||
                (group->physicalDeviceCount && group->pPhysicalDevices[0] != physical))
                return VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }
        size_t size = feature_struct_size(next->sType);
        const VkBool32 *flags = (const VkBool32 *)((const uint8_t *)next + sizeof *next);
        for (size_t i = 0; size && i < (size - sizeof *next) / sizeof(VkBool32); i++)
            if (flags[i])
                return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    return VK_SUCCESS;
}

static void fill_vulkan11_properties(VkPhysicalDeviceVulkan11Properties *p)
{
    memcpy(p->deviceUUID, mx_device_uuid, VK_UUID_SIZE);
    memcpy(p->driverUUID, mx_driver_uuid, VK_UUID_SIZE);
    memset(p->deviceLUID, 0, VK_LUID_SIZE);
    p->deviceNodeMask = 0;
    p->deviceLUIDValid = VK_FALSE;
    p->subgroupSize = 1;
    p->subgroupSupportedStages = 0;
    p->subgroupSupportedOperations = 0;
    p->subgroupQuadOperationsInAllStages = VK_FALSE;
    p->pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES;
    p->maxMultiviewViewCount = 0;
    p->maxMultiviewInstanceIndex = 0;
    p->protectedNoFault = VK_FALSE;
    p->maxPerSetDescriptors = MXGPU_VK_MAX_PER_SET_DESCRIPTORS;
    p->maxMemoryAllocationSize = MXGPU_VK_HEAP_BYTES;
}

static void device_props2(VkPhysicalDevice physical, VkPhysicalDeviceProperties2 *properties)
{
    VkPhysicalDeviceVulkan11Properties core = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    fill_vulkan11_properties(&core);
    device_props(physical, &properties->properties);
    for (VkBaseOutStructure *next = properties->pNext; next; next = next->pNext) {
        switch (next->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES:
            fill_vulkan11_properties((VkPhysicalDeviceVulkan11Properties *)next);
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: {
            VkPhysicalDeviceIDProperties *p = (VkPhysicalDeviceIDProperties *)next;
            memcpy(p->deviceUUID, core.deviceUUID, VK_UUID_SIZE);
            memcpy(p->driverUUID, core.driverUUID, VK_UUID_SIZE);
            memcpy(p->deviceLUID, core.deviceLUID, VK_LUID_SIZE);
            p->deviceNodeMask = core.deviceNodeMask;
            p->deviceLUIDValid = core.deviceLUIDValid;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES: {
            VkPhysicalDeviceMaintenance3Properties *p = (VkPhysicalDeviceMaintenance3Properties *)next;
            p->maxPerSetDescriptors = core.maxPerSetDescriptors;
            p->maxMemoryAllocationSize = core.maxMemoryAllocationSize;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES: {
            VkPhysicalDeviceMultiviewProperties *p = (VkPhysicalDeviceMultiviewProperties *)next;
            p->maxMultiviewViewCount = core.maxMultiviewViewCount;
            p->maxMultiviewInstanceIndex = core.maxMultiviewInstanceIndex;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES:
            ((VkPhysicalDevicePointClippingProperties *)next)->pointClippingBehavior = core.pointClippingBehavior;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES:
            ((VkPhysicalDeviceProtectedMemoryProperties *)next)->protectedNoFault = core.protectedNoFault;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: {
            VkPhysicalDeviceSubgroupProperties *p = (VkPhysicalDeviceSubgroupProperties *)next;
            p->subgroupSize = core.subgroupSize;
            p->supportedStages = core.subgroupSupportedStages;
            p->supportedOperations = core.subgroupSupportedOperations;
            p->quadOperationsInAllStages = core.subgroupQuadOperationsInAllStages;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: {
            VkPhysicalDeviceDriverProperties *p = (VkPhysicalDeviceDriverProperties *)next;
            p->driverID = MXGPU_VK_DRIVER_ID;
            snprintf(p->driverName, sizeof p->driverName, "MXGPU");
            snprintf(p->driverInfo, sizeof p->driverInfo, "MXGPU Vulkan %s", MXGPU_DRIVER_VERSION);
            p->conformanceVersion = (VkConformanceVersion){0, 0, 0, 0};
            break;
        }
        default:
            break;
        }
    }
}

static void queue_props2(VkPhysicalDevice gpu, uint32_t *count, VkQueueFamilyProperties2 *props)
{
    if (!props) {
        queue_props(gpu, count, NULL);
        return;
    }
    if (!*count)
        return;
    queue_props(gpu, count, &props[0].queueFamilyProperties);
}

static void mem_props2(VkPhysicalDevice gpu, VkPhysicalDeviceMemoryProperties2 *props)
{
    mem_props(gpu, &props->memoryProperties);
}

static void format_properties2(VkPhysicalDevice physical, VkFormat format, VkFormatProperties2 *properties)
{
    format_properties(physical, format, &properties->formatProperties);
}

static VkResult image_format_properties2(VkPhysicalDevice physical, const VkPhysicalDeviceImageFormatInfo2 *info,
                                        VkImageFormatProperties2 *properties)
{
    bool external = false;
    for (const VkBaseInStructure *next = info->pNext; next; next = next->pNext)
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO &&
            ((const VkPhysicalDeviceExternalImageFormatInfo *)next)->handleType)
            external = true;
    VkResult result = image_format_properties(physical, info->format, info->type, info->tiling, info->usage,
                                              info->flags, &properties->imageFormatProperties);
    for (VkBaseOutStructure *next = properties->pNext; next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES)
            memset(&((VkExternalImageFormatProperties *)next)->externalMemoryProperties, 0,
                   sizeof(VkExternalMemoryProperties));
        else if (next->sType == VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES)
            ((VkSamplerYcbcrConversionImageFormatProperties *)next)->combinedImageSamplerDescriptorCount = 1;
    }
    if (result == VK_SUCCESS && external) {
        memset(&properties->imageFormatProperties, 0, sizeof properties->imageFormatProperties);
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    return result;
}

static void sparse_format_properties2(VkPhysicalDevice physical, const VkPhysicalDeviceSparseImageFormatInfo2 *info,
                                      uint32_t *count, VkSparseImageFormatProperties2 *properties)
{
    (void)physical;
    (void)info;
    (void)properties;
    *count = 0;
}

static void external_buffer_properties(VkPhysicalDevice physical, const VkPhysicalDeviceExternalBufferInfo *info,
                                       VkExternalBufferProperties *properties)
{
    (void)physical;
    (void)info;
    memset(&properties->externalMemoryProperties, 0, sizeof properties->externalMemoryProperties);
}

static void external_fence_properties(VkPhysicalDevice physical, const VkPhysicalDeviceExternalFenceInfo *info,
                                      VkExternalFenceProperties *properties)
{
    (void)physical;
    (void)info;
    properties->exportFromImportedHandleTypes = 0;
    properties->compatibleHandleTypes = 0;
    properties->externalFenceFeatures = 0;
}

static void external_semaphore_properties(VkPhysicalDevice physical, const VkPhysicalDeviceExternalSemaphoreInfo *info,
                                          VkExternalSemaphoreProperties *properties)
{
    (void)physical;
    (void)info;
    properties->exportFromImportedHandleTypes = 0;
    properties->compatibleHandleTypes = 0;
    properties->externalSemaphoreFeatures = 0;
}

static VkResult enum_device_groups(VkInstance instance, uint32_t *count, VkPhysicalDeviceGroupProperties *groups)
{
    struct mx_instance *owner = (struct mx_instance *)instance;
    uint32_t available = mxgpu_device_available() ? 1 : 0;
    if (!count)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!groups) {
        *count = available;
        return VK_SUCCESS;
    }
    uint32_t written = *count < available ? *count : available;
    for (uint32_t i = 0; i < written; i++) {
        groups[i].physicalDeviceCount = 1;
        memset(groups[i].physicalDevices, 0, sizeof groups[i].physicalDevices);
        groups[i].physicalDevices[0] = (VkPhysicalDevice)&owner->physical;
        groups[i].subsetAllocation = VK_FALSE;
    }
    *count = written;
    return written < available ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL enumerate_instance_version(uint32_t *version)
{
    *version = MXGPU_VK_API_VERSION;
    return VK_SUCCESS;
}

static VkResult instance_extensions(const char *layer, uint32_t *count, VkExtensionProperties *properties)
{
    if (layer)
        return VK_ERROR_LAYER_NOT_PRESENT;
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    static const VkExtensionProperties extensions[] = {
        {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_SURFACE_SPEC_VERSION},
        {VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME, VK_KHR_WAYLAND_SURFACE_SPEC_VERSION},
    };
    if (!properties) {
        *count = 2;
        return VK_SUCCESS;
    }
    uint32_t written = *count < 2 ? *count : 2;
    memcpy(properties, extensions, written * sizeof *properties);
    *count = written;
    return written < 2 ? VK_INCOMPLETE : VK_SUCCESS;
#else
    (void)properties;
    *count = 0;
    return VK_SUCCESS;
#endif
}

static const VkExtensionProperties mx_device_extensions[] = {
    {VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, VK_KHR_DRIVER_PROPERTIES_SPEC_VERSION},
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_SWAPCHAIN_SPEC_VERSION},
#endif
};

static VkResult device_extensions(VkPhysicalDevice physical, const char *layer, uint32_t *count, VkExtensionProperties *properties)
{
    const uint32_t total = sizeof mx_device_extensions / sizeof mx_device_extensions[0];
    (void)physical;
    if (layer)
        return VK_ERROR_LAYER_NOT_PRESENT;
    if (!properties) {
        *count = total;
        return VK_SUCCESS;
    }
    uint32_t written = *count < total ? *count : total;
    memcpy(properties, mx_device_extensions, written * sizeof *properties);
    *count = written;
    return written < total ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice device, const char *name)
{
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    if (name && wsi_device_proc(name) &&
        (!device || !((struct mx_device *)device)->swapchain_enabled))
        return NULL;
#else
    (void)device;
#endif
    return name ? device_proc(name) : NULL;
}

static PFN_vkVoidFunction instance_proc(const char *name)
{
    PFN_vkVoidFunction dev;
    if (!name)
        return NULL;
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    PFN_vkVoidFunction wsi = wsi_instance_proc(name);
    if (wsi)
        return wsi;
#endif
    if (strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0) return (PFN_vkVoidFunction)instance_extensions;
    if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) return (PFN_vkVoidFunction)device_extensions;
    if (strcmp(name, "vkGetPhysicalDeviceFeatures") == 0) return (PFN_vkVoidFunction)device_features;
    if (strcmp(name, "vkGetPhysicalDeviceFormatProperties") == 0) return (PFN_vkVoidFunction)format_properties;
    if (strcmp(name, "vkGetPhysicalDeviceImageFormatProperties") == 0) return (PFN_vkVoidFunction)image_format_properties;
    if (strcmp(name, "vkGetPhysicalDeviceSparseImageFormatProperties") == 0) return (PFN_vkVoidFunction)sparse_format_properties;
    if (strcmp(name, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)create_instance;
    if (strcmp(name, "vkDestroyInstance") == 0) return (PFN_vkVoidFunction)destroy_instance;
    if (strcmp(name, "vkEnumeratePhysicalDevices") == 0) return (PFN_vkVoidFunction)enum_devices;
    if (strcmp(name, "vkGetPhysicalDeviceProperties") == 0) return (PFN_vkVoidFunction)device_props;
    if (strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties") == 0) return (PFN_vkVoidFunction)queue_props;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties") == 0) return (PFN_vkVoidFunction)mem_props;
    if (strcmp(name, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)create_device;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)get_device_proc;
    if (strcmp(name, "vkEnumerateInstanceVersion") == 0) return (PFN_vkVoidFunction)enumerate_instance_version;
    if (strcmp(name, "vkEnumeratePhysicalDeviceGroups") == 0) return (PFN_vkVoidFunction)enum_device_groups;
    if (strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0) return (PFN_vkVoidFunction)device_features2;
    if (strcmp(name, "vkGetPhysicalDeviceProperties2") == 0) return (PFN_vkVoidFunction)device_props2;
    if (strcmp(name, "vkGetPhysicalDeviceFormatProperties2") == 0) return (PFN_vkVoidFunction)format_properties2;
    if (strcmp(name, "vkGetPhysicalDeviceImageFormatProperties2") == 0) return (PFN_vkVoidFunction)image_format_properties2;
    if (strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties2") == 0) return (PFN_vkVoidFunction)queue_props2;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0) return (PFN_vkVoidFunction)mem_props2;
    if (strcmp(name, "vkGetPhysicalDeviceSparseImageFormatProperties2") == 0)
        return (PFN_vkVoidFunction)sparse_format_properties2;
    if (strcmp(name, "vkGetPhysicalDeviceExternalBufferProperties") == 0)
        return (PFN_vkVoidFunction)external_buffer_properties;
    if (strcmp(name, "vkGetPhysicalDeviceExternalFenceProperties") == 0)
        return (PFN_vkVoidFunction)external_fence_properties;
    if (strcmp(name, "vkGetPhysicalDeviceExternalSemaphoreProperties") == 0)
        return (PFN_vkVoidFunction)external_semaphore_properties;
    dev = device_proc(name);
    return dev;
}

static PFN_vkVoidFunction enabled_instance_proc(VkInstance instance, const char *name)
{
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    if (name && wsi_instance_proc(name)) {
        struct mx_instance *owner = (void *)instance;
        bool wayland = !strcmp(name, "vkCreateWaylandSurfaceKHR") ||
            !strcmp(name, "vkGetPhysicalDeviceWaylandPresentationSupportKHR");
        if (!owner || !owner->surface_enabled || (wayland && !owner->wayland_enabled))
            return NULL;
    }
#else
    (void)instance;
#endif
    return instance_proc(name);
}

MX_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name)
{
    if (name && strcmp(name, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    return enabled_instance_proc(instance, name);
}

MX_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetInstanceProcAddr(VkInstance instance, const char *name)
{
    if (name && strcmp(name, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)vk_icdGetInstanceProcAddr;
    return enabled_instance_proc(instance, name);
}

MX_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char *name)
{
    (void)instance;
    return enabled_instance_proc(instance, name);
}

MX_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t *version)
{
    if (!version)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (*version > 5)
        *version = 5;
    return VK_SUCCESS;
}
