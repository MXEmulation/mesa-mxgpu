/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */

#include <stddef.h>
#include <wayland-client.h>
#include <pthread.h>
#include <poll.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <unistd.h>
#include <sys/mman.h>

struct mx_swapchain;

struct mx_surface {
    VkIcdSurfaceWayland base;
    atomic_uint references;
    VkAllocationCallbacks allocator;
    bool custom_allocator;
    struct wl_display *display;
    struct wl_surface *window;
    struct wl_proxy *window_wrapper, *display_wrapper;
    struct wl_event_queue *events;
    struct wl_registry *registry;
    struct wl_shm *shm;
    struct wl_callback *frame;
    struct mx_swapchain *current;
    pthread_mutex_t lock;
    VkResult status;
    bool argb, xrgb;
};

struct mx_present_image {
    struct mx_img image;
    struct mx_mem memory;
    struct wl_buffer *buffer;
    void *mapping;
    size_t bytes;
    bool acquired, busy;
};

struct mx_swapchain {
    struct mx_device *device;
    struct mx_surface *surface;
    VkAllocationCallbacks allocator;
    bool custom_allocator, retired;
    struct mx_present_image *images;
    uint32_t count, cursor;
    VkPresentModeKHR mode;
};

static void *wsi_alloc(const VkAllocationCallbacks *allocator, size_t bytes)
{
    void *result = allocator ? allocator->pfnAllocation(allocator->pUserData, bytes,
        _Alignof(max_align_t), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) : malloc(bytes);
    if (result)
        memset(result, 0, bytes);
    return result;
}

static void wsi_free(const VkAllocationCallbacks *allocator, void *memory)
{
    if (!memory)
        return;
    if (allocator)
        allocator->pfnFree(allocator->pUserData, memory);
    else
        free(memory);
}

static uint64_t wsi_now(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (uint64_t)time.tv_sec * UINT64_C(1000000000) + time.tv_nsec;
}

static uint64_t wsi_deadline(uint64_t timeout)
{
    uint64_t now = wsi_now();
    return timeout > UINT64_MAX - now ? UINT64_MAX : now + timeout;
}

static VkResult wsi_events(struct mx_surface *surface, uint64_t deadline)
{
    int dispatched = wl_display_dispatch_queue_pending(surface->display, surface->events);
    if (dispatched < 0)
        return surface->status = VK_ERROR_SURFACE_LOST_KHR;
    if (dispatched)
        return VK_SUCCESS;
    while (wl_display_prepare_read_queue(surface->display, surface->events)) {
        dispatched = wl_display_dispatch_queue_pending(surface->display, surface->events);
        if (dispatched < 0 || wl_display_get_error(surface->display))
            return surface->status = VK_ERROR_SURFACE_LOST_KHR;
        if (dispatched)
            return VK_SUCCESS;
    }
    int flushed = wl_display_flush(surface->display);
    if (flushed < 0 && errno != EAGAIN) {
        wl_display_cancel_read(surface->display);
        return surface->status = VK_ERROR_SURFACE_LOST_KHR;
    }
    for (;;) {
        uint64_t now = wsi_now();
        uint64_t remaining = deadline > now ? deadline - now : 0;
        int milliseconds = deadline == UINT64_MAX ? -1 :
            remaining / 1000000 >= INT_MAX ? INT_MAX : (int)((remaining + 999999) / 1000000);
        struct pollfd fd = {.fd = wl_display_get_fd(surface->display),
            .events = POLLIN | (flushed < 0 ? POLLOUT : 0)};
        int ready = poll(&fd, 1, milliseconds);
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready <= 0) {
            wl_display_cancel_read(surface->display);
            return ready == 0 ? VK_TIMEOUT : (surface->status = VK_ERROR_SURFACE_LOST_KHR);
        }
        if (fd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            wl_display_cancel_read(surface->display);
            return surface->status = VK_ERROR_SURFACE_LOST_KHR;
        }
        if (fd.revents & POLLOUT) {
            flushed = wl_display_flush(surface->display);
            if (flushed < 0 && errno != EAGAIN) {
                wl_display_cancel_read(surface->display);
                return surface->status = VK_ERROR_SURFACE_LOST_KHR;
            }
        }
        if (fd.revents & POLLIN) {
            if (wl_display_read_events(surface->display) < 0 ||
                wl_display_dispatch_queue_pending(surface->display, surface->events) < 0)
                return surface->status = VK_ERROR_SURFACE_LOST_KHR;
            return VK_SUCCESS;
        }
        if (deadline != UINT64_MAX && wsi_now() >= deadline) {
            wl_display_cancel_read(surface->display);
            return VK_TIMEOUT;
        }
    }
}

static void wsi_sync_done(void *data, struct wl_callback *callback, uint32_t serial)
{
    (void)callback;
    (void)serial;
    *(bool *)data = true;
}

static const struct wl_callback_listener wsi_sync_listener = {wsi_sync_done};

static VkResult wsi_roundtrip(struct mx_surface *surface)
{
    bool done = false;
    struct wl_callback *callback = wl_display_sync((struct wl_display *)surface->display_wrapper);
    if (!callback)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    wl_callback_add_listener(callback, &wsi_sync_listener, &done);
    uint64_t deadline = wsi_deadline(UINT64_C(5000000000));
    VkResult result = VK_SUCCESS;
    while (!done && result == VK_SUCCESS)
        result = wsi_events(surface, deadline);
    wl_callback_destroy(callback);
    return result;
}

static void wsi_shm_format(void *data, struct wl_shm *shm, uint32_t format)
{
    (void)shm;
    struct mx_surface *surface = data;
    surface->argb |= format == WL_SHM_FORMAT_ARGB8888;
    surface->xrgb |= format == WL_SHM_FORMAT_XRGB8888;
}

static const struct wl_shm_listener wsi_shm_listener = {wsi_shm_format};

static void wsi_global(void *data, struct wl_registry *registry, uint32_t name,
                        const char *interface, uint32_t version)
{
    (void)version;
    struct mx_surface *surface = data;
    if (!strcmp(interface, wl_shm_interface.name) && !surface->shm) {
        surface->shm = wl_registry_bind(registry, name, &wl_shm_interface, 1);
        if (surface->shm)
            wl_shm_add_listener(surface->shm, &wsi_shm_listener, surface);
        else
            surface->status = VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

static void wsi_global_remove(void *data, struct wl_registry *registry, uint32_t name)
{
    (void)data;
    (void)registry;
    (void)name;
}

static const struct wl_registry_listener wsi_registry_listener = {wsi_global, wsi_global_remove};

static void wsi_release_surface(struct mx_surface *surface)
{
    if (!surface || atomic_fetch_sub_explicit(&surface->references, 1, memory_order_acq_rel) != 1)
        return;
    if (surface->frame)
        wl_callback_destroy(surface->frame);
    if (surface->shm)
        wl_shm_destroy(surface->shm);
    if (surface->registry)
        wl_registry_destroy(surface->registry);
    if (surface->window_wrapper)
        wl_proxy_wrapper_destroy(surface->window_wrapper);
    if (surface->display_wrapper)
        wl_proxy_wrapper_destroy(surface->display_wrapper);
    if (surface->events)
        wl_event_queue_destroy(surface->events);
    pthread_mutex_destroy(&surface->lock);
    wsi_free(surface->custom_allocator ? &surface->allocator : NULL, surface);
}

static VkResult create_wayland_surface(VkInstance instance, const VkWaylandSurfaceCreateInfoKHR *info,
    const VkAllocationCallbacks *allocator, VkSurfaceKHR *out)
{
    struct mx_instance *owner = (void *)instance;
    *out = VK_NULL_HANDLE;
    if (!owner || !owner->wayland_enabled || !info || info->flags || !info->display || !info->surface)
        return VK_ERROR_INITIALIZATION_FAILED;
    struct mx_surface *surface = wsi_alloc(allocator, sizeof *surface);
    if (!surface)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    atomic_init(&surface->references, 1);
    if (allocator) {
        surface->allocator = *allocator;
        surface->custom_allocator = true;
    }
    if (pthread_mutex_init(&surface->lock, NULL)) {
        wsi_free(allocator, surface);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    surface->display = info->display;
    surface->window = info->surface;
    surface->base = (VkIcdSurfaceWayland){
        .base = {.platform = VK_ICD_WSI_PLATFORM_WAYLAND},
        .display = info->display, .surface = info->surface,
    };
    surface->events = wl_display_create_queue(surface->display);
    surface->window_wrapper = wl_proxy_create_wrapper(surface->window);
    surface->display_wrapper = wl_proxy_create_wrapper(surface->display);
    VkResult result = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (!surface->events || !surface->window_wrapper || !surface->display_wrapper)
        goto fail;
    wl_proxy_set_queue(surface->window_wrapper, surface->events);
    wl_proxy_set_queue(surface->display_wrapper, surface->events);
    surface->registry = wl_display_get_registry((struct wl_display *)surface->display_wrapper);
    if (!surface->registry)
        goto fail;
    wl_registry_add_listener(surface->registry, &wsi_registry_listener, surface);
    result = wsi_roundtrip(surface);
    if (result == VK_SUCCESS && surface->status != VK_SUCCESS)
        result = surface->status;
    if (result == VK_SUCCESS && surface->shm)
        result = wsi_roundtrip(surface);
    if (result != VK_SUCCESS || !surface->shm || !surface->argb || !surface->xrgb) {
        if (result == VK_SUCCESS)
            result = VK_ERROR_SURFACE_LOST_KHR;
        goto fail;
    }
    *out = (VkSurfaceKHR)surface;
    return VK_SUCCESS;
fail:
    wsi_release_surface(surface);
    return result == VK_TIMEOUT ? VK_ERROR_SURFACE_LOST_KHR : result;
}

static void destroy_surface(VkInstance instance, VkSurfaceKHR handle, const VkAllocationCallbacks *allocator)
{
    (void)instance;
    (void)allocator;
    wsi_release_surface((void *)handle);
}

static VkBool32 wayland_presentation_support(VkPhysicalDevice physical, uint32_t family, struct wl_display *display)
{
    (void)physical;
    return family == 0 && display && !wl_display_get_error(display) && mxgpu_device_available();
}

static VkResult surface_support(VkPhysicalDevice physical, uint32_t family, VkSurfaceKHR handle, VkBool32 *supported)
{
    struct mx_surface *surface = (void *)handle;
    *supported = false;
    if (!surface || surface->status != VK_SUCCESS || wl_display_get_error(surface->display))
        return VK_ERROR_SURFACE_LOST_KHR;
    *supported = wayland_presentation_support(physical, family, surface->display);
    return VK_SUCCESS;
}

static VkResult surface_capabilities(VkPhysicalDevice physical, VkSurfaceKHR handle, VkSurfaceCapabilitiesKHR *caps)
{
    VkBool32 supported;
    VkResult result = surface_support(physical, 0, handle, &supported);
    if (result != VK_SUCCESS)
        return result;
    struct mx_surface *surface = (void *)handle;
    VkPhysicalDeviceProperties properties;
    device_props(physical, &properties);
    *caps = (VkSurfaceCapabilitiesKHR){
        .minImageCount = 2, .maxImageCount = 0,
        .currentExtent = {UINT32_MAX, UINT32_MAX},
        .minImageExtent = {1, 1},
        .maxImageExtent = {properties.limits.maxImageDimension2D, properties.limits.maxImageDimension2D},
        .maxImageArrayLayers = 1,
        .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .supportedCompositeAlpha = (surface->xrgb ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : 0) |
                                  (surface->argb ? VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR : 0),
        .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    };
    return VK_SUCCESS;
}

static VkResult surface_formats(VkPhysicalDevice physical, VkSurfaceKHR handle, uint32_t *count, VkSurfaceFormatKHR *formats)
{
    VkBool32 supported;
    VkResult result = surface_support(physical, 0, handle, &supported);
    if (result != VK_SUCCESS)
        return result;
    if (!formats) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (!*count)
        return VK_INCOMPLETE;
    formats[0] = (VkSurfaceFormatKHR){VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    *count = 1;
    return VK_SUCCESS;
}

static VkResult surface_present_modes(VkPhysicalDevice physical, VkSurfaceKHR handle, uint32_t *count, VkPresentModeKHR *modes)
{
    VkBool32 supported;
    VkResult result = surface_support(physical, 0, handle, &supported);
    if (result != VK_SUCCESS)
        return result;
    static const VkPresentModeKHR available[] = {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR};
    if (!modes) {
        *count = 2;
        return VK_SUCCESS;
    }
    uint32_t written = *count < 2 ? *count : 2;
    memcpy(modes, available, written * sizeof *modes);
    *count = written;
    return written < 2 ? VK_INCOMPLETE : VK_SUCCESS;
}

static void wsi_buffer_release(void *data, struct wl_buffer *buffer)
{
    (void)buffer;
    ((struct mx_present_image *)data)->busy = false;
}

static const struct wl_buffer_listener wsi_buffer_listener = {wsi_buffer_release};

static void destroy_swapchain(VkDevice device, VkSwapchainKHR handle, const VkAllocationCallbacks *allocator)
{
    (void)device;
    (void)allocator;
    struct mx_swapchain *chain = (void *)handle;
    if (!chain)
        return;
    struct mx_surface *surface = chain->surface;
    pthread_mutex_lock(&surface->lock);
    if (surface->current == chain)
        surface->current = NULL;
    for (uint32_t i = 0; i < chain->count; i++) {
        struct mx_present_image *image = &chain->images[i];
        if (image->buffer)
            wl_buffer_destroy(image->buffer);
        if (image->mapping)
            munmap(image->mapping, image->bytes);
    }
    pthread_mutex_unlock(&surface->lock);
    const VkAllocationCallbacks *a = chain->custom_allocator ? &chain->allocator : NULL;
    wsi_free(a, chain->images);
    wsi_release_surface(surface);
    wsi_free(a, chain);
}

static VkResult create_swapchain(VkDevice device, const VkSwapchainCreateInfoKHR *info,
    const VkAllocationCallbacks *allocator, VkSwapchainKHR *out)
{
    struct mx_device *owner = (void *)device;
    *out = VK_NULL_HANDLE;
    if (!owner || !owner->swapchain_enabled || !info)
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    struct mx_surface *surface = (void *)info->surface;
    if (!surface)
        return VK_ERROR_SURFACE_LOST_KHR;
    VkSurfaceCapabilitiesKHR caps;
    VkResult result = surface_capabilities(VK_NULL_HANDLE, info->surface, &caps);
    if (result != VK_SUCCESS)
        return result;
    if (info->flags || info->minImageCount < caps.minImageCount || info->imageArrayLayers != 1 ||
        info->imageFormat != VK_FORMAT_B8G8R8A8_UNORM || info->imageColorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
        !info->imageExtent.width || !info->imageExtent.height ||
        info->imageExtent.width > caps.maxImageExtent.width || info->imageExtent.height > caps.maxImageExtent.height ||
        (info->imageUsage & ~caps.supportedUsageFlags) ||
        info->preTransform != VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR ||
        !(info->compositeAlpha & caps.supportedCompositeAlpha) ||
        (info->compositeAlpha & (info->compositeAlpha - 1)) ||
        (info->presentMode != VK_PRESENT_MODE_FIFO_KHR && info->presentMode != VK_PRESENT_MODE_MAILBOX_KHR) ||
        info->imageSharingMode != VK_SHARING_MODE_EXCLUSIVE)
        return VK_ERROR_INITIALIZATION_FAILED;
    struct mx_swapchain *old = (void *)info->oldSwapchain;
    pthread_mutex_lock(&surface->lock);
    if ((old && (old->surface != surface || old->device != owner || old->retired)) ||
        (surface->current && surface->current != old)) {
        pthread_mutex_unlock(&surface->lock);
        return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
    }
    if (old)
        old->retired = true;
    struct mx_swapchain *chain = wsi_alloc(allocator, sizeof *chain);
    if (!chain) {
        pthread_mutex_unlock(&surface->lock);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (allocator) {
        chain->allocator = *allocator;
        chain->custom_allocator = true;
    }
    chain->device = owner;
    chain->surface = surface;
    chain->mode = info->presentMode;
    atomic_fetch_add_explicit(&surface->references, 1, memory_order_relaxed);
    if (info->minImageCount > SIZE_MAX / sizeof *chain->images) {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto fail;
    }
    chain->images = wsi_alloc(allocator, (size_t)info->minImageCount * sizeof *chain->images);
    if (!chain->images) {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto fail;
    }
    chain->count = info->minImageCount;
    uint32_t stride = info->imageExtent.width * 4;
    size_t bytes = (size_t)stride * info->imageExtent.height;
    uint32_t format = info->compositeAlpha == VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR ?
        WL_SHM_FORMAT_XRGB8888 : WL_SHM_FORMAT_ARGB8888;
    for (uint32_t i = 0; i < chain->count; i++) {
        struct mx_present_image *image = &chain->images[i];
        int fd = memfd_create("mxgpu-present", MFD_CLOEXEC);
        if (fd < 0) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
        }
        if (ftruncate(fd, bytes)) {
            close(fd);
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
        }
        image->mapping = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (image->mapping == MAP_FAILED) {
            image->mapping = NULL;
            close(fd);
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
        }
        image->bytes = bytes;
        struct wl_shm_pool *pool = wl_shm_create_pool(surface->shm, fd, bytes);
        close(fd);
        if (!pool) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
        }
        image->buffer = wl_shm_pool_create_buffer(pool, 0, info->imageExtent.width,
            info->imageExtent.height, stride, format);
        wl_shm_pool_destroy(pool);
        if (!image->buffer) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto fail;
        }
        wl_buffer_add_listener(image->buffer, &wsi_buffer_listener, image);
        image->memory = (struct mx_mem){.size = bytes, .ptr = image->mapping};
        image->image = (struct mx_img){.width = info->imageExtent.width, .height = info->imageExtent.height,
            .mem = &image->memory, .size = bytes, .depth = 1, .levels = 1, .layers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .format = info->imageFormat};
    }
    surface->current = chain;
    pthread_mutex_unlock(&surface->lock);
    *out = (VkSwapchainKHR)chain;
    return VK_SUCCESS;
fail:
    pthread_mutex_unlock(&surface->lock);
    destroy_swapchain(device, (VkSwapchainKHR)chain, allocator);
    return result;
}

static VkResult swapchain_images(VkDevice device, VkSwapchainKHR handle, uint32_t *count, VkImage *images)
{
    struct mx_swapchain *chain = (void *)handle;
    if (!chain || chain->device != (void *)device)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!images) {
        *count = chain->count;
        return VK_SUCCESS;
    }
    uint32_t written = *count < chain->count ? *count : chain->count;
    for (uint32_t i = 0; i < written; i++)
        images[i] = (VkImage)&chain->images[i].image;
    *count = written;
    return written < chain->count ? VK_INCOMPLETE : VK_SUCCESS;
}

static VkResult acquire_image(VkDevice device, VkSwapchainKHR handle, uint64_t timeout,
    VkSemaphore semaphore, VkFence fence, uint32_t *index)
{
    struct mx_swapchain *chain = (void *)handle;
    struct mx_device *owner = (void *)device;
    if (!chain || chain->device != owner || (!semaphore && !fence))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (atomic_load_explicit(&owner->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    struct mx_surface *surface = chain->surface;
    uint64_t deadline = wsi_deadline(timeout);
    pthread_mutex_lock(&surface->lock);
    VkResult result = VK_SUCCESS;
    for (;;) {
        if (chain->retired) {
            result = VK_ERROR_OUT_OF_DATE_KHR;
            break;
        }
        if (surface->status != VK_SUCCESS || wl_display_get_error(surface->display)) {
            result = VK_ERROR_SURFACE_LOST_KHR;
            break;
        }
        if (wl_display_dispatch_queue_pending(surface->display, surface->events) < 0) {
            result = surface->status = VK_ERROR_SURFACE_LOST_KHR;
            break;
        }
        bool acquired = false;
        for (uint32_t n = 0; n < chain->count; n++) {
            uint32_t at = (chain->cursor + n) % chain->count;
            struct mx_present_image *image = &chain->images[at];
            if (image->busy || image->acquired)
                continue;
            if (semaphore)
                result = semaphore_transition(owner, semaphore, 1);
            if (result == VK_SUCCESS) {
                if (fence)
                    atomic_store_explicit(&((struct mx_fence *)fence)->signaled, 1, memory_order_release);
                image->acquired = true;
                chain->cursor = (at + 1) % chain->count;
                *index = at;
                acquired = true;
            }
            break;
        }
        if (acquired || result != VK_SUCCESS)
            break;
        result = wsi_events(surface, deadline);
        if (result != VK_SUCCESS) {
            if (result == VK_TIMEOUT && !timeout)
                result = VK_NOT_READY;
            break;
        }
    }
    pthread_mutex_unlock(&surface->lock);
    return result;
}

static void wsi_frame_done(void *data, struct wl_callback *callback, uint32_t time)
{
    (void)time;
    struct mx_surface *surface = data;
    surface->frame = NULL;
    wl_callback_destroy(callback);
}

static const struct wl_callback_listener wsi_frame_listener = {wsi_frame_done};

static VkResult wsi_present_image(struct mx_swapchain *chain, uint32_t index)
{
    struct mx_surface *surface = chain->surface;
    if (index >= chain->count || !chain->images[index].acquired)
        return VK_ERROR_DEVICE_LOST;
    VkResult result = surface->status;
    if (result != VK_SUCCESS || wl_display_get_error(surface->display))
        return VK_ERROR_SURFACE_LOST_KHR;
    if (chain->mode == VK_PRESENT_MODE_FIFO_KHR) {
        while (surface->frame && result == VK_SUCCESS)
            result = wsi_events(surface, UINT64_MAX);
        if (result != VK_SUCCESS)
            return result;
        surface->frame = wl_surface_frame((struct wl_surface *)surface->window_wrapper);
        if (!surface->frame) {
            atomic_store_explicit(&chain->device->lost, 1, memory_order_release);
            return VK_ERROR_DEVICE_LOST;
        }
        wl_callback_add_listener(surface->frame, &wsi_frame_listener, surface);
    }
    struct mx_present_image *image = &chain->images[index];
    image->busy = true;
    image->acquired = false;
    wl_surface_attach((struct wl_surface *)surface->window_wrapper, image->buffer, 0, 0);
    wl_surface_damage((struct wl_surface *)surface->window_wrapper, 0, 0, INT32_MAX, INT32_MAX);
    wl_surface_commit((struct wl_surface *)surface->window_wrapper);
    if (wl_display_flush(surface->display) < 0 && errno != EAGAIN)
        return surface->status = VK_ERROR_SURFACE_LOST_KHR;
    return VK_SUCCESS;
}

static VkResult queue_present(VkQueue queue, const VkPresentInfoKHR *info)
{
    struct mx_device *owner = queue ? ((struct mx_queue *)queue)->device : NULL;
    if (!owner || !info || !owner->swapchain_enabled)
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    if (atomic_load_explicit(&owner->lost, memory_order_acquire))
        return VK_ERROR_DEVICE_LOST;
    for (uint32_t i = 0; i < info->waitSemaphoreCount; i++) {
        VkResult result = semaphore_transition(owner, info->pWaitSemaphores[i], 0);
        if (result != VK_SUCCESS)
            return result;
    }
    VkResult result = VK_SUCCESS;
    for (uint32_t i = 0; i < info->swapchainCount; i++) {
        struct mx_swapchain *chain = (void *)info->pSwapchains[i];
        VkResult current = VK_ERROR_DEVICE_LOST;
        if (chain && chain->device == owner) {
            pthread_mutex_lock(&chain->surface->lock);
            current = wsi_present_image(chain, info->pImageIndices[i]);
            pthread_mutex_unlock(&chain->surface->lock);
        }
        if (info->pResults)
            info->pResults[i] = current;
        if (result == VK_SUCCESS && current != VK_SUCCESS)
            result = current;
    }
    return result;
}

static PFN_vkVoidFunction wsi_device_proc(const char *name)
{
    if (!strcmp(name, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)create_swapchain;
    if (!strcmp(name, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)destroy_swapchain;
    if (!strcmp(name, "vkGetSwapchainImagesKHR")) return (PFN_vkVoidFunction)swapchain_images;
    if (!strcmp(name, "vkAcquireNextImageKHR")) return (PFN_vkVoidFunction)acquire_image;
    if (!strcmp(name, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)queue_present;
    return NULL;
}

static PFN_vkVoidFunction wsi_instance_proc(const char *name)
{
    if (!strcmp(name, "vkCreateWaylandSurfaceKHR")) return (PFN_vkVoidFunction)create_wayland_surface;
    if (!strcmp(name, "vkDestroySurfaceKHR")) return (PFN_vkVoidFunction)destroy_surface;
    if (!strcmp(name, "vkGetPhysicalDeviceWaylandPresentationSupportKHR")) return (PFN_vkVoidFunction)wayland_presentation_support;
    if (!strcmp(name, "vkGetPhysicalDeviceSurfaceSupportKHR")) return (PFN_vkVoidFunction)surface_support;
    if (!strcmp(name, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")) return (PFN_vkVoidFunction)surface_capabilities;
    if (!strcmp(name, "vkGetPhysicalDeviceSurfaceFormatsKHR")) return (PFN_vkVoidFunction)surface_formats;
    if (!strcmp(name, "vkGetPhysicalDeviceSurfacePresentModesKHR")) return (PFN_vkVoidFunction)surface_present_modes;
    return NULL;
}
