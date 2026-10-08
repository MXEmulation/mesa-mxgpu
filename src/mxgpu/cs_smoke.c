/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_driver.h"
#include "mxsb.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELEMENTS 4096u
#define WORKGROUP 64u
#define REC(count, opcode) (((uint32_t)(count) << 16) | (opcode))

static const uint32_t r_global[] = {REC(4, MXSB_OP_BUILTIN), 10, MXSB_TYPE_U32X3, MXSB_BUILTIN_GLOBAL_INVOCATION_ID};
static const uint32_t r_index[] = {REC(5, MXSB_OP_EXTRACT), 11, MXSB_TYPE_U32, 10, 0};
static const uint32_t r_three[] = {REC(4, MXSB_OP_CONSTANT), 12, MXSB_TYPE_U32, 3};
static const uint32_t r_scaled[] = {REC(5, MXSB_OP_MUL), 13, MXSB_TYPE_U32, 11, 12};
static const uint32_t r_one[] = {REC(4, MXSB_OP_CONSTANT), 14, MXSB_TYPE_U32, 1};
static const uint32_t r_value[] = {REC(5, MXSB_OP_ADD), 15, MXSB_TYPE_U32, 13, 14};
static const uint32_t r_store[] = {REC(4, MXSB_OP_BUFFER_STORE), 1, 11, 15};
static const uint32_t r_load_src[] = {REC(5, MXSB_OP_BUFFER_LOAD), 12, MXSB_TYPE_U32, 1, 11};
static const uint32_t r_load_dst[] = {REC(5, MXSB_OP_BUFFER_LOAD), 13, MXSB_TYPE_U32, 2, 11};
static const uint32_t r_sum[] = {REC(5, MXSB_OP_ADD), 14, MXSB_TYPE_U32, 12, 13};
static const uint32_t r_store_sum[] = {REC(4, MXSB_OP_BUFFER_STORE), 2, 11, 14};
static const uint32_t r_return[] = {REC(1, MXSB_OP_RETURN_VOID)};

static const uint32_t *const store_records[] = {r_global, r_index, r_three, r_scaled, r_one, r_value, r_store, r_return};
static const uint32_t store_words[] = {4, 5, 4, 5, 4, 5, 4, 1};
static const uint32_t *const sum_records[] = {r_global, r_index, r_load_src, r_load_dst, r_sum, r_store_sum, r_return};
static const uint32_t sum_words[] = {4, 5, 5, 5, 5, 4, 1};

static uint32_t g_words[1024];
static uint8_t g_module[sizeof g_words];
static uint32_t g_values[ELEMENTS];
static char g_failure[256];

static int fail(const char *what)
{
    snprintf(g_failure, sizeof g_failure, "%s", what);
    return -1;
}

static int build_module(int sum, uint32_t *module_len)
{
    struct mxsb_writer writer;
    struct mxsb_limits limits;
    int status = mxsb_writer_init(&writer, g_words, sizeof g_words / sizeof g_words[0], MXSB_VERSION_MINOR);
    if (!status && sum)
        status = mxsb_writer_binding(&writer, 1, 0, MXSB_BINDING_STORAGE, MXSB_ACCESS_READ, MXSB_TYPE_U32, ELEMENTS);
    if (!status)
        status = mxsb_writer_binding(&writer, sum ? 2 : 1, sum ? 1 : 0, MXSB_BINDING_STORAGE,
                                     MXSB_ACCESS_READ_WRITE, MXSB_TYPE_U32, ELEMENTS);
    if (!status)
        status = mxsb_writer_entry_workgroup(&writer, 1, MXSB_STAGE_COMPUTE, 1, WORKGROUP, 1, 1);
    if (!status)
        status = mxsb_writer_block(&writer, 1, 1, sum ? sum_records : store_records, sum ? sum_words : store_words,
                                   sum ? 7 : 8);
    if (!status)
        status = mxsb_writer_finish(&writer, g_module, sizeof g_module, module_len);
    if (status)
        return fail("mxsb writer refused the module");
    if (mxsb_limits_default(&limits) != MXSB_OK || mxsb_verify(g_module, *module_len, &limits) != MXSB_OK)
        return fail("mxsb_verify refused the module");
    return 0;
}

static int check(const char *name, uint32_t buffer, uint32_t (*expected)(uint32_t))
{
    memset(g_values, 0, sizeof g_values);
    if (mxgpu_storage_buffer_read(buffer, 0, g_values, sizeof g_values))
        return fail("storage buffer readback failed");
    for (uint32_t i = 0; i < ELEMENTS; i++) {
        if (g_values[i] != expected(i)) {
            snprintf(g_failure, sizeof g_failure, "%s element %u is %u, expected %u", name, i, g_values[i], expected(i));
            return -1;
        }
    }
    return 0;
}

static uint32_t expect_store(uint32_t i) { return i * 3u + 1u; }
static uint32_t expect_sum(uint32_t i) { return i * 7u + (i + 5u); }

static int run_store(void)
{
    uint32_t module_len = 0, groups[3] = {ELEMENTS / WORKGROUP, 1, 1};
    if (build_module(0, &module_len))
        return -1;
    uint32_t pipeline = mxgpu_compute_pipeline_create(g_module, module_len, 1);
    uint32_t buffer = mxgpu_storage_buffer_create(ELEMENTS * 4u);
    struct mxgpu_compute_binding binding = {0, MXGPU_BIND_ACCESS_READ_WRITE, buffer, 0, ELEMENTS * 4u};
    int result = 0;
    if (!pipeline || !buffer)
        result = fail(!pipeline ? "store pipeline creation failed" : "store buffer creation failed");
    else if (mxgpu_compute_dispatch(pipeline, MXGPU_DISPATCH_THREADGROUPS, groups, &binding, 1))
        result = fail("store dispatch failed");
    else
        result = check("store", buffer, expect_store);
    if (buffer && mxgpu_storage_buffer_destroy(buffer) && !result)
        result = fail("store buffer destroy failed");
    if (pipeline && mxgpu_compute_pipeline_destroy(pipeline) && !result)
        result = fail("store pipeline destroy failed");
    return result;
}

static int run_sum(void)
{
    uint32_t module_len = 0, threads[3] = {ELEMENTS, 1, 1};
    if (build_module(1, &module_len))
        return -1;
    uint32_t pipeline = mxgpu_compute_pipeline_create(g_module, module_len, 1);
    uint32_t src = mxgpu_storage_buffer_create(ELEMENTS * 4u);
    uint32_t dst = mxgpu_storage_buffer_create(ELEMENTS * 4u);
    struct mxgpu_compute_binding bindings[2] = {
        {0, MXGPU_BIND_ACCESS_READ, src, 0, ELEMENTS * 4u},
        {1, MXGPU_BIND_ACCESS_READ_WRITE, dst, 0, ELEMENTS * 4u},
    };
    int result = 0;
    if (!pipeline || !src || !dst) {
        result = fail(!pipeline ? "sum pipeline creation failed" : "sum buffer creation failed");
        goto done;
    }
    for (uint32_t i = 0; i < ELEMENTS; i++)
        g_values[i] = i * 7u;
    if (mxgpu_storage_buffer_upload(src, 0, g_values, sizeof g_values)) {
        result = fail("source upload failed");
        goto done;
    }
    for (uint32_t i = 0; i < ELEMENTS; i++)
        g_values[i] = i + 5u;
    if (mxgpu_storage_buffer_upload(dst, 0, g_values, sizeof g_values))
        result = fail("destination upload failed");
    else if (mxgpu_compute_dispatch(pipeline, MXGPU_DISPATCH_THREADS, threads, bindings, 2))
        result = fail("sum dispatch failed");
    else
        result = check("load-add-store", dst, expect_sum);
done:
    if (src && mxgpu_storage_buffer_destroy(src) && !result)
        result = fail("source destroy failed");
    if (dst && mxgpu_storage_buffer_destroy(dst) && !result)
        result = fail("destination destroy failed");
    if (pipeline && mxgpu_compute_pipeline_destroy(pipeline) && !result)
        result = fail("sum pipeline destroy failed");
    return result;
}

int main(void)
{
    int result;
    if (mxgpu_device_open()) {
        printf("cs_smoke FAIL: render node open failed\n");
        return 1;
    }
    if (!mxgpu_compute_available()) {
        mxgpu_device_close();
        printf("cs_smoke FAIL: device does not offer compute\n");
        return 1;
    }
    result = run_store();
    if (!result)
        result = run_sum();
    if (!result && mxgpu_device_lost())
        result = fail("device lost");
    mxgpu_device_close();
    if (result) {
        printf("cs_smoke FAIL: %s\n", g_failure);
        return 1;
    }
    printf("cs_smoke PASS: store %u elements in %u workgroups, load-add-store %u threads over two storage buffers\n",
           ELEMENTS, ELEMENTS / WORKGROUP, ELEMENTS);
    return 0;
}
