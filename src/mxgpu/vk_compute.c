/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#include "mxgpu_driver.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define STORE_COUNT 1000u
#define STORE_CAPACITY 1024u
#define CHAIN_COUNT 4096u
#define REDUCE_GROUP 64u
#define ATOMIC_GROUP 32u
#define ATOMIC_GROUPS 8u
#define SENTINEL 0xdeadbeefu

static const uint32_t store_spirv[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000002b, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000011,
    0x696d694c, 0x00007374, 0x00050006, 0x00000011, 0x00000000, 0x6e756f63, 0x00000074, 0x00040005,
    0x00000013, 0x696d696c, 0x00007374, 0x00040005, 0x0000001e, 0x756c6156, 0x00007365, 0x00050006,
    0x0000001e, 0x00000000, 0x756c6176, 0x00007365, 0x00030005, 0x00000020, 0x00000000, 0x00040047,
    0x0000000b, 0x0000000b, 0x0000001c, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011,
    0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000001d, 0x00000006, 0x00000004, 0x00030047,
    0x0000001e, 0x00000003, 0x00050048, 0x0000001e, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x00000020, 0x00000021, 0x00000000, 0x00040047, 0x00000020, 0x00000022, 0x00000000, 0x00040047,
    0x0000002a, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006,
    0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009,
    0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000,
    0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0003001e, 0x00000011, 0x00000006, 0x00040020,
    0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000009, 0x00040015,
    0x00000014, 0x00000020, 0x00000001, 0x0004002b, 0x00000014, 0x00000015, 0x00000000, 0x00040020,
    0x00000016, 0x00000009, 0x00000006, 0x00020014, 0x00000019, 0x0003001d, 0x0000001d, 0x00000006,
    0x0003001e, 0x0000001e, 0x0000001d, 0x00040020, 0x0000001f, 0x00000002, 0x0000001e, 0x0004003b,
    0x0000001f, 0x00000020, 0x00000002, 0x0004002b, 0x00000006, 0x00000023, 0x00000003, 0x0004002b,
    0x00000006, 0x00000025, 0x00000001, 0x00040020, 0x00000027, 0x00000002, 0x00000006, 0x0004002b,
    0x00000006, 0x00000029, 0x00000040, 0x0006002c, 0x00000009, 0x0000002a, 0x00000029, 0x00000025,
    0x00000025, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b,
    0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f,
    0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x00050041, 0x00000016, 0x00000017, 0x00000013,
    0x00000015, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x000500b0, 0x00000019, 0x0000001a,
    0x00000010, 0x00000018, 0x000300f7, 0x0000001c, 0x00000000, 0x000400fa, 0x0000001a, 0x0000001b,
    0x0000001c, 0x000200f8, 0x0000001b, 0x0004003d, 0x00000006, 0x00000021, 0x00000008, 0x0004003d,
    0x00000006, 0x00000022, 0x00000008, 0x00050084, 0x00000006, 0x00000024, 0x00000022, 0x00000023,
    0x00050080, 0x00000006, 0x00000026, 0x00000024, 0x00000025, 0x00060041, 0x00000027, 0x00000028,
    0x00000020, 0x00000015, 0x00000021, 0x0003003e, 0x00000028, 0x00000026, 0x000200f9, 0x0000001c,
    0x000200f8, 0x0000001c, 0x000100fd, 0x00010038,
};

static const uint32_t reduce_spirv[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000050, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000001b, 0x00000049,
    0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000008, 0x61636f6c,
    0x0000006c, 0x00080005, 0x0000000b, 0x4c5f6c67, 0x6c61636f, 0x6f766e49, 0x69746163, 0x44496e6f,
    0x00000000, 0x00040005, 0x00000013, 0x74726170, 0x006c6169, 0x00040005, 0x00000016, 0x75706e49,
    0x00000074, 0x00050006, 0x00000016, 0x00000000, 0x61746164, 0x00000000, 0x00030005, 0x00000018,
    0x00000000, 0x00080005, 0x0000001b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69,
    0x00000044, 0x00040005, 0x00000025, 0x69727473, 0x00006564, 0x00040005, 0x00000046, 0x736d7553,
    0x00000000, 0x00050006, 0x00000046, 0x00000000, 0x736d7573, 0x00000000, 0x00030005, 0x00000048,
    0x00000000, 0x00060005, 0x00000049, 0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449, 0x00040047,
    0x0000000b, 0x0000000b, 0x0000001b, 0x00040047, 0x00000015, 0x00000006, 0x00000004, 0x00030047,
    0x00000016, 0x00000003, 0x00040048, 0x00000016, 0x00000000, 0x00000018, 0x00050048, 0x00000016,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000018, 0x00000018, 0x00040047, 0x00000018,
    0x00000021, 0x00000000, 0x00040047, 0x00000018, 0x00000022, 0x00000000, 0x00040047, 0x0000001b,
    0x0000000b, 0x0000001c, 0x00040047, 0x00000045, 0x00000006, 0x00000004, 0x00030047, 0x00000046,
    0x00000003, 0x00050048, 0x00000046, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000048,
    0x00000021, 0x00000001, 0x00040047, 0x00000048, 0x00000022, 0x00000000, 0x00040047, 0x00000049,
    0x0000000b, 0x0000001a, 0x00040047, 0x0000004f, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020,
    0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020,
    0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
    0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0004002b,
    0x00000006, 0x00000010, 0x00000040, 0x0004001c, 0x00000011, 0x00000006, 0x00000010, 0x00040020,
    0x00000012, 0x00000004, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000004, 0x0003001d,
    0x00000015, 0x00000006, 0x0003001e, 0x00000016, 0x00000015, 0x00040020, 0x00000017, 0x00000002,
    0x00000016, 0x0004003b, 0x00000017, 0x00000018, 0x00000002, 0x00040015, 0x00000019, 0x00000020,
    0x00000001, 0x0004002b, 0x00000019, 0x0000001a, 0x00000000, 0x0004003b, 0x0000000a, 0x0000001b,
    0x00000001, 0x00040020, 0x0000001e, 0x00000002, 0x00000006, 0x00040020, 0x00000021, 0x00000004,
    0x00000006, 0x0004002b, 0x00000006, 0x00000023, 0x00000002, 0x0004002b, 0x00000006, 0x00000024,
    0x00000108, 0x0004002b, 0x00000006, 0x00000026, 0x00000020, 0x00020014, 0x0000002d, 0x0004002b,
    0x00000006, 0x0000003e, 0x00000001, 0x0003001d, 0x00000045, 0x00000006, 0x0003001e, 0x00000046,
    0x00000045, 0x00040020, 0x00000047, 0x00000002, 0x00000046, 0x0004003b, 0x00000047, 0x00000048,
    0x00000002, 0x0004003b, 0x0000000a, 0x00000049, 0x00000001, 0x0006002c, 0x00000009, 0x0000004f,
    0x00000010, 0x0000003e, 0x0000003e, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x00000007,
    0x00000025, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d,
    0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006,
    0x00000014, 0x00000008, 0x00050041, 0x0000000d, 0x0000001c, 0x0000001b, 0x0000000c, 0x0004003d,
    0x00000006, 0x0000001d, 0x0000001c, 0x00060041, 0x0000001e, 0x0000001f, 0x00000018, 0x0000001a,
    0x0000001d, 0x0004003d, 0x00000006, 0x00000020, 0x0000001f, 0x00050041, 0x00000021, 0x00000022,
    0x00000013, 0x00000014, 0x0003003e, 0x00000022, 0x00000020, 0x000400e0, 0x00000023, 0x00000023,
    0x00000024, 0x0003003e, 0x00000025, 0x00000026, 0x000200f9, 0x00000027, 0x000200f8, 0x00000027,
    0x000400f6, 0x00000029, 0x0000002a, 0x00000000, 0x000200f9, 0x0000002b, 0x000200f8, 0x0000002b,
    0x0004003d, 0x00000006, 0x0000002c, 0x00000025, 0x000500ac, 0x0000002d, 0x0000002e, 0x0000002c,
    0x0000000c, 0x000400fa, 0x0000002e, 0x00000028, 0x00000029, 0x000200f8, 0x00000028, 0x0004003d,
    0x00000006, 0x0000002f, 0x00000008, 0x0004003d, 0x00000006, 0x00000030, 0x00000025, 0x000500b0,
    0x0000002d, 0x00000031, 0x0000002f, 0x00000030, 0x000300f7, 0x00000033, 0x00000000, 0x000400fa,
    0x00000031, 0x00000032, 0x00000033, 0x000200f8, 0x00000032, 0x0004003d, 0x00000006, 0x00000034,
    0x00000008, 0x0004003d, 0x00000006, 0x00000035, 0x00000008, 0x0004003d, 0x00000006, 0x00000036,
    0x00000025, 0x00050080, 0x00000006, 0x00000037, 0x00000035, 0x00000036, 0x00050041, 0x00000021,
    0x00000038, 0x00000013, 0x00000037, 0x0004003d, 0x00000006, 0x00000039, 0x00000038, 0x00050041,
    0x00000021, 0x0000003a, 0x00000013, 0x00000034, 0x0004003d, 0x00000006, 0x0000003b, 0x0000003a,
    0x00050080, 0x00000006, 0x0000003c, 0x0000003b, 0x00000039, 0x00050041, 0x00000021, 0x0000003d,
    0x00000013, 0x00000034, 0x0003003e, 0x0000003d, 0x0000003c, 0x000200f9, 0x00000033, 0x000200f8,
    0x00000033, 0x000400e0, 0x00000023, 0x00000023, 0x00000024, 0x000200f9, 0x0000002a, 0x000200f8,
    0x0000002a, 0x0004003d, 0x00000006, 0x0000003f, 0x00000025, 0x000500c2, 0x00000006, 0x00000040,
    0x0000003f, 0x0000003e, 0x0003003e, 0x00000025, 0x00000040, 0x000200f9, 0x00000027, 0x000200f8,
    0x00000029, 0x0004003d, 0x00000006, 0x00000041, 0x00000008, 0x000500aa, 0x0000002d, 0x00000042,
    0x00000041, 0x0000000c, 0x000300f7, 0x00000044, 0x00000000, 0x000400fa, 0x00000042, 0x00000043,
    0x00000044, 0x000200f8, 0x00000043, 0x00050041, 0x0000000d, 0x0000004a, 0x00000049, 0x0000000c,
    0x0004003d, 0x00000006, 0x0000004b, 0x0000004a, 0x00050041, 0x00000021, 0x0000004c, 0x00000013,
    0x0000001a, 0x0004003d, 0x00000006, 0x0000004d, 0x0000004c, 0x00060041, 0x0000001e, 0x0000004e,
    0x00000048, 0x0000001a, 0x0000004b, 0x0003003e, 0x0000004e, 0x0000004d, 0x000200f9, 0x00000044,
    0x000200f8, 0x00000044, 0x000100fd, 0x00010038,
};

static const uint32_t atomics_spirv[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000003f, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00000008, 0x0000001e, 0x0000003a,
    0x00060010, 0x00000004, 0x00000011, 0x00000020, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00080005, 0x00000008, 0x4c5f6c67,
    0x6c61636f, 0x6f766e49, 0x69746163, 0x6e496e6f, 0x00786564, 0x00050005, 0x00000010, 0x69727261,
    0x736c6176, 0x00000000, 0x00050005, 0x00000015, 0x6e756f43, 0x73726574, 0x00000000, 0x00050006,
    0x00000015, 0x00000000, 0x61746f74, 0x0000006c, 0x00050006, 0x00000015, 0x00000001, 0x6978616d,
    0x006d756d, 0x00060006, 0x00000015, 0x00000002, 0x6c6c7566, 0x6f72675f, 0x00737075, 0x00050006,
    0x00000015, 0x00000003, 0x756f7267, 0x00007370, 0x00030005, 0x00000017, 0x00000000, 0x00080005,
    0x0000001e, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00070005,
    0x0000003a, 0x4e5f6c67, 0x6f576d75, 0x72476b72, 0x7370756f, 0x00000000, 0x00040047, 0x00000008,
    0x0000000b, 0x0000001d, 0x00030047, 0x00000015, 0x00000003, 0x00050048, 0x00000015, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x00000015, 0x00000001, 0x00000023, 0x00000004, 0x00050048,
    0x00000015, 0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x00000015, 0x00000003, 0x00000023,
    0x0000000c, 0x00040047, 0x00000017, 0x00000021, 0x00000000, 0x00040047, 0x00000017, 0x00000022,
    0x00000000, 0x00040047, 0x0000001e, 0x0000000b, 0x0000001c, 0x00040047, 0x0000003a, 0x0000000b,
    0x00000018, 0x00040047, 0x0000003e, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007,
    0x00000001, 0x00000006, 0x0004003b, 0x00000007, 0x00000008, 0x00000001, 0x0004002b, 0x00000006,
    0x0000000a, 0x00000000, 0x00020014, 0x0000000b, 0x00040020, 0x0000000f, 0x00000004, 0x00000006,
    0x0004003b, 0x0000000f, 0x00000010, 0x00000004, 0x0004002b, 0x00000006, 0x00000011, 0x00000002,
    0x0004002b, 0x00000006, 0x00000012, 0x00000108, 0x0004002b, 0x00000006, 0x00000013, 0x00000001,
    0x0006001e, 0x00000015, 0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x00000016,
    0x00000002, 0x00000015, 0x0004003b, 0x00000016, 0x00000017, 0x00000002, 0x00040015, 0x00000018,
    0x00000020, 0x00000001, 0x0004002b, 0x00000018, 0x00000019, 0x00000000, 0x00040020, 0x0000001a,
    0x00000002, 0x00000006, 0x00040017, 0x0000001c, 0x00000006, 0x00000003, 0x00040020, 0x0000001d,
    0x00000001, 0x0000001c, 0x0004003b, 0x0000001d, 0x0000001e, 0x00000001, 0x0004002b, 0x00000018,
    0x00000022, 0x00000001, 0x0004002b, 0x00000006, 0x0000002c, 0x00000020, 0x0004002b, 0x00000018,
    0x00000031, 0x00000002, 0x0004002b, 0x00000018, 0x00000039, 0x00000003, 0x0004003b, 0x0000001d,
    0x0000003a, 0x00000001, 0x0006002c, 0x0000001c, 0x0000003e, 0x0000002c, 0x00000013, 0x00000013,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
    0x00000006, 0x00000009, 0x00000008, 0x000500aa, 0x0000000b, 0x0000000c, 0x00000009, 0x0000000a,
    0x000300f7, 0x0000000e, 0x00000000, 0x000400fa, 0x0000000c, 0x0000000d, 0x0000000e, 0x000200f8,
    0x0000000d, 0x0003003e, 0x00000010, 0x0000000a, 0x000200f9, 0x0000000e, 0x000200f8, 0x0000000e,
    0x000400e0, 0x00000011, 0x00000011, 0x00000012, 0x000700ea, 0x00000006, 0x00000014, 0x00000010,
    0x00000013, 0x0000000a, 0x00000013, 0x000400e0, 0x00000011, 0x00000011, 0x00000012, 0x00050041,
    0x0000001a, 0x0000001b, 0x00000017, 0x00000019, 0x00050041, 0x00000007, 0x0000001f, 0x0000001e,
    0x0000000a, 0x0004003d, 0x00000006, 0x00000020, 0x0000001f, 0x000700ea, 0x00000006, 0x00000021,
    0x0000001b, 0x00000013, 0x0000000a, 0x00000020, 0x00050041, 0x0000001a, 0x00000023, 0x00000017,
    0x00000022, 0x00050041, 0x00000007, 0x00000024, 0x0000001e, 0x0000000a, 0x0004003d, 0x00000006,
    0x00000025, 0x00000024, 0x000700ef, 0x00000006, 0x00000026, 0x00000023, 0x00000013, 0x0000000a,
    0x00000025, 0x0004003d, 0x00000006, 0x00000027, 0x00000008, 0x000500aa, 0x0000000b, 0x00000028,
    0x00000027, 0x0000000a, 0x000300f7, 0x0000002a, 0x00000000, 0x000400fa, 0x00000028, 0x00000029,
    0x0000002a, 0x000200f8, 0x00000029, 0x0004003d, 0x00000006, 0x0000002b, 0x00000010, 0x000500aa,
    0x0000000b, 0x0000002d, 0x0000002b, 0x0000002c, 0x000200f9, 0x0000002a, 0x000200f8, 0x0000002a,
    0x000700f5, 0x0000000b, 0x0000002e, 0x00000028, 0x0000000e, 0x0000002d, 0x00000029, 0x000300f7,
    0x00000030, 0x00000000, 0x000400fa, 0x0000002e, 0x0000002f, 0x00000030, 0x000200f8, 0x0000002f,
    0x00050041, 0x0000001a, 0x00000032, 0x00000017, 0x00000031, 0x000700ea, 0x00000006, 0x00000033,
    0x00000032, 0x00000013, 0x0000000a, 0x00000013, 0x000200f9, 0x00000030, 0x000200f8, 0x00000030,
    0x00050041, 0x00000007, 0x00000034, 0x0000001e, 0x0000000a, 0x0004003d, 0x00000006, 0x00000035,
    0x00000034, 0x000500aa, 0x0000000b, 0x00000036, 0x00000035, 0x0000000a, 0x000300f7, 0x00000038,
    0x00000000, 0x000400fa, 0x00000036, 0x00000037, 0x00000038, 0x000200f8, 0x00000037, 0x00050041,
    0x00000007, 0x0000003b, 0x0000003a, 0x0000000a, 0x0004003d, 0x00000006, 0x0000003c, 0x0000003b,
    0x00050041, 0x0000001a, 0x0000003d, 0x00000017, 0x00000039, 0x0003003e, 0x0000003d, 0x0000003c,
    0x000200f9, 0x00000038, 0x000200f8, 0x00000038, 0x000100fd, 0x00010038,
};

static const uint32_t loop_spirv[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000031, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00000027, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x006d7573, 0x00030005, 0x0000000a,
    0x00000069, 0x00040005, 0x00000011, 0x696d694c, 0x00007374, 0x00050006, 0x00000011, 0x00000000,
    0x6e756f63, 0x00000074, 0x00040005, 0x00000013, 0x696d696c, 0x00007374, 0x00040005, 0x00000022,
    0x756c6156, 0x00007365, 0x00050006, 0x00000022, 0x00000000, 0x756c6176, 0x00007365, 0x00030005,
    0x00000024, 0x00000000, 0x00080005, 0x00000027, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f,
    0x496e6f69, 0x00000044, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000,
    0x00000023, 0x00000000, 0x00040047, 0x00000021, 0x00000006, 0x00000004, 0x00030047, 0x00000022,
    0x00000003, 0x00050048, 0x00000022, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000024,
    0x00000021, 0x00000000, 0x00040047, 0x00000024, 0x00000022, 0x00000000, 0x00040047, 0x00000027,
    0x0000000b, 0x0000001c, 0x00040047, 0x00000030, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020,
    0x00000007, 0x00000007, 0x00000006, 0x0004002b, 0x00000006, 0x00000009, 0x00000000, 0x0003001e,
    0x00000011, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012,
    0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b, 0x00000014,
    0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014, 0x00000019,
    0x0004002b, 0x00000014, 0x0000001f, 0x00000001, 0x0003001d, 0x00000021, 0x00000006, 0x0003001e,
    0x00000022, 0x00000021, 0x00040020, 0x00000023, 0x00000002, 0x00000022, 0x0004003b, 0x00000023,
    0x00000024, 0x00000002, 0x00040017, 0x00000025, 0x00000006, 0x00000003, 0x00040020, 0x00000026,
    0x00000001, 0x00000025, 0x0004003b, 0x00000026, 0x00000027, 0x00000001, 0x00040020, 0x00000028,
    0x00000001, 0x00000006, 0x00040020, 0x0000002c, 0x00000002, 0x00000006, 0x0004002b, 0x00000006,
    0x0000002e, 0x00000040, 0x0004002b, 0x00000006, 0x0000002f, 0x00000001, 0x0006002c, 0x00000025,
    0x00000030, 0x0000002e, 0x0000002f, 0x0000002f, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b,
    0x00000007, 0x0000000a, 0x00000007, 0x0003003e, 0x00000008, 0x00000009, 0x0003003e, 0x0000000a,
    0x00000009, 0x000200f9, 0x0000000b, 0x000200f8, 0x0000000b, 0x000400f6, 0x0000000d, 0x0000000e,
    0x00000000, 0x000200f9, 0x0000000f, 0x000200f8, 0x0000000f, 0x0004003d, 0x00000006, 0x00000010,
    0x0000000a, 0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006,
    0x00000018, 0x00000017, 0x000500b0, 0x00000019, 0x0000001a, 0x00000010, 0x00000018, 0x000400fa,
    0x0000001a, 0x0000000c, 0x0000000d, 0x000200f8, 0x0000000c, 0x0004003d, 0x00000006, 0x0000001b,
    0x0000000a, 0x0004003d, 0x00000006, 0x0000001c, 0x00000008, 0x00050080, 0x00000006, 0x0000001d,
    0x0000001c, 0x0000001b, 0x0003003e, 0x00000008, 0x0000001d, 0x000200f9, 0x0000000e, 0x000200f8,
    0x0000000e, 0x0004003d, 0x00000006, 0x0000001e, 0x0000000a, 0x00050080, 0x00000006, 0x00000020,
    0x0000001e, 0x0000001f, 0x0003003e, 0x0000000a, 0x00000020, 0x000200f9, 0x0000000b, 0x000200f8,
    0x0000000d, 0x00050041, 0x00000028, 0x00000029, 0x00000027, 0x00000009, 0x0004003d, 0x00000006,
    0x0000002a, 0x00000029, 0x0004003d, 0x00000006, 0x0000002b, 0x00000008, 0x00060041, 0x0000002c,
    0x0000002d, 0x00000024, 0x00000015, 0x0000002a, 0x0003003e, 0x0000002d, 0x0000002b, 0x000100fd,
    0x00010038,
};

#define INSTANCE_FUNCTIONS(X) \
    X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
    X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) \
    X(CreateDevice) X(GetDeviceProcAddr)
#define DEVICE_FUNCTIONS(X) \
    X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer) \
    X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
    X(MapMemory) X(UnmapMemory) X(CreateShaderModule) X(DestroyShaderModule) \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreateDescriptorPool) \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
    X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) \
    X(EndCommandBuffer) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) \
    X(CmdDispatch) X(CmdPipelineBarrier) X(CreateFence) X(DestroyFence) X(QueueSubmit) \
    X(WaitForFences) X(ResetFences) X(DeviceWaitIdle)

struct compute_api {
#define DECLARE_FUNCTION(name) PFN_vk##name name;
    INSTANCE_FUNCTIONS(DECLARE_FUNCTION)
    DEVICE_FUNCTIONS(DECLARE_FUNCTION)
#undef DECLARE_FUNCTION
};

struct compute_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint32_t *values;
};

struct compute_context {
    struct compute_api api;
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t memory_type;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool descriptor_pool;
    VkCommandPool command_pool;
    VkFence fence;
};

static char g_failure[256];

static int fail(const char *what)
{
    snprintf(g_failure, sizeof g_failure, "%s", what);
    return -1;
}

static int create_buffer(struct compute_context *ctx, uint32_t words, struct compute_buffer *out)
{
    struct compute_api *api = &ctx->api;
    VkMemoryRequirements requirements;
    void *mapped = NULL;
    memset(out, 0, sizeof *out);
    if (api->CreateBuffer(ctx->device, &(VkBufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = words * 4u,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        }, NULL, &out->buffer) != VK_SUCCESS)
        return fail("vkCreateBuffer failed");
    api->GetBufferMemoryRequirements(ctx->device, out->buffer, &requirements);
    if (!(requirements.memoryTypeBits & (1u << ctx->memory_type)) ||
        api->AllocateMemory(ctx->device, &(VkMemoryAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size,
            .memoryTypeIndex = ctx->memory_type
        }, NULL, &out->memory) != VK_SUCCESS ||
        api->BindBufferMemory(ctx->device, out->buffer, out->memory, 0) != VK_SUCCESS ||
        api->MapMemory(ctx->device, out->memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
        return fail("buffer memory setup failed");
    out->values = mapped;
    return 0;
}

static void destroy_buffer(struct compute_context *ctx, struct compute_buffer *buffer)
{
    if (buffer->memory) {
        ctx->api.UnmapMemory(ctx->device, buffer->memory);
        ctx->api.FreeMemory(ctx->device, buffer->memory, NULL);
    }
    if (buffer->buffer)
        ctx->api.DestroyBuffer(ctx->device, buffer->buffer, NULL);
    memset(buffer, 0, sizeof *buffer);
}

static int create_pipeline(struct compute_context *ctx, const uint32_t *code, size_t size, VkPipeline *pipeline)
{
    struct compute_api *api = &ctx->api;
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result;
    if (api->CreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code
        }, NULL, &module) != VK_SUCCESS)
        return fail("vkCreateShaderModule failed");
    result = api->CreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"
        },
        .layout = ctx->pipeline_layout
    }, NULL, pipeline);
    api->DestroyShaderModule(ctx->device, module, NULL);
    if (result != VK_SUCCESS) {
        snprintf(g_failure, sizeof g_failure, "vkCreateComputePipelines returned %d", (int)result);
        return -1;
    }
    return 0;
}

static int bind_buffers(struct compute_context *ctx, VkDescriptorSet set,
                        const struct compute_buffer *first, const struct compute_buffer *second)
{
    VkDescriptorBufferInfo infos[2] = {
        {first->buffer, 0, VK_WHOLE_SIZE},
        {second ? second->buffer : first->buffer, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[2];
    for (unsigned i = 0; i < 2; i++)
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i]
        };
    ctx->api.UpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
    return 0;
}

static int allocate_set(struct compute_context *ctx, VkDescriptorSet *set)
{
    if (ctx->api.AllocateDescriptorSets(ctx->device, &(VkDescriptorSetAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = ctx->descriptor_pool,
            .descriptorSetCount = 1, .pSetLayouts = &ctx->set_layout
        }, set) != VK_SUCCESS)
        return fail("vkAllocateDescriptorSets failed");
    return 0;
}

static int begin(struct compute_context *ctx, VkCommandBuffer *command)
{
    if (ctx->api.AllocateCommandBuffers(ctx->device, &(VkCommandBufferAllocateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ctx->command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1
        }, command) != VK_SUCCESS ||
        ctx->api.BeginCommandBuffer(*command, &(VkCommandBufferBeginInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        }) != VK_SUCCESS)
        return fail("command buffer begin failed");
    return 0;
}

static int submit(struct compute_context *ctx, VkCommandBuffer command)
{
    VkResult result = ctx->api.EndCommandBuffer(command);
    if (result != VK_SUCCESS) {
        snprintf(g_failure, sizeof g_failure, "vkEndCommandBuffer returned %d", (int)result);
        return -1;
    }
    if (ctx->api.ResetFences(ctx->device, 1, &ctx->fence) != VK_SUCCESS)
        return fail("vkResetFences failed");
    result = ctx->api.QueueSubmit(ctx->queue, 1, &(VkSubmitInfo){
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command
    }, ctx->fence);
    if (result != VK_SUCCESS) {
        snprintf(g_failure, sizeof g_failure, "vkQueueSubmit returned %d", (int)result);
        return -1;
    }
    if (ctx->api.WaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return fail("vkWaitForFences failed");
    return 0;
}

static void storage_barrier(struct compute_context *ctx, VkCommandBuffer command)
{
    ctx->api.CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &(VkMemoryBarrier){
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        }, 0, NULL, 0, NULL);
}

static int refuse_unsupported(struct compute_context *ctx)
{
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (ctx->api.CreateShaderModule(ctx->device, &(VkShaderModuleCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof loop_spirv, .pCode = loop_spirv
        }, NULL, &module) != VK_SUCCESS)
        return fail("vkCreateShaderModule failed for the data-dependent loop");
    VkResult result = ctx->api.CreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"
        },
        .layout = ctx->pipeline_layout
    }, NULL, &pipeline);
    ctx->api.DestroyShaderModule(ctx->device, module, NULL);
    if (result == VK_SUCCESS || pipeline != VK_NULL_HANDLE) {
        if (pipeline)
            ctx->api.DestroyPipeline(ctx->device, pipeline, NULL);
        return fail("a data-dependent loop was accepted");
    }
    return 0;
}

static int run_store(struct compute_context *ctx, VkPipeline store)
{
    struct compute_buffer values;
    VkDescriptorSet set;
    VkCommandBuffer command;
    uint32_t count = STORE_COUNT;
    int result = -1;
    if (create_buffer(ctx, STORE_CAPACITY, &values) || allocate_set(ctx, &set) || bind_buffers(ctx, set, &values, NULL))
        goto done;
    for (uint32_t i = 0; i < STORE_CAPACITY; i++)
        values.values[i] = SENTINEL;
    if (begin(ctx, &command))
        goto done;
    ctx->api.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, store);
    ctx->api.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &set, 0, NULL);
    ctx->api.CmdPushConstants(command, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &count);
    ctx->api.CmdDispatch(command, (STORE_COUNT + 63u) / 64u, 1, 1);
    if (submit(ctx, command))
        goto done;
    for (uint32_t i = 0; i < STORE_CAPACITY; i++) {
        uint32_t expected = i < STORE_COUNT ? i * 3u + 1u : SENTINEL;
        if (values.values[i] != expected) {
            snprintf(g_failure, sizeof g_failure, "store element %u is 0x%x, expected 0x%x", i, values.values[i], expected);
            goto done;
        }
    }
    result = 0;
done:
    destroy_buffer(ctx, &values);
    return result;
}

static int run_chain(struct compute_context *ctx, VkPipeline store, VkPipeline reduce)
{
    struct compute_buffer values, sums;
    VkDescriptorSet sets[2];
    VkCommandBuffer command;
    uint32_t count = CHAIN_COUNT;
    int result = -1;
    memset(&sums, 0, sizeof sums);
    if (create_buffer(ctx, CHAIN_COUNT, &values) || create_buffer(ctx, CHAIN_COUNT / REDUCE_GROUP, &sums) ||
        allocate_set(ctx, &sets[0]) || allocate_set(ctx, &sets[1]) ||
        bind_buffers(ctx, sets[0], &values, NULL) || bind_buffers(ctx, sets[1], &values, &sums))
        goto done;
    memset(values.values, 0, CHAIN_COUNT * 4u);
    memset(sums.values, 0xff, CHAIN_COUNT / REDUCE_GROUP * 4u);
    if (begin(ctx, &command))
        goto done;
    ctx->api.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, store);
    ctx->api.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &sets[0], 0, NULL);
    ctx->api.CmdPushConstants(command, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &count);
    ctx->api.CmdDispatch(command, CHAIN_COUNT / 64u, 1, 1);
    storage_barrier(ctx, command);
    ctx->api.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, reduce);
    ctx->api.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &sets[1], 0, NULL);
    ctx->api.CmdDispatch(command, CHAIN_COUNT / REDUCE_GROUP, 1, 1);
    if (submit(ctx, command))
        goto done;
    for (uint32_t group = 0; group < CHAIN_COUNT / REDUCE_GROUP; group++) {
        uint32_t expected = 0;
        for (uint32_t i = group * REDUCE_GROUP; i < (group + 1) * REDUCE_GROUP; i++)
            expected += i * 3u + 1u;
        if (sums.values[group] != expected) {
            snprintf(g_failure, sizeof g_failure, "workgroup %u reduced to %u, expected %u", group, sums.values[group], expected);
            goto done;
        }
    }
    result = 0;
done:
    destroy_buffer(ctx, &values);
    destroy_buffer(ctx, &sums);
    return result;
}

static int run_atomics(struct compute_context *ctx, VkPipeline atomics)
{
    struct compute_buffer counters;
    VkDescriptorSet set;
    VkCommandBuffer command;
    const uint32_t invocations = ATOMIC_GROUP * ATOMIC_GROUPS;
    int result = -1;
    if (create_buffer(ctx, 4, &counters) || allocate_set(ctx, &set) || bind_buffers(ctx, set, &counters, NULL))
        goto done;
    memset(counters.values, 0, 16);
    if (begin(ctx, &command))
        goto done;
    ctx->api.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, atomics);
    ctx->api.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &set, 0, NULL);
    ctx->api.CmdDispatch(command, ATOMIC_GROUPS, 1, 1);
    if (submit(ctx, command))
        goto done;
    const uint32_t expected[4] = {invocations * (invocations - 1u) / 2u, invocations - 1u, ATOMIC_GROUPS, ATOMIC_GROUPS};
    static const char *const names[4] = {"total", "maximum", "full groups", "groups"};
    for (unsigned i = 0; i < 4; i++) {
        if (counters.values[i] != expected[i]) {
            snprintf(g_failure, sizeof g_failure, "atomic %s is %u, expected %u", names[i], counters.values[i], expected[i]);
            goto done;
        }
    }
    result = 0;
done:
    destroy_buffer(ctx, &counters);
    return result;
}

static int setup(struct compute_context *ctx)
{
    struct compute_api *api = &ctx->api;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory;
    VkQueueFamilyProperties family;
    uint32_t count = 1;
    PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_instance || create_instance(&(VkInstanceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &(VkApplicationInfo){
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0
            }
        }, NULL, &ctx->instance) != VK_SUCCESS)
        return fail("vkCreateInstance failed");
#define LOAD_INSTANCE(name) api->name = (PFN_vk##name)vkGetInstanceProcAddr(ctx->instance, "vk" #name); \
    if (!api->name) return fail("missing vk" #name);
    INSTANCE_FUNCTIONS(LOAD_INSTANCE)
#undef LOAD_INSTANCE
    if (api->EnumeratePhysicalDevices(ctx->instance, &count, &ctx->physical) < 0 || !count)
        return fail("no physical device");
    api->GetPhysicalDeviceProperties(ctx->physical, &properties);
    count = 1;
    api->GetPhysicalDeviceQueueFamilyProperties(ctx->physical, &count, &family);
    if (!count || !(family.queueFlags & VK_QUEUE_COMPUTE_BIT))
        return fail("queue family 0 does not offer compute");
    printf("device %s, workgroup size %u x %u x %u, invocations %u, shared %u bytes, storage range %u, storage buffers %u\n",
           properties.deviceName, properties.limits.maxComputeWorkGroupSize[0],
           properties.limits.maxComputeWorkGroupSize[1], properties.limits.maxComputeWorkGroupSize[2],
           properties.limits.maxComputeWorkGroupInvocations, properties.limits.maxComputeSharedMemorySize,
           properties.limits.maxStorageBufferRange, properties.limits.maxPerStageDescriptorStorageBuffers);
    api->GetPhysicalDeviceMemoryProperties(ctx->physical, &memory);
    ctx->memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount && ctx->memory_type == UINT32_MAX; i++)
        if ((memory.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            ctx->memory_type = i;
    if (ctx->memory_type == UINT32_MAX)
        return fail("no host-visible coherent memory type");
    if (api->CreateDevice(ctx->physical, &(VkDeviceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
                .pQueuePriorities = &(float){1.0f}
            }
        }, NULL, &ctx->device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");
#define LOAD_DEVICE(name) api->name = (PFN_vk##name)api->GetDeviceProcAddr(ctx->device, "vk" #name); \
    if (!api->name) return fail("missing vk" #name);
    DEVICE_FUNCTIONS(LOAD_DEVICE)
#undef LOAD_DEVICE
    api->GetDeviceQueue(ctx->device, 0, 0, &ctx->queue);
    VkDescriptorSetLayoutBinding bindings[2];
    for (unsigned i = 0; i < 2; i++)
        bindings[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    if (api->CreateDescriptorSetLayout(ctx->device, &(VkDescriptorSetLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bindings
        }, NULL, &ctx->set_layout) != VK_SUCCESS ||
        api->CreatePipelineLayout(ctx->device, &(VkPipelineLayoutCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &ctx->set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, 0, 4}
        }, NULL, &ctx->pipeline_layout) != VK_SUCCESS ||
        api->CreateDescriptorPool(ctx->device, &(VkDescriptorPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 8, .poolSizeCount = 1,
            .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}
        }, NULL, &ctx->descriptor_pool) != VK_SUCCESS ||
        api->CreateCommandPool(ctx->device, &(VkCommandPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0
        }, NULL, &ctx->command_pool) != VK_SUCCESS ||
        api->CreateFence(ctx->device, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
                         NULL, &ctx->fence) != VK_SUCCESS)
        return fail("device object creation failed");
    return 0;
}

static void teardown(struct compute_context *ctx)
{
    struct compute_api *api = &ctx->api;
    if (ctx->device) {
        api->DeviceWaitIdle(ctx->device);
        if (ctx->fence)
            api->DestroyFence(ctx->device, ctx->fence, NULL);
        if (ctx->command_pool)
            api->DestroyCommandPool(ctx->device, ctx->command_pool, NULL);
        if (ctx->descriptor_pool)
            api->DestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
        if (ctx->pipeline_layout)
            api->DestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
        if (ctx->set_layout)
            api->DestroyDescriptorSetLayout(ctx->device, ctx->set_layout, NULL);
        api->DestroyDevice(ctx->device, NULL);
    }
    if (ctx->instance && api->DestroyInstance)
        api->DestroyInstance(ctx->instance, NULL);
}

int main(void)
{
    struct compute_context ctx;
    VkPipeline pipelines[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    int result;
    memset(&ctx, 0, sizeof ctx);
    result = setup(&ctx);
    if (!result)
        result = create_pipeline(&ctx, store_spirv, sizeof store_spirv, &pipelines[0]);
    if (!result)
        result = create_pipeline(&ctx, reduce_spirv, sizeof reduce_spirv, &pipelines[1]);
    if (!result)
        result = create_pipeline(&ctx, atomics_spirv, sizeof atomics_spirv, &pipelines[2]);
    if (!result)
        result = refuse_unsupported(&ctx);
    if (!result)
        result = run_store(&ctx, pipelines[0]);
    if (!result)
        result = run_chain(&ctx, pipelines[0], pipelines[1]);
    if (!result)
        result = run_atomics(&ctx, pipelines[2]);
    for (unsigned i = 0; i < 3; i++)
        if (pipelines[i])
            ctx.api.DestroyPipeline(ctx.device, pipelines[i], NULL);
    teardown(&ctx);
    if (!result && mxgpu_device_lost())
        result = fail("device lost");
    mxgpu_device_close();
    if (result) {
        printf("vk_compute FAIL: %s\n", g_failure);
        return 1;
    }
    printf("vk_compute PASS: refused a data-dependent loop, guarded store of %u of %u elements, store then barrier then shared-memory reduction of %u elements "
           "in %u workgroups, buffer and workgroup atomics over %u workgroups\n",
           STORE_COUNT, STORE_CAPACITY, CHAIN_COUNT, CHAIN_COUNT / REDUCE_GROUP, ATOMIC_GROUPS);
    return 0;
}
