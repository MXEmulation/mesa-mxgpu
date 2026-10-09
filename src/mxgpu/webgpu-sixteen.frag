// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(set = 0, binding = 0) uniform sampler2D t0;
layout(set = 0, binding = 1) uniform sampler2D t1;
layout(set = 0, binding = 2) uniform sampler2D t2;
layout(set = 0, binding = 3) uniform sampler2D t3;
layout(set = 0, binding = 4) uniform sampler2D t4;
layout(set = 0, binding = 5) uniform sampler2D t5;
layout(set = 0, binding = 6) uniform sampler2D t6;
layout(set = 0, binding = 7) uniform sampler2D t7;
layout(set = 0, binding = 8) uniform sampler2D t8;
layout(set = 0, binding = 9) uniform sampler2D t9;
layout(set = 0, binding = 10) uniform sampler2D t10;
layout(set = 0, binding = 11) uniform sampler2D t11;
layout(set = 0, binding = 12) uniform sampler2D t12;
layout(set = 0, binding = 13) uniform sampler2D t13;
layout(set = 0, binding = 14) uniform sampler2D t14;
layout(set = 0, binding = 15) uniform sampler2D t15;
layout(location = 0) out vec4 color;
#define PICK(i) (texture(t##i, uv) * clamp(1.0 - abs(x - float(i)), 0.0, 1.0))
void main()
{
    float x = floor(gl_FragCoord.x);
    vec2 uv = vec2(0.5);
    color = PICK(0) + PICK(1) + PICK(2) + PICK(3) + PICK(4) + PICK(5) + PICK(6) + PICK(7) +
            PICK(8) + PICK(9) + PICK(10) + PICK(11) + PICK(12) + PICK(13) + PICK(14) + PICK(15);
}
