// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(location = 0) in vec4 position_uv;
layout(location = 0) out vec2 uv;
void main()
{
    gl_Position = vec4(position_uv.xy, 0.0, 1.0);
    uv = position_uv.zw;
}
