// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(location = 0) out vec4 color;
void main()
{
    vec2 p = floor(gl_FragCoord.xy);
    color = vec4(mod(p.x, 256.0), mod(p.y, 256.0), floor(p.x / 256.0) * 8.0, floor(p.y / 256.0) * 8.0) / 255.0;
}
