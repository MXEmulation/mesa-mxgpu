// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) out vec4 color;
void main()
{
    vec2 size = vec2(textureSize(source, 0));
    color = texture(source, (floor(gl_FragCoord.xy) * (size / 8.0) + vec2(300.5, 700.5)) / size);
}
