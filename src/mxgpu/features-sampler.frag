// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(set = 0, binding = 0) uniform texture2D image;
layout(set = 0, binding = 1) uniform sampler point;
layout(location = 0) out vec4 color;
void main()
{
    vec2 size = vec2(textureSize(sampler2D(image, point), 0));
    color = texture(sampler2D(image, point), gl_FragCoord.xy / (4.0 * size));
}
