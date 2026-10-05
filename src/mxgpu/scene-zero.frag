// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(location = 0) out vec4 color;
void main()
{
    color = texture(source_texture, vec2(0.0));
}
