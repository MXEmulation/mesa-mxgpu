// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(set = 0, binding = 0) uniform Colors { vec4 values[4096]; } colors;
layout(location = 0) out vec4 color;
void main()
{
    color = colors.values[4095];
}
