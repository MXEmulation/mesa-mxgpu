// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Zak Noble-Clarke
#version 450
layout(location = 0) out vec4 first;
layout(location = 1) out vec4 second;
layout(location = 2) out vec4 third;
void main()
{
    first = vec4(1.0, 0.0, 0.0, 1.0);
    second = vec4(0.0, 0.5, 0.0, 1.0);
    third = vec4(0.0, 0.0, 1.0, 0.5);
}
