/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Zak Noble-Clarke */
#ifndef MXGPU_PUBLIC_H
#define MXGPU_PUBLIC_H

struct sw_winsys;
struct pipe_screen;

struct pipe_screen *mxgpu_create_screen(struct sw_winsys *winsys);
struct pipe_screen *mxgpu_drm_screen_create(int fd);

#endif
