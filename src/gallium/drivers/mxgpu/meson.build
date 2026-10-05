# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Zak Noble-Clarke

libmxgpu = static_library(
  'mxgpu',
  files('mxgpu_screen.c', 'mxgpu_link.c', '../../../mxgpu/mxgpu_compiler.c',
        '../../../mxgpu/mxgpu_scene.c'),
  include_directories : [
    inc_include, inc_src, inc_gallium, inc_gallium_aux, inc_gallium_drivers,
    include_directories('../../../mxgpu'),
  ],
  c_args : c_msvc_compat_args,
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil, mxgpu_core_dep, mxgpu_linux_common_dep],
)

driver_mxgpu = declare_dependency(
  compile_args : '-DGALLIUM_MXGPU',
  link_whole : libmxgpu,
)
