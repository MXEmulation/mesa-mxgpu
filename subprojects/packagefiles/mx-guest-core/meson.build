# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Zak Noble-Clarke

project('mx-guest-core', 'c', license : 'MIT', default_options : ['c_std=c11'])

fs = import('fs')
core_sources = []
foreach entry : fs.read('sources.list').strip().split('\n')
  if entry.startswith('src/') and entry.endswith('.c')
    core_sources += files(entry)
  endif
endforeach
assert(core_sources.length() > 0, 'Core sources.list must name its implementation sources')

core_library = static_library('mx-guest-core', core_sources, pic : true,
                              include_directories : include_directories('include', 'src'),
                              gnu_symbol_visibility : 'hidden')
mxgpu_core_dep = declare_dependency(
  include_directories : include_directories('include', 'src'),
  link_with : core_library,
)
