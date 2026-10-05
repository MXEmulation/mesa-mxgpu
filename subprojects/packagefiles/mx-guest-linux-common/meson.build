# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Zak Noble-Clarke

project('mx-guest-linux-common', 'c', license : 'MIT', default_options : ['c_std=c11'])

fs = import('fs')
common_sources = []
foreach entry : fs.read('sources.list').strip().split('\n')
  if entry.startswith('src/') and entry.endswith('.c')
    common_sources += files(entry)
  endif
endforeach
assert(common_sources.length() > 0, 'Linux Common sources.list must name its implementation sources')

common_library = static_library('mx-guest-linux-common', common_sources, pic : true,
                                include_directories : include_directories('include', 'src'),
                                gnu_symbol_visibility : 'hidden')
mxgpu_linux_common_dep = declare_dependency(
  include_directories : include_directories('include', 'src'),
  link_with : common_library,
)
