# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Zak Noble-Clarke

mxgpu_protocol_deps = [mxgpu_core_dep, mxgpu_linux_common_dep]
if with_mxgpu_vk
  mxgpu_vk_platform_deps = []
  mxgpu_vk_platform_args = []
  if with_platform_wayland
    mxgpu_vk_platform_deps += dep_wayland_client
    mxgpu_vk_platform_args += '-DVK_USE_PLATFORM_WAYLAND_KHR'
  endif
  mxgpu_vk_deps = [idep_nir, idep_vtn, idep_mesautil, idep_vulkan_util] + mxgpu_protocol_deps + mxgpu_vk_platform_deps
  mxgpu_vk_impl = static_library(
    'mxgpu_vk_impl',
    files('mxgpu_vk.c', 'mxgpu_compiler.c', 'mxgpu_scene.c', '../gallium/drivers/mxgpu/mxgpu_link.c'),
    include_directories : [inc_include, inc_src, include_directories('.')],
    dependencies : mxgpu_vk_deps,
    c_args : mxgpu_vk_platform_args,
    gnu_symbol_visibility : 'hidden',
    pic : true,
  )
  libvulkan_mxgpu = shared_library(
    'vulkan_mxgpu',
    link_whole : mxgpu_vk_impl,
    dependencies : mxgpu_vk_deps,
    gnu_symbol_visibility : 'hidden',
    install : true,
  )
  install_data(
    'mxgpu_icd.json',
    install_dir : join_paths(get_option('datadir'), 'vulkan/icd.d'),
  )
endif

if get_option('mxgpu-tests')
  if with_gallium_mxgpu
    mxgpu_egl_deps = [dependency('egl'), dependency('glesv2')]
    foreach validation : ['gl_scene', 'gl_stress']
      executable(validation, files(validation + '.c', 'egl_client.c'),
                 dependencies : mxgpu_egl_deps)
    endforeach
    executable('egl_client', 'egl_client.c', c_args : '-DMXGPU_EGL_CLIENT_MAIN',
               dependencies : mxgpu_egl_deps)
    executable('egl_instancing', 'egl_instancing.c', dependencies : mxgpu_egl_deps)
    executable('gl_textures', 'gl_textures.c', dependencies : mxgpu_egl_deps)
  endif
  if with_mxgpu_vk
    glslang = find_program('glslangValidator', required : true)
    scene_shaders = []
    foreach shader : [
      ['scene.vert', 'scene-vert.h', 'scene_vertex'],
      ['scene.frag', 'scene-frag.h', 'scene_fragment'],
      ['scene-zero.frag', 'scene-zero-frag.h', 'scene_zero_fragment'],
    ]
      scene_shaders += custom_target(
        shader[1], input : shader[0], output : shader[1],
        command : [glslang, '-V', '--target-env', 'vulkan1.0', '--vn', shader[2], '-o', '@OUTPUT@', '@INPUT@'],
      )
    endforeach
    foreach validation : ['vk_scene', 'vk_stress']
      executable(validation, files(validation + '.c', 'vk_frame.c') + scene_shaders,
                 include_directories : include_directories('.'),
                 link_with : mxgpu_vk_impl, dependencies : mxgpu_vk_deps)
    endforeach
    foreach validation : ['cs_smoke', 'vk_compute']
      executable(validation, files(validation + '.c'),
                 include_directories : include_directories('.'),
                 link_with : mxgpu_vk_impl, dependencies : mxgpu_vk_deps)
    endforeach
    vulkan_loader = dependency('vulkan')
    foreach validation : ['vk_client', 'vk_system']
      executable(validation, files(validation + '.c', 'vk_frame.c') + scene_shaders,
                 include_directories : include_directories('.'),
                 dependencies : [vulkan_loader] + mxgpu_protocol_deps + (validation == 'vk_client' ? [dep_dl] : []))
    endforeach
    core11_shaders = []
    foreach shader : [
      ['core11-copy.comp', 'core11-copy-comp.h', 'core11_copy'],
      ['core11-alias.comp', 'core11-alias-comp.h', 'core11_alias'],
      ['core11-uniform.comp', 'core11-uniform-comp.h', 'core11_uniform'],
      ['core11.vert', 'core11-vert.h', 'core11_vertex'],
      ['core11.frag', 'core11-frag.h', 'core11_fragment'],
    ]
      core11_shaders += custom_target(
        shader[1], input : shader[0], output : shader[1],
        command : [glslang, '-V', '--target-env', 'vulkan1.1', '--vn', shader[2], '-o', '@OUTPUT@', '@INPUT@'],
      )
    endforeach
    executable('vk_core11', files('vk_core11.c') + core11_shaders,
               dependencies : [vulkan_loader])
    features_shaders = []
    foreach shader : [
      ['features.vert', 'features-vert.h', 'features_vertex'],
      ['features-mrt.frag', 'features-mrt-frag.h', 'features_mrt'],
      ['features-sampler.frag', 'features-sampler-frag.h', 'features_sampler'],
      ['features-color.frag', 'features-color-frag.h', 'features_color'],
    ]
      features_shaders += custom_target(
        shader[1], input : shader[0], output : shader[1],
        command : [glslang, '-V', '--target-env', 'vulkan1.1', '--vn', shader[2], '-o', '@OUTPUT@', '@INPUT@'],
      )
    endforeach
    executable('vk_features', files('vk_features.c') + features_shaders,
               dependencies : [vulkan_loader, cc.find_library('m', required : false)])
    webgpu_shaders = [features_shaders[0]]
    foreach shader : [
      ['webgpu-sixteen.frag', 'webgpu-sixteen-frag.h', 'webgpu_sixteen'],
      ['webgpu-gradient.frag', 'webgpu-gradient-frag.h', 'webgpu_gradient'],
      ['webgpu-probe.frag', 'webgpu-probe-frag.h', 'webgpu_probe'],
      ['webgpu-float.frag', 'webgpu-float-frag.h', 'webgpu_float'],
      ['webgpu-unorm.frag', 'webgpu-unorm-frag.h', 'webgpu_unorm'],
    ]
      webgpu_shaders += custom_target(
        shader[1], input : shader[0], output : shader[1],
        command : [glslang, '-V', '--target-env', 'vulkan1.1', '--vn', shader[2], '-o', '@OUTPUT@', '@INPUT@'],
      )
    endforeach
    executable('vk_webgpu', files('vk_webgpu.c') + webgpu_shaders,
               dependencies : [vulkan_loader, cc.find_library('m', required : false)])
  endif
endif
