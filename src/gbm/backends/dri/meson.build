# Modified by Zak Noble-Clarke for MX, 2026-09-28: record $ORIGIN/.. as the RUNPATH for libgallium and libgbm.
# Modified by Zak Noble-Clarke for MX, 2026-10-05: retain relative runtime library lookup on Linux.

deps_gbm_dri = []
flags_gbm_dri = []
gbm_dri_link_args = [ld_args_gc_sections]
gbm_dri_link_depends = []
if host_machine.system() == 'linux'
  gbm_dri_link_with = [libloader]
  gbm_dri_link_args += [libgallium_dri.full_path()]
  gbm_dri_link_depends += [libgallium_dri]
  if not get_option('libgbm-external')
    gbm_dri_link_depends += [libgbm]
  endif
else
  gbm_dri_link_with = [libloader, libgallium_dri]
endif

# TODO: fix includes to delete this
incs_gbm_dri = [inc_mesa]

if with_platform_wayland
  deps_gbm_dri += dep_wayland_server
  if with_wayland_bind_display
    flags_gbm_dri += ['-DHAVE_BIND_WL_DISPLAY']
    incs_gbm_dri += inc_wayland_drm
  endif
endif

shared_library(
  'dri_gbm',
  files('gbm_dri.c', 'gbm_driint.h'),
  include_directories : [inc_gallium, incs_gbm_dri, inc_loader, inc_st_dri, inc_gallium_aux],
  c_args : [flags_gbm_dri],
  link_args : gbm_dri_link_args,
  link_depends : gbm_dri_link_depends,
  link_with : gbm_dri_link_with,
  build_rpath : host_machine.system() == 'linux' ? '$ORIGIN/..' : '',
  install_rpath : host_machine.system() == 'linux' ? '$ORIGIN/..' : '',
  dependencies : [deps_gbm_dri, dep_dl, dep_gbm, dep_libdrm, idep_mesautil, idep_xmlconfig],
  gnu_symbol_visibility : 'hidden',
  install : true,
  install_dir: join_paths(get_option('libdir'), 'gbm'),
  name_prefix : '',
)

devenv.set('GBM_BACKENDS_PATH', meson.current_build_dir())
