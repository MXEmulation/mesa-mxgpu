# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Zak Noble-Clarke

MXGPU_BUILD ?= ../../build
MESON ?= meson
MESON_OPTIONS ?= -Dgallium-drivers=mxgpu -Dvulkan-drivers=mxgpu -Dllvm=disabled -Dplatforms=x11,wayland -Dglx=dri -Degl=enabled -Dgbm=enabled -Dglvnd=enabled -Dvideo-codecs= -Dgallium-va=disabled
PROGRAMS := gl_scene gl_stress vk_scene vk_stress vk_system vk_client egl_client
REQUESTED_PROGRAMS := $(filter $(PROGRAMS),$(MAKECMDGOALS))
BUILD_PROGRAMS := $(if $(REQUESTED_PROGRAMS),$(REQUESTED_PROGRAMS),$(PROGRAMS))
MXGPU_BUILD_JOBS ?= 2

.PHONY: all clean configure validation-build $(PROGRAMS)

all: $(PROGRAMS)

configure:
	if test -f "$(MXGPU_BUILD)/build.ninja"; then $(MESON) configure "$(MXGPU_BUILD)" -Dmxgpu-tests=true; else $(MESON) setup "$(MXGPU_BUILD)" ../.. $(MESON_OPTIONS) -Dmxgpu-tests=true; fi

validation-build: configure
	$(MESON) compile -C "$(MXGPU_BUILD)" -j $(MXGPU_BUILD_JOBS) $(BUILD_PROGRAMS)

$(PROGRAMS): validation-build
	cp "$(MXGPU_BUILD)/src/mxgpu/$@" "$@"

clean:
	rm -f $(PROGRAMS)
