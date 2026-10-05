<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: 2026 Zak Noble-Clarke -->
<!-- REUSE-IgnoreStart -->

# mesa-mxgpu

MX's Mesa fork for MXGPU in Linux guests.

- Repository: https://github.com/MXEmulation/mesa-mxgpu
- Upstream Mesa: https://mesa3d.org
- Fork support: support@coretools.app

The upstream source baseline is Mesa commit `503e8fe6077474c912876c8298fc7b839d90ceba`. Mesa's [README.rst](README.rst) describes the upstream project.

## Capabilities

The tested rendering paths support OpenGL 2.1 and OpenGL ES 2.0, including textures, indexed draws and instancing. The Vulkan ICD supports shader translation, rendering and readback for a graphics subset and is not conformant. OpenGL ES 3, WebGL 2 and WebGPU support are incomplete.

`GL_ANGLE_instanced_arrays` requires native instancing and attribute divisors. Its EGL fixture checks indexed widths, divisors, pixels and error recovery. Firefox 157 renders the browser fixture's valid scenes; its disabled active attribute check returns no error where `INVALID_OPERATION` is expected, so that check remains failed.

## Build and tests

```sh
meson setup build -Dgallium-drivers=mxgpu -Dvulkan-drivers=mxgpu -Dplatforms=x11,wayland -Dllvm=disabled
meson compile -C build
```

Meson fetches the pinned dependencies and uses the overlays in [subprojects/packagefiles/](subprojects/packagefiles/).

Enable `-Dmxgpu-tests=true` for the EGL/GLES and Vulkan rendering fixtures under [src/mxgpu/](src/mxgpu/). They require EGL/GLES and Vulkan development packages plus `glslangValidator`. Running them requires a compatible loaded `mxgpu` DRM driver and selected MXGPU libraries. The fixtures check actual rendering, pixels and error recovery.

## Dependencies

The exact public [Core](https://github.com/MXEmulation/mx-guest-core) and [Linux Common](https://github.com/MXEmulation/mx-guest-linux-common) commits are pinned in [subprojects/mx-guest-core.wrap](subprojects/mx-guest-core.wrap) and [subprojects/mx-guest-linux-common.wrap](subprojects/mx-guest-linux-common.wrap). They supply the protocol constants, codecs and DRM ioctl ABI.

## Licences and contributions

MX additions are MIT. Mesa contains multiple licences; the fork as a whole is not MIT. See [docs/license.rst](docs/license.rst), [licenses/](licenses/) and each file's SPDX and copyright notices.

Source distributions retain all licence files, attribution files and per-file notices. Binary packages retain the applicable licences and attributions and meet their corresponding source obligations. Modified upstream files retain their existing notices and carry a dated MX change notice.

Contributions require a matching author `Signed-off-by` under [DCO 1.1](https://developercertificate.org). AI contributions follow [AGENTS.md](AGENTS.md) and carry `Generated-by: LLM` or `Assisted-by: LLM`, as applicable.

<!-- REUSE-IgnoreEnd -->
