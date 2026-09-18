<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: 2026 Zak Noble-Clarke -->
<!-- REUSE-IgnoreStart -->

# mesa-mxgpu

MX's Mesa fork for MXGPU in Linux guests.

- Repository: https://github.com/MXEmulation/mesa-mxgpu
- Upstream Mesa: https://mesa3d.org
- Fork support: support@coretools.app

The upstream source baseline is Mesa commit `503e8fe6077474c912876c8298fc7b839d90ceba`. Mesa's [README.rst](README.rst) describes the upstream project.

This revision contains fork documentation and dependency wraps. The MXGPU driver and Vulkan ICD are not implemented.

## Dependencies

The exact public [Core](https://github.com/MXEmulation/mx-guest-core) and [Linux Common](https://github.com/MXEmulation/mx-guest-linux-common) commits are pinned in [subprojects/mx-guest-core.wrap](subprojects/mx-guest-core.wrap) and [subprojects/mx-guest-linux-common.wrap](subprojects/mx-guest-linux-common.wrap). Those dependency revisions contain repository metadata only; no codecs or DRM ioctl ABI implementation is present.

## Licences and contributions

MX additions are MIT. Mesa contains multiple licences; the fork as a whole is not MIT. See [docs/license.rst](docs/license.rst), [licenses/](licenses/) and each file's SPDX and copyright notices.

Source distributions retain all licence files, attribution files and per-file notices. Binary packages retain the applicable licences and attributions and meet their corresponding source obligations. Modified upstream files retain their existing notices and carry a dated MX change notice.

Contributions require a matching author `Signed-off-by` under [DCO 1.1](https://developercertificate.org). AI contributions follow [AGENTS.md](AGENTS.md) and carry `Generated-by: LLM` or `Assisted-by: LLM`, as applicable.

<!-- REUSE-IgnoreEnd -->
