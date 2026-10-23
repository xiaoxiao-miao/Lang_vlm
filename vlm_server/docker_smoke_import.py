#!/usr/bin/env python3
"""Build/runtime helper: nvidia lib path + import smoke test."""

from __future__ import annotations

import glob
import os
import sys


def nvidia_lib_paths(*, include_cu13: bool = True) -> list[str]:
    """Collect pip nvidia wheel lib dirs.

    vLLM 0.26 wheels link against libcudart.so.13 — keep cu13 on the path for
    import on new stacks. Cloud535 (cu121) should set VLM_ALLOW_CUDART13=0.
    """
    roots = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
    ]
    paths: list[str] = []
    for root in roots:
        paths.extend(
            glob.glob(os.path.join(root, "python*", "site-packages", "nvidia", "*", "lib"))
        )
        paths.extend(
            glob.glob(os.path.join(root, "python*", "site-packages", "nvidia", "cu*", "lib"))
        )

    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if not os.path.isdir(p) or p in seen:
            continue
        norm = p.replace("\\", "/")
        if not include_cu13 and ("/nvidia/cu13/" in norm or "/nvidia/cuda_runtime/" in norm):
            so13 = glob.glob(os.path.join(p, "libcudart.so.13*"))
            if so13:
                continue
        seen.add(p)
        out.append(p)
    return out


def system_cuda_lib_paths() -> list[str]:
    return [p for p in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib") if os.path.isdir(p)]


def print_libpath() -> None:
    include_cu13 = os.environ.get("VLM_ALLOW_CUDART13", "1") != "0"
    parts = system_cuda_lib_paths() + nvidia_lib_paths(include_cu13=include_cu13)
    print(":".join(dict.fromkeys(parts)))


def smoke() -> None:
    include_cu13 = os.environ.get("VLM_ALLOW_CUDART13", "1") != "0"
    require_fa = os.environ.get("VLM_REQUIRE_FLASH_ATTN", "1") != "0"
    libpath = ":".join(system_cuda_lib_paths() + nvidia_lib_paths(include_cu13=include_cu13))
    if libpath:
        os.environ["LD_LIBRARY_PATH"] = (
            libpath + ((":" + os.environ["LD_LIBRARY_PATH"]) if os.environ.get("LD_LIBRARY_PATH") else "")
        )

    import torch
    import torchaudio
    import vllm  # noqa: F401

    fa_ver = "skipped"
    if require_fa:
        import flash_attn  # noqa: F401

        fa_ver = flash_attn.__version__
    else:
        try:
            import flash_attn  # noqa: F401

            fa_ver = flash_attn.__version__
        except Exception as e:
            fa_ver = f"unavailable ({e})"

    print(
        f"smoke ok: torch={torch.__version__} cuda={torch.version.cuda} "
        f"torchaudio={torchaudio.__version__} flash_attn={fa_ver} vllm_import=ok"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--print-libpath":
        print_libpath()
    else:
        smoke()
