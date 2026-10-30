# SPDX-License-Identifier: Apache-2.0
"""Build fissionkv._C (native core) alongside the pure-python package.

``pip install .`` compiles csrc/*.cpp into fissionkv/_C<abi>.so with g++/clang
(pybind11 headers from the build env; the module dlopen()s libcuda at
runtime, so no CUDA toolkit is needed to build).

Env knobs:
  FISSION_KV_SKIP_BUILD=1   skip the native build (pure-python install; tiering
                        falls back to torch copies, elastic is unavailable)
  FISSION_KV_CXX            compiler override
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent
SRC = sorted(f"csrc/{p.name}" for p in (ROOT / "csrc").glob("*.cpp"))


def _pybind_include() -> str:
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        out = subprocess.run([sys.executable, "-m", "pybind11", "--includes"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()


class NativeBuildPy(build_py):
    """build_py hook so ``pip install`` (no editable CMake path) compiles."""

    def run(self):
        super().run()
        if os.environ.get("FISSION_KV_SKIP_BUILD"):
            self.announce("FISSION_KV_SKIP_BUILD set: skipping native core")
            return
        self.announce(f"fissionkv: building native core from {len(SRC)} files")
        try:
            _build_native(self.build_lib)
        except Exception as exc:  # noqa: BLE001 - mirror graceful-degrade
            self.warn(f"fissionkv native build FAILED ({exc}); installing "
                      "pure-python only (native APIs unavailable)")


def _build_native(build_lib: str) -> None:
    cxx = os.environ.get("FISSION_KV_CXX") or "c++"
    py_inc = os.environ.get("FISSION_KV_PY_INC") or sysconfig.get_paths()["include"]
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = Path(build_lib) / "fissionkv" / f"_C{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx, "-std=c++17", "-O2", "-fPIC", "-shared",
        "-DTORCH_EXTENSION_NAME=_C",  # csrc/module.cpp uses PYBIND11_MODULE(name)")
        f"-I{py_inc}", f"-I{_pybind_include()}",
        f"-o{out}", *(str(ROOT / s_) for s_ in SRC), "-ldl", "-lpthread",
    ]
    subprocess.run(cmd, check=True)


setup(
    ext_modules=[
        # declared for `python setup.py build_ext` workflows; pip uses the
        # build_py hook above (plain Extension needs pybind11 at build time
        # too, harmless duplicate target)
        Extension(name="fissionkv._C", sources=SRC, optional=True,
                  include_dirs=[], libraries=["dl", "pthread"],
                  extra_compile_args=["-std=c++17", "-O2",
                                     "-DTORCH_EXTENSION_NAME=_C"]),
    ],
    cmdclass={"build_py": NativeBuildPy},
)
