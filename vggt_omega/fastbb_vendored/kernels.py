# vendored from fastforks/fastbb: fastbb/kernels.py @ a61ba9c -- do not edit here, see fastbb/VENDORED.md
# loader copied from exp 004 mono_fast/kernels.py (originally RAFT-Stereo exp 002)
"""Custom CUDA kernels of fastbb/fastbb_kernels.cu for PyTorch:
compiled once (shipped fatbin in prebuilt/, else nvcc of CUDA_HOME, cached in ~/.cache/fastbb-torch),
loaded through the driver API, launched on torch's current stream. Size-independent: no per-shape compilation.
"""
import ctypes
import hashlib
import os
import struct
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastbb_kernels.cu")
_PREBUILT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fastbb-torch")

# p: pointer (tensor or int address), i: int32, l: int64, f: float32
SIGNATURES = {
    "flash64_r1_hh": "pppp" + "ii" + "lii" * 4 + "f",
    "flash64_r1_fh": "pppp" + "ii" + "lii" * 4 + "f",
    "flash64_r1_ff": "pppp" + "ii" + "lii" * 4 + "f",
    "flash64_r2_hh": "pppp" + "ii" + "lii" * 4 + "f",
    "flash64_r2_fh": "pppp" + "ii" + "lii" * 4 + "f",
    "flash64_r2_ff": "pppp" + "ii" + "lii" * 4 + "f",
}
DYN_SMEM = {}
_CTYPE = {"d": ctypes.c_double, "i": ctypes.c_int32, "l": ctypes.c_int64, "f": ctypes.c_float, "u": ctypes.c_uint32}


def _srctag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build_prebuilt(archs=(80, 86, 87, 89, 90, 110, 120)):
    """Ship the kernels for the current source: one fatbin (SASS for archs + compute_80 PTX) in fastbb/prebuilt/."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("fastbb_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"fastbb_kernels_{_srctag()}.fatbin")
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + ["-gencode=arch=compute_80,code=compute_80"]
    subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "-fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                    *gencode, f"-I{cuda_home}/include", "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)
    return out


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        src = open(_SRC, "rb").read()
        tag = hashlib.sha1(src + f"sm_{major}{minor}".encode()).hexdigest()[:16]
        cubin = os.path.join(_PREBUILT, f"fastbb_kernels_{tag}.cubin")
        fat = os.path.join(_PREBUILT, f"fastbb_kernels_{_srctag()}.fatbin")   # shipped: sm_80/86/87/89/90/110/120 SASS + PTX
        if os.path.exists(fat):
            cubin = fat
        if not os.path.exists(cubin):
            cubin = os.path.join(_CACHE, f"fastbb_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            os.makedirs(_CACHE, exist_ok=True)
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "-cubin", "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                            f"-arch=sm_{major}{minor}", f"-I{cuda_home}/include", "-o", cubin + ".tmp", _SRC], check=True)
            os.replace(cubin + ".tmp", cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(cubin, "rb").read()
        _check(self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData")
        self.funcs = {}
        self._packers = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()), f"cuModuleGetFunction {name}")
            self.funcs[name] = f
            smem = DYN_SMEM.get(name, 0)
            if smem > 48 * 1024:  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
                _check(self.lib.cuFuncSetAttribute(f, 8, smem), f"cuFuncSetAttribute {name}")

    @staticmethod
    def linear(n):
        """Rust's Launch::linear: 256 threads, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args):
        if torch.jit.is_tracing():
            raise RuntimeError("raw fastbb kernels cannot be traced; export the portable model")
        grid, block = cfg
        sig = SIGNATURES[name]
        assert len(sig) == len(args), f"{name}: expected {len(sig)} args, got {len(args)}"
        holders = []
        for t, a in zip(sig, args):
            if t == "p":
                holders.append(ctypes.c_void_p(a.data_ptr() if isinstance(a, torch.Tensor) else (int(a) or None)))
            else:
                holders.append(_CTYPE[t](a))
        arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = torch.cuda.current_stream().cuda_stream
        _check(self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, DYN_SMEM.get(name, 0), stream, arr, None), f"launch {name}")


    def launch_packed(self, name, grid, block, *args):
        """Same as launch, but the arguments go as one packed buffer (CU_LAUNCH_PARAM_BUFFER_POINTER, C alignment via
        struct '@'): one struct.pack instead of one ctypes object per argument (~3x less CPU per launch)."""
        st = self._packers.get(name)
        if st is None:
            fmt = "@" + "".join({"p": "Q", "i": "i", "l": "q", "f": "f", "u": "I", "d": "d"}[t] for t in SIGNATURES[name])
            size = struct.calcsize(fmt)  # exact parameter size: the driver rejects a tail-padded size (error 701)
            buf = ctypes.create_string_buffer(size + (-size) % 8)
            sz = ctypes.c_size_t(size)
            extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(sz), 0)
            st = self._packers[name] = (struct.Struct(fmt), buf, sz, extra)
        if torch.jit.is_tracing():
            raise RuntimeError("raw fastbb kernels cannot be traced; export the portable model")
        S, buf, sz, extra = st
        S.pack_into(buf, 0, *[x.data_ptr() if isinstance(x, torch.Tensor) else x for x in args])
        stream = torch.cuda.current_stream().cuda_stream
        r = self.lib.cuLaunchKernel(self.funcs[name], grid[0], grid[1], grid[2], block[0], block[1], block[2], DYN_SMEM.get(name, 0), stream, None, extra)
        if r:
            _check(r, f"launch {name}")


if __name__ == "__main__":
    print(build_prebuilt())
