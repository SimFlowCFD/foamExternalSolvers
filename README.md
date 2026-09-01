# foamExternalSolvers

foamExternalSolvers is a plugin for OpenFOAM collecting interface modules
towards third-party linear algebra solvers.
AmgX4Foam is the only supported package.
This library works both with OpenFOAM vanilla and SPUMA,
the GPU porting of OpenFOAM released by CINECA.

## Building

```
./Allwmake [-cuda [N]] [-clean] [wmake options]
```

The library is installed as `libAmgX4Foam` into `FOAM_USER_LIBBIN`
(or the location given by `-prefix=...` / `FOAM_MODULE_LIBBIN`).

| Option | Meaning |
| --- | --- |
| `-cuda [N]`, `-cu [N]` | Also build the CUDA ldu-to-csr executor (`.cu` sources) for compute capability N (80, 86, 90, 120, ...; a single digit 5-9 means N0). Default: `NVARCH` or 80 |
| `-clean`, `-cl` | Remove the library, objects and generated files |
| `-j N`, `-prefix=...`, ... | Standard `wmake`/`Allwmake` options |

### Locating AmgX and CUDA

| Variable | Purpose |
| --- | --- |
| `AMGX_ARCH_PATH` | AmgX install prefix containing `include/` and `lib/`. SPUMA resolves it through `etc/config.sh/amgx` and derives `AMGX_INC`/`AMGX_LIB`; elsewhere set `AMGX_INC` and `AMGX_LIB` directly |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit prefix for clang-based toolchains on Linux (headers in `include/`, `libcudart` in `lib64/`). Not needed for nvc++, nor when the toolkit is on the compiler's default search paths |
| `CUDA_PATH` (Windows) | CUDA toolkit prefix; SPUMA's wmake derives the space-free `CUDA_PATH_SHORT` from it |
| `CUDA_CLANG` | clang++ used for the `.cu` sources on clang-based toolchains (default: `clang++` from `PATH`). On Windows it must be clang 20 or newer (the AdaptiveCpp-bundled clang is not) and the build stops otherwise; a path with spaces is 8.3-shortened by `etc/config.sh/amgx` |
| `NVARCH` | Compute capability for `-cuda` when no value is given on the command line (default 80) |

The build is skipped (exit 0) when AmgX is not found, so the module can be
part of an unconditional `Allwmake-modules` sweep.

At run time `libamgxsh.so` must be found by the dynamic loader: the library
is not linked with an rpath, so add `$AMGX_LIB` (SPUMA: `$AMGX_ARCH_PATH/lib`)
to `LD_LIBRARY_PATH` before running a case.

Example, SPUMA `Sycl` build on Linux with the system clang and a CUDA 12.6
toolkit:

```
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_CLANG=/usr/lib/llvm-19/bin/clang++
export AMGX_ARCH_PATH=/path/to/amgx-2.5.0     # include/ lib/
./Allwmake -cuda 86
export LD_LIBRARY_PATH=$AMGX_ARCH_PATH/lib:$LD_LIBRARY_PATH
```

### Toolchains

- **nvc++** (`WM_COMPILER=Nvidia*`, the upstream/SPUMA-Linux configuration):
  the compiler's `-cuda` mode provides the CUDA runtime. The rules bundled in
  `wmake/` add `-gpu=ccNN` and the `nvcc` rule for `.cu` sources when
  `-cuda` is given.
- **clang-based compilers, including AdaptiveCpp** (SPUMA `Sycl` builds on
  Linux and native Windows): the CUDA runtime headers and `libcudart` are
  taken from the toolkit prefix above. With `-cuda` the `.cu` sources are
  compiled by `CUDA_CLANG` (default `clang++` from `PATH`) in `-x cuda` mode
  and linked into the same library; that clang must support the CUDA toolkit
  in use, which the clang bundled with an AdaptiveCpp install may not
  (SPUMA on Windows: LLVM 20 for CUDA 12.x; on Linux the Ubuntu 24.04
  clang 19 builds them against CUDA 12.6).

The `-cuda` executor keeps the CSR arrays and the per-solve coefficient
permutation on the GPU (CUB sort/scan and CUDA kernels). Without it the
conversion runs on the host; with SPUMA the arrays still live in managed
memory, so every matrix update migrates them to the host and back.

Native Windows builds use OpenFOAM's dummy Pstream: the library is compiled
with `AMGX4FOAM_NO_MPI` and parallel runs are rejected with a FatalError.

### Parallel runs

One AmgX instance is created per GPU. When several ranks of a node share a
GPU, one of them owns the AmgX instance and the others hand their matrix
partitions over through CUDA IPC (consolidation). That path needs the
`-cuda` executor: the CPU executor does not implement the IPC copy
(`cpuCsrMatrixExecutor::offsetCopy`) and stops with a FatalError. With one
rank per GPU no consolidation takes place and AmgX receives the partitions
directly (`AMGX_matrix_upload_distributed`).

## Usage

At run time the loader must find the AmgX shared library (`libamgxsh.so` /
`amgxsh.dll`), the CUDA runtime and `libAmgX4Foam` itself. With SPUMA,
`etc/config.sh/amgx` adds the AmgX library directory to `LD_LIBRARY_PATH`
(`PATH` on Windows) when sourced from the OpenFOAM environment, or on demand:

```
eval "$(foamEtcFile -sh -config amgx -- -force)"
```

On native Windows also put `FOAM_USER_LIBBIN` (or the `-prefix` location)
and the CUDA `bin` directory on `PATH`. Elsewhere add `AMGX_LIB` to the
library path yourself.

Load the library from `system/controlDict` and select the solver per field
in `system/fvSolution`:

```
libs (AmgX4Foam);
```

```
p
{
    solver          AmgX;
    matrixType      CSR;
    dataLocation    device;     // or host
    mode            dDDI;
    allowHostFallback false;    // optional, see below
    AmgXconfig
    {
        // AmgX configuration, see the AmgX reference manual
    }
}
```

`dataLocation device` hands the CSR arrays to AmgX without a copy and
requires them to live in CUDA device or managed memory (SPUMA GPU builds
allocate them through the memory pool). `dataLocation host` copies them to
the device first and works with any allocation.

At initialisation the library checks that AmgX uses the GPU the application
computes on. With SPUMA this is the SYCL (or native CUDA) device: in serial
runs AmgX is placed on that device, in parallel runs a mismatch between the
rank's AmgX device and its compute device is a FatalError. When the
application does not compute on a CUDA device at all (host OpenMP or HIP
backend) the run stops as well, because every solve would copy the matrix
and vectors across host and device. Set `allowHostFallback true;` to accept
that (typically for debugging); `dataLocation` is then forced to `host`.
