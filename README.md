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
| `CUDA_PATH` (Windows) | CUDA toolkit prefix; SPUMA derives the space-free `CUDA_PATH_SHORT` from it |

The build is skipped (exit 0) when AmgX is not found, so the module can be
part of an unconditional `Allwmake-modules` sweep.

### Toolchains

- **nvc++** (`WM_COMPILER=Nvidia*`, the upstream/SPUMA-Linux configuration):
  the compiler's `-cuda` mode provides the CUDA runtime. The rules bundled in
  `wmake/` add `-gpu=ccNN` and the `nvcc` rule for `.cu` sources when
  `-cuda` is given.
- **clang-based compilers, including AdaptiveCpp** (SPUMA `Sycl` builds on
  Linux and native Windows): the CUDA runtime headers and `libcudart` are
  taken from the toolkit prefix above. The `.cu` executor is not yet
  available for these toolchains (the CPU executor is used).

Native Windows builds use OpenFOAM's dummy Pstream: the library is compiled
with `AMGX4FOAM_NO_MPI` and parallel runs are rejected with a FatalError.

## Usage

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
