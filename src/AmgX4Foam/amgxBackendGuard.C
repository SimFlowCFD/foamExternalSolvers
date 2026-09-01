/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2026 SPUMA contributors
-------------------------------------------------------------------------------
License
    This file is part of foamExternalSolvers.

    foamExternalSolvers is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    foamExternalSolvers is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with foamExternalSolvers. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "amgxBackendGuard.H"
#include "error.H"
#include "messageStream.H"

#include <cuda_runtime_api.h>

#ifdef have_sycl
#include <sycl/sycl.hpp>
#include "syclDeviceInit.H"     // getSyclQueue()
#endif

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

namespace
{

// "CUDA device N 'name' [PCI bus id]" for diagnostics
std::string cudaDeviceLabel(int dev)
{
    std::string label = "CUDA device " + std::to_string(dev);

    cudaDeviceProp prop{};
    if (dev >= 0 && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        label += " '" + std::string(prop.name) + "'";

        char bus[32] = {0};
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) == cudaSuccess)
        {
            label += " [" + std::string(bus) + "]";
        }
    }
    cudaGetLastError();     // clear a possible sticky error

    return label;
}

} // End anonymous namespace


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

Foam::amgxComputeBackend Foam::amgxQueryComputeBackend()
{
    amgxComputeBackend info;

#if defined(have_sycl)

    const sycl::device dev = getSyclQueue().get_device();
    info.deviceName = dev.get_info<sycl::info::device::name>();

    #if __has_include(<hipSYCL/runtime/device_id.hpp>)
    // AdaptiveCpp: the runtime device id carries the backend and, for the
    // CUDA backend, the CUDA runtime ordinal of the device
    const hipsycl::rt::device_id id = dev.AdaptiveCpp_device_id();
    switch (id.get_backend())
    {
        case hipsycl::rt::backend_id::cuda:
            info.name = "cuda";
            info.cudaDevice = id.get_id();
            break;
        case hipsycl::rt::backend_id::hip:
            info.name = "hip";
            break;
        case hipsycl::rt::backend_id::level_zero:
            info.name = "level_zero";
            break;
        case hipsycl::rt::backend_id::ocl:
            info.name = "ocl";
            break;
        case hipsycl::rt::backend_id::omp:
            info.name = "omp";
            break;
    }
    #else
    info.name = "sycl";
    #endif

#elif defined(have_cuda)

    info.name = "cuda";
    if (cudaGetDevice(&info.cudaDevice) == cudaSuccess)
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, info.cudaDevice) == cudaSuccess)
        {
            info.deviceName = prop.name;
        }
    }
    else
    {
        info.cudaDevice = -1;
    }
    cudaGetLastError();

#elif defined(have_hip)

    info.name = "hip";

#endif

    return info;
}


int Foam::amgxCheckComputeBackend
(
    int cudaDevice,
    bool allowHostFallback,
    word& dataLocation
)
{
    const amgxComputeBackend cb = amgxQueryComputeBackend();

    if (cb.name == "none")
    {
        // Plain OpenFOAM: nothing to align with, AmgX uses the current device
        if (cudaDevice < 0)
        {
            cudaGetDevice(&cudaDevice);
        }
        return cudaDevice;
    }

    if (cb.isCuda())
    {
        if (cudaDevice < 0)
        {
            cudaDevice = cb.cudaDevice;
        }
        else if (cudaDevice != cb.cudaDevice)
        {
            FatalErrorInFunction
                << "AmgX was assigned " << cudaDeviceLabel(cudaDevice).c_str()
                << " but the application computes on "
                << cudaDeviceLabel(cb.cudaDevice).c_str() << nl
                << "Matrix and field memory would be accessed across GPUs."
                << " Make the per-rank GPU assignment consistent"
                << " (eg, CUDA_VISIBLE_DEVICES or ACPP_VISIBILITY_MASK)."
                << exit(FatalError);
        }

        if (cudaSetDevice(cudaDevice) != cudaSuccess)
        {
            FatalErrorInFunction
                << "cudaSetDevice(" << cudaDevice << ") failed: "
                << cudaGetErrorString(cudaGetLastError())
                << exit(FatalError);
        }

        Info<< "AmgX: sharing " << cudaDeviceLabel(cudaDevice).c_str()
            << " with the " << cb.name.c_str() << " compute backend" << nl;

        return cudaDevice;
    }

    // The application does not compute on a CUDA device

    if (!allowHostFallback)
    {
        FatalErrorInFunction
            << "The application computes on the '" << cb.name.c_str()
            << "' backend (" << cb.deviceName.c_str()
            << ") while AmgX runs on an NVIDIA GPU." << nl
            << "The matrix and the solution vectors would be copied between"
            << " host and device on every solve." << nl
            << "Either run on a CUDA device (eg, ACPP_VISIBILITY_MASK=cuda)"
            << " or set 'allowHostFallback true;' in the solver dictionary"
            << " to accept the copies (dataLocation is then 'host')."
            << exit(FatalError);
    }

    if (dataLocation != "host")
    {
        WarningInFunction
            << "allowHostFallback: the '" << cb.name.c_str()
            << "' compute backend cannot provide device-resident matrix"
            << " storage, using dataLocation 'host' instead of '"
            << dataLocation << "'" << endl;
        dataLocation = "host";
    }

    if (cudaDevice < 0)
    {
        cudaGetDevice(&cudaDevice);
    }
    cudaGetLastError();

    Info<< "AmgX: host fallback from the " << cb.name.c_str()
        << " compute backend (" << cb.deviceName.c_str() << ") to "
        << cudaDeviceLabel(cudaDevice).c_str()
        << ", data copied on every solve" << nl;

    return cudaDevice;
}


// ************************************************************************* //
