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

Description
    Consistency check between the compute backend of the host application
    (where OpenFOAM/SPUMA runs its own kernels and allocates its fields)
    and the NVIDIA GPU that AmgX is about to use.

    AmgX always executes on a CUDA device. When the application itself runs
    on that same device (SPUMA SYCL build with the CUDA backend, or a native
    CUDA build) the matrix and vectors can be shared in place. When the
    application runs elsewhere (host OpenMP, HIP, ...) every solve has to
    move the data between host and device, which is normally a configuration
    mistake: it is rejected unless explicitly allowed.

SourceFiles
    amgxBackendGuard.C

\*---------------------------------------------------------------------------*/

#ifndef amgxBackendGuard_H
#define amgxBackendGuard_H

#include "word.H"
#include <string>

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Struct amgxComputeBackend Declaration
\*---------------------------------------------------------------------------*/

struct amgxComputeBackend
{
    //- Backend name: "cuda", "hip", "omp", "level_zero", "ocl", "sycl"
    //  (unidentified SYCL implementation) or "none" (no device backend
    //  compiled into the application)
    std::string name = "none";

    //- Device name reported by the backend (empty for "none")
    std::string deviceName;

    //- CUDA ordinal of the compute device (name == "cuda"), otherwise -1
    int cudaDevice = -1;

    //- The application computes on a CUDA device
    bool isCuda() const noexcept
    {
        return cudaDevice >= 0;
    }
};


//- Query the compute backend of the running application
amgxComputeBackend amgxQueryComputeBackend();

//- Make AmgX use the GPU the application computes on, or fail.
//
//  \param cudaDevice the CUDA device already chosen for AmgX, or -1 to
//      select the application's compute device (serial runs)
//  \param allowHostFallback accept a non-CUDA compute backend; the data
//      are then copied on every solve and dataLocation is forced to "host"
//  \param dataLocation the requested dataLocation (may be changed)
//  \return the CUDA device AmgX shall use (also made current)
int amgxCheckComputeBackend
(
    int cudaDevice,
    bool allowHostFallback,
    word& dataLocation
);

} // End namespace Foam

#endif

// ************************************************************************* //
