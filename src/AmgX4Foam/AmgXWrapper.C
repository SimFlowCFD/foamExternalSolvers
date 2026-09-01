/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
    Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
    Copyright (C) 2023-2024 Cineca
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

#include "AmgXWrapper.H"

#ifndef AMGX4FOAM_NO_MPI
#include "PstreamGlobals.H"
#endif

#include "amgxBackendGuard.H"
#include "global.cuh"
#include "OSspecific.H"

/*---------------------------------------------------------------------------*\
License
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

\*---------------------------------------------------------------------------*/

// ************************************************************************* //

// initialize AmgXWrapper::count to 0
int Foam::AmgXWrapper::count = 0;

// initialize AmgXWrapper::rsrc to nullptr;
AMGX_resources_handle Foam::AmgXWrapper::rsrc = nullptr;

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

/* \implements AmgXWrapper::AmgXWrapper */
/*Foam::AmgXWrapper::AmgXWrapper
(
    const MPI_Comm &comm,
    const std::string &modeStr,
    const std::string &cfgFile
)
{
    initialize(comm, modeStr, cfgFile);
}*/

// * * * * * * * * * * * * * * * Destructor * * * * * * * * * * * * * * * * * //

/* \implements AmgXWrapper::~AmgXWrapper */
Foam::AmgXWrapper::~AmgXWrapper()
{
    if (isInitialised)
        finalize();
}

// * * * * * * * * * * * * * * * Utilities * * * * * * * * * * * * * * * * * * //

void checkAmgXerror(AMGX_RC code, const std::string& function)
{
    char buff[256];
    AMGX_get_error_string(code, buff, 256);
    if(code != AMGX_RC_OK){
        AMGX_get_error_string(code, buff, 256);
        Foam::Info << function << " returned: " << buff << Foam::nl;
    }
}

namespace Foam
{
bool amgxFoamMemoryIsCudaAccessible(const void* ptr);
}

bool Foam::amgxFoamMemoryIsCudaAccessible(const void* ptr)
{
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess)
    {
        cudaGetLastError();  // clear the sticky error
        return false;
    }
    return
        attr.type == cudaMemoryTypeManaged
     || attr.type == cudaMemoryTypeDevice;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " (error code " << err << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// * * * * * * * * * * * * * * Member functions  * * * * * * * * * * * * * * * //

/* \implements AmgXWrapper::initialize*/
void Foam::AmgXWrapper::initialize(
    const word &modeStr,
    const word &dataLocation,
    const string &configStr,
    const bool allowHostFallback
)
{
    //- increase the number of AmgXWrapper instances
    count += 1;

    //- get the mode of AmgX solver
    setMode(modeStr);

    //- use the GPU the application computes on (may adjust dataLocation)
    word location(dataLocation);
    devID_ = amgxCheckComputeBackend(-1, allowHostFallback, location);

    initAmgX(configStr);

    dataOrigin_ = location;

    gpuProc_ = true;

    isInitialised = true;
}


/* \implements AmgXWrapper::initialize*/
void Foam::AmgXWrapper::initialize(
    const label &commId,
    const word &modeStr,
    const word &dataLocation,
    const string &configStr,
    const bool allowHostFallback
)
{
    //- increase the number of AmgXWrapper instances
    count += 1;

    //- get the mode of AmgX solver
    setMode(modeStr);

    //- initialize communicators and corresponding information
    initComms(commId);

    word location(dataLocation);

    if(gpuProc_)
    {
        //- the device chosen by initComms must be the one the application
        //  computes on (may adjust dataLocation)
        devID_ = amgxCheckComputeBackend(devID_, allowHostFallback, location);

        initAmgX(configStr);
    }

    dataOrigin_ = location;

    isInitialised = true;
}


void Foam::AmgXWrapper::initialiseMatrixComms(csrMatrix* matrix)
{
    matrix->initializeComms(gpuWorld_, gpuProc_);

    Pstream::barrier(globalWorld_);
}


/* \implements AmgXWrapper::setMode */
void Foam::AmgXWrapper::setMode(const word &modeStr)
{
    if (modeStr == "dDDI")
        mode = AMGX_mode_dDDI;
    else if (modeStr == "dDFI")
        mode = AMGX_mode_dDFI;
    else if (modeStr == "dFFI")
        mode = AMGX_mode_dFFI;
    else
        FatalErrorInFunction
            << modeStr.c_str() << " is not an available mode! Available modes are: dDDI, dDFI, dFFI." <<  nl 
            << abort(FatalError);
}


/* \implements AmgXWrapper::initComms */
void Foam::AmgXWrapper::initComms(const int &commId)
{   
    //- Assign global communicator
    globalWorld_ = commId;

    //- Get size and rank for communicator
    globalWorldSize_ = Pstream::nProcs(commId);
    myGlobalWorldRank_ = Pstream::myProcNo(commId);

    //- Get the communicator for processors on the same node (local world)
    localWorld_ = Pstream::commLocalNode();

    //- Get size and rank for local communicator
    localWorldSize_ = Pstream::nProcs(localWorld_);
    myLocalWorldRank_ = Pstream::myProcNo(localWorld_);

    cudaGetDeviceCount(&nDevs_);
    
    if (localWorldSize_ == nDevs_)
    {
        devID_ = myLocalWorldRank_;
        gpuProc_ = true;
    }
    else if (localWorldSize_ > nDevs_)
    {
        int nBasic = localWorldSize_ / nDevs_,
            nRemain = localWorldSize_ % nDevs_;

        if (myLocalWorldRank_ < (nBasic+1)*nRemain)
        {
            devID_ = myLocalWorldRank_ / (nBasic + 1);
            if (myLocalWorldRank_ % (nBasic + 1) == 0)  gpuProc_ = 0;
        }
        else
        {
            devID_ = (myLocalWorldRank_ - (nBasic+1)*nRemain) / nBasic + nRemain;
            if ((myLocalWorldRank_ - (nBasic+1)*nRemain) % nBasic == 0) gpuProc_ = true;
        }

    }
    else
    {
        Info << "CUDA devices per node are more than the MPI processes launched on the node. Only " 
             << localWorldSize_ << " CUDA devices will be used." << nl;
        
        devID_ = myLocalWorldRank_;
        gpuProc_ = true;
        nDevs_ = localWorldSize_;
    }

    cudaSetDevice(devID_);

    UPstream::barrier(globalWorld_);

    //- split the global world into a world involved in AmgX and a null world
    List<bool>  gpuProcList(globalWorldSize_, false);
    gpuProcList.data()[myGlobalWorldRank_] = gpuProc_;
    Pstream::allGatherList(gpuProcList);
    DynamicList<label> globalGpuWolrdProcs;
    for(label i=0; i<globalWorldSize_; ++i){
        if(gpuProcList[i]) globalGpuWolrdProcs.append(i);
    }
    globalGpuWorld_ = Pstream::newCommunicator(globalWorld_, globalGpuWolrdProcs);

    //- Get size and rank for the communicator corresponding to gpuWorld
    if (gpuProc_)
    {
        globalGpuWorldSize_ = Pstream::nProcs(globalGpuWorld_);
        myGlobalGpuWorldRank_ = Pstream::myProcNo(globalGpuWorld_);
    }

    //- Split local world into worlds corresponding to each CUDA device
    labelList devIds(localWorldSize_);
    devIds[myLocalWorldRank_] = devID_;
    Pstream::allGatherList(devIds, UPstream::msgType(), localWorld_);
    DynamicList<label> gpuWorldProcs;
    for(label i=0; i<localWorldSize_; ++i)
    {
        if(devIds[i] == devID_) gpuWorldProcs.append(i);
    }
    gpuWorld_ = Pstream::newCommunicator(localWorld_, gpuWorldProcs);

    //- Get size and rank for the communicator corresponding to myWorld
    gpuWorldSize_ = Pstream::nProcs(gpuWorld_);
    myGpuWorldRank_ = Pstream::myProcNo(gpuWorld_);

    Pstream::barrier(globalWorld_);
}


/* \implements AmgXWrapper::initAmgX */
void Foam::AmgXWrapper::initAmgX(const string &configStr)
{
    //- only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        //- initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        //- only the master process can output something on the screen
        AMGX_SAFE_CALL(AMGX_register_print_callback(
                    [](const char *msg, int length)->void
                    {Info << msg << nl;}));

        //- let AmgX to handle errors returned
        // Not on Windows: AmgX's handler swallows faults (incl. access
        // violations) and exits silently, hiding FOAM diagnostics
#ifndef _WIN32
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
#endif
    }

    //- create an AmgX configure object
    if(configStr.contains("system"))
    {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, configStr.c_str()));
    }
    else
    {
        AMGX_SAFE_CALL(AMGX_config_create(&cfg, configStr.c_str()));
    }

    //- let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    //- create an AmgX resource object, only the first instance is in charge
    if (count == 1)
    {
        if (!Pstream::parRun())
        {
            AMGX_resources_create_simple(&rsrc, cfg);
        }
        else
        {
#ifdef AMGX4FOAM_NO_MPI
            FatalErrorInFunction
                << "Parallel AmgX requires an MPI build of foamExternalSolvers"
                << abort(FatalError);
#else
            AMGX_resources_create(&rsrc, cfg, &PstreamGlobals::MPICommunicators_[globalGpuWorld_], 1, &devID_);
#endif
        }
    }

    //- create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&AmgXP, rsrc, mode);
    AMGX_vector_create(&AmgXRHS, rsrc, mode);

    //- create AmgX matrix object for unknowns and RHS
    checkAmgXerror(AMGX_matrix_create(&AmgXA, rsrc, mode), "Matrix creation");

    //- create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    //- obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);
}

/* \implements AmgXWrapper::finalize */
void Foam::AmgXWrapper::finalize()
{
    //- skip if this instance has not been initialised
    if (!isInitialised)
    {
        fprintf(stderr,
                "This AmgXWrapper has not been initialised. "
                "Please initialise it before finalization.\n");
    }

    if(gpuProc_)
    {
        //- destroy solver instance
        AMGX_solver_destroy(solver);

        //- destroy matrix instance
        AMGX_matrix_destroy(AmgXA);

        //- destroy RHS and unknown vectors
        AMGX_vector_destroy(AmgXP);
        AMGX_vector_destroy(AmgXRHS);

        //- only the last instance need to destroy resource and finalizing AmgX
        if (count == 1)
        {
            AMGX_resources_destroy(rsrc);
            AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

            AMGX_SAFE_CALL(AMGX_finalize());
        }
        else
        {
            AMGX_config_destroy(cfg);
        }
    }

    if (Pstream::parRun()) 
    {   
        Pstream::freeCommunicator(gpuWorld_);
        Pstream::freeCommunicator(localWorld_);
        Pstream::freeCommunicator(globalGpuWorld_);
    }
        
    //- decrease the number of instances
    count -= 1;

    //- change status
    isInitialised = false;
}

/* \implements AmgXWrapper::setOperator */
void Foam::AmgXWrapper::setOperator
(
    const label nGlobalRows,
    const csrMatrix* matrix
)
{
    if(gpuProc_)
    {    
        const label nLocalRows = matrix->nOwnerStart() - 1;
        const label nLocalNz = matrix->nLocalNz();
        const label nBlocks = matrix->nBlocks();
        
        //- Check the matrix size is not larger than tolerated by AmgX
        if(nLocalRows > std::numeric_limits<int>::max())
        {
            fprintf(stderr,
                    "AmgX does not support a global number of rows greater than "
                    "what can be stored in 32 bits (nGlobalRows = %d).\n",
                    nLocalRows);
        }

        if (nLocalNz > std::numeric_limits<int>::max())
        {
            fprintf(stderr,
                    "AmgX does not support non-zeros per (consolidated) rank greater than"
                    "what can be stored in 32 bits (nLocalNz = %d).\n",
                    nLocalNz);
        }

        const int * ownStart; // = matrix->ownerStart().cdata();
        const int * colInd; // = matrix->colIndices().cdata();
        const void * matValues; // = matrix->values().cdata();

        // dataLocation=host: copy the CSR arrays to AmgX-owned device
        // storage. cudaMemcpyDefault infers the direction, the source may be
        // host memory (CPU executor) or device memory (CUDA executor).
        if(dataOrigin_ == "host")
        {
            cudaMalloc((void**) &ownStart, sizeof(int)*(nLocalRows+1));
            cudaMalloc((void**) &colInd, sizeof(int)*nLocalNz);
            cudaMalloc((void**) &matValues, sizeof(double)*nLocalNz);
            cudaMemcpy((void*) ownStart, (const void*) matrix->ownerStart(), sizeof(int)*(nLocalRows+1), cudaMemcpyDefault);
            cudaMemcpy((void*) colInd, (const void*) matrix->colIndices(), sizeof(int)*nLocalNz, cudaMemcpyDefault);
            cudaMemcpy((void*) matValues, (const void*) matrix->values(), sizeof(double)*nLocalNz, cudaMemcpyDefault);
        }
        else
        {
            ownStart = matrix->ownerStart();
            colInd = matrix->colIndices();
            matValues = matrix->values();

            // dataLocation=device hands these pointers straight to AmgX:
            // they must be CUDA device or managed (USM) allocations.
            // Fails loudly on host-heap arrays, OMP-host or HIP runs.
            if (!amgxFoamMemoryIsCudaAccessible(matValues)
             || !amgxFoamMemoryIsCudaAccessible(ownStart)
             || !amgxFoamMemoryIsCudaAccessible(colInd))
            {
                FatalErrorInFunction
                    << "dataLocation is 'device' but the CSR matrix arrays"
                    << " are not CUDA-accessible (device/managed) memory."
                    << nl
                    << "Use 'dataLocation host', or provide device-resident"
                    << " matrix storage (NVIDIA GPU run with USM pool"
                    << " allocations)." << nl
                    << exit(FatalError);
            }
        }

        //- upload matrix A to AmgX
        if (globalGpuWorldSize_ == 1 || !Pstream::parRun())
        {
            AMGX_matrix_upload_all(
                AmgXA, nLocalRows, nLocalNz, nBlocks, nBlocks,
                ownStart, colInd, matValues, nullptr);
        }
        else
        {
            AMGX_distribution_handle dist;
            AMGX_distribution_create(&dist, cfg);

            //- Must persist until after we call upload
            labelList offsets(globalGpuWorldSize_ + 1, Zero);

            //- Determine the number of rows per GPU
            labelList nRowsPerGPU(globalGpuWorldSize_, nLocalRows);
            Pstream::allGatherList(nRowsPerGPU, UPstream::msgType(), globalGpuWorld_);
 
            //- Calculate the global offsets
            for(int i = 0; i < globalGpuWorldSize_; ++i)
            {
                offsets.data()[i+1] = offsets.data()[i] + nRowsPerGPU.data()[i];
            }

            AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, offsets.data());

            //- Set the column indices size, 32- / 64-bit
            AMGX_distribution_set_32bit_colindices(dist, true);

            AMGX_matrix_upload_distributed(
                AmgXA, nGlobalRows, nLocalRows, nLocalNz, nBlocks, nBlocks,
                ownStart, colInd, matValues, nullptr, dist);

            AMGX_distribution_destroy(dist);
        }

        //- bind the matrix A to the solver
        AMGX_solver_setup(solver, AmgXA);

        //- connect (bind) vectors to the matrix
        AMGX_vector_bind(AmgXP, AmgXA);
        AMGX_vector_bind(AmgXRHS, AmgXA);
    }
}


/* \implements AmgXWrapper::updateOperator */
void Foam::AmgXWrapper::updateOperator
(
    const csrMatrix* matrix
)
{
    if(gpuProc_)
    {
        const label nLocalRows = matrix->nOwnerStart() - 1;
        const label nLocalNz = matrix->nLocalNz();
        const void * matValues = matrix->values();

        //- Replace the coefficients for the CSR matrix A within AmgX
        AMGX_matrix_replace_coefficients(AmgXA, nLocalRows, nLocalNz, matValues, nullptr);

        //- Re-setup the solver (a reduced overhead setup that accounts for consistent matrix structure)
        AMGX_solver_resetup(solver, AmgXA);
    }
}


/* \implements AmgXSolver::updateConfig */
void Foam::AmgXWrapper::updateConfig(const string& newConfig)
{
    AMGX_config_add_parameters(&cfg, newConfig.c_str());
}


/* \implements AmgXWrapper::solve */
void Foam::AmgXWrapper::solve
(
    const csrMatrix* matrix
)
{    
    scalar * p = matrix->psiCons();
    const scalar * b = matrix->rhsCons();
    label nRows = matrix->nConsRows();

    // nLocalRows = matrix->ownerStart().size() - 1;
    label nBlocks = matrix->nBlocks();
    
    if (gpuProc_)
    {    
        //- Upload vectors to AmgX
        AMGX_vector_upload(AmgXP, nRows, nBlocks, p);
        AMGX_vector_upload(AmgXRHS, nRows, nBlocks, b);

        //- Solve
        AMGX_solver_solve(solver, AmgXRHS, AmgXP);

        //- Get the status of the solver
        AMGX_SOLVE_STATUS status;
        AMGX_solver_get_status(solver, &status);

        //- Check whether the solver successfully solved the problem
        if (status != AMGX_SOLVE_SUCCESS)
        {
            fprintf(stderr, "AmgX solver failed to solve the system! "
                            "The error code is %d.\n",
                    status);
        }

        // Download data from device
        AMGX_vector_download(AmgXP, p);

        if(matrix->isConsolidated()) cudaDeviceSynchronize();
    }
    if(Pstream::parRun()) Pstream::barrier(gpuWorld_); //necessary
}


/* \implements AmgXWrapper::getIters */
void Foam::AmgXWrapper::getIters(label &iter)
{
    if (gpuProc_) AMGX_solver_get_iterations_number(solver, &iter);
}


/* \implements AmgXWrapper::getResidual */
void Foam::AmgXWrapper::getResidual(const label &iter, scalarField &res, label nBlocks)
{
    if (gpuProc_)
    {
	    for (label nres=0;nres<nBlocks; nres++)
	    {
            AMGX_solver_get_iteration_residual(solver, iter, nres, res.data()+nres);
	    }
    }
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //


// ************************************************************************* //
