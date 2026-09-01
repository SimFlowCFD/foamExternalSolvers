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

Class
    Foam::AmgXWrapper

Description
     Interface to the external package AmgX (https://github.com/NVIDIA/AMGX).

SourceFiles
    AmgXWrapper.H
    AmgXWrapper.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_AmgXWrapper_H
#define Foam_AmgXWrapper_H

// AmgX
#include <amgx_c.h>

#include "word.H"
#include "scalar.H"
#include "label.H"
#include "IOstreams.H"
#include "csrMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations


/*---------------------------------------------------------------------------*\
                           Class AmgXWrapper Declaration
\*---------------------------------------------------------------------------*/

class AmgXWrapper
{
    public:

        /** \brief Default constructor. */
        AmgXWrapper() = default;


        /** \brief Construct a AmgXWrapper instance.
         *
         * \param comm [in] MPI communicator.
         * \param modeStr [in] A string; target mode of AmgX (e.g., dDDI).
         * \param cfgFile [in] A string; the path to AmgX configuration file.
         */
        /*AmgXWrapper
        (
            // const MPI_Comm &comm,
            // const std::string &modeStr,
            // const std::string &cfgFile
        );*/


        /** \brief Destructor. */
        ~AmgXWrapper();


        /** \brief Initialize a AmgXWrapper instance.
         *
         * \param commId [in] MPI communicator.
         * \param modeStr [in] A string; target mode of AmgX (e.g., dDDI).
         * \param dataLocation [in] A string; indicates where are data.
         * \param allowHostFallback [in] accept a non-CUDA compute backend
         *        of the application (data copied on every solve).
         * \param configStr [in] A string; the path to AmgX configuration file
         *                  or a configuration string already concatenated.
         */
        void initialize
        (
            const word &modeStr,
            const word &dataLocation,
            const string &configStr,
            const bool allowHostFallback = false
        );

        void initialize
        (
            const label &commId,
            const word &modeStr,
            const word &dataLocation,
            const string &configStr,
            const bool allowHostFallback = false
        );

        void initialiseMatrixComms(csrMatrix* matrix);


        /** \brief Finalize this instance.
         *
         * This function destroys AmgX data. When there are more than one
         * AmgXWrapper instances, the last one destroyed is also in charge of
         * destroying the shared resource object and finalizing AmgX.
         */
        void finalize();


        /** \brief Set up the matrix used by AmgX.
         *
         * This function sets up the AmgX matrix from the provided CSR data
         * structures and partition data.
         *
         * \param nLocalRows [in] The number of local rows on this rank.
         * \param nGlobalRows [in] The number of global rows.
         * \param nLocalNz [in] The total number of non zero entries locally.
         * \param matrix [in] CSR matrix.
         */
        void setOperator
        (
            const label nGlobalRows,
            const csrMatrix* matrix
        );


        /** \brief Re-sets up an existing AmgX matrix.
         *
         * Replaces the matrix coefficients with the provided values and performs
         * a resetup for the AmgX matrix.
         *
         * \param nLocalRows [in] The number of local rows on this rank.
         * \param nLocalNz [in] The total number of non zero entries locally.
         * \param matrix [in] CSR matrix.
         */
        void updateOperator
        (
            const csrMatrix* matrix
        );


        /** \brief Re-sets up an existing AmgX config.
         *
         * Replaces the existing configuration.
         *
         * \param newConfig [in] 
         */
        void updateConfig
        (
            const string& newConfig
        );


        /** \brief Solve the linear system.
         *
         * \p p vector will be used as an initial guess and will be updated to the
         * solution by the end of solving.
         *
         * For cases that use more MPI processes than the number of GPUs, this
         * function will do data gathering before solving and data scattering
         * after the solving.
         *
         * \param nLocalRows [in] The number of rows owned by this rank.
         * \param pscalar [in, out] The unknown array.
         * \param bscalar [in] The RHS array.
         */
        void solve
        (
            const csrMatrix* matrix
        );

        /** \brief Get the number of iterations of the last solving.
         *
         * \param iter [out] Number of iterations.
         */
        void getIters
        (
            label &iter
        );

        /** \brief Get the residual at a specific iteration during the last solving.
         *
         * \param iter [in] Target iteration.
         * \param res [out] Returned residual.
         */
        void getResidual
        (
            const label &iter,
            scalarField &res,
			label nres = 1
        );


    private:

        /** \brief Current count of AmgXWrapper instances.
         *
         * This static variable is used to count the number of instances. The
         * fisrt instance is responsable for initializing AmgX library and the
         * resource instance.
         */
        static label            count;

        /** \brief A flag indicating if this instance has been initialized. */
        bool                    isInitialised = false;

        /** \brief the global MPI communicator processes. */
        label                   globalWorld_;

        /** \brief the index of the global MPI communicator processes. */
        // label                   globalWorldIdx_;

        /** \brief Size of \ref AmgXWrapper::globalWorld "globalWorld". */
        label                   globalWorldSize_;

        /** \brief Rank in \ref AmgXWrapper::globalWorld "globalWorld". */
        label                   myGlobalWorldRank_;

        /** \brief the index of the local node MPI communicator processes. */
        label                   localWorld_;

        /** \brief Size of \ref AmgXWrapper::globalWorld "globalWorld". */
        label                   localWorldSize_;

        /** \brief Rank in \ref AmgXWrapper::globalWorld "globalWorld". */
        label                   myLocalWorldRank_;

        /** \brief the index of the MPI communicator processes that talk to a GPUs. */
        label                   gpuWorld_;

        /** \brief Size of \ref AmgXWrapper::gpuWorld "gpuWorld". */
        label                   gpuWorldSize_;

        /** \brief Rank in \ref AmgXWrapper::gpuWorld "gpuWorld". */
        label                   myGpuWorldRank_;

        /** \brief the index of the MPI communicator af all the processes that tolk with GPU. */
        label                   globalGpuWorld_;

        /** \brief Size of \ref AmgXWrapper::myGpuWorldRank_ "myGpuWorldRank_". */
        label                   globalGpuWorldSize_;

        /** \brief Rank in \ref AmgXWrapper::myGpuWorldRank_ "myGpuWorldRank_". */
        label                   myGlobalGpuWorldRank_;

        /** \brief Number of GPU devices in the node.*/
        label                   nDevs_;

        /** \brief The ID of corresponding GPU device used by this MPI process. */
        label                   devID_;

        /** \brief Flag that indicates if the process is assigned to a GPU. */
        bool                    gpuProc_ = false;

        /** \brief A parameter used by AmgX. */
        label                   ring;

        /** \brief Data origin. */
        word                    dataOrigin_;

        /** \brief AmgX solver mode. */
        AMGX_Mode               mode;

        /** \brief AmgX config object. */
        AMGX_config_handle      cfg = nullptr;

        /** \brief AmgX matrix object. */
        AMGX_matrix_handle      AmgXA = nullptr;

        /** \brief AmgX vector object representing unknowns. */
        AMGX_vector_handle      AmgXP = nullptr;

        /** \brief AmgX vector object representing RHS. */
        AMGX_vector_handle      AmgXRHS = nullptr;

        /** \brief AmgX solver object. */
        AMGX_solver_handle      solver = nullptr;

        /** \brief AmgX resource object.
         *
         * Due to the design of AmgX library, using more than one resource
         * instance may cause some problems. So we make the resource instance
         * as a static member to keep only one instance.
         */
        static AMGX_resources_handle   rsrc;

        /** \brief Set AmgX solver mode based on the user-provided string.
         *
         * Available modes are: dDDI, dDFI, dFFI, hDDI, hDFI, hFFI.
         *
         * \param modeStr [in] a std::string.
         */
        void setMode(const word &modeStr);

        /** \brief Set indices relatively to MPI comunications.
         *
         * \param commId [in] Global communicator index.
         */
        void initComms(const int &commId);

        /** \brief Perform necessary initialization of AmgX.
         *
         * This function initializes AmgX for current instance. Based on
         * \ref AmgXWrapper::count "count", only the instance initialized first
         * is in charge of initializing AmgX and the resource instance.
         *
         * \param configStr [in] configuration string.
         * \return PetscErrorCode.
         */
        void initAmgX(const string &configStr);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

