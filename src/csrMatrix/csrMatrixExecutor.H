/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2025 CINECA
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
    csrMatrixExecutor

Description
    List of executors used as Visitors in the csrMatrix class.

SourceFiles
    csrMatrixExecutor.H

\*---------------------------------------------------------------------------*/


#ifndef csrMatrix_Executor_H
#define csrMatrix_Executor_H

#ifdef AMGX4FOAM_CUDA_EXECUTOR
#include "cudaCsrMatrixExecutor.H"
#endif
#include "cpuCsrMatrixExecutor.H"
#include <variant>

namespace Foam
{

using csrMatrixExecutor = std::variant
    <
#ifdef AMGX4FOAM_CUDA_EXECUTOR
    cudaCsrMatrixExecutor,
#endif
    cpuCsrMatrixExecutor
    >;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

