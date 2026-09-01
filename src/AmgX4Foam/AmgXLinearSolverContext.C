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

\*---------------------------------------------------------------------------*/

#include "AmgXLinearSolverContext.H"
#include "csrMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    // defineTypeNameAndDebug(AmgXLinearSolverContext, 0);
    defineTemplateTypeNameAndDebug
    (
        AmgXLinearSolverContext<csrMatrix>,
        0
    );

    // const word AmgXLinearSolverContext::packageName = "AmgX";
    template<class matrix> const word AmgXLinearSolverContext<matrix>::packageName = "AmgX";
}

// * * * * * * * * * * * * * explicit instantiation * * * * * * * * * * * * //

template class Foam::AmgXLinearSolverContext<Foam::csrMatrix>;

// ************************************************************************* //
