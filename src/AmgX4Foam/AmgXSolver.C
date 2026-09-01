/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
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

#include "AmgXSolver.H"
#include "direction.H"
#include "AmgXLinearSolverContext.H"
#include "linearSolverContextTable.H"
#include "csrMatrix.H"

#include "globalIndex.H"

// #include <iostream>
// #include <fstream>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(AmgXSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<AmgXSolver>
        addAmgXSolverSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<AmgXSolver>
        addAmgXSolverAsymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::AmgXSolver::AmgXSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    eqName_(fieldName)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::AmgXSolver::solve
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{   
    const fvMesh& fvm = dynamicCast<const fvMesh>(this->matrix_.mesh().thisDb());
    
    label nCells = psi.size();

    const linearSolverContextTable<AmgXLinearSolverContext<csrMatrix>>& contexts =
        linearSolverContextTable<AmgXLinearSolverContext<csrMatrix>>::New(fvm);

    AmgXLinearSolverContext<csrMatrix>& ctx = contexts.getContext(eqName_, typeName);

    if (!ctx.loaded())
    {
        FatalErrorInFunction
            << "Could not initialize AMGx" << nl << abort(FatalError);
    }

    AmgXWrapper& amgx = ctx.amgx_;
    
    csrMatrix& Amat = ctx.Amat_;

    if (!Pstream::parRun())
    {
	if (!ctx.initialized() || ctx.doUpdateMatrixCoefficients())
	{
            Amat.applyPermutation(matrix_);
	}
    }
    else
    {
        if (!ctx.initialized()) 
	{
            amgx.initialiseMatrixComms(&Amat);
	}

	if (!ctx.initialized() || ctx.doUpdateMatrixCoefficients())
        {
            Amat.applyPermutation(matrix_, interfaceBouCoeffs_);
	}
    }

    if (!ctx.initialized())
    {
        Info << "Initializing AmgX Linear Solver " << eqName_ << nl;

        //- Compute global number of equations
        const label nGlobalCells = returnReduce(nCells, sumOp<label>());

        amgx.setOperator(nGlobalCells, &Amat);

        Amat.clearAddressing();

        ctx.initialized() = true;
    }
    else
    {
        if (ctx.doUpdateMatrixCoefficients())	
        {
            amgx.updateOperator(&Amat);
        }
    }

    if(!controlDict_.dictName().ends_with("Final") && ctx.updated())
    {
        ctx.updateConfig(controlDict_);
        ctx.updated() = false;
    }
    else if(controlDict_.dictName().ends_with("Final"))
    {
        ctx.updateConfig(controlDict_);
        ctx.updated() = true;
    }

    Amat.createConsVect(source, psi);
    amgx.solve(&Amat);

    Amat.distributeSolution(psi);

    scalarField iNorm(1, 0.0);
    amgx.getResidual(0, iNorm);
    ctx.performance_.initialResidual() = iNorm[0];

    label nIters = 0;
    amgx.getIters(nIters);
    ctx.performance_.nIterations() = nIters;

    scalarField fNorm(1, 0.0);
    amgx.getResidual(nIters, fNorm);
    ctx.performance_.finalResidual() = fNorm[0];

    return ctx.performance_;
}

// ************************************************************************* //
