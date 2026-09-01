/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2025 Cineca
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
    cudaCsrMatrixExecutor

Description
    CUDA visitor of the csrMatrix class.

SourceFiles
    cudaCsrMatrixExecutor.H
    cudaCsrMatrixExecutor.cu

\*---------------------------------------------------------------------------*/

#ifdef AMGX4FOAM_CUDA_EXECUTOR

#ifndef cuda_CsrMatrix_Executor_H
#define cuda_CsrMatrix_Executor_H

#include "scalar.H"
#include "List.H"
#include "primitiveFieldsFwd.H"

namespace Foam
{

// Forward Declarations
class csrMatrix;

class cudaCsrMatrixExecutor
{
public:

    cudaCsrMatrixExecutor() {};

    ~cudaCsrMatrixExecutor() {};

    template <class Type>
    bool isDeviceValid(const Type* ptr) const;

    template<class Type>
    Type* alloc(Foam::label size) const;

    template<class Type>
    Type* allocZero(Foam::label size) const;

    template<class Type>
    const Type* copyFromFoam(Foam::label size, const Type* hostPtr) const;

    template<class Type>
    void copyToFoam(Foam::label size, Type* devPtr, Type** hostPtr) const;

    template<class Type>
    void clear(Type* ptr) const;

    template<class Type>
    void clear(const Type* ptr) const;

    template<class Type>
    void concatenate
    (
        label globSize,
        List<List<Type>> lst,
        Type * ptr
    ) const;

    void offsetCopy
    (
        const List<scalar>& lst,
        scalar * ptr,
	    label consDispl
    ) const;

    void initializeSequence
    (
    const label len,
          label * vect
    ) const;

    void initializeAddressing
    (
        const label   nCells,
        const label   nInternalFaces,
        const label * const owner,
        const label * const neighbour,
              label * rowIndTmp,
              label * colIndTmp
    ) const;

    void initializeAddressingExt
    (
        const label   nCells,
        const label   nInternalFaces,
        const label   nnzExt,
        const label * const owner,
        const label * const neighbour,
        const label * const extRows,
        const label * const extCols,
              label * rowIndTmp,
              label * colIndTmp
    ) const;

    void computeSorting
    (
        const label   totNnz,
              label * tmpPerm,
              label * rowIndTmp,
              label * rowInd,
              label * ldu2csr
    ) const;

    void localToGlobalColIndices
    (
        const label   nConsRows,
        const label   nConsIntFaces,
        const label   nRows,
        const label   nIntFaces,
        const label   diagIndexGlobal,
        const label   lowOffGlobal,
        const label   uppOffGlobal,
              label * colIndicesGlobal,
        const label   rowsDispl = 0,
        const label   intFacesDispl = 0
    ) const;

    void localToConsRowIndex
    (
        const label nConsRows,
        const label nConsIntFaces,
        const label nIntFaces,
        const label nExtNz,
        const label intFacesDipl,
        const label extDispl,
        const label offset,
            label * rowIndices
    ) const;

    void applyAddressingPermutation
    (
        const label   nCells,
        const label   totNnz,
        const label * const ldu2csr,
        const label * const colIndTmp,
        const label * const rowInd,
              label * colInd,
              label * ownStart
    ) const;

    void initializeValue
    (
        const label   nCells,
        const label   nIntFaces,
        const scalar * const diag,
        const scalar * const upper,
        const scalar * const lower,
              scalar * valuesTmp                  
    ) const;

    void initializeValueExt
    (
        const label    nCells,
        const label    nIntFaces,
        const label    nnzExt,
        const scalar * const diag,
        const scalar * const upper,
        const scalar * const lower,
        const scalar * const extValue,
              scalar * valuesTmp
    ) const;

    void applyValuePermutation
    (
        const label    totNnz,
        const label  * const ldu2csr,
        const scalar * const valuesTmp,
              scalar * values,
        const label    nBlocks = 1
    ) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

#endif

// ************************************************************************* //

