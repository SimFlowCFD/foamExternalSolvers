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

\*---------------------------------------------------------------------------*/

#include "cpuCsrMatrixExecutor.H"
#ifdef have_sycl
// SPUMA: route executor allocations through the memory pool so the CSR
// arrays land in USM (device-visible) storage - enables the zero-copy
// dataLocation=device path. Non-SPUMA builds keep plain host new/delete.
#include "MemoryPoolBase.H"
#endif
#include "zero.H"
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

// * * * * * * * * * * * * * * * * Member functions * * * * * * * * * * * * * //

template<class Type>
Type* Foam::cpuCsrMatrixExecutor::alloc
(
    Foam::label size
) const
{
    if (size <= 0) return nullptr;
#ifdef have_sycl
    return static_cast<Type*>
    (
        Spuma::MemoryPool::getInstance()->allocate(size*sizeof(Type))
    );
#else
    Type* ptr = new Type[size];
	return ptr;
#endif
}

template<class Type>
Type* Foam::cpuCsrMatrixExecutor::allocZero
(
    Foam::label size
) const
{
    if (size <= 0) return nullptr;
#ifdef have_sycl
    Type* ptr = static_cast<Type*>
    (
        Spuma::MemoryPool::getInstance()->allocate(size*sizeof(Type))
    );
#else
    Type* ptr = new Type[size];
#endif
    for(label i=0; i<size; i++)
    {
    	ptr[i] = Type(0); //vi sy  Foam::Zero);
    }
	return ptr;
}

template<class Type>
const Type* Foam::cpuCsrMatrixExecutor::copyFromFoam
(
    Foam::label size,
	const Type* hostPtr
) const
{
    const Type* ptr = (Type*) hostPtr;
	return ptr;
}

template<class Type>
void Foam::cpuCsrMatrixExecutor::copyToFoam
(
    Foam::label size,
	Type* devPtr,
	Type** hostPtr
) const
{
    *hostPtr = devPtr;
}

template<class Type>
void Foam::cpuCsrMatrixExecutor::clear(Type* ptr) const
{
#ifdef have_sycl
    Spuma::MemoryPool::getInstance()->free(ptr);
#else
    delete[] ptr;
#endif
}

template<class Type>
void Foam::cpuCsrMatrixExecutor::clear(const Type* ptr) const
{
}

template<class Type>
void Foam::cpuCsrMatrixExecutor::concatenate
(
    label globSize,
    List<List<Type>> lst,
    Type * ptr
) const
{
    label ptrIdx = 0;

    for(label i=0; i<lst.size(); ++i)
    {
        for(label j=0; j<lst[i].size(); ++j) 
        {
            if(ptrIdx > globSize)
            {
                FatalErrorInFunction << "Concatenate size mismatch" << nl;
            }
            ptr[ptrIdx++] = lst[i].cdata()[j];
        }
    }
}

void Foam::cpuCsrMatrixExecutor::offsetCopy
(
    const scalarField& lst,
    scalar * ptr,
	label consDispl
) const
{
	NotImplemented;
}

void Foam::cpuCsrMatrixExecutor::initializeSequence
(
    const label len,
          label * vect
) const
{
    // Initialize vect = [0, 1, ... len-1]
    for(label i = 0; i < len; ++i) vect[i] = i;
}

void Foam::cpuCsrMatrixExecutor::initializeAddressing
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label * const owner,
    const label * const neighbour,
          label * rowIndTmp,
          label * colIndTmp
) const
{
    // Initialize: rowIndecesTmp = [0, ... nConsRows, (owner), (neighbour)]
    //             colIndecesTmp = [0, ... nRows1, .. 0 ... nRowsN, (neighbour), (owner)]
    for(label i=0; i<nConsIntFaces; ++i)
    {
        rowIndTmp[nConsRows + i] = owner[i];
        colIndTmp[nConsRows + i] = neighbour[i];

        rowIndTmp[nConsRows + nConsIntFaces + i] = neighbour[i];
        colIndTmp[nConsRows + nConsIntFaces + i] = owner[i];
    }

    return;
}

void Foam::cpuCsrMatrixExecutor::initializeAddressingExt
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nConsExtNz,
    const label * const owner,
    const label * const neighbour,
    const label * const extRows,
    const label * const extCols,
          label * rowIndTmp,
          label * colIndTmp
) const
{
    this->initializeAddressing
    (
        nConsRows,
        nConsIntFaces,
        owner,
        neighbour,
        rowIndTmp,
        colIndTmp
    );

    for(int i=0; i<nConsExtNz; ++i)
    {
        rowIndTmp[nConsRows + 2*nConsIntFaces + i] = extRows[i];
        colIndTmp[nConsRows + 2*nConsIntFaces + i] = extCols[i];
    }

    return;
}

void Foam::cpuCsrMatrixExecutor::computeSorting
(
    const label   totNnz,
          label * tmpPerm,
          label * rowIndTmp,
          label * rowInd,
          label * ldu2csr
) const
{
    std::pair<label, label> *pairTmp = new std::pair<label, label>[totNnz];
    for(label i=0; i<totNnz; ++i)
    {
        pairTmp[i].first = rowIndTmp[i];
        pairTmp[i].second = tmpPerm[i];
    }
    std::vector< std::pair<label,label>> pairVect(pairTmp, pairTmp+totNnz);

    // Find the permutation vector
    std::sort(pairVect.begin(), pairVect.end());

    for(label i=0; i<totNnz; ++i)
    {
        rowInd[i] = pairVect[i].first;
        ldu2csr[pairVect[i].second] = i;
    }
    delete[] pairTmp;
}


void Foam::cpuCsrMatrixExecutor::localToGlobalColIndices
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nRows,
    const label   nIntFaces,
    const label   diagIndexGlobal,
    const label   lowOffGlobal,
    const label   uppOffGlobal,
          label * colIndicesGlobal,
    const label   rowDispl, // default = 0
    const label   intFacesDispl // default = 0
) const
{
    for(label i=0; i<nRows; ++i)
    {
        colIndicesGlobal[rowDispl + i] += diagIndexGlobal;
    }

    for(label i=0; i<nIntFaces; ++i)
    {
        colIndicesGlobal[nConsRows + intFacesDispl + i] += uppOffGlobal;
        colIndicesGlobal[nConsRows + nConsIntFaces + intFacesDispl + i] += lowOffGlobal;
    }
}


void Foam::cpuCsrMatrixExecutor::localToConsRowIndex
(
    const label nConsRows,
    const label nConsIntFaces,
    const label nIntFaces,
    const label nExtNz,
    const label intFacesDipl,
    const label extDispl,
    const label offset,
          label * rowIndices
) const
{
    for(label i=0; i<nIntFaces; ++i)
    {
        rowIndices[nConsRows + intFacesDipl + i] += offset;
        rowIndices[nConsRows + nConsIntFaces + intFacesDipl + i] += offset;
    }

    for(label i=0; i<nExtNz; ++i)
    {
        rowIndices[nConsRows + 2 * nConsIntFaces + extDispl + i] += offset;
    }
}


void Foam::cpuCsrMatrixExecutor::applyAddressingPermutation
(
    const label   nCells, //NOTE: it is not used but is need for the cuda kernel
    const label   totNnz,
    const label * const ldu2csr,
    const label * const colIndTmp,
    const label * const rowInd,
          label * colInd,
          label * ownStart
) const
{
    label curRow = 0;

    ownStart[0] = 0;
    for(label i=0; i<totNnz; ++i)
    {
        colInd[ldu2csr[i]] = colIndTmp[i];
        if(curRow < rowInd[i])
        {
            ownStart[rowInd[i]] = i;
        }
        curRow = rowInd[i];
    }

    ownStart[nCells] = totNnz;
}


void Foam::cpuCsrMatrixExecutor::initializeValue
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const double * const diag,
    const double * const upper,
    const double * const lower,
          double * valuesTmp
) const
{
    for(label i=0; i<nConsRows; ++i)
    {
        valuesTmp[i] = diag[i];
    }

    for(label i=0; i<nConsIntFaces; ++i)
    {
        valuesTmp[nConsRows + i] = upper[i];
        valuesTmp[nConsRows + nConsIntFaces + i] = lower[i];
    }
}


void Foam::cpuCsrMatrixExecutor::initializeValueExt
(
    const label nConsRows,
    const label nConsIntFaces,
    const label nConsExtNz,
    const double * const diag,
    const double * const upper,
    const double * const lower,
    const double * const extValue,
          double * valuesTmp
) const
{
    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]

    initializeValue
    (
        nConsRows,
        nConsIntFaces,
        diag,
        upper,
        lower,
        valuesTmp
    );

    for(label i=0; i<nConsExtNz; ++i)
    {
        valuesTmp[nConsRows + 2*nConsIntFaces + i] = extValue[i];
    }
}

void Foam::cpuCsrMatrixExecutor::applyValuePermutation
(
    const label    totNnz,
    const label *  const ldu2csr,
    const scalar * const valuesTmp,
          scalar * values,
    const label    nBlocks
) const
{
    label blockLen = nBlocks * nBlocks;
    
    for(label i=0; i<totNnz; ++i)
    {
        for(label j=0; j<blockLen; ++j)
        {
            values[ldu2csr[i]*blockLen + j] = valuesTmp[i*blockLen + j];
        }
    }
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

#define makecpuCsrMatrixExecutor(Type)                                    \
    template Type* Foam::cpuCsrMatrixExecutor::alloc<Type>                \
    (                                                                     \
        Foam::label size                                                  \
    ) const;                                                              \
    template Type* Foam::cpuCsrMatrixExecutor::allocZero<Type>            \
    (                                                                     \
        Foam::label size                                                  \
    ) const;                                                              \
    template const Type* Foam::cpuCsrMatrixExecutor::copyFromFoam<Type>   \
    (                                                                     \
        Foam::label size,                                                 \
        const Type* hostPtr                                               \
    ) const;                                                              \
    template void Foam::cpuCsrMatrixExecutor::copyToFoam<Type>            \
    (                                                                     \
        Foam::label size,                                                 \
        Type* devPtr,                                                     \
        Type** hostPtr                                                    \
    ) const;                                                              \
    template void  Foam::cpuCsrMatrixExecutor::clear<Type>                \
    (                                                                     \
        Type* ptr                                                         \
    ) const;                                                              \
    template void  Foam::cpuCsrMatrixExecutor::clear<Type>                \
    (                                                                     \
        const Type* ptr                                                   \
    ) const;                                                              \
    template void Foam::cpuCsrMatrixExecutor::concatenate<Type>           \
    (                                                                     \
        label globSize,                                                   \
        List<List<Type>> lst,                                             \
        Type * ptr                                                        \
    ) const;

makecpuCsrMatrixExecutor(Foam::label)
makecpuCsrMatrixExecutor(Foam::scalar)

// ************************************************************************* //
