/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2025      Cineca
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
    Foam::csrMatrix

Description
    Class that stores an OpenFOAM matrix in the CSR format 
    (Compressed Sparse Row format).

SourceFiles
    csrMatrix.H
    csrMatrix.C
    csrMatrixFunctions.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_csrMatrix_H
#define Foam_csrMatrix_H

#include "Enum.H"
#include "scalar.H"
#include "lduMatrix.H"
#include "csrMatrixExecutor.H"

// Member data below uses CUDA runtime types (cudaIpcMemHandle_t) directly.
// nvc++ -cuda includes the CUDA runtime header implicitly; other compilers
// (clang, acpp) need it spelled out.
#include <cuda_runtime_api.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/* Enumeration for the status of matrix consolidation for the solver.*/
enum class ConsolidationStatus
{
    uninitialised,
    necessary,
    notNecessary,
    initialized,
    consolidated
};

// Forward Declarations
class csrMatrix;


/*---------------------------------------------------------------------------*\
                           Class csrMatrix Declaration
\*---------------------------------------------------------------------------*/

class csrMatrix
{
    //- Private Data

    //- Private methods

protected:
    //- Protected Data

        //- Data executor
        csrMatrixExecutor csrMatExec_;

        //- Number of blocks
        label nBlocks_ = 1;

        //- Number of local rows
        label nOwnerStart_ = 0;

        //- Number of local Nz
        label nLocalNz_ = 0;

        //- Total number of rows in the consolidated matrix
        label nConsRows_;

        //- Total number of internal faces in the consolidated matrix
        label nConsIntFaces_;
        
        //- Total number of external values in the consolidated matrix
        label nConsExtNz_;

        //- Total number of non-zero values in the consolidated matrix
        // label nConsTotNz_;

        //- Numbers of non-zeros per row
        label* ownerStartPtr_;

        //- Column indices array
        label* colIndicesPtr_;

        //- Permutation ldu2csr array
        label* ldu2csrPerm_;

        //- Coefficients
        scalar* valuesPtr_;

        //- Temporary coefficients
        scalar* valuesTmpPtr_;

        //- Consolidated variable array
        scalar* psiCons_;

        //- Consolidated RHS array
        scalar* rhsCons_;

        //- Rows diplacement between consolidated processes
        labelList* rowsConsDispPtr_;

        //- Internal faces displcament between consolidated processes
        labelList* intFacesConsDispPtr_;
    
        //- External values displcament between consolidated processes
        labelList* extNzConsDispPtr_;

        //- Flag for computed permutation
        bool       hasPermutation_ = false;

        //- A flag indicating the type of consolidation applied
        ConsolidationStatus consolidationStatus_ = ConsolidationStatus::uninitialised;

        //- Index of the MPI communicator processes that talk to a GPU
        label gpuWorld_;

        //- Size of gpuWorld
        label gpuWorldSize_;

        //- Rank in gpuWorld
        label myGpuWorldRank_;

        //- Indication if the process talk directly to GPU
        bool gpuProc_ = false;

        cudaIpcMemHandle_t psiConsHandle_;
        cudaIpcMemHandle_t rhsConsHandle_;   

    //- Protected functions
        void initializeConsolidation
        (
            const label nLocalRows,
            const label nLocalIntFaces,
            const label nLocalExtNz,
            const label diagIndexGlobal,
            const label lowOffGlobal,
            const label uppOffGlobal,
            const labelList& own,
            const labelList& neigh,
            const labelList& extRows,
            const labelList& extCols,
                label& nConsTotNz,
                labelList* consDiagOffGlob,
                labelList* consLowOffGlob, 
                labelList* consUppOffGlob,
                label * &ownCons,
                label * &neighCons,
                label * &extRowsCons,
                label * &extColsConst
        );

        void initializeSequence
        (
            const label len,
                  label * vect
        ) const;

        void initializeAddressing
        (
            const label   nConsRows,
            const label   nConsIntFaces,
            const label * const owner,
            const label * const neighbour,
                  label * rowIndTmp,
                  label * colIndTmp
        );

        void initializeAddressingExt
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
        );

        void computeSorting
        (
            const label   totNnz,
                  label * tmpPerm,
                  label * rowIndTmp,
                  label * rowInd,
                  label * ldu2csr
        );

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
            const label   rowDispl = 0,
            const label   intFacesDispl = 0
        );

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
        );

        void applyAddressingPermutation
        (
            const label   nCells,
            const label   totNnz,
            const label * const ldu2csr,
            const label * const colIndTmp,
            const label * const rowInd,
                  label * colInd,
                  label * ownStart
        );

        void initializeValuesConsolidation
        (
            const scalarField& diag,
            const scalarField& upper,
            const scalarField& lower,
            const scalarField& extVal,
            scalar*& diagCons,
            scalar*& upperCons,
            scalar*& lowerCons,
            scalar*& extValCons
        );
           
        void initializeValue
        (
            const label   nConsRows,
            const label   nConsIntFaces,
            const double * const diag,
            const double * const upper,
            const double * const lower,
                  double * valuesTmp            
        );

        void initializeValueExt
        (
            const label nConsRows,
            const label nConsIntFaces,
            const label nConsExtNz,
            const double * const diag,
            const double * const upper,
            const double * const lower,
            const double * const extValue,
                  double * valuesTmp
        );

        void applyValuePermutation
        (
            const label totNnz,
            const label  * const ldu2csr,
            const scalar * const valuesTmp,
                  scalar * values,
            const label    nBlocks = 1
        );

        //- Computation of the permutatrion array
        void computePermutation(const lduAddressing * addr);

        //- Computation of the permutatrion array
        void computePermutation
        (
            const lduAddressing& addr,
            const lduInterfacePtrsList& interfaces,
                  label& nExtNz
        );

public:

    //- Public Types

    //- Static Data

        //- Declare name of the class and its debug switch
        ClassName("csrMatrix");

    //- Constructors

        //- Construct default.
        //- The coefficients are initialized empty.
        csrMatrix(word mode);

//        //- Construct as copy
//        csrMatrix(const csrMatrix& A);
//
//        //- Construct as copy or re-use as specified.
//        csrMatrix(csrMatrix& A, bool reuse);

        //- Construct given an LDU addressed mesh and an Istream
        //- from which the coefficients are read
        // lduMatrix(const lduMesh&, Istream&);


    //- Destructor
        ~csrMatrix() = default;


    //- Member Functions

        //- Access to addressing

            //- Number of blocks
            label nBlocks() const
            {
                return nBlocks_;
            }

            //- Number of local rows
            label nOwnerStart() const
            {
                return nOwnerStart_;
            }
            //- Number of blocks
            label nLocalNz() const
            {
                return nLocalNz_;
            }

            //- Return vector of the ownerStart
            const label* ownerStart() const
            {
                return ownerStartPtr_;
            }

            //- Return the column index
            const label* colIndices() const
            {
                return colIndicesPtr_;
            }

            //- Return the column index
            const label* ldu2csrPerm() const
            {
                return ldu2csrPerm_;
            }

            //- Retrun rows displacement in consolidated matrix
            label nConsRows() const
            {
                return nConsRows_;
            }

            const labelList& rowsConsDisp() const
            {
                return *rowsConsDispPtr_;
            }

            const labelList& intFaceConsDisp() const
            {
                return *intFacesConsDispPtr_;
            }

            const labelList& extNzConsDisp() const
            {
                return *extNzConsDispPtr_;
            }

            //- Return true if the permutation has been computed
            bool hasPermutation() const noexcept
            {
                return hasPermutation_;
            }

            bool isConsolidated() const
            {
                return consolidationStatus_ == ConsolidationStatus::initialized;
            }

        //- Access to coefficients
            virtual const scalar* values() const;

            bool hasValues() const noexcept
            {
                return (valuesPtr_);
            }

            virtual const scalar* rhsCons() const
            {
                return rhsCons_;
            }

            virtual scalar* psiCons() const
            {
                return psiCons_;
            }

        //- Operations

            //- Set number of blocks
            void setNblocks(label n)
            {
                nBlocks_ = n;
            }

            //- Set number of blocks
            void setGpuProc(bool proc)
            {
                gpuProc_ = proc;
            }

            //- Deallocate useless addressing pointer
            void clearAddressing();

            //- Initialize matrix comunications for consolidation
            void initializeComms(label commId, bool gpuProc);

            //- Apply permutation from LDU to CSR
            void applyPermutation(const lduMatrix& lduMatrix);

            //- Apply permutation from LDU to CSR considering the interface values
            void applyPermutation
            (
                const lduMatrix& lduMatrix,
                const FieldField<Field, scalar> interfaceBouCoeffs
            );

            //- Allocate consolidated variable and source vector
            void createConsVect
            (
                const scalarField& rhs,
                      scalarField& psi
            );

            //- Redistribute solution vector
            void distributeSolution(scalarField& psi);

            //- Deallocate useless addressing pointer
            void finalize();

            void printValues()
            {
            	label size = nLocalNz_*nBlocks_*nBlocks_;
            	Info << "SIZE: " << size << endl;
            	scalarField val(size);
            	scalar* valData = val.data();
            	scalar** valPtr = &valData;
                std::visit([this, &valPtr, size](const auto& exec)
                           { exec.template copyToFoam<scalar>(size, this->valuesPtr_, valPtr);},
                           csrMatExec_);
                Info << "printing values :" << val << endl;

            };

            void printLdu2csr()
            {
            	label size = nLocalNz_;
            	Info << "SIZE: " << size << endl;
            	labelField val(size);
            	label* valData = val.data();
            	label** valPtr = &valData;
                std::visit([this, &valPtr, size](const auto& exec)
                           { exec.template copyToFoam<label>(size, this->ldu2csrPerm_, valPtr);},
                           csrMatExec_);
                Info << "printing ldu2csr :" << val << endl;

            };
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
