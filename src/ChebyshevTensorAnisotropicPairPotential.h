// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevTensorAnisotropicPairPotential.h
 * \brief Declaration of ChebyshevTensorAnisotropicPairPotential
 */

#ifndef AZPLUGINS_CHEBYSHEV_TENSOR_ANISO_PAIR_POTENTIAL_H_
#define AZPLUGINS_CHEBYSHEV_TENSOR_ANISO_PAIR_POTENTIAL_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include "hoomd/md/NeighborList.h"

namespace hoomd
    {
namespace azplugins
    {

class R0Interpolator;

class PYBIND11_EXPORT ChebyshevTensorAnisotropicPairPotential : public ForceCompute
    {
    public:
    //! Constructor
    ChebyshevTensorAnisotropicPairPotential(std::shared_ptr<SystemDefinition> sysdef,
                                            std::shared_ptr<hoomd::md::NeighborList> nlist);

    //! Destructor
    virtual ~ChebyshevTensorAnisotropicPairPotential();

    // Getters
    std::shared_ptr<hoomd::md::NeighborList> getNeighborList() const
        {
        return m_nlist;
        }

    /// 6x2 domain: stored as 6 entries of Scalar2 = (min,max)
    const GPUArray<Scalar2>& getApproximationDomain() const
        {
        return m_domain;
        }

    /// Term degrees (Nterms x 6)
    const GPUArray<unsigned int>& getChebyshevTermList() const
        {
        return m_terms;
        }

    /// Coefficients (length = Nterms)
    const GPUArray<Scalar>& getCoefficients() const
        {
        return m_coeffs;
        }

    /// Max degree per dimension length (length = 6)
    const GPUArray<unsigned int>& getMaxDegreePerDim() const
        {
        return m_max_degree;
        }

    /// Number of terms (int)
    unsigned int getNTerms() const
        {
        return m_Nterms;
        }

    /// R0 interpolator object
    std::shared_ptr<R0Interpolator> getR0Interpolator() const
        {
        return m_r0_interp;
        }

    /// Allocate storage for term list and coefficients.
    void resizeTerms(unsigned int Nterms);

    void setR0Interpolator(std::shared_ptr<R0Interpolator> interp)
        {
        m_r0_interp = interp;
        }

    protected:
    void computeForces(uint64_t timestep) override;

    private:
    // 1) neighbor list object
    std::shared_ptr<hoomd::md::NeighborList> m_nlist;

    // 2) approximation domain (6x2): 6 rows, each is (min,max)
    GPUArray<Scalar2> m_domain;

    // 3) r0 interpolation object
    std::shared_ptr<R0Interpolator> m_r0_interp;

    // 4) Chebyshev term list (Nterms x 6)
    GPUArray<unsigned int> m_terms;

    // 5) coeffs (Nterms)
    GPUArray<Scalar> m_coeffs;

    // 6) max degree per dimension (6)
    GPUArray<unsigned int> m_max_degree;

    // 7) number of terms
    unsigned int m_Nterms;
    };

namespace detail
    {
///! exports to Python
void export_ChebyshevTensorAnisotropicPairPotential(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_CHEBYSHEV_TENSOR_ANISO_PAIR_POTENTIAL_H_
