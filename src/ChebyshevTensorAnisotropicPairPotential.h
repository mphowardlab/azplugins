// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevTensorAnisotropicPairPotential.h
 * \brief Declaration of ChebyshevTensorAnisotropicPairPotential
 */

#ifndef AZPLUGINS_CHEBYSHEV_TENSOR_ANISOTROPIC_PAIR_POTENTIAL_H_
#define AZPLUGINS_CHEBYSHEV_TENSOR_ANISOTROPIC_PAIR_POTENTIAL_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <array>
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

class LinearInterpolator5D;

class PYBIND11_EXPORT ChebyshevTensorAnisotropicPairPotential : public ForceCompute
    {
    public:
    //! Constructor
    ChebyshevTensorAnisotropicPairPotential(std::shared_ptr<SystemDefinition> sysdef,
                                            std::shared_ptr<hoomd::md::NeighborList> nlist,
                                            const std::array<Scalar2, 6>& domain,
                                            const std::vector<Scalar>& r0_data,
                                            const std::vector<unsigned int>& terms,
                                            const std::vector<Scalar>& coeffs);

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

    /// r0 data (theta, phi, alpha, beta, gamma) (N x 6)
    const GPUArray<Scalar>& getR0Data() const
        {
        return m_r0_data;
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

    unsigned int getNTerms() const
        {
        return m_Nterms;
        }

    /// Allocate storage for term list and coefficients.
    void resizeTerms(unsigned int Nterms);

    protected:
    void computeForces(uint64_t timestep) override;

    private:
    // 1) neighbor list object
    std::shared_ptr<hoomd::md::NeighborList> m_nlist;

    // 2) approximation domain (6x2): 6 rows, each is (min,max)
    GPUArray<Scalar2> m_domain;

    // 3) intenal r0 linear interpolator
    std::unique_ptr<LinearInterpolator5D> m_r0_interp;

    // 4) r0_data
    GPUArray<Scalar> m_r0_data;

    // 5) Chebyshev term list (Nterms x 6)
    GPUArray<unsigned int> m_terms;

    // 6) coeffs (Nterms)
    GPUArray<Scalar> m_coeffs;

    // 7) number of terms
    unsigned int m_Nterms = 0;
    };

namespace detail
    {
///! exports to Python
void export_ChebyshevTensorAnisotropicPairPotential(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_CHEBYSHEV_TENSOR_ANISOTROPIC_PAIR_POTENTIAL_H_
