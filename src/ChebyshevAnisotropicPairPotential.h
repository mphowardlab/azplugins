// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevAnisotropicPairPotential.h
 * \brief Declaration of ChebyshevAnisotropicPairPotential
 */

#ifndef AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_
#define AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_

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

class PYBIND11_EXPORT ChebyshevAnisotropicPairPotential : public ForceCompute
    {
    public:
    //! Constructor
    ChebyshevAnisotropicPairPotential(std::shared_ptr<SystemDefinition> sysdef,
                                      std::shared_ptr<hoomd::md::NeighborList> nlist,
                                      const Scalar* domain,
                                      const Scalar* r0_data,
                                      const unsigned int* r0_shape,
                                      unsigned int Nterms,
                                      const unsigned int* terms,
                                      const Scalar* coeffs);

    //! Destructor
    virtual ~ChebyshevAnisotropicPairPotential();

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

    protected:
    void computeForces(uint64_t timestep) override;

    // neighbor list object
    std::shared_ptr<hoomd::md::NeighborList> m_nlist;

    // approximation domain (6x2): 6 rows, each is (min,max)
    GPUArray<Scalar2> m_domain;

    // intenal r0 linear interpolator
    std::unique_ptr<LinearInterpolator5D> m_r0_interp;

    // r0_data
    GPUArray<Scalar> m_r0_data;

    std::array<unsigned int, 5> m_r0_shape;

    // Chebyshev term list (Nterms x 6)
    GPUArray<unsigned int> m_terms;

    // coeffs (Nterms)
    GPUArray<Scalar> m_coeffs;

    // number of terms
    unsigned int m_Nterms = 0;
    };

namespace detail
    {
///! exports to Python
void export_ChebyshevAnisotropicPairPotential(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_
