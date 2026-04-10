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

class PYBIND11_EXPORT ChebyshevAnisotropicPairPotential : public ForceCompute
    {
    public:
    //! Constructor
    ChebyshevAnisotropicPairPotential(std::shared_ptr<SystemDefinition> sysdef,
                                      std::shared_ptr<hoomd::md::NeighborList> nlist,
                                      const Scalar* domain,
                                      const Scalar r_cut,
                                      const unsigned int* terms,
                                      const Scalar* coeffs,
                                      unsigned int Nterms,
                                      const Scalar* r0_data,
                                      const unsigned int* r0_shape);

    //! Destructor
    virtual ~ChebyshevAnisotropicPairPotential();

    //! Detach from the neighbor list (called when removing from simulation)
    virtual void notifyDetach();

    // Getters
    std::shared_ptr<hoomd::md::NeighborList> getNeighborList() const
        {
        return m_nlist;
        }

    /// 5x2 domain: stored as 5 entries of Scalar2 = (min,max)
    const GPUArray<Scalar2>& getApproximationDomain() const
        {
        return m_domain;
        }

    /// Read-only cutoff radius
    Scalar getRCut() const
        {
        return m_r_cut;
        }

    /// Read-only number of Chebyshev terms
    unsigned int getNTerms() const
        {
        return m_Nterms;
        }

    protected:
    // member variables

    std::shared_ptr<hoomd::md::NeighborList> m_nlist; //!< Neighbor list

    GPUArray<Scalar2> m_domain; //!< Approximation domain (5x2): 5 rows, each is (min, max)

    Scalar m_r_cut; //!< Cut-off distance in approximation domain

    Scalar m_nlist_r_cut; //!< Effective neighbor-list cutoff = ceil(max(r0_data) + r_cut)

    /// r_cut matrix shared with the neighbor list (subscriber pattern)
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist;

    /// Track whether we have attached to the Simulation object
    bool m_attached = true;

    GPUArray<unsigned int> m_terms; //!< Chebyshev term list (Nterms x 6)
    GPUArray<Scalar> m_coeffs;      //!< Coefficients corresponding to each term
    unsigned int m_Nterms;          //!< Number of terms

    GPUArray<Scalar> m_r0_data;        //!< R0 data
    GPUArray<unsigned int> m_r0_shape; //!< Points per dimension to sample r0

    // methods

    void computeForces(uint64_t timestep) override;
    };

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_
