// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_
#define AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_

#include <cmath>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "hoomd/BoxDim.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Variant.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include "LinearInterpolator2D.h"
#include "PairEvaluatorPerturbedLennardJones.h"
#include "VariantInterpolated.h"

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
class PerturbedLennardJonesEvap : public ForceCompute
    {
    public:
    typedef detail::PairParametersPerturbedLennardJones param_type;

    PerturbedLennardJonesEvap(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<hoomd::md::NeighborList> nlist,
                              const Scalar r_cut,
                              const Scalar time_scale_factor,
                              const param_type& params,
                              bool energy_shift,
                              const Scalar* attraction_scale_factor_data,
                              const unsigned int* attraction_scale_factor_shape,
                              const Scalar* domain,
                              std::shared_ptr<VariantInterpolated> variant);

    //! Destructor
    ~PerturbedLennardJonesEvap()
        {
        if (m_r_cut_nlist)
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }

    Scalar scaleTime(uint64_t timestep) const
        {
        return Scalar(static_cast<Scalar>(timestep) / m_time_scale_factor);
        }

    protected:
    std::shared_ptr<hoomd::md::NeighborList> m_nlist; //!< Neighbor list
    Scalar epsilon_x_4;
    Scalar m_rcut;
    Scalar m_time_scale_factor; //!< Time scaling factor
    param_type m_params;
    bool m_energy_shift;
    GPUArray<Scalar> m_domain;                              //!< [t_lo, t_hi]
    GPUArray<Scalar> m_attraction_scale_factor_data;        //!< Flattened (y, t) data
    GPUArray<unsigned int> m_attraction_scale_factor_shape; //!< [ny, nt]
    GPUArray<Scalar> m_scale_factor;
    std::shared_ptr<VariantInterpolated> m_variant;

    std::shared_ptr<GPUArray<Scalar>>
        m_r_cut_nlist; //!< Cutoff matrix shared with the neighbor list

    void computeForces(uint64_t timestep) override;
    };

void export_PerturbedLennardJonesEvap(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_
