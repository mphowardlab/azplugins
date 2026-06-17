// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_H_
#define AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_H_

#include "PerturbedLennardJonesEvapGPU.cuh"
#include "PerturbedLennardJonesEvap.h"

namespace hoomd
    {
namespace azplugins
    {
class PYBIND11_EXPORT PerturbedLennardJonesEvapGPU : public PerturbedLennardJonesEvap
    {
    public:
    //! Constructor
    PerturbedLennardJonesEvapGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<hoomd::md::NeighborList> nlist,
                                 const Scalar r_cut,
                                 const Scalar scale_factor,
                                 const param_type& params,
                                 bool energy_shift,
                                 const Scalar* attraction_scale_factor_data,
                                 const unsigned int* attraction_scale_factor_shape,
                                 const Scalar* domain,
                                 std::shared_ptr<VariantInterpolated> variant);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner;
    void computeForces(uint64_t timestep) override;
    };

namespace detail
    {
void export_PerturbedLennardJonesEvapGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_H_
