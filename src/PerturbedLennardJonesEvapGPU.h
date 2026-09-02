// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2026, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_H_
#define AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_H_

#ifdef ENABLE_HIP

#include "PerturbedLennardJonesEvap.h"
#include "PerturbedLennardJonesEvapGPU.cuh"
#include "hoomd/Autotuner.h"

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
class PerturbedLennardJonesEvapGPU : public PerturbedLennardJonesEvap
    {
    public:
    //! Constructor

    typedef PairParametersPerturbedLennardJones param_type;

    PerturbedLennardJonesEvapGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<hoomd::md::NeighborList> nlist,
                                 const Scalar r_cut,
                                 const Scalar scale_factor,
                                 bool energy_shift,
                                 const Scalar* attraction_scale_factor_data,
                                 const unsigned int* attraction_scale_factor_shape,
                                 const Scalar* domain,
                                 std::shared_ptr<VariantInterpolated> variant);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner;
    void computeForces(uint64_t timestep) override;
    };

void export_PerturbedLennardJonesEvapGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_H_
