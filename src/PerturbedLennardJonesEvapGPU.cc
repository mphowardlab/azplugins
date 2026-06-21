// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PerturbedLennardJonesEvapGPU.h"

#include <algorithm>

namespace hoomd
    {
namespace azplugins
    {

PerturbedLennardJonesEvapGPU::PerturbedLennardJonesEvapGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar r_cut,
    const Scalar scale_factor,
    const param_type& params,
    bool energy_shift,
    const Scalar* attraction_scale_factor_data,
    const unsigned int* attraction_scale_factor_shape,
    const Scalar* domain,
    std::shared_ptr<VariantInterpolated> variant)
    : PerturbedLennardJonesEvap(sysdef,
                                nlist,
                                r_cut,
                                scale_factor,
                                params,
                                energy_shift,
                                attraction_scale_factor_data,
                                attraction_scale_factor_shape,
                                domain,
                                variant)
    {
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "perturbed_lennard_jones_evap"));
    this->m_autotuners.push_back(m_tuner);
    }

void PerturbedLennardJonesEvapGPU::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);

    const Scalar interface_height = (*m_variant)(timestep);
    const Scalar scaled_t = scaleTime(timestep);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    // neighbor-list arrays
    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    ArrayHandle<Scalar> d_data(m_attraction_scale_factor_data,
                               access_location::device,
                               access_mode::read);
    ArrayHandle<unsigned int> h_shape(m_attraction_scale_factor_shape,
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<Scalar> h_domain(m_domain, access_location::host, access_mode::read);

    // build the interpolator: lo = {y_lo, t_lo}, hi = {y_hi, t_hi}
    const Scalar lo[2] = {Scalar(0.0), h_domain.data[0]};
    const Scalar hi[2] = {Scalar(1.0), h_domain.data[1]};
    const LinearInterpolator2D<Scalar> interp(d_data.data, h_shape.data, lo, hi);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    m_tuner->begin();
    gpu::perturbed_lennard_jones_evap_args_t args(d_force.data,
                                                  d_pos.data,
                                                  this->m_pdata->getGlobalBox(),
                                                  this->m_pdata->getN(),
                                                  d_n_neigh.data,
                                                  d_nlist.data,
                                                  d_head_list.data,
                                                  interp,
                                                  scaled_t,
                                                  interface_height,
                                                  lj1,
                                                  lj2,
                                                  epsilon_x_4,
                                                  rcutsq,
                                                  rwcasq,
                                                  m_energy_shift,
                                                  m_tuner->getParam()[0]);

    gpu::compute_perturbed_lennard_jones_evap_forces(args);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

    } // end namespace azplugins
    } // end namespace hoomd
