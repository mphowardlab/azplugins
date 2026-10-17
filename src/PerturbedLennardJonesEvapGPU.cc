// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2026, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PerturbedLennardJonesEvapGPU.h"

#include <algorithm>

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
PerturbedLennardJonesEvapGPU::PerturbedLennardJonesEvapGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar r_cut,
    const Scalar scale_factor,
    bool energy_shift,
    const Scalar* attraction_scale_factor_data,
    const unsigned int* attraction_scale_factor_shape,
    const Scalar* domain,
    std::shared_ptr<VariantInterpolated> variant)
    : PerturbedLennardJonesEvap(sysdef,
                                nlist,
                                r_cut,
                                scale_factor,
                                energy_shift,
                                attraction_scale_factor_data,
                                attraction_scale_factor_shape,
                                domain,
                                variant)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PerturbedLennardJonesPotentialGPU with no GPU in the "
            << "execution configuration" << std::endl;
        throw std::runtime_error("Error initializing PerturbedLennardJonesEvapPotentialGPU");
        }

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "perturbed_lennard_jones_evap"));
    this->m_autotuners.push_back(m_tuner);
    }

void PerturbedLennardJonesEvapGPU::computeForces(uint64_t timestep)
    {
    this->m_nlist->compute(timestep);
    this->m_force.zeroFill();
    const Scalar interface_height = (*m_variant)(timestep);
    const Scalar scaled_t = Scalar(static_cast<Scalar>(timestep) / m_time_scale_factor);

    const BoxDim box = this->m_pdata->getGlobalBox();
    const unsigned int N = this->m_pdata->getN();
    const unsigned int n_ghost = this->m_pdata->getNGhosts();
    const unsigned int Ntot = N + n_ghost;
    const unsigned int ntypes = this->m_pdata->getNTypes();

    if (m_scale_factor.getNumElements() < Ntot)
        {
        GPUArray<Scalar> tmp(Ntot, m_exec_conf);
        m_scale_factor.swap(tmp);
        }

    ArrayHandle<Scalar> d_data(m_attraction_scale_factor_data,
                               access_location::device,
                               access_mode::read);
    ArrayHandle<unsigned int> h_shape(m_attraction_scale_factor_shape,
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<Scalar> h_domain(m_domain, access_location::host, access_mode::read);

    const Scalar lo[2] = {Scalar(0.0), h_domain.data[0]};
    const Scalar hi[2] = {Scalar(1.0), h_domain.data[1]};

    LinearInterpolator2D<Scalar> interp(d_data.data, h_shape.data, lo, hi);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar> d_scale_factor(m_scale_factor,
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<param_type> d_params(m_params, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    const Scalar rcutsq = m_rcut * m_rcut;
    this->m_tuner->begin();
    gpu::perturbed_lennard_jones_evap_args_t args(d_force.data,
                                                  d_pos.data,
                                                  d_scale_factor.data,
                                                  box,
                                                  N,
                                                  n_ghost,
                                                  d_n_neigh.data,
                                                  d_nlist.data,
                                                  d_head_list.data,
                                                  interp,
                                                  scaled_t,
                                                  interface_height,
                                                  rcutsq,
                                                  d_params.data,
                                                  ntypes,
                                                  m_energy_shift,
                                                  m_tuner->getParam()[0]);

    gpu::compute_perturbed_lennard_jones_evap_forces(args);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner->end();
    }
namespace py = pybind11;

void export_PerturbedLennardJonesEvapGPU(py::module& m)
    {
    py::class_<PerturbedLennardJonesEvapGPU,
               PerturbedLennardJonesEvap,
               std::shared_ptr<PerturbedLennardJonesEvapGPU>>(m, "PerturbedLennardJonesEvapGPU")
        .def(py::init(
            [](std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<hoomd::md::NeighborList> nlist,
               Scalar rcut,
               Scalar scale_factor,
               bool energy_shift,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast>
                   attraction_scale_factor_data,
               py::array_t<unsigned int, py::array::c_style | py::array::forcecast>
                   attraction_scale_factor_shape,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> domain,
               std::shared_ptr<VariantInterpolated> variant)
            {
                if (attraction_scale_factor_shape.size() != 2)
                    throw std::runtime_error("lambda_shape must have 2 elements");
                if (domain.size() != 2)
                    throw std::runtime_error(
                        "domain must have 2 elements [y_lo, y_hi, t_lo, t_hi]");

                const unsigned int* shape_ptr = attraction_scale_factor_shape.data();
                const Scalar* data_ptr = attraction_scale_factor_data.data();
                const Scalar* dom_ptr = domain.data();

                if (attraction_scale_factor_data.size()
                    != static_cast<py::ssize_t>(shape_ptr[0] * shape_ptr[1]))
                    throw std::runtime_error(
                        "attraction_scale_factor_data size does not match lambda_shape");

                return std::make_shared<PerturbedLennardJonesEvapGPU>(sysdef,
                                                                      nlist,
                                                                      rcut,
                                                                      scale_factor,
                                                                      energy_shift,
                                                                      data_ptr,
                                                                      shape_ptr,
                                                                      dom_ptr,
                                                                      variant);
            }));
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
