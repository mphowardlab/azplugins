// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_HARMONIC_BARRIER_GPU_H_
#define AZPLUGINS_HARMONIC_BARRIER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "HarmonicBarrier.h"
#include "HarmonicBarrierGPU.cuh"

#include "hoomd/Autotuner.h"

namespace hoomd
    {

namespace azplugins
    {

//! Harmonic barrier on the GPU
template<class BarrierEvaluatorT>
class PYBIND11_EXPORT HarmonicBarrierGPU : public HarmonicBarrier<BarrierEvaluatorT>
    {
    public:
    //! Constructor
    HarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Variant> location)
        : HarmonicBarrier<BarrierEvaluatorT>(sysdef, location)
        {
        m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                       this->m_exec_conf,
                                       "harmonic_barrier"));
        this->m_autotuners.push_back(m_tuner);
        }

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    //! Method to compute the forces
    void computeForces(uint64_t timestep) override;
    };

template<class BarrierEvaluatorT>
void HarmonicBarrierGPU<BarrierEvaluatorT>::computeForces(uint64_t timestep)
    {
    const BarrierEvaluatorT evaluator = this->makeEvaluator(timestep);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar2> d_params(this->m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    gpu::compute_harmonic_barrier(d_force.data,
                                  d_virial.data,
                                  d_pos.data,
                                  d_params.data,
                                  evaluator,
                                  this->m_pdata->getN(),
                                  this->m_pdata->getNTypes(),
                                  m_tuner->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    // virial is zeroed by GPU function, warn here
    this->warnVirialOnce();
    }

namespace detail
    {

template<class BarrierEvaluatorT>
void export_HarmonicBarrierGPU(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    py::class_<HarmonicBarrierGPU<BarrierEvaluatorT>,
               HarmonicBarrier<BarrierEvaluatorT>,
               std::shared_ptr<HarmonicBarrierGPU<BarrierEvaluatorT>>>(m, name.c_str())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }

    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_HARMONIC_BARRIER_GPU_H_
