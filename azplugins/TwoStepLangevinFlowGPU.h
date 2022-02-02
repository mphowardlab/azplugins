// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file TwoStepLangevinFlowGPU.h
 * \brief Declaration of TwoStepLangevinFlowGPU
 */

#ifndef AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_GPU_H_
#define AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TwoStepLangevinFlow.h"
#include "TwoStepLangevinFlowGPU.cuh"
#include "hoomd/Autotuner.h"

namespace azplugins
{

template<class FlowField>
class PYBIND11_EXPORT TwoStepLangevinFlowGPU : public azplugins::TwoStepLangevinFlow<FlowField>
    {
    public:
        //! Constructor
        TwoStepLangevinFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ParticleGroup> group,
                               std::shared_ptr<Variant> T,
                               std::shared_ptr<FlowField> flow_field,
                               unsigned int seed,
                               bool use_lambda,
                               Scalar lambda,
                               bool noiseless)
            : azplugins::TwoStepLangevinFlow<FlowField>(sysdef, group, T, flow_field, seed, use_lambda, lambda, noiseless)
            {
            this->m_tuner_1.reset(new Autotuner(32, 1024, 32, 5, 100000, "langevin_flow_1", this->m_exec_conf));
            this->m_tuner_2.reset(new Autotuner(32, 1024, 32, 5, 100000, "langevin_flow_2", this->m_exec_conf));
            }

        //! Destructor
        virtual ~TwoStepLangevinFlowGPU() {};

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepLangevinFlow<FlowField>::setAutotunerParams(enable, period);
            this->m_tuner_1->setPeriod(period); this->m_tuner_1->setEnabled(enable);
            this->m_tuner_2->setPeriod(period); this->m_tuner_2->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner_1;   //!< Kernel tuner for step 1
        std::unique_ptr<Autotuner> m_tuner_2;   //!< Kernel tuner for step 2
    };

template<class FlowField>
void TwoStepLangevinFlowGPU<FlowField>::integrateStepOne(unsigned int timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "azplugins.integrate: anisotropic particles are not supported with langevin flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with langevin flow");
        }
    if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Langevin step 1");

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(), access_location::device, access_mode::read);
    const BoxDim& box = this->m_pdata->getBox();
    unsigned int group_size = this->m_group->getNumMembers();

    // perform the update on the GPU
    this->m_tuner_1->begin();
    azplugins::gpu::langevin_flow_step1(d_pos.data,
                                        d_image.data,
                                        d_vel.data,
                                        d_accel.data,
                                        d_group.data,
                                        box,
                                        group_size,
                                        this->m_deltaT,
                                        this->m_tuner_1->getParam());
    if(this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner_1->end();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }


template<class FlowField>
void TwoStepLangevinFlowGPU<FlowField>::integrateStepTwo(unsigned int timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "azplugins.integrate: anisotropic particles are not supported with langevin flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with langevin flow");
        }
    if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Langevin step 2");

    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_net_force(this->m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(this->m_gamma, access_location::device, access_mode::read);

    // this template could be implicitly deduced, but writing it explicitly to make it clear to the caller
    this->m_tuner_2->begin();
    azplugins::gpu::langevin_flow_step2<FlowField>(d_vel.data,
                                                   d_accel.data,
                                                   d_pos.data,
                                                   d_net_force.data,
                                                   d_tag.data,
                                                   d_group.data,
                                                   d_diameter.data,
                                                   this->m_lambda,
                                                   d_gamma.data,
                                                   this->m_pdata->getNTypes(),
                                                   *(this->m_flow_field),
                                                   this->m_group->getNumMembers(),
                                                   this->m_deltaT,
                                                   this->m_T->getValue(timestep),
                                                   timestep,
                                                   this->m_seed,
                                                   this->m_noiseless,
                                                   this->m_use_lambda,
                                                   this->m_tuner_2->getParam());
    if(this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner_2->end();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

namespace detail
{
//! Export TwoStepLangevinFlowGPU to python
template<class FlowField>
void export_TwoStepLangevinFlowGPU(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    typedef TwoStepLangevinFlow<FlowField> LangevinFlow;
    typedef TwoStepLangevinFlowGPU<FlowField> LangevinFlowGPU;

    py::class_<LangevinFlowGPU, std::shared_ptr<LangevinFlowGPU> >(m, name.c_str(), py::base<LangevinFlow>())
        .def(py::init<std::shared_ptr<SystemDefinition>,
                      std::shared_ptr<ParticleGroup>,
                      std::shared_ptr<Variant>,
                      std::shared_ptr<FlowField>,
                      unsigned int,
                      bool,
                      Scalar,
                      bool>());
    }
} // end namespace detail
}
#endif // AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_GPU_H_
