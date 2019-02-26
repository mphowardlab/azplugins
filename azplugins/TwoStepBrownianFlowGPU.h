// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TwoStepBrownianFlowGPU.h
 * \brief Declaration of TwoStepBrownianFlowGPU
 */

#ifndef AZPLUGINS_TWO_STEP_BROWNIAN_FLOW_GPU_H_
#define AZPLUGINS_TWO_STEP_BROWNIAN_FLOW_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TwoStepBrownianFlow.h"
#include "TwoStepBrownianFlowGPU.cuh"
#include "hoomd/Autotuner.h"

namespace azplugins
{

template<class FlowField>
class PYBIND11_EXPORT TwoStepBrownianFlowGPU : public azplugins::TwoStepBrownianFlow<FlowField>
    {
    public:
        //! Constructor
        TwoStepBrownianFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ParticleGroup> group,
                               std::shared_ptr<Variant> T,
                               std::shared_ptr<FlowField> flow_field,
                               unsigned int seed,
                               bool use_lambda,
                               Scalar lambda,
                               bool noiseless)
            : azplugins::TwoStepBrownianFlow<FlowField>(sysdef, group, T, flow_field, seed, use_lambda, lambda, noiseless)
            {
            this->m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "langevin_flow_1", this->m_exec_conf));
            }

        //! Destructor
        virtual ~TwoStepBrownianFlowGPU() {};

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepBrownianFlow<FlowField>::setAutotunerParams(enable, period);
            this->m_tuner->setPeriod(period); this->m_tuner->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner; //!< Kernel tuner
    };

template<class FlowField>
void TwoStepBrownianFlowGPU<FlowField>::integrateStepOne(unsigned int timestep)
    {
    if (this->m_aniso)
        {
        this->m_exec_conf->msg->error() << "azplugins.integrate: anisotropic particles are not supported with brownian flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with brownian flow");
        }
    if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Brownian step");

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_net_force(this->m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(this->m_gamma, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    this->m_tuner->begin();
    azplugins::gpu::brownian_flow<FlowField>(d_pos.data,
                                             d_image.data,
                                             this->m_pdata->getBox(),
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
                                             this->m_tuner->getParam());
    if(this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner->end();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

namespace detail
{
//! Export TwoStepBrownianFlowGPU to python
template<class FlowField>
void export_TwoStepBrownianFlowGPU(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    typedef TwoStepBrownianFlow<FlowField> BrownianFlow;
    typedef TwoStepBrownianFlowGPU<FlowField> BrownianFlowGPU;

    py::class_<BrownianFlowGPU, std::shared_ptr<BrownianFlowGPU> >(m, name.c_str(), py::base<BrownianFlow>())
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
#endif // AZPLUGINS_TWO_STEP_BROWNIAN_FLOW_GPU_H_
