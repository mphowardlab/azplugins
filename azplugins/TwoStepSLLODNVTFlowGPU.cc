// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


#include "TwoStepSLLODNVTFlowGPU.h"
#include "TwoStepSLLODNVTFlowGPU.cuh"


#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif


namespace azplugins
{

/*! \file TwoStepSLLODNVTFlowGPU.h
    \brief Contains code for the TwoStepSLLODNVTFlowGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepSLLODNVTFlowGPU::TwoStepSLLODNVTFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ComputeThermo> thermo,
                             Scalar tau,
                             std::shared_ptr<Variant> T,
                             Scalar shear_rate,
                             const std::string& suffix)
    : TwoStepSLLODNVTFlow(sysdef, group, thermo, tau, T, shear_rate, suffix)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepSLLODNVTFlowPU when CUDA is disabled" << std::endl;
        throw std::runtime_error("Error initializing TwoStepSLLODNVTFlowGPU");
        }

    // initialize autotuner
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params.push_back(block_size);

    m_tuner_one.reset(new Autotuner(valid_params, 5, 100000, "sllod_nvt_step_one", this->m_exec_conf));
    m_tuner_two.reset(new Autotuner(valid_params, 5, 100000, "sllod_nvt_step_two", this->m_exec_conf));
    m_tuner_add_flowfield.reset(new Autotuner(valid_params, 5, 100000, "sllod_nvt_add_flow_field", this->m_exec_conf));
    m_tuner_rm_flowfield.reset(new Autotuner(valid_params, 5, 100000, "sllod_nvt_remove_flow_field", this->m_exec_conf));
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover method
*/
void TwoStepSLLODNVTFlowGPU::integrateStepOne(unsigned int timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        m_exec_conf->msg->error() << "azplugins.sllod_nvt(): Integration group empty." << std::endl;
        throw std::runtime_error("Error during NVT integration.");
        }

    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        {
        m_prof->push(m_exec_conf, "SLLOD NVT  step 1");
        }

        {
        // access all the needed data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

        // box deformation: update tilt factor of global box
        bool flipped = deformGlobalBox();

        BoxDim box = m_pdata->getBox();
        ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();

        // perform the update on the GPU
        m_tuner_one->begin();
        gpu::sllod_nvt_step_one(d_pos.data,
                         d_vel.data,
                         d_accel.data,
                         d_image.data,
                         d_index_array.data,
                         group_size,
                         box,
                         m_tuner_one->getParam(),
                         m_exp_thermo_fac,
                         m_deltaT,
                         m_shear_rate,
                         flipped,
                         m_boundary_shear_velocity,
                         m_group->getGPUPartition());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_one->end();

        m_exec_conf->endMultiGPU();
        }

    // advance thermostat
    advanceThermostat(timestep, false);

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepSLLODNVTFlowGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "SLLOD NVT step 2");

    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();

        // perform the update on the GPU
        m_tuner_two->begin();
        gpu::sllod_nvt_step_two(d_vel.data,
                         d_accel.data,
                         d_index_array.data,
                         group_size,
                         d_net_force.data,
                         m_tuner_two->getParam(),
                         m_deltaT,
                         m_shear_rate,
                         m_exp_thermo_fac,
                         m_group->getGPUPartition());


        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_two->end();

        m_exec_conf->endMultiGPU();
        }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void TwoStepSLLODNVTFlowGPU::removeFlowField()
{
  unsigned int group_size = m_group->getNumMembers();

  // profile this step
  if (m_prof)
      m_prof->push("SLLOD NVT remove flowfield");

  ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
  ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
  ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

  m_exec_conf->beginMultiGPU();

  // perform the removal of the flow field on the GPU
  m_tuner_rm_flowfield ->begin();
  gpu::sllod_nvt_remove_flow_field(d_vel.data,
                   d_pos.data,
                   d_index_array.data,
                   group_size,
                   m_tuner_rm_flowfield->getParam(),
                   m_shear_rate,
                   m_group->getGPUPartition());


  if(m_exec_conf->isCUDAErrorCheckingEnabled())
      CHECK_CUDA_ERROR();
  m_tuner_rm_flowfield->end();

  m_exec_conf->endMultiGPU();


  // done profiling
  if (m_prof)
      m_prof->pop();
}

void TwoStepSLLODNVTFlowGPU::addFlowField()
{
  unsigned int group_size = m_group->getNumMembers();

  // profile this step
  if (m_prof)
      m_prof->push("SLLOD NVT remove flowfield");

  ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
  ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
  ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

  m_exec_conf->beginMultiGPU();

  // perform the removal of the flow field on the GPU
  m_tuner_add_flowfield ->begin();
  gpu::sllod_nvt_add_flow_field(d_vel.data,
                   d_pos.data,
                   d_index_array.data,
                   group_size,
                   m_tuner_add_flowfield->getParam(),
                   m_shear_rate,
                   m_group->getGPUPartition());


  if(m_exec_conf->isCUDAErrorCheckingEnabled())
      CHECK_CUDA_ERROR();
  m_tuner_add_flowfield->end();

  m_exec_conf->endMultiGPU();


  // done profiling
  if (m_prof)
      m_prof->pop();
}

namespace detail {

void export_TwoStepSLLODNVTFlowGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepSLLODNVTFlowGPU, std::shared_ptr<TwoStepSLLODNVTFlowGPU> >(m, "TwoStepSLLODNVTFlowGPU", pybind11::base<TwoStepSLLODNVTFlow>())
    .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                          std::shared_ptr<ParticleGroup>,
                          std::shared_ptr<ComputeThermo>,
                          Scalar,
                          std::shared_ptr<Variant>,
                          Scalar,
                          const std::string&
                          >())
        ;
    }

} // end namespace detail
} // end namespace azplugins
