// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "TwoStepSLLODLangevinFlowGPU.h"
#include "TwoStepSLLODLangevinFlowGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif



/*! \file TwoStepSLLODLangevinFlowGPU.h
    \brief Contains code for the TwoStepSLLODLangevinFlowGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param suffix Suffix to attach to the end of log quantity names
*/
namespace azplugins
{
TwoStepSLLODLangevinFlowGPU::TwoStepSLLODLangevinFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                                       std::shared_ptr<Variant> T,
                                       Scalar shear_rate,
                                       unsigned int seed,
                                       bool use_lambda,
                                       Scalar lambda,
                                       bool noiseless,
                                       const std::string& suffix)
    : TwoStepSLLODLangevinFlow(sysdef, group, T, shear_rate, seed, use_lambda, lambda, noiseless, suffix)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepSLLODLangevinFlowGPU while CUDA is disabled" << std::endl;
        throw std::runtime_error("Error initializing TwoStepSLLODLangevinFlowGPU");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum(1, m_exec_conf);
    m_sum.swap(sum);

    // initialize the partial sum array
    m_block_size = 256;
    unsigned int group_size = m_group->getNumMembers();
    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<Scalar> partial_sum1(m_num_blocks, m_exec_conf);
    m_partial_sum1.swap(partial_sum1);

    cudaDeviceProp dev_prop = m_exec_conf->dev_prop;
    m_tuner_1.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, "langevin_step_1", this->m_exec_conf));
    m_tuner_2.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, "langevin_step_2", this->m_exec_conf));
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.

    This method is copied directly from TwoStepNVEGPU::integrateStepOne() and reimplemented here to avoid multiple.
*/
void TwoStepSLLODLangevinFlowGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "Langevin step 1");

    // box deformation: update tilt factor of global box
    bool flipped = deformGlobalBox();

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    unsigned int group_size = m_group->getNumMembers();

    m_exec_conf->beginMultiGPU();
    m_tuner_1->begin();
    // perform the update on the GPU

    gpu::langevin_sllod_step1(d_pos.data,
                                        d_image.data,
                                        d_vel.data,
                                        d_accel.data,
                                        d_index_array.data,
                                        box,
                                        group_size,
                                        m_deltaT,
                                        m_shear_rate,
                                        flipped,
                                        m_boundary_shear_velocity,
                                        m_tuner_1->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_1->end();
    m_exec_conf->endMultiGPU();

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepSLLODLangevinFlowGPU::integrateStepTwo(unsigned int timestep)
    {
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "Langevin step 2");

    // get the dimensionality of the system
    const unsigned int D = m_sysdef->getNDimensions();

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);

    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        {
        ArrayHandle<Scalar> d_partial_sumBD(m_partial_sum1, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sumBD(m_sum, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        unsigned int group_size = m_group->getNumMembers();
        m_num_blocks = group_size / m_block_size + 1;


        // this template could be implicitly deduced, but writing it explicitly to make it clear to the caller
        this->m_tuner_2->begin();
        gpu::langevin_sllod_step2(d_vel.data,
                                                       d_accel.data,
                                                       d_pos.data,
                                                       d_net_force.data,
                                                       d_tag.data,
                                                       d_index_array.data,
                                                       d_diameter.data,
                                                       m_lambda,
                                                       d_gamma.data,
                                                       m_pdata->getNTypes(),
                                                       m_group->getNumMembers(),
                                                       m_deltaT,
                                                       m_T->getValue(timestep),
                                                       timestep,
                                                       m_seed,
                                                       m_noiseless,
                                                       m_use_lambda,
                                                       m_shear_rate,
                                                       m_tuner_2->getParam());
        if(this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        this->m_tuner_2->end();

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        }



    if (m_tally)
        {
        ArrayHandle<Scalar> h_sumBD(m_sum, access_location::host, access_mode::read);
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &h_sumBD.data[0], 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        #endif
        m_reservoir_energy -= h_sumBD.data[0]*m_deltaT;
        m_extra_energy_overdeltaT= 0.5*h_sumBD.data[0];
        }
    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

namespace detail
{
void export_TwoStepSLLODLangevinFlowGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepSLLODLangevinFlowGPU, std::shared_ptr<TwoStepSLLODLangevinFlowGPU> >(m, "TwoStepSLLODLangevinFlowGPU", pybind11::base<TwoStepSLLODLangevinFlow>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                               std::shared_ptr<ParticleGroup>,
                               std::shared_ptr<Variant>,
                               Scalar,
                               unsigned int,
                               bool,
                               Scalar,
                               bool,
                               const std::string&>()
                           );
    }
} // end namespace detail
} // end namespace azplugins
