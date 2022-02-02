// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*! \file WallRestraintComputeGPU.h
 *  \brief Computes harmonic restraint forces relative to a plane, on the GPU
 */

#ifndef AZPLUGINS_WALL_RESTRAINT_COMPUTE_GPU_H_
#define AZPLUGINS_WALL_RESTRAINT_COMPUTE_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "WallRestraintCompute.h"
#include "WallRestraintComputeGPU.cuh"

#include "hoomd/Autotuner.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace azplugins
{
//! Applies a harmonic force relative to a WallData object for a group of particles on the GPU
template<class T>
class PYBIND11_EXPORT WallRestraintComputeGPU : public WallRestraintCompute<T>
    {
    public:
        //! Constructs the compute
        /*!
         * \param sysdef HOOMD system definition.
         * \param group Particle group to compute on.
         * \param wall Restraint wall.
         * \param k Force constant.
         */
        WallRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<ParticleGroup> group,
                                std::shared_ptr<T> wall,
                                Scalar k)
            : WallRestraintCompute<T>(sysdef, group, wall, k)
            {
            this->m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "harmonic_plane", this->m_exec_conf));
            }

    protected:
        //! Actually compute the forces on the GPU
        /*!
         * \param timestep Current timestep
         *
         * Harmonic forces are computed on all particles in group based on their distance from the surface.
         */
        virtual void computeForces(unsigned int timestep)
            {
            ArrayHandle<unsigned int> d_group(this->m_group->getIndexArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

            // zero the forces and virial before calling
            cudaMemset((void*)d_force.data, 0, sizeof(Scalar4)*this->m_force.getNumElements());
            cudaMemset((void*)d_virial.data, 0, sizeof(Scalar)*this->m_virial.getNumElements());

            this->m_tuner->begin();
            gpu::compute_wall_restraint(d_force.data,
                                        d_virial.data,
                                        d_group.data,
                                        d_pos.data,
                                        d_image.data,
                                        this->m_pdata->getGlobalBox(),
                                        *(this->m_wall),
                                        this->m_k,
                                        this->m_group->getNumMembers(),
                                        this->m_virial_pitch,
                                        this->m_tuner->getParam());
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            this->m_tuner->end();
            }

    private:
        std::shared_ptr<Autotuner> m_tuner; //!< Tuner for force kernel
    };

namespace detail
{
//! Exports the WallRestraintComputeGPU to python
/*!
 * \param m Python module to export to.
 * \param name Name for the potential.
 */
template<class T>
void export_WallRestraintComputeGPU(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    py::class_<WallRestraintComputeGPU<T>,std::shared_ptr<WallRestraintComputeGPU<T>>>(m,name.c_str(),py::base<WallRestraintCompute<T>>())
    .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,std::shared_ptr<T>,Scalar>())
    ;
    }
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_WALL_RESTRAINT_COMPUTE_GPU_H_
