// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*! \file WallRestraintCompute.h
 *  \brief Computes harmonic restraint forces relative to a WallData object
 */

#ifndef AZPLUGINS_WALL_RESTRAINT_COMPUTE_H_
#define AZPLUGINS_WALL_RESTRAINT_COMPUTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/md/WallData.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace azplugins
{
//! Applies a harmonic force relative to a WallData object for a group of particles
/*
 * Particles are restrained to a surface defined by a WallData object, which currently
 * includes a plane, a cylinder, and a sphere using a harmonic potential that is a
 * function of the distance \a d from this surface:
 *
 * \f[ V(d) = \frac{k}{2} d^2 \f]
 *
 * This restraint is implemented as a ForceCompute (and not an external potential)
 * because it acts on a ParticleGroup (and not all particles by type).
 */
template<class T>
class PYBIND11_EXPORT WallRestraintCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        /*!
         * \param sysdef HOOMD system definition.
         * \param group Particle group to compute on.
         * \param wall Restraint wall.
         * \param k Force constant.
         */
        WallRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<T> wall,
                             Scalar k)
            : ForceCompute(sysdef), m_group(group), m_wall(wall), m_k(k)
            {
            m_exec_conf->msg->notice(5) << "Constructing WallRestraintCompute[T = " << typeid(T).name() << "]" << std::endl;
            }

        //! Destructor
        virtual ~WallRestraintCompute()
            {
            m_exec_conf->msg->notice(5) << "Destroying WallRestraintCompute[T = " << typeid(T).name() << "]" << std::endl;
            }

        //! Get the force constant
        Scalar getForceConstant() const
            {
            return m_k;
            }

        //! Set the force constant
        void setForceConstant(Scalar k)
            {
            m_k = k;
            }

        //! Get the wall object for this class
        std::shared_ptr<T> getWall() const
            {
            return m_wall;
            }

        //! Set the wall object for this class
        void setWall(std::shared_ptr<T> wall)
            {
            m_wall = wall;
            }

    protected:
        std::shared_ptr<ParticleGroup> m_group; //!< Group to apply forces to
        std::shared_ptr<T> m_wall;              //!< WallData object for restraint
        Scalar m_k;                             //!< Spring constant

        //! Actually compute the forces
        /*!
         * \param timestep Current timestep
         *
         * Harmonic forces are computed on all particles in group based on their distance from the surface.
         */
        virtual void computeForces(unsigned int timestep)
            {
            ArrayHandle<unsigned int> h_group(m_group->getIndexArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
            const BoxDim box = m_pdata->getGlobalBox();

            ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

            // zero the forces and virial before calling
            memset((void*)h_force.data, 0, sizeof(Scalar4)*m_force.getNumElements());
            memset((void*)h_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

            const T wall = *m_wall;

            for (unsigned int idx=0; idx < m_group->getNumMembers(); ++idx)
                {
                const unsigned int pidx = h_group.data[idx];

                // unwrapped particle coordinate
                const Scalar4 pos = h_pos.data[pidx];
                const int3 image = h_image.data[pidx];
                const Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

                // vector to point from surface (inside is required but not used by this potential)
                bool inside;
                const vec3<Scalar> dr = vecPtToWall(wall, vec3<Scalar>(box.shift(r, image)), inside);

                // force points along the point-to-wall vector (cancellation of minus signs)
                const Scalar3 force = vec_to_scalar3(m_k*dr);

                // squared-distance gives energy
                const Scalar energy = Scalar(0.5)*m_k*dot(dr,dr);

                // virial is dyadic product of force with position (in this box)
                Scalar virial[6];
                virial[0] = force.x * r.x;
                virial[1] = force.x * r.y;
                virial[2] = force.x * r.z;
                virial[3] = force.y * r.y;
                virial[4] = force.y * r.z;
                virial[5] = force.z * r.z;

                h_force.data[pidx] = make_scalar4(force.x, force.y, force.z, energy);
                for (unsigned int j=0; j < 6; ++j)
                    h_virial.data[m_virial_pitch*j+pidx] = virial[j];
                }
            }
    };

namespace detail
{
//! Exports the WallRestraintCompute to python
/*!
 * \param m Python module to export to.
 * \param name Name for the potential.
 */
template<class T>
void export_WallRestraintCompute(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    py::class_<WallRestraintCompute<T>,std::shared_ptr<WallRestraintCompute<T>>>(m,name.c_str(),py::base<ForceCompute>())
    .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,std::shared_ptr<T>,Scalar>())
    .def("getForceConstant", &WallRestraintCompute<T>::getForceConstant)
    .def("setForceConstant", &WallRestraintCompute<T>::setForceConstant)
    .def("getWall", &WallRestraintCompute<T>::getWall)
    .def("setWall", &WallRestraintCompute<T>::setWall)
    ;
    }

//! Exports the PlaneWall to python
/*!
 * \param m Python module to export to.
 */
void export_PlaneWall(pybind11::module& m);

//! Exports the CylinderWall to python
/*!
 * \param m Python module to export to.
 */
void export_CylinderWall(pybind11::module& m);

//! Exports the SphereWall to python
/*!
 * \param m Python module to export to.
 */
void export_SphereWall(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_WALL_RESTRAINT_COMPUTE_H_
