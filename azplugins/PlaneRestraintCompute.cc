// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "PlaneRestraintCompute.h"

namespace azplugins
{
/*!
 * \param sysdef HOOMD system definition.
 * \param group Particle group to compute on.
 * \param point Point in the plane.
 * \param normal Normal to the plane.
 * \param k Force constant.
 */
PlaneRestraintCompute::PlaneRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<ParticleGroup> group,
                                             Scalar3 point,
                                             Scalar3 normal,
                                             Scalar k)
    : ForceCompute(sysdef), m_group(group), m_k(k)
    {
    m_exec_conf->msg->notice(5) << "Constructing PlaneRestraintCompute" << std::endl;

    setPoint(point);
    setNormal(normal);
    }

PlaneRestraintCompute::~PlaneRestraintCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PlaneRestraintCompute" << std::endl;
    }

/*!
 * \param timestep Current timestep
 *
 * Harmonic forces are computed on all particles in group based on their distance from the plane.
 */
void PlaneRestraintCompute::computeForces(unsigned int timestep)
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

    for (unsigned int idx=0; idx < m_group->getNumMembers(); ++idx)
        {
        const unsigned int pidx = h_group.data[idx];

        // unwrapped particle coordinate
        const Scalar4 pos = h_pos.data[pidx];
        const int3 image = h_image.data[pidx];
        const Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

        // distance to point from plane
        const Scalar3 dr = box.shift(r, image) - m_o;
        const Scalar d = dot(dr,m_n);

        // force points along normal vector
        const Scalar3 force = -m_k*(d*m_n);

        // squared-distance gives energy
        const Scalar energy = Scalar(0.5)*m_k*(d*d);

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

namespace detail
{
void export_PlaneRestraintCompute(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<PlaneRestraintCompute,std::shared_ptr<PlaneRestraintCompute>>(m,"PlaneRestraintCompute",py::base<ForceCompute>())
    .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,Scalar3,Scalar3,Scalar>())
    .def("getPoint", &PlaneRestraintCompute::getPoint)
    .def("setPoint", &PlaneRestraintCompute::setPoint)
    .def("getNormal", &PlaneRestraintCompute::getNormal)
    .def("setNormal", &PlaneRestraintCompute::setNormal)
    .def("getForceConstant", &PlaneRestraintCompute::getForceConstant)
    .def("setForceConstant", &PlaneRestraintCompute::setForceConstant)
    ;
    }
} // end namespace detail
} // end namespace azplugins
