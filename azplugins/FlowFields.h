// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParabolicFlow.h
 * \brief Declaration of ParabolicFlow
 */

#ifndef AZPLUGINS_PARABOLIC_FLOW_H_
#define AZPLUGINS_PARABOLIC_FLOW_H_

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#endif

namespace azplugins
{

//! Quiescent (motionless) fluid flow field
class QuiescentFluid
    {
    public:
        //! Construct quiescent fluid
        QuiescentFluid() { }

        //! Evaluate the flow field
        /*!
         * \param r position to evaluate flow
         */
        HOSTDEVICE Scalar3 operator()(const Scalar3& r) const
            {
            return make_scalar3(0.0, 0.0, 0.0);
            }
    };

//! Position-independent flow along a vector
class ConstantFlow
    {
    public:
        //! Constructor
        /*!
         *\param U_ Flow field
         */
        ConstantFlow(Scalar3 U_)
            : U(U_)
            {}

        //! Evaluate the flow field
        /*!
         * \param r position to evaluate flow
         *
         * This is just a constant, independent of \a r.
         */
        HOSTDEVICE Scalar3 operator()(const Scalar3& r) const
            {
            return U;
            }

    private:
        Scalar3 U;  //!< Flow field
    };

//! Unidirectional parabolic flow field
/*!
 * 1d flow along the \a x axis. The geometry is a parallel plate channel with
 * the plates centered around \f$ z = 0 \f$ and positioned at \f$ \pm L \f$.
 * The \a y axis is the vorticity direction and periodic. The flow profile in
 * this geometry is then
 *
 * \f[
 * u_x(z) = \frac{3}{2} U \left[1 - \left(\frac{z}{L}\right)^2 \right]
 * \f]
 *
 * Here, \f$ U \f$ is the mean velocity, which is related to the pressure drop
 * and viscosity.
 *
 * \note The user must properly establish no flux of particles through the channel
 *       walls through an appropriate wall potential.
 */
class ParabolicFlow
    {
    public:
        //! Construct parabolic flow profile
        /*!
         * \param U_ Mean velocity
         * \param L_ Channel half width
         */
        ParabolicFlow(Scalar U_, Scalar L_)
            : Umax(Scalar(1.5)*U_), L(L_) { }

        //! Evaluate the flow field
        /*!
         * \param r position to evaluate flow
         */
        HOSTDEVICE Scalar3 operator()(const Scalar3& r) const
            {
            const Scalar zr = (r.z / L);
            return make_scalar3(Umax * (1. - zr*zr), 0.0, 0.0);
            }

    private:
        Scalar Umax;    //<! Mean velocity
        Scalar L;       //!< Half width
    };

#ifndef NVCC

namespace detail
{
//! Export QuiescentFluid to python
/*!
 * \param m Python module to export to
 */
void export_QuiescentFluid(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<QuiescentFluid, std::shared_ptr<QuiescentFluid> >(m, "QuiescentFluid")
        .def(py::init<>())
        .def("__call__", &QuiescentFluid::operator());
    }

//! Export ConstantFlow to python
/*!
 * \param m Python module to export to
 */
void export_ConstantFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ConstantFlow, std::shared_ptr<ConstantFlow> >(m, "ConstantFlow")
        .def(py::init<Scalar3>())
        .def("__call__", &ConstantFlow::operator());
    }

//! Export ParabolicFlow to python
/*!
 * \param m Python module to export to
 */
void export_ParabolicFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ParabolicFlow, std::shared_ptr<ParabolicFlow> >(m, "ParabolicFlow")
        .def(py::init<Scalar,Scalar>())
        .def("__call__", &ParabolicFlow::operator());
    }
} // end namespace detail
#endif // NVCC

} // end namespace azplugins

#undef HOSTDEVICE

#endif // AZPLUGINS_PARABOLIC_FLOW_H_
