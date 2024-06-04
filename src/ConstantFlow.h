// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_CONSTANT_FLOW_H_
#define AZPLUGINS_CONSTANT_FLOW_H_

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include <pybind11/pybind11.h>
#endif
namespace hoomd
    {
namespace azplugins
    {

//! Position-independent flow along a vector
class ConstantFlow
    {
    public:
    //! Constructor
    /*!
     *\param U_ Flow field
     */
    ConstantFlow(Scalar3 U_) : U(U_) { }
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

    HOSTDEVICE Scalar3 getVelocity() const
        {
        return U;
        }

    HOSTDEVICE void setVelocity(const Scalar3& U_)
        {
        U = U_;
        }

    private:
    Scalar3 U; //!< Flow field
    };

namespace detail
    {
void export_ConstantFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ConstantFlow, std::shared_ptr<ConstantFlow>>(m, "ConstantFlow")
        .def(py::init<Scalar3>())
        .def_property(
            "mean_velocity",
            [](const ConstantFlow& U)
            {
                const auto field = U.getVelocity();
                return pybind11::make_tuple(field.x, field.y, field.z);
            },
            [](ConstantFlow& U, const pybind11::tuple& field)
            {
                U.setVelocity(make_scalar3(pybind11::cast<Scalar>(field[0]),
                                           pybind11::cast<Scalar>(field[1]),
                                           pybind11::cast<Scalar>(field[2])));
            });
    }
    } // end namespace detail

    } // namespace azplugins
    } // namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_CONSTANT_FLOW_H_
