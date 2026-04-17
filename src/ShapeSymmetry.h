// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ShapeSymmetry.h
 * \brief Symmetry evaluators that reduce angular coordinates to a fundamental domain.
 */

#ifndef AZPLUGINS_SHAPE_SYMMETRY_H_
#define AZPLUGINS_SHAPE_SYMMETRY_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include <cmath>

#ifndef __HIPCC__
#include <string>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define AZPLUGINS_HOSTDEVICE __host__ __device__
#define AZPLUGINS_FORCEINLINE __forceinline__
#else
#define AZPLUGINS_HOSTDEVICE
#define AZPLUGINS_FORCEINLINE inline
#endif

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

//! Convert spherical coordinates to Cartesian coordinates.
AZPLUGINS_HOSTDEVICE inline vec3<Scalar> sphericalToCartesian(Scalar r, Scalar theta, Scalar phi)
    {
    Scalar s_th, c_th, s_ph, c_ph;
    fast::sincos(theta, s_th, c_th);
    fast::sincos(phi, s_ph, c_ph);
    return vec3<Scalar>(r * s_ph * c_th, r * s_ph * s_th, r * c_ph);
    }

//! Convert Cartesian coordinates to spherical coordinates.
AZPLUGINS_HOSTDEVICE inline void
cartesianToSpherical(const vec3<Scalar>& v, Scalar& r, Scalar& theta, Scalar& phi)
    {
    r = fast::sqrt(dot(v, v));
    if (r > Scalar(0))
        {
        theta = std::atan2(v.y, v.x);
        if (theta < Scalar(0))
            theta += Scalar(2) * Scalar(M_PI);
        Scalar cp = v.z / r;
        if (cp < Scalar(-1))
            cp = Scalar(-1);
        else if (cp > Scalar(1))
            cp = Scalar(1);
        phi = slow::acos(cp);
        }
    else
        {
        theta = Scalar(0);
        phi = Scalar(0);
        }
    }

//! Build a quaternion from an axis and an angle.
AZPLUGINS_HOSTDEVICE inline quat<Scalar> quatFromAxisAngle(const vec3<Scalar>& axis, Scalar angle)
    {
    Scalar s, c;
    fast::sincos(Scalar(0.5) * angle, s, c);
    return quat<Scalar>(c, s * axis);
    }

//! Build a quaternion from intrinsic ZXZ Euler angles.
AZPLUGINS_HOSTDEVICE inline quat<Scalar> quatFromEulerZXZ(Scalar alpha, Scalar beta, Scalar gamma)
    {
    const quat<Scalar> qz_a = quatFromAxisAngle(vec3<Scalar>(0, 0, 1), alpha);
    const quat<Scalar> qx_b = quatFromAxisAngle(vec3<Scalar>(1, 0, 0), beta);
    const quat<Scalar> qz_g = quatFromAxisAngle(vec3<Scalar>(0, 0, 1), gamma);
    return qz_a * qx_b * qz_g;
    }

//! Extract intrinsic ZXZ Euler angles from a quaternion.
AZPLUGINS_HOSTDEVICE inline void
eulerFromQuat(const quat<Scalar>& q, Scalar& alpha, Scalar& beta, Scalar& gamma)
    {
    const rotmat3<Scalar> R(q);
    const Scalar tol = Scalar(1e-7);

    if (R.row2.z < Scalar(-1))
        {
        beta = Scalar(M_PI);
        }
    else if (R.row2.z > Scalar(1))
        {
        beta = Scalar(0);
        }
    else
        {
        beta = std::acos(R.row2.z);
        }

    if (beta > tol && beta < Scalar(M_PI) - tol)
        {
        alpha = std::atan2(R.row0.z, -R.row1.z);
        gamma = std::atan2(R.row2.x, R.row2.y);
        }
    else if (beta <= tol)
        {
        alpha = Scalar(0);
        gamma = std::atan2(R.row1.x, R.row0.x);
        }
    else
        {
        alpha = Scalar(0);
        gamma = std::atan2(-R.row1.x, R.row0.x);
        }

    if (alpha < Scalar(0))
        alpha += Scalar(2) * Scalar(M_PI);
    if (gamma < Scalar(0))
        gamma += Scalar(2) * Scalar(M_PI);
    }

    } // namespace detail

//! Null symmetry: no reduction.
/*! Full natural domain:
    theta in [0, 2 pi], phi in [0, pi], alpha in [0, 2 pi],
    beta in [0, pi], gamma in [0, 2 pi].
*/
class ShapeSymmetryNull
    {
    public:
    //! Upper bounds of the reduced domain (lower bounds are always zero).
    static constexpr Scalar domain_upper[5]
        = {Scalar(2.0 * M_PI), Scalar(M_PI), Scalar(2.0 * M_PI), Scalar(M_PI), Scalar(2.0 * M_PI)};

    AZPLUGINS_HOSTDEVICE ShapeSymmetryNull() { }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "Null";
        }
#endif

    AZPLUGINS_HOSTDEVICE quat<Scalar> reduce(Scalar& /*theta*/,
                                             Scalar& /*phi*/,
                                             Scalar& /*alpha*/,
                                             Scalar& /*beta*/,
                                             Scalar& /*gamma*/) const
        {
        return quat<Scalar>(Scalar(1), vec3<Scalar>(0, 0, 0));
        }
    };

//! Tetrahedron symmetry evaluator.
/*! Reduced domain:
    theta in [0, 2 pi/3], phi in [0, pi], alpha in [0, 2 pi],
    beta in [0, pi], gamma in [0, 2 pi/3].
*/
class ShapeSymmetryTetrahedron
    {
    public:
    //! Upper bounds of the domain.
    static constexpr Scalar domain_upper[5] = {Scalar(2.0 * M_PI / 3.0),
                                               Scalar(M_PI),
                                               Scalar(2.0 * M_PI),
                                               Scalar(M_PI),
                                               Scalar(2.0 * M_PI / 3.0)};

    AZPLUGINS_HOSTDEVICE ShapeSymmetryTetrahedron() { }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "Tetrahedron";
        }
#endif

    AZPLUGINS_HOSTDEVICE quat<Scalar>
    reduce(Scalar& theta, Scalar& phi, Scalar& alpha, Scalar& beta, Scalar& gamma) const
        {
        quat<Scalar> transformation(Scalar(1), vec3<Scalar>(0, 0, 0));

        const Scalar theta_fold = Scalar(2) * Scalar(M_PI) / Scalar(3);

        // fold theta into [0, 2 pi/3] by rotating around z.
        if (theta > theta_fold)
            {
            vec3<Scalar> pos = detail::sphericalToCartesian(Scalar(1), theta, phi);

            const Scalar n = slow::floor(theta / theta_fold);
            const Scalar angle = -n * theta_fold;
            const quat<Scalar> rot_z = detail::quatFromAxisAngle(vec3<Scalar>(0, 0, 1), angle);
            pos = rotate(rot_z, pos);
            transformation = rot_z * transformation;

            Scalar r_tmp;
            detail::cartesianToSpherical(pos, r_tmp, theta, phi);
            }

        // apply transformation to orientation, extract Euler angles.
        quat<Scalar> q_orient = detail::quatFromEulerZXZ(alpha, beta, gamma);
        quat<Scalar> q_transformed = transformation * q_orient;
        detail::eulerFromQuat(q_transformed, alpha, beta, gamma);

        // 3-fold gamma symmetry.
        gamma = std::fmod(gamma, theta_fold);

        return transformation;
        }
    };

    } // namespace azplugins
    } // namespace hoomd

#undef AZPLUGINS_HOSTDEVICE
#undef AZPLUGINS_FORCEINLINE

#endif // AZPLUGINS_SHAPE_SYMMETRY_H_
