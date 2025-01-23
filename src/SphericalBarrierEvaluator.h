// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_SPHERICAL_BARRIER_EVALUATOR_H_
#define AZPLUGINS_SPHERICAL_BARRIER_EVALUATOR_H_

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {

class SphericalBarrierEvaluator
    {
    public:
    HOSTDEVICE SphericalBarrierEvaluator(Scalar R) : m_R(R) { }

    HOSTDEVICE Scalar4 operator()(const Scalar3& pos, Scalar k, Scalar offset) const
        {
        const Scalar r = fast::sqrt(dot(pos, pos));
        const Scalar dr = r - (m_R + offset);

        // don't compute if below minimum of potential
        if (dr <= Scalar(0.0))
            {
            return make_scalar4(0, 0, 0, 0);
            }

        const Scalar k_dr = k * dr;
        const Scalar3 f = -(k_dr / r) * pos;      // F = -k dr \hat r
        const Scalar e = Scalar(0.5) * k_dr * dr; // U = 0.5 k dr^2
        return make_scalar4(f.x, f.y, f.z, e);
        }

    HOSTDEVICE bool valid(const BoxDim& box) const
        {
        const Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();
        const Scalar two_R = Scalar(2.0) * m_R;
        return (m_R >= Scalar(0.0) && nearest_plane_distance.x >= two_R
                && nearest_plane_distance.y >= two_R && nearest_plane_distance.z >= two_R);
        }

    private:
    Scalar m_R; // Radius of sphere
    };

    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_SPHERICAL_BARRIER_EVALUATOR_H_
