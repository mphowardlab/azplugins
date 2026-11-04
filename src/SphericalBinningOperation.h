// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2026, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_SPHERICAL_BINNING_OPERATION_H_
#define AZPLUGINS_SPHERICAL_BINNING_OPERATION_H_

#include "BinningOperation.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {

//! Binning operation in spherical coordinates
class SphericalBinningOperation : public BinningOperation
    {
    public:
    using BinningOperation::BinningOperation;

    HOSTDEVICE bool bin(uint3& bin,
                        Scalar3& transformed_vector,
                        const Scalar3& coordinates,
                        const Scalar3& vector) const
        {
        uint3 bin_ = make_uint3(0, 0, 0);

        // bin with respect to r first because it is needed for the vector transform
        const Scalar r = fast::sqrt(dot(coordinates, coordinates));

        if (m_should_bin[0] && !bin1D(bin_.x, r, 0))
            {
            return false;
            }

        // bin with respect to theta second because its internals aren't needed later
        if (m_should_bin[1])
            {
#if HOOMD_LONGREAL_SIZE == 32
            Scalar theta = ::atan2f(coordinates.y, coordinates.x);
#else
            Scalar theta = ::atan2(coordinates.y, coordinates.x);
#endif
            if (theta < Scalar(0))
                {
                theta += Scalar(2) * M_PI;
                }
            if (!bin1D(bin_.y, theta, 1))
                {
                return false;
                }
            }

        // cos and sin of phi, defaulting phi to 0 if r == 0
        const Scalar rho
            = fast::sqrt(coordinates.x * coordinates.x + coordinates.y * coordinates.y);

        Scalar cos_phi(1), sin_phi(0);
        if (r > Scalar(0))
            {
            cos_phi = coordinates.z / r;
            sin_phi = rho / r;
            }

        // bin with respect to phi last
        if (m_should_bin[2])
            {
#if HOOMD_LONGREAL_SIZE == 32
            const Scalar phi = ::acosf(cos_phi);
#else
            const Scalar phi = ::acos(cos_phi);
#endif
            if (!bin1D(bin_.z, phi, 2))
                {
                return false;
                }
            }

        // cos and sin of theta, defaulting theta to 0 if rho == 0 (point is on the z axis)
        Scalar cos_theta(1), sin_theta(0);
        if (rho > Scalar(0))
            {
            cos_theta = coordinates.x / rho;
            sin_theta = coordinates.y / rho;
            }

        // transform Cartesian vector to spherical coordinates using the unit vectors
        //   r_hat     = ( sin phi cos theta, sin phi sin theta,  cos phi)
        //   theta_hat = (        -sin theta,         cos theta,        0)
        //   phi_hat   = ( cos phi cos theta, cos phi sin theta, -sin phi)
        const Scalar v_rho = cos_theta * vector.x + sin_theta * vector.y;
        transformed_vector = make_scalar3(sin_phi * v_rho + cos_phi * vector.z,
                                          -sin_theta * vector.x + cos_theta * vector.y,
                                          cos_phi * v_rho - sin_phi * vector.z);

        bin = bin_;
        return true;
        }
    };

    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_SPHERICAL_BINNING_OPERATION_H_
