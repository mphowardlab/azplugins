// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_CYLINDRICAL_BINNING_OPERATION_H_
#define AZPLUGINS_CYLINDRICAL_BINNING_OPERATION_H_

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

//! Binning operation in cylindrical coordinates
class CylindricalBinningOperation : public BinningOperation
    {
    public:
    using BinningOperation::BinningOperation;

    HOSTDEVICE bool bin(uint3& bin,
                        Scalar3& transformed_vector,
                        const Scalar3& coordinates,
                        const Scalar3& vector) const
        {
        uint3 bin_ = make_uint3(0, 0, 0);

        // bin with respect to z first because it is cheapest in case of early exit
        if (m_should_bin[2] && !bin1D(bin_.z, coordinates.z, 2))
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

        // bin with respect to r last, r will be used for vector transform so always do it
        const Scalar r = fast::sqrt(coordinates.x * coordinates.x + coordinates.y * coordinates.y);
        if (m_should_bin[0] && !bin1D(bin_.x, r, 0))
            {
            return false;
            }

        // transform Cartesian vector to cylindrical coordinates,
        // defaulting angle to 0 if r == 0 (point is exactly at center)
        Scalar cos_theta(1), sin_theta(0);
        if (r > Scalar(0))
            {
            cos_theta = coordinates.x / r;
            sin_theta = coordinates.y / r;
            }
        transformed_vector = make_scalar3(cos_theta * vector.x + sin_theta * vector.y,
                                          -sin_theta * vector.x + cos_theta * vector.y,
                                          vector.z);

        bin = bin_;
        return true;
        }
    };

    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_CYLINDRICAL_BINNING_OPERATION_H_
