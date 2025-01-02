// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_CARTESIAN_BINNING_OPERATION_H_
#define AZPLUGINS_CARTESIAN_BINNING_OPERATION_H_

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

//! Binning operation in Cartesian coordinates
class CartesianBinningOperation : public BinningOperation
    {
    public:
    using BinningOperation::BinningOperation;

    HOSTDEVICE bool bin(uint3& bin,
                        Scalar3& transformed_vector,
                        const Scalar3& coordinates,
                        const Scalar3& vector) const
        {
        uint3 bin_ = make_uint3(0, 0, 0);

        if (m_should_bin[0] && !bin1D(bin_.x, coordinates.x, 0))
            {
            return false;
            }

        if (m_should_bin[1] && !bin1D(bin_.y, coordinates.y, 1))
            {
            return false;
            }

        if (m_should_bin[2] && !bin1D(bin_.z, coordinates.z, 2))
            {
            return false;
            }

        bin = bin_;
        transformed_vector = vector;

        return true;
        }
    };

    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_CARTESIAN_BINNING_OPERATION_H_
