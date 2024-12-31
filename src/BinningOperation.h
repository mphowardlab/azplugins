// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_BINNING_OPERATION_H_
#define AZPLUGINS_BINNING_OPERATION_H_

#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {

//! Interface for binning operations in different 3D coordinate systems.
class BinningOperation
    {
    public:
    HOSTDEVICE BinningOperation(const uint3& num_bins,
                                const Scalar3& lower_bounds,
                                const Scalar3& upper_bounds)
        : m_lower_bounds {lower_bounds.x, lower_bounds.y, lower_bounds.z},
          m_upper_bounds {upper_bounds.x, upper_bounds.y, upper_bounds.z}, m_num_bins {1, 1, 1},
          m_should_bin {false, false, false}
        {
        if (num_bins.x > 0)
            {
            m_num_bins[0] = num_bins.x;
            m_should_bin[0] = true;
            }
        if (num_bins.y > 0)
            {
            m_num_bins[1] = num_bins.y;
            m_should_bin[1] = true;
            }
        if (num_bins.z > 0)
            {
            m_num_bins[2] = num_bins.z;
            m_should_bin[2] = true;
            }
        }

    HOSTDEVICE bool bin(uint3& bin,
                        Scalar3& transformed_vector,
                        const Scalar3& coordinates,
                        const Scalar3& vector) const
        {
        return false;
        }

    HOSTDEVICE size_t getTotalNumBins() const
        {
        return static_cast<size_t>(m_num_bins[0]) * m_num_bins[1] * m_num_bins[2];
        }

    HOSTDEVICE size_t ravelBin(const uint3& bin) const
        {
        return static_cast<size_t>(bin.z) + m_num_bins[2] * (bin.y + m_num_bins[1] * bin.x);
        }

    protected:
    Scalar m_lower_bounds[3];
    Scalar m_upper_bounds[3];
    unsigned int m_num_bins[3];
    bool m_should_bin[3];

    HOSTDEVICE bool bin1D(unsigned int& bin_1d, const Scalar x, unsigned int dim) const
        {
        int bin_ = static_cast<int>(
            slow::floor(((x - m_lower_bounds[dim]) / (m_upper_bounds[dim] - m_lower_bounds[dim]))
                        * m_num_bins[dim]));
        if (bin_ >= 0 && bin_ < static_cast<int>(m_num_bins[dim]))
            {
            bin_1d = bin_;
            return true;
            }
        else
            {
            return false;
            }
        }
    };

    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_BINNING_OPERATION_H_
