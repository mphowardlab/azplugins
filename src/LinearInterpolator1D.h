// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_LINEAR_INTERPOLATOR_1D_H_
#define AZPLUGINS_LINEAR_INTERPOLATOR_1D_H_

#include <cassert>
#include <cmath>
#include <cstdint>

#include "hoomd/HOOMDMath.h"

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

template<typename T> class LinearInterpolator1D
    {
    public:
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE LinearInterpolator1D()
        : m_data(nullptr), m_lo(Scalar(0)), m_hi(Scalar(0)), m_dx(Scalar(0)), m_n(0)
        {
        }

    //! Constructor
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE
    LinearInterpolator1D(const T* data, unsigned int n, Scalar lo, Scalar hi)
        : m_data(data), m_lo(lo), m_hi(hi), m_n(n)
        {
        assert(n >= 2);
        m_dx = (m_hi - m_lo) / Scalar(n - 1);
        }

    //! Implement piecewise linear interpolation
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar operator()(Scalar x) const
        {
        const Scalar f = (x - m_lo) / m_dx;
        int b = static_cast<int>(std::floor(f));

        // If exactly at the top boundary, shift into the last valid cell
        if (f == Scalar(m_n - 1) && x == m_hi)
            {
            --b;
            }

        assert(b >= 0);
        assert(b < static_cast<int>(m_n - 1));
        const Scalar frac = f - Scalar(b);

        const Scalar c0 = m_data[b];
        const Scalar c1 = m_data[b + 1];

        // Linear interpolation
        return c0 * (Scalar(1) - frac) + c1 * frac;
        }

    //! Lower bound
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar getLo() const
        {
        return m_lo;
        }
    //! Upper bound
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar getHi() const
        {
        return m_hi;
        }

    private:
    const T* m_data;
    Scalar m_lo;
    Scalar m_hi;
    Scalar m_dx;
    unsigned int m_n;
    };

    } // namespace azplugins
    } // namespace hoomd

#undef AZPLUGINS_HOSTDEVICE
#undef AZPLUGINS_FORCEINLINE

#endif // AZPLUGINS_LINEAR_INTERPOLATOR_1D_H_
