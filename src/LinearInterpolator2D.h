// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_LINEAR_INTERPOLATOR_2D_H_
#define AZPLUGINS_LINEAR_INTERPOLATOR_2D_H_

#include <cassert>
#include <cmath>
#include <cstdint>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

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

template<typename T> class LinearInterpolator2D
    {
    public:
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE LinearInterpolator2D() : m_data(nullptr), m_indexer()
        {
        m_n[0] = 0;
        m_n[1] = 0;
        for (int d = 0; d < 2; ++d)
            {
            m_lo[d] = Scalar(0);
            m_hi[d] = Scalar(0);
            m_dx[d] = Scalar(0);
            }
        }

    //! Constructor
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE
    LinearInterpolator2D(const T* data, const unsigned int* n, const Scalar* lo, const Scalar* hi)
        : m_data(data), m_indexer(n[0], n[1])
        {
        for (int d = 0; d < 2; ++d)
            {
            const unsigned int nd = n[d];
            assert(nd >= 2);

            m_n[d] = nd;
            m_lo[d] = lo[d];
            m_hi[d] = hi[d];
            m_dx[d] = (m_hi[d] - m_lo[d]) / Scalar(nd - 1);
            }
        }

    //! Destructor
    ~LinearInterpolator2D()
        {
        delete[] m_data;
        }

    //! Interpolate at (x0, x1)
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar operator()(Scalar x0, Scalar x1) const
        {
        const Scalar x[2] = {x0, x1};

        int bin[2];
        Scalar frac[2];

        for (int d = 0; d < 2; ++d)
            {
            const unsigned int nd = m_n[d];
            const Scalar f = (x[d] - m_lo[d]) / m_dx[d];
            int b = static_cast<int>(std::floor(f));

            // If exactly at the top boundary, shift into the last valid cell
            if (f == Scalar(nd - 1) && x[d] == m_hi[d])
                {
                --b;
                }

            assert(b >= 0);
            assert(b < static_cast<int>(nd - 1));
            bin[d] = b;
            frac[d] = f - Scalar(b);
            }

        const unsigned int b0 = static_cast<unsigned int>(bin[0]);
        const unsigned int b1 = static_cast<unsigned int>(bin[1]);
        const Scalar c00 = m_data[m_indexer(b0, b1)];
        const Scalar c01 = m_data[m_indexer(b0, b1 + 1)];
        const Scalar c10 = m_data[m_indexer(b0 + 1, b1)];
        const Scalar c11 = m_data[m_indexer(b0 + 1, b1 + 1)];

        // Bilinear interpolation
        const Scalar t0 = frac[0];
        const Scalar omt0 = Scalar(1) - t0;
        const Scalar c0 = c00 * omt0 + c10 * t0;
        const Scalar c1 = c01 * omt0 + c11 * t0;

        const Scalar t1 = frac[1];
        return c0 * (Scalar(1) - t1) + c1 * t1;
        }

    //! Lower bound for a given dimension
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar getLo(int dim) const
        {
        return m_lo[dim];
        }
    //! Upper bound for a given dimension
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar getHi(int dim) const
        {
        return m_hi[dim];
        }

    private:
    const T* m_data;
    Scalar m_lo[2];
    Scalar m_hi[2];
    Scalar m_dx[2];
    unsigned int m_n[2];
    Index2D m_indexer;
    };

    } // namespace azplugins
    } // namespace hoomd

#undef AZPLUGINS_HOSTDEVICE
#undef AZPLUGINS_FORCEINLINE

#endif // AZPLUGINS_LINEAR_INTERPOLATOR_2D_H_
