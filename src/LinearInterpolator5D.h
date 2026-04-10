// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_LINEAR_INTERPOLATOR_5D_H_
#define AZPLUGINS_LINEAR_INTERPOLATOR_5D_H_

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

class Index5D
    {
    public:
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Index5D() : m_n {0, 0, 0, 0, 0} { }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE explicit Index5D(const unsigned int* n)
        : m_n {n[0], n[1], n[2], n[3], n[4]}
        {
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE
    Index5D(unsigned int n0, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4)
        : m_n {n0, n1, n2, n3, n4}
        {
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE unsigned int operator()(unsigned int i0,
                                                                       unsigned int i1,
                                                                       unsigned int i2,
                                                                       unsigned int i3,
                                                                       unsigned int i4) const
        {
        unsigned int idx = i0;
        idx = idx * m_n[1] + i1;
        idx = idx * m_n[2] + i2;
        idx = idx * m_n[3] + i3;
        idx = idx * m_n[4] + i4;
        return idx;
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE unsigned int size() const
        {
        return m_n[0] * m_n[1] * m_n[2] * m_n[3] * m_n[4];
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE unsigned int getN(unsigned int dim) const
        {
        return m_n[dim];
        }

    private:
    unsigned int m_n[5];
    };

/*! \brief 5D multilinear interpolation on a uniform rectilinear grid.

    This is an extension of three-dimensional linear interpolation
    from (https://github.com/mphowardlab/flyft/blob/main/src/grid_interpolator.cc).

*/
template<typename T> class LinearInterpolator5D
    {
    public:
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE LinearInterpolator5D() : m_data(nullptr), m_indexer()
        {
        for (int d = 0; d < 5; ++d)
            {
            m_lo[d] = Scalar(0);
            m_hi[d] = Scalar(0);
            m_dx[d] = Scalar(0);
            }
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE
    LinearInterpolator5D(const T* data, const unsigned int* n, const Scalar* lo, const Scalar* hi)
        : m_data(data), m_indexer(n)
        {
        for (int d = 0; d < 5; ++d)
            {
            const unsigned int nd = n[d];
            assert(nd >= 2);

            m_lo[d] = lo[d];
            m_hi[d] = hi[d];
            m_dx[d] = (m_hi[d] - m_lo[d]) / Scalar(nd - 1);
            }

        assert(m_indexer.size() > 0);
        }

    //! Constructor accepting the domain as a Scalar2 array.
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE LinearInterpolator5D(const T* data,
                                                                    const unsigned int* n,
                                                                    const Scalar2* domain_s2)
        : m_data(data), m_indexer(n)
        {
        for (int d = 0; d < 5; ++d)
            {
            const unsigned int nd = n[d];
            assert(nd >= 2);

            m_lo[d] = domain_s2[d].x;
            m_hi[d] = domain_s2[d].y;
            m_dx[d] = (m_hi[d] - m_lo[d]) / Scalar(nd - 1);
            }

        assert(m_indexer.size() > 0);
        }

    //! Interpolate at (x0, x1, x2, x3, x4).
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar
    operator()(Scalar x0, Scalar x1, Scalar x2, Scalar x3, Scalar x4) const
        {
        const Scalar x[5] = {x0, x1, x2, x3, x4};

        // Compute the cell bin and fractional coordinate in each dimension.
        int bin[5];
        Scalar frac[5];

        for (int d = 0; d < 5; ++d)
            {
            const unsigned int nd = m_indexer.getN(static_cast<unsigned int>(d));
            const Scalar f = (x[d] - m_lo[d]) / m_dx[d];

            int b = static_cast<int>(std::floor(f));

            // If exactly at the top boundary, shift into the last valid cell so
            // that (b+1) remains in bounds.
            if (f == Scalar(nd - 1) && x[d] == m_hi[d])
                {
                --b;
                }

            assert(b >= 0);
            assert(b < static_cast<int>(nd) - 1);

            bin[d] = b;
            frac[d] = f - Scalar(b);
            }

        // Load the 2^5=32 corners of the surrounding 5D cell.
        Scalar corners[32];

        for (unsigned int mask = 0; mask < 32; ++mask)
            {
            const unsigned int i0
                = static_cast<unsigned int>(bin[0] + static_cast<int>((mask >> 0) & 1u));
            const unsigned int i1
                = static_cast<unsigned int>(bin[1] + static_cast<int>((mask >> 1) & 1u));
            const unsigned int i2
                = static_cast<unsigned int>(bin[2] + static_cast<int>((mask >> 2) & 1u));
            const unsigned int i3
                = static_cast<unsigned int>(bin[3] + static_cast<int>((mask >> 3) & 1u));
            const unsigned int i4
                = static_cast<unsigned int>(bin[4] + static_cast<int>((mask >> 4) & 1u));

            // Implicit conversion from T to Scalar is intended.
            corners[mask] = m_data[m_indexer(i0, i1, i2, i3, i4)];
            }

        // For each dimension d, collapse pairs of points that differ in bit d.
        Scalar scratch[16];
        Scalar* in = corners;
        Scalar* out = scratch;
        unsigned int len = 32;

        for (int d = 0; d < 5; ++d)
            {
            const Scalar t = frac[d];
            const Scalar omt = Scalar(1) - t;
            const unsigned int out_len = len / 2;

            for (unsigned int i = 0; i < out_len; ++i)
                {
                out[i] = in[2 * i] * omt + in[2 * i + 1] * t;
                }
            // Swap input/output
            std::swap(in, out);
            len = out_len;
            }

        // After 5 reductions, len==1 and in[0] holds the interpolated value.
        return in[0];
        }

    //! Compute the finite-difference derivative with respect to dimensions.
    /*! Uses central differences when possible, falling back to forward or backward
        differences at the domain boundaries.

        \param x0  Coordinate along dimension 0
        \param x1  Coordinate along dimension 1
        \param x2  Coordinate along dimension 2
        \param x3  Coordinate along dimension 3
        \param x4  Coordinate along dimension 4
        \param dim Which dimension (0-4) to differentiate with respect to
        \param h   Finite-difference step size
    */
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar
    derivative(Scalar x0, Scalar x1, Scalar x2, Scalar x3, Scalar x4, int dim, Scalar h) const
        {
        Scalar x[5] = {x0, x1, x2, x3, x4};
        const Scalar val = (*this)(x[0], x[1], x[2], x[3], x[4]);

        const bool at_lo = (x[dim] - h < m_lo[dim]);
        const bool at_hi = (x[dim] + h > m_hi[dim]);

        if (!at_lo && !at_hi)
            {
            // central difference
            x[dim] += h;
            const Scalar f_plus = (*this)(x[0], x[1], x[2], x[3], x[4]);
            x[dim] -= Scalar(2) * h;
            const Scalar f_minus = (*this)(x[0], x[1], x[2], x[3], x[4]);
            return (f_plus - f_minus) / (Scalar(2) * h);
            }
        else if (at_lo)
            {
            // forward difference
            x[dim] += h;
            const Scalar f_plus = (*this)(x[0], x[1], x[2], x[3], x[4]);
            return (f_plus - val) / h;
            }
        else
            {
            // backward difference
            x[dim] -= h;
            const Scalar f_minus = (*this)(x[0], x[1], x[2], x[3], x[4]);
            return (val - f_minus) / h;
            }
        }

    //! Return the lower bound for a given dimension.
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar getLo(int dim) const
        {
        return m_lo[dim];
        }

    //! Return the upper bound for a given dimension.
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar getHi(int dim) const
        {
        return m_hi[dim];
        }

    private:
    const T* m_data;
    Scalar m_lo[5];
    Scalar m_hi[5];
    Scalar m_dx[5];
    Index5D m_indexer;
    };

    } // namespace azplugins
    } // namespace hoomd

#undef AZPLUGINS_HOSTDEVICE
#undef AZPLUGINS_FORCEINLINE

#endif // AZPLUGINS_LINEAR_INTERPOLATOR_5D_H_
