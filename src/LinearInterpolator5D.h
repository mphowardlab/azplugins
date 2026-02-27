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

class FiveDimensionalIndex
    {
    public:
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE FiveDimensionalIndex()
        : n0_(0), n1_(0), n2_(0), n3_(0), n4_(0)
        {
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE FiveDimensionalIndex(unsigned int n0,
                                                                    unsigned int n1,
                                                                    unsigned int n2,
                                                                    unsigned int n3,
                                                                    unsigned int n4)
        : n0_(n0), n1_(n1), n2_(n2), n3_(n3), n4_(n4)
        {
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE unsigned int operator()(unsigned int i0,
                                                                       unsigned int i1,
                                                                       unsigned int i2,
                                                                       unsigned int i3,
                                                                       unsigned int i4) const
        {
        unsigned int idx = i0;
        idx = idx * n1_ + i1;
        idx = idx * n2_ + i2;
        idx = idx * n3_ + i3;
        idx = idx * n4_ + i4;
        return idx;
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE unsigned int size() const
        {
        return n0_ * n1_ * n2_ * n3_ * n4_;
        }

    private:
    unsigned int n0_, n1_, n2_, n3_, n4_;
    };

// T is the stored data type.
template<typename T> class LinearInterpolator5D
    {
    public:
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE LinearInterpolator5D() : data_(nullptr), nindex_()
        {
        for (int d = 0; d < 5; ++d)
            {
            n_[d] = 0;
            lo_[d] = Scalar(0);
            hi_[d] = Scalar(0);
            dx_[d] = Scalar(0);
            }
        }

    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE
    LinearInterpolator5D(const T* data, const unsigned int* n, const Scalar* lo, const Scalar* hi)
        : data_(data)
        {
        for (int d = 0; d < 5; ++d)
            {
            n_[d] = n[d];
            lo_[d] = lo[d];
            hi_[d] = hi[d];

            assert(n_[d] >= 2);
            dx_[d] = (hi_[d] - lo_[d]) / Scalar(n_[d] - 1);
            }

        nindex_ = FiveDimensionalIndex(n_[0], n_[1], n_[2], n_[3], n_[4]);
        }

    // Interpolate at (x0, x1, x2, x3, x4).
    AZPLUGINS_HOSTDEVICE AZPLUGINS_FORCEINLINE Scalar
    operator()(Scalar x0, Scalar x1, Scalar x2, Scalar x3, Scalar x4) const
        {
        const Scalar x[5] = {x0, x1, x2, x3, x4};

        Scalar f[5];
        for (int d = 0; d < 5; ++d)
            {
            f[d] = (x[d] - lo_[d]) / dx_[d];
            }

        int bin[5];
        Scalar dloc[5];

        for (int dim = 0; dim < 5; ++dim)
            {
            bin[dim] = (int)std::floor((double)f[dim]);

            if (f[dim] == Scalar(n_[dim] - 1) && x[dim] == hi_[dim])
                {
                --bin[dim];
                }

            dloc[dim] = f[dim] - Scalar(bin[dim]);

            assert(bin[dim] >= 0);
            assert(bin[dim] < (int)n_[dim] - 1);
            }

        const Scalar xd = dloc[0];
        const Scalar yd = dloc[1];
        const Scalar zd = dloc[2];
        const Scalar wd = dloc[3];
        const Scalar vd = dloc[4];

        Scalar c[32];
        for (unsigned int mask = 0; mask < 32; ++mask)
            {
            const unsigned int i0 = (unsigned int)(bin[0] + ((mask >> 0) & 1u));
            const unsigned int i1 = (unsigned int)(bin[1] + ((mask >> 1) & 1u));
            const unsigned int i2 = (unsigned int)(bin[2] + ((mask >> 2) & 1u));
            const unsigned int i3 = (unsigned int)(bin[3] + ((mask >> 3) & 1u));
            const unsigned int i4 = (unsigned int)(bin[4] + ((mask >> 4) & 1u));

            c[mask] = Scalar(data_[nindex_(i0, i1, i2, i3, i4)]);
            }

        Scalar c0[16];
        for (unsigned int i = 0; i < 16; ++i)
            {
            c0[i] = c[2 * i] * (Scalar(1) - xd) + c[2 * i + 1] * xd;
            }

        Scalar c1[8];
        for (unsigned int i = 0; i < 8; ++i)
            {
            c1[i] = c0[2 * i] * (Scalar(1) - yd) + c0[2 * i + 1] * yd;
            }

        Scalar c2[4];
        for (unsigned int i = 0; i < 4; ++i)
            {
            c2[i] = c1[2 * i] * (Scalar(1) - zd) + c1[2 * i + 1] * zd;
            }

        Scalar c3[2];
        for (unsigned int i = 0; i < 2; ++i)
            {
            c3[i] = c2[2 * i] * (Scalar(1) - wd) + c2[2 * i + 1] * wd;
            }

        return c3[0] * (Scalar(1) - vd) + c3[1] * vd;
        }

    private:
    const T* data_;
    unsigned int n_[5];
    Scalar lo_[5];
    Scalar hi_[5];
    Scalar dx_[5];
    FiveDimensionalIndex nindex_;
    };

    } // namespace azplugins
    } // namespace hoomd

#endif // AZPLUGINS_LINEAR_INTERPOLATOR_5D_H_
