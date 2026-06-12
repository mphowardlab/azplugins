// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VARIANT_INTERPOLATED_H_
#define AZPLUGINS_VARIANT_INTERPOLATED_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc / hipcc
#endif
#include <cstdint>
#include <iostream>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

#include "hoomd/GPUArray.h"
#include "hoomd/Variant.h"
#include "hoomd/VectorMath.h"

#include "LinearInterpolator1D.h"

namespace hoomd
    {
namespace azplugins
    {

class PYBIND11_EXPORT VariantInterpolated : public Variant
    {
    public:
    //! Construct from a 1D values array and the time domain.
    VariantInterpolated(const Scalar* data, // Input interface height values
                        unsigned int n,     // Size of the array
                        Scalar t_lo,        // Low of the unscaled time
                        Scalar t_hi)        // Hi of the unscaled time

        : m_n(n), m_t_lo(t_lo), m_t_hi(t_hi)
        {
        GPUArray<Scalar> data_arr(n, m_exec_conf);
        m_data.swap(data_arr);
        ArrayHandle<Scalar> h_data(m_data, access_location::host, access_mode::overwrite);
        std::copy(data, data + n, h_data.data);
        }

    Scalar operator()(uint64_t timestep)
        {
        ArrayHandle<Scalar> h_data(m_data, access_location::host, access_mode::read);

        // LinearInterpolator1D
        LinearInterpolator1D<Scalar> interp(h_data.data, m_n, m_t_lo, m_t_hi);
        return interp(static_cast<Scalar>(timestep));
        }

    Scalar getTLo() const
        {
        return m_t_lo;
        }

    Scalar getTHi() const
        {
        return m_t_hi;
        }

    Scalar setTLo(Scalar t_lo)
        {
        m_t_lo = t_lo;
        return m_t_lo;
        }

    Scalar setTHi(Scalar t_hi)
        {
        m_t_hi = t_hi;
        return m_t_hi;
        }

    Scalar min()
        {
        ArrayHandle<Scalar> h_data(m_data, hoomd::access_location::host, hoomd::access_mode::read);
        m_min = h_data.data[0];
        for (unsigned int i = 1; i < m_n; ++i)
            {
            if (h_data.data[i] < m_min)
                m_min = h_data.data[i];
            }
        return m_min;
        }

    Scalar max()
        {
        ArrayHandle<Scalar> h_data(m_data, hoomd::access_location::host, hoomd::access_mode::read);
        m_max = h_data.data[0];
        for (unsigned int i = 1; i < m_n; ++i)
            {
            if (h_data.data[i] > m_max)
                m_max = h_data.data[i];
            }
        return m_max;
        }

    protected:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    GPUArray<Scalar> m_data;
    unsigned int m_n;
    Scalar m_t_lo;
    Scalar m_t_hi;
    Scalar m_min;
    Scalar m_max;
    };

namespace detail
    {
void export_VariantInterpolated(pybind11::module& m);
    } // namespace detail
    } // namespace azplugins
    } // namespace hoomd

#endif // AZPLUGINS_VARIANT_INTERPOLATED_H_
