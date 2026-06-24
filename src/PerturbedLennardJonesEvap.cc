// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PerturbedLennardJonesEvap.h"

#include <algorithm>

namespace hoomd
    {
namespace azplugins
    {

PerturbedLennardJonesEvap::PerturbedLennardJonesEvap(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar rcut,
    const Scalar time_scale_factor,
    const param_type& params,
    bool energy_shift,
    const Scalar* attraction_scale_factor_data,
    const unsigned int* attraction_scale_factor_shape,
    const Scalar* domain,
    std::shared_ptr<VariantInterpolated> variant)
    : ForceCompute(sysdef), m_nlist(nlist), epsilon_x_4(params.epsilon_x_4), m_rcut(rcut),
      m_time_scale_factor(time_scale_factor),
      lj1(params.epsilon_x_4 * params.sigma_6 * params.sigma_6),
      lj2(params.epsilon_x_4 * params.sigma_6), rcutsq(rcut * rcut), rwcasq(params.rwcasq),
      sigma_6(params.sigma_6), m_energy_shift(energy_shift), m_variant(variant)
    {
        // Allocate and fill the domain for time: [t_lo, t_hi]
        {
        GPUArray<Scalar> domain_arr(2, m_exec_conf);
        m_domain.swap(domain_arr);

        ArrayHandle<Scalar> h_domain(m_domain, access_location::host, access_mode::overwrite);
        std::copy(domain, domain + 2, h_domain.data);
        }

        // Allocate and fill the table shape (ny, nt)
        {
        GPUArray<unsigned int> shape_arr(2, m_exec_conf);
        m_attraction_scale_factor_shape.swap(shape_arr);

        ArrayHandle<unsigned int> h_shape(m_attraction_scale_factor_shape,
                                          access_location::host,
                                          access_mode::overwrite);
        std::copy(attraction_scale_factor_shape, attraction_scale_factor_shape + 2, h_shape.data);
        }

        {
        const unsigned int n_data
            = attraction_scale_factor_shape[0] * attraction_scale_factor_shape[1];

        GPUArray<Scalar> attraction_scale_factor_arr(n_data, m_exec_conf);
        m_attraction_scale_factor_data.swap(attraction_scale_factor_arr);

        ArrayHandle<Scalar> h_data(m_attraction_scale_factor_data,
                                   access_location::host,
                                   access_mode::overwrite);

        std::copy(attraction_scale_factor_data, attraction_scale_factor_data + n_data, h_data.data);
        }

        {
        const Index2D typpair_idx(m_pdata->getNTypes());
        m_r_cut_nlist
            = std::make_shared<GPUArray<Scalar>>(typpair_idx.getNumElements(), m_exec_conf);
            {
            ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                              access_location::host,
                                              access_mode::overwrite);

            for (unsigned int i = 0; i < m_r_cut_nlist->getNumElements(); ++i)
                h_r_cut_nlist.data[i] = m_rcut;
            }
        m_nlist->addRCutMatrix(m_r_cut_nlist);
        m_nlist->notifyRCutMatrixChange();
        }
    }

void PerturbedLennardJonesEvap::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);

    const bool third_law
        = (m_nlist->getStorageMode() == hoomd::md::NeighborList::storageMode::half);

    Scalar scaled_t = scaleTime(timestep);
    Scalar interface_height = (*m_variant)(timestep);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    m_force.zeroFill();

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_data(m_attraction_scale_factor_data,
                               access_location::host,
                               access_mode::read);
    ArrayHandle<unsigned int> h_shape(m_attraction_scale_factor_shape,
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<Scalar> h_domain(m_domain, access_location::host, access_mode::read);

    // Neighbor-list arrays
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    // Build the interpolator: lo = {y_lo, t_lo}, hi = {y_hi, t_hi}
    const Scalar lo[2] = {Scalar(0.0), h_domain.data[0]};
    const Scalar hi[2] = {Scalar(1.0), h_domain.data[1]}; // y is always scaled between 0 and 1
    LinearInterpolator2D<Scalar> interp(h_data.data, h_shape.data, lo, hi);

    auto clamp_scaled_y = [](Scalar y, Scalar height, Scalar ylo)
    {
        const Scalar s = (y - ylo) / (height - ylo);
        if (!(s > Scalar(0.0)))
            return Scalar(0.0);
        if (s > Scalar(1.0))
            return Scalar(1.0);
        return s;
    };

    const BoxDim box = m_pdata->getGlobalBox();
    const unsigned int N = m_pdata->getN();
    const unsigned int N_tot = N + m_pdata->getNGhosts();

    GPUArray<Scalar> scale_factor(N_tot, m_exec_conf);
    ArrayHandle<Scalar> h_scale_factor(scale_factor, access_location::host, access_mode::readwrite);

    for (unsigned int k = 0; k < N_tot; k++)
        {
        const Scalar scaled_pos_y
            = clamp_scaled_y(h_pos.data[k].y, interface_height, box.getLo().y);
        h_scale_factor.data[k] = interp(scaled_pos_y, scaled_t);
        }

    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar3 pos_i = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0;

        const Scalar attraction_scale_factor_i = h_scale_factor.data[i];

        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        const size_t head = h_head_list.data[i];

        for (unsigned int k = 0; k < size; ++k)
            {
            const unsigned int j = h_nlist.data[head + k];
            if (j == i)
                continue;
            Scalar3 pos_j = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

            const Scalar attraction_scale_factor_j = h_scale_factor.data[j];

            // Minimum-image
            Scalar3 dx = pos_i - pos_j;
            dx = box.minImage(dx);

            const Scalar rsq = dot(dx, dx);
            const Scalar attraction_scale_factor_avg
                = Scalar(0.5) * (attraction_scale_factor_i + attraction_scale_factor_j);

            const Scalar wca_shift
                = epsilon_x_4 * (Scalar(1.0) - attraction_scale_factor_avg) / Scalar(4.0);

            if (rsq < rcutsq && lj1 != 0)
                {
                const Scalar r2inv = Scalar(1) / rsq;
                const Scalar r6inv = r2inv * r2inv * r2inv;

                Scalar pair_eng = r6inv * (lj1 * r6inv - lj2);
                Scalar force_divr
                    = r6inv * r2inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);

                if (rsq < rwcasq)
                    {
                    pair_eng += wca_shift;
                    }
                else
                    {
                    pair_eng *= attraction_scale_factor_avg;
                    force_divr *= attraction_scale_factor_avg;
                    }

                if (m_energy_shift)
                    {
                    const Scalar rcut2inv = Scalar(1.0) / rcutsq;
                    const Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;

                    Scalar pair_eng_shift = rcut6inv * (lj1 * rcut6inv - lj2);

                    if (rcutsq < rwcasq)
                        {
                        pair_eng_shift += wca_shift;
                        }
                    else
                        {
                        pair_eng_shift *= attraction_scale_factor_avg;
                        }

                    // Apply the shift to the pair energy
                    pair_eng -= pair_eng_shift;
                    }

                fi.x += force_divr * dx.x;
                fi.y += force_divr * dx.y;
                fi.z += force_divr * dx.z;

                pei += pair_eng;

                if (third_law)
                    {
                    h_force.data[j].x -= force_divr * dx.x;
                    h_force.data[j].y -= force_divr * dx.y;
                    h_force.data[j].z -= force_divr * dx.z;
                    h_force.data[j].w += Scalar(0.5) * pair_eng;
                    }
                }
            }

        h_force.data[i].x += fi.x;
        h_force.data[i].y += fi.y;
        h_force.data[i].z += fi.z;
        h_force.data[i].w += Scalar(0.5) * pei;
        }
    }

    } // end namespace azplugins
    } // end namespace hoomd
