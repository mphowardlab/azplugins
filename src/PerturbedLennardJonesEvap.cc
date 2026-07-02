// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PerturbedLennardJonesEvap.h"

#include <algorithm>
#include <stdexcept>

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
PerturbedLennardJonesEvap::PerturbedLennardJonesEvap(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar rcut,
    const Scalar time_scale_factor,
    bool energy_shift,
    const Scalar* attraction_scale_factor_data,
    const unsigned int* attraction_scale_factor_shape,
    const Scalar* domain,
    std::shared_ptr<VariantInterpolated> variant)
    : ForceCompute(sysdef), m_nlist(nlist), m_rcut(rcut), m_time_scale_factor(time_scale_factor),
      m_typpair_idx(m_pdata->getNTypes()), m_energy_shift(energy_shift), m_variant(variant)
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
        GPUArray<param_type> params_arr(m_typpair_idx.getNumElements(), m_exec_conf);
        m_params.swap(params_arr);
        ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::overwrite);
        }

        {
        m_r_cut_nlist
            = std::make_shared<GPUArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
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

void PerturbedLennardJonesEvap::validateTypes(unsigned int typ1,
                                              unsigned int typ2,
                                              std::string action) const
    {
    const unsigned int n_types = m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        throw std::runtime_error("Invalid type encountered when " + action);
    }

void PerturbedLennardJonesEvap::setParams(unsigned int typ1,
                                          unsigned int typ2,
                                          const param_type& param)
    {
    validateTypes(typ1, typ2, "setting params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);

    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }

void PerturbedLennardJonesEvap::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    const auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    const auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());

    const Scalar epsilon = params["epsilon"].cast<Scalar>();
    const Scalar sigma = params["sigma"].cast<Scalar>();
    const Scalar sigma_2 = sigma * sigma;

    param_type p;
    p.sigma_6 = sigma_2 * sigma_2 * sigma_2;
    p.epsilon_x_4 = Scalar(4.0) * epsilon;
    p.rwcasq = std::pow(Scalar(2.0), Scalar(1.0) / Scalar(3.0)) * sigma_2;
    p.attraction_scale_factor = Scalar(0.0);

    setParams(typ1, typ2, p);
    }

pybind11::dict PerturbedLennardJonesEvap::getParams(pybind11::tuple typ)
    {
    const auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    const auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting params");

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    const param_type& p = h_params.data[m_typpair_idx(typ1, typ2)];

    pybind11::dict v;
    v["epsilon"] = p.epsilon_x_4 / Scalar(4.0);
    v["sigma"] = std::pow(p.sigma_6, Scalar(1.0) / Scalar(6.0));
    return v;
    }

void PerturbedLennardJonesEvap::computeForces(uint64_t timestep)
    {
    m_nlist->compute(timestep);

    const bool third_law
        = (m_nlist->getStorageMode() == hoomd::md::NeighborList::storageMode::half);

    Scalar scaled_t = Scalar(static_cast<Scalar>(timestep) / m_time_scale_factor);

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
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

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

    const BoxDim box = m_pdata->getGlobalBox();
    const unsigned int N = m_pdata->getN();
    const unsigned int Ntot = N + m_pdata->getNGhosts();

    if (m_scale_factor.getNumElements() < Ntot)
        {
        GPUArray<Scalar> tmp(Ntot, m_exec_conf);
        m_scale_factor.swap(tmp);
        }
    ArrayHandle<Scalar> h_scale_factor(m_scale_factor,
                                       access_location::host,
                                       access_mode::readwrite);

    for (unsigned int k = 0; k < Ntot; k++)
        {
        Scalar scaled_pos_y
            = (h_pos.data[k].y - box.getLo().y) / (interface_height - box.getLo().y);
        if (scaled_pos_y < 0)
            {
            scaled_pos_y = Scalar(0.0);
            }
        if (scaled_pos_y > 1)
            {
            scaled_pos_y = Scalar(1.0);
            }
        h_scale_factor.data[k] = interp(scaled_pos_y, scaled_t);
        }

    Scalar rcutsq = m_rcut * m_rcut;
    for (unsigned int i = 0; i < N; ++i)
        {
        const Scalar3 pos_i = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int type_i = __scalar_as_int(h_pos.data[i].w);

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
            const unsigned int type_j = __scalar_as_int(h_pos.data[j].w);

            const Scalar attraction_scale_factor_j = h_scale_factor.data[j];

            // Minimum-image
            Scalar3 dx = pos_i - pos_j;
            dx = box.minImage(dx);

            const Scalar rsq = dot(dx, dx);
            param_type param = h_params.data[m_typpair_idx(type_i, type_j)];

            // Setting averaged attraction scale factor
            param.attraction_scale_factor
                = Scalar(0.5) * (attraction_scale_factor_i + attraction_scale_factor_j);

            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            PairEvaluatorPerturbedLennardJones eval(rsq, rcutsq, param);
            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, m_energy_shift);

            if (evaluated)
                {
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

namespace py = pybind11;

void export_PerturbedLennardJonesEvap(py::module& m)
    {
    py::class_<PerturbedLennardJonesEvap, ForceCompute, std::shared_ptr<PerturbedLennardJonesEvap>>(
        m,
        "PerturbedLennardJonesEvap")
        .def(py::init(
            [](std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<hoomd::md::NeighborList> nlist,
               Scalar rcut,
               Scalar time_scale_factor,
               bool energy_shift,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast>
                   attraction_scale_factor_data,
               py::array_t<unsigned int, py::array::c_style | py::array::forcecast>
                   attraction_scale_factor_shape,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> domain,
               std::shared_ptr<VariantInterpolated> variant)
            {
                if (attraction_scale_factor_shape.size() != 2)
                    throw std::runtime_error("lambda_shape must have 2 elements");
                if (domain.size() != 2)
                    throw std::runtime_error("domain must have 2 elements [t_lo, t_hi]");

                const unsigned int* shape_ptr = attraction_scale_factor_shape.data();
                const Scalar* data_ptr = attraction_scale_factor_data.data();
                const Scalar* dom_ptr = domain.data();

                if (attraction_scale_factor_data.size()
                    != static_cast<py::ssize_t>(shape_ptr[0] * shape_ptr[1]))
                    throw std::runtime_error(
                        "attraction_scale_factor_data size does not match lambda_shape");

                return std::make_shared<PerturbedLennardJonesEvap>(sysdef,
                                                                   nlist,
                                                                   rcut,
                                                                   time_scale_factor,
                                                                   energy_shift,
                                                                   data_ptr,
                                                                   shape_ptr,
                                                                   dom_ptr,
                                                                   variant);
            }))
        .def("setParams", &PerturbedLennardJonesEvap::setParamsPython)
        .def("getParams", &PerturbedLennardJonesEvap::getParams);
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
