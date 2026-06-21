// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_
#define AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_

#include <cmath>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "hoomd/BoxDim.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Variant.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include "LinearInterpolator2D.h"
#include "VariantInterpolated.h"

namespace hoomd
    {
namespace azplugins
    {

struct PairParametersPerturbedLennardJonesEvap
    {
    Scalar epsilon_x_4;
    Scalar sigma_6;
    Scalar rwcasq;

#ifndef __HIPCC__

    //! Default constructor
    PairParametersPerturbedLennardJonesEvap() : epsilon_x_4(0), sigma_6(0), rwcasq(0) { }

    PairParametersPerturbedLennardJonesEvap(Scalar epsilon, Scalar sigma)
        {
        const Scalar sigma_2 = sigma * sigma;
        const Scalar sigma_4 = sigma_2 * sigma_2;
        sigma_6 = sigma_2 * sigma_4;
        epsilon_x_4 = Scalar(4.0) * epsilon;
        rwcasq = std::pow(Scalar(2.0), Scalar(1.0) / Scalar(3.0)) * sigma_2;
        }

    PairParametersPerturbedLennardJonesEvap(pybind11::dict v, bool managed = false)
        {
        auto sigma = v["sigma"].cast<Scalar>();
        auto epsilon = v["epsilon"].cast<Scalar>();

        const Scalar sigma_2 = sigma * sigma;
        const Scalar sigma_4 = sigma_2 * sigma_2;
        sigma_6 = sigma_2 * sigma_4;
        epsilon_x_4 = Scalar(4.0) * epsilon;
        rwcasq = std::pow(Scalar(2.0), Scalar(1.0) / Scalar(3.0)) * sigma_2;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["sigma"] = std::pow(sigma_6, Scalar(1.0) / Scalar(6.0));
        v["epsilon"] = epsilon_x_4 / Scalar(4.0);
        return v;
        }
#endif // __HIPCC__
    };

class PerturbedLennardJonesEvap : public ForceCompute
    {
    public:
    typedef PairParametersPerturbedLennardJonesEvap param_type;

    PerturbedLennardJonesEvap(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<hoomd::md::NeighborList> nlist,
                              const Scalar r_cut,
                              const Scalar time_scale_factor,
                              const param_type& params,
                              bool energy_shift,
                              const Scalar* attraction_scale_factor_data,
                              const unsigned int* attraction_scale_factor_shape,
                              const Scalar* domain,
                              std::shared_ptr<VariantInterpolated> variant);

    //! Destructor
    ~PerturbedLennardJonesEvap()
        {
        if (m_r_cut_nlist)
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }

    Scalar scaleTime(uint64_t timestep) const
        {
        return Scalar(static_cast<Scalar>(timestep) / m_time_scale_factor);
        }

    Scalar getRCut() const
        {
        return m_rcut;
        }

    Scalar getEpsilon() const
        {
        return epsilon_x_4 / Scalar(4.0);
        }

    Scalar getSigma() const
        {
        return std::pow(sigma_6, Scalar(1.0) / Scalar(6.0));
        }

    protected:
    std::shared_ptr<hoomd::md::NeighborList> m_nlist; //!< Neighbor list
    Scalar epsilon_x_4;
    Scalar m_rcut;
    Scalar m_time_scale_factor; //!< Time scaling factor
    Scalar lj1;
    Scalar lj2;
    Scalar rcutsq;
    Scalar rwcasq;
    Scalar sigma_6;
    bool m_energy_shift;
    GPUArray<Scalar> m_domain;                              //!< [y_lo, y_hi, t_lo, t_hi]
    GPUArray<Scalar> m_attraction_scale_factor_data;        //!< Flattened (y, t) data
    GPUArray<unsigned int> m_attraction_scale_factor_shape; //!< [ny, nt]
    std::shared_ptr<VariantInterpolated> m_variant;

    std::shared_ptr<GPUArray<Scalar>>
        m_r_cut_nlist; //!< Cutoff matrix shared with the neighbor list

    void computeForces(uint64_t timestep) override;
    };

namespace detail
    {
void export_PerturbedLennardJonesEvap(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_
