// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_
#define AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "hoomd/BoxDim.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include "LinearInterpolator2D.h"

namespace hoomd
    {
namespace azplugins
    {

class PYBIND11_EXPORT PerturbedLennardJonesEvap : public ForceCompute
    {
    public:
    PerturbedLennardJonesEvap(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<hoomd::md::NeighborList> nlist,
                              Scalar r_cut,
                              Scalar epsilon,
                              Scalar sigma,
                              Scalar dt,
                              Scalar initial_delta,
                              Scalar solventD,
                              const Scalar* lambda_data,
                              const unsigned int* lambda_shape,
                              const Scalar* domain);

    virtual ~PerturbedLennardJonesEvap();

    Scalar getRCut() const
        {
        return m_r_cut;
        }

    Scalar getEpsilon() const
        {
        return m_epsilon;
        }

    Scalar getSigma() const
        {
        return m_sigma;
        }

    Scalar getInitial_delta() const
        {
        return m_initial_delta;
        }

    Scalar getSolventD() const
        {
        return m_solventD;
        }

    const GPUArray<Scalar>& getLambdaDomain() const
        {
        return m_domain;
        }

    protected:
    std::shared_ptr<hoomd::md::NeighborList> m_nlist; //!< Neighbor list

    // Lennard-Jones parameters
    Scalar m_r_cut;
    Scalar m_r_cutsq;
    Scalar m_epsilon;
    Scalar m_sigma;
    Scalar m_dt;
    Scalar m_initial_delta;
    Scalar m_solventD;

    GPUArray<Scalar> m_domain;             //!< Scaled coordinate domain [y_lo, y_hi, t_lo, t_hi]
    GPUArray<Scalar> m_lambda_data;        //!< Flattened scaled position scaled time data
    GPUArray<unsigned int> m_lambda_shape; //!< [ny, nt]

    void computeForces(uint64_t timestep) override;
    };

namespace detail
    {
void export_PerturbedLennardJonesEvap(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_H_
