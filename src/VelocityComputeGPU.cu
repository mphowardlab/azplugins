// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "VelocityComputeGPU.cuh"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

//! Transform to go from an index to a momentum+mass Scalar4 using a particle data loader
template<class LoadOpT> class MomentumMassTransform
    {
    public:
    __host__ __device__ MomentumMassTransform(const LoadOpT& load_op) : m_load_op(load_op) { }

    __host__ __device__ Scalar4 operator()(unsigned int idx) const
        {
        Scalar3 v;
        Scalar m;
        m_load_op(v, m, idx);

        return make_scalar4(m * v.x, m * v.y, m * v.z, m);
        }

    private:
    const LoadOpT m_load_op;
    };

//! Summation for Scalar4s
struct AddScalar4
    {
    __host__ __device__ Scalar4 operator()(const Scalar4& a, const Scalar4& b) const
        {
        return make_scalar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
    };

//! Thrust implementation of summation for different particle data loaders
template<class LoadOpT>
void thrust_sum_momentum_and_mass(Scalar3& momentum,
                                  Scalar& mass,
                                  const LoadOpT& load_op,
                                  unsigned int N)
    {
    thrust::transform_iterator momentum_mass_iterator(thrust::counting_iterator<unsigned int>(0),
                                                      MomentumMassTransform(load_op));

    const Scalar4 momentum_mass = thrust::reduce(thrust::device,
                                                 momentum_mass_iterator,
                                                 momentum_mass_iterator + N,
                                                 make_scalar4(0, 0, 0, 0),
                                                 AddScalar4());

    momentum = make_scalar3(momentum_mass.x, momentum_mass.y, momentum_mass.z);
    mass = momentum_mass.w;
    }

void sum_momentum_and_mass(Scalar3& momentum,
                           Scalar& mass,
                           const detail::LoadParticleGroupVelocityMass& load_op,
                           unsigned int N)
    {
    thrust_sum_momentum_and_mass(momentum, mass, load_op, N);
    }

void sum_momentum_and_mass(Scalar3& momentum,
                           Scalar& mass,
                           const detail::LoadMPCDParticleVelocityMass& load_op,
                           unsigned int N)
    {
    thrust_sum_momentum_and_mass(momentum, mass, load_op, N);
    }

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd
