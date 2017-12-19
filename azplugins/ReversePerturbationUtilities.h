// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file ReversePerturbationUtilities.h
 * \brief Helper functions for Reverse perturbation
 */

#ifndef AZPLUGINS_REVERSE_PERTURBATION_UTILITIES_H_
#define AZPLUGINS_REVERSE_PERTURBATION_UTILITIES_H_

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define HOSTDEVICE
#endif

namespace azplugins
{
namespace detail
{

//! Custom comparison operator for sorting arrays by their momentum closest to a target value.
class ReversePerturbationSorter
    {
    public:
        //! Constructor
        /*!
         * \param p_target Target momentum
         */
        HOSTDEVICE ReversePerturbationSorter(Scalar p_target)
            : p(p_target)
            { }

        //! Sorts two momenta
        /*!
         * \param in0 Momentum of particle 0
         * \param in1 Momentum of particle 1
         *
         * \returns True if \a in0 has momentum closer to the target value than \a in1
         *
         * Particles to be sorted \b must contain only the values with the appropriate
         * sign. (Particles to sort in the top slab have all negative momentum,
         * particles in the bottom slab have all positive momentum.)
         *
         * Particles in the top slab get sorted according to the closest value to -p,
         * and particles in the bottom slab get sorted according to the closest value to +p.
         */
        HOSTDEVICE bool operator()(const Scalar &in0, const Scalar &in1 ) const
            {
            signed char sign = (in0 < 0) - (in0 > 0); // +1 = bottom , -1 = top
            return sign*fabs(in0 + sign*p) < sign*fabs(in1 + sign*p);
            }

        //! Sorts particles by slab and momenta
        /*
         * \param in0 Signed, shifted tag and momentum of particle 0
         * \param in1 Signed, shifted tag and momentum of particle 1
         *
         * \returns True if \a in0 is in the top slab and \a in1 is not.
         *          If both are in the top slab, returns True if \a in0 is closer to -p
         *          than \a in 1. If both are in the bottom slab, returns False if \a in0
         *          is closer to +p than \a in1.
         *
         * The effective sorting order becomes: top slab closest to -p at the beginning and
         * bottom slab clostest to +p at the end of the sorted array.
         */
        HOSTDEVICE bool operator()(const Scalar2 &in0, const Scalar2 &in1 ) const
            {
            int tag0 = __scalar_as_int(in0.x);
            int tag1 = __scalar_as_int(in1.x);

            if ((tag0 ^ tag1) >=0) // same sign
                {
                signed char sign = (tag1 < 0) - (tag1 > 0);
                return sign*fabs(in0.y + sign*p) < sign*fabs(in1.y +sign*p);
                }
            else //different sign
                {
                return tag0 < tag1;
                }
            }

    private:
        Scalar p; //!< Target momentum value
    };

} // end namespace detail
} // end namespace azplugins

#undef HOSTDEVICE
#endif // AZPLUGINS_REVERSE_PERTURBATION_UTILITIES_H_
