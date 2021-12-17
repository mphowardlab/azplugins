// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file PairEvaluator.h
 * \brief Base class for pair evaluators.
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_H_
#define AZPLUGINS_PAIR_EVALUATOR_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <string>
#endif

namespace azplugins
{
namespace detail
{

//! Base class for isotropic pair potential evaluator
/*!
 * This class covers the default case of a simple potential that doesn't require
 * additional data. Its constructor stores the common variables. The class can
 * be overridden for more complex potentials.
 *
 * Deriving classes \b must define a \a param_type . They must also
 * give the potential a name and implement the evaluate() method.
 */
class PairEvaluator
    {
    public:
        DEVICE PairEvaluator(const Scalar _rsq,
                             const Scalar _rcutsq)
        : rsq(_rsq), rcutsq(_rcutsq)
        {}

        //! Base potential does not need diameter
        DEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Accept the optional diameter values
        /*!
         * \param di Diameter of particle i
         * \param dj Diameter of particle j
         */
        DEVICE void setDiameter(Scalar di, Scalar dj) {}

        //! Base potential does not need charge
        DEVICE static bool needsCharge()
            {
            return false;
            }

        //! Accept the optional charge values
        /*!
         * \param qi Charge of particle i
         * \param qj Charge of particle j
         */
        DEVICE void setCharge(Scalar qi, Scalar qj) {}

        //! Evaluate the force and energy
        /*!
         * \param force_divr Holds the computed force divided by r
         * \param pair_eng Holds the computed pair energy
         * \param energy_shift If true, the potential is shifted to zero at the cutoff
         *
         * \returns True if the energy calculation occurs
         *
         * The calculation does not occur if the pair distance is greater than the cutoff
         * or if the potential is scaled to zero.
         */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            return false;
            }

    #ifndef NVCC
        //! Get the name of this potential
        /*!
         * This method must be overridden by deriving classes.
         */
        static std::string getName()
            {
            throw std::runtime_error("Name not defined for this pair potential.");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
    #endif // NVCC

    protected:
        Scalar rsq;     //!< Squared distance between particles
        Scalar rcutsq;  //!< Squared cutoff distance
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_H_
