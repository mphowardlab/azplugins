// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file SpecialPairEvaluator.h
 * \brief Defines the special pair evaluator class which templates on an existing evaluator.
 */

#ifndef AZPLUGINS_SPECIAL_PAIR_EVALUATOR_H_
#define AZPLUGINS_SPECIAL_PAIR_EVALUATOR_H_

#include "hoomd/HOOMDMath.h"

#ifndef NVCC
#include <string>
#endif

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace azplugins
{
namespace detail
{

//! Class for evaluating a special pair potential
/*!
 * \tparam evaluator Base pair potential evaluator
 *
 * SpecialPairEvaluator is a convenience wrapper around an existing pair potential. Internally,
 * an \a evaluator is created when the force is computed. The \a evaluator parameters are wrapped
 * in an internal struct that also stores the cutoff radius and energy shifting mode, which are
 * necessary to evaluate a standard pair potential. This saves considerable development overhead
 * since the same evaluators can be reused between pair potentials and special pair potentials.
 */
template<class evaluator>
class SpecialPairEvaluator
    {
    public:
        //! Define the parameter type used by the special pair potential evaluator
        typedef struct
            {
            typename evaluator::param_type params; //!< Parameters for the evaluator it templates on
            Scalar rcutsq; //!< The square of the cutoff distance
            bool energy_shift; //!< Whether the energy should be cut and shifted or just cut
            } param_type;

        //! Constructor
        /*!
         * \param rsq Squared distance between particles
         * \param params Special pair potential parameters, given by typedef above
         */
        DEVICE SpecialPairEvaluator<evaluator>(Scalar rsq, const param_type& params)
            : m_eval(rsq, params.rcutsq, params.params), m_shift(params.energy_shift)
            { }

        //! Special pair potential needs diameter if the potential it templates on needs diameter
        DEVICE static bool needsDiameter()
            {
            return evaluator::needsDiameter();
            }

        //! Accept and stash the diameter values
        /*!
         * \param di Diameter of particle i
         * \param dj Diameter of particle j
         */
        DEVICE void setDiameter(Scalar di, Scalar dj)
            {
            m_eval.setDiameter(di, dj);
            }

        //! Special pair potential needs charge if the potential it templates on needs charge
        DEVICE static bool needsCharge()
            {
            return evaluator::needsCharge();
            }

        //! Accept and stash the charge values
        /*!
         * \param qi Charge of particle i
         * \param qj Charge of particle j
         */
        DEVICE void setCharge(Scalar qi, Scalar qj)
            {
            m_eval.setCharge(qi,qj);
            }

        //! Evaluate the force and energy
        /*!
         * \param force_divr Holds the computed force divided by r
         * \param pair_eng Holds the computed pair energy
         *
         * \returns True if force and energy are evaluated or false if the energy is not defined
         *
         * The calculation does not occur if the pair distance is greater than the cutoff
         * or if the potential is scaled to zero.
         */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng)
            {
            m_eval.evalForceAndEnergy(force_divr, pair_eng, m_shift);
            return true;
            }

        #ifndef NVCC
        //! Return the name of the potential on which the special pair potential is templated
        static std::string getName()
            {
            return evaluator::getName();
            }
        #endif

    private:
        evaluator m_eval;   //!< Evaluator for this pair potential
        bool m_shift;       //!< If true, shift the energy to zero at the cutoff
    };

//! Convenience function for making special pair parameters.
/*!
 * \param params Pair potential parameters
 * \param rcutsq Cutoff radius squared
 * \param energy_shift Flag for energy shifting mode.
 * \tparam evaluator Pair potential evaluator to template on
 *
 * This convenience method is required for making the special pair parameters on the python level.
 */
template<class evaluator>
typename SpecialPairEvaluator<evaluator>::param_type make_special_pair_params(typename evaluator::param_type params,
                                                                              Scalar rcutsq,
                                                                              bool energy_shift)
    {
    typename SpecialPairEvaluator<evaluator>::param_type p;
    p.params = params;
    p.rcutsq = rcutsq;
    p.energy_shift = energy_shift;
    return p;
    }

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_SPECIAL_PAIR_EVALUATOR_H_
