// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: sjiao

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
 * necessary to evaluate a standard pair potential.
 *
 * This class increases the overhead of special pair evaluation because diameter and charge member
 * variables are always created, and the parameters are also stored both in the SpecialPairEvaluator
 * and in the instance of \a evaluator. However, the development convenience is worth it.
 * Eventually, \a rcutsq and \a energy_shift parameters should be moved into the base potential
 * so that the same pair evaluator can be used for both.
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
            : m_rsq(rsq), m_params(params) { }

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
            m_di = di;
            m_dj = dj;
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
            m_qi = qi;
            m_qj = qj;
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
            //! Create the evaluator on which this special pair evaluator is templated
            evaluator eval(m_rsq, m_params.rcutsq, m_params.params);

            //! Gives the created evaluator the stashed diameter and charge values, if needed
            if (evaluator::needsDiameter())
                eval.setDiameter(m_di, m_dj);
            if (evaluator::needsCharge())
                eval.setCharge(m_qi, m_qj);

            eval.evalForceAndEnergy(force_divr, pair_eng, m_params.energy_shift);
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
        Scalar m_rsq;           //!< Distance between particles in pair
        param_type m_params;    //!< Parameters for special pair potential
        Scalar m_di;            //!< Diameter of i-th particle
        Scalar m_dj;            //!< Diameter of j-th particle
        Scalar m_qi;            //!< Charge of i-th particle
        Scalar m_qj;            //!< Charge of j-th particle
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
