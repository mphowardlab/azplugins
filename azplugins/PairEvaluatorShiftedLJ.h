// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file PairEvaluatorShiftedLJ.h
 * \brief Defines the pair force evaluator class for core-shifted Lennard-Jones potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_SHIFTED_LJ_H_
#define AZPLUGINS_PAIR_EVALUATOR_SHIFTED_LJ_H_

#include "PairEvaluator.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace azplugins
{

namespace detail
{
//! Class for evaluating the core-shifted Lennard-Jones pair potential
/*!
 * PairEvaluatorShiftedLJ evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & 4 \varepsilon \left[ \left(\frac{\sigma}{r-\Delta} \right)^12
                  - \alpha \left(\frac{\sigma}{r-\Delta} \right)^6 \right] & r < r_{\rm cut} \\
 *            = & 0 & r \ge r_{\rm cut}
 *      \f}
 *
 * where \f$\varepsilon\f$, \f$\sigma\f$, and \f$\alpha\f$ (default: 1.0) are the standard Lennard-Jones parameters.
 * \f$\Delta\f$ is a shift factor for the minimum of the potential, which moves to \f$r_{\rm min} = 2^{1/6} \sigma + \Delta \f$.
 * This potential is analogous to the HOOMD shifted Lennard-Jones potential (see EvaluatorPairSLJ), but does
 * not rely on reading the diameter to set the shift factor. This can be more convenient for certain systems where
 * the shift factor is a parameterization, but not necessarily connected to the diameter of the particle.
 *
 * The core-shifted Lennard-Jones potential does not need diameter or charge. Three parameters are specified and stored in a
 * Scalar3. These are related to the standard Lennard-Jones parameters by:
 * - \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
 * - \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);
 * - \a Delta is the amount to shift the potential minimum by.
 */
class PairEvaluatorShiftedLJ : public PairEvaluator
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar3 param_type;

        //! Constructor
        /*!
         * \param _rsq Squared distance between particles
         * \param _rcutsq Cutoff radius squared
         * \param _params Pair potential parameters, given by typedef above
         *
         * The functor initializes its members from \a _params.
         */
        DEVICE PairEvaluatorShiftedLJ(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : PairEvaluator(_rsq,_rcutsq), lj1(_params.x), lj2(_params.y), delta(_params.z)
            {}

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
            if (rsq < rcutsq && lj1 != 0)
                {
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r = Scalar(1.0) / rinv;

                Scalar rmd = r - delta;
                Scalar rmdinv = Scalar(1.0) / rmd;
                Scalar rmd2inv = rmdinv * rmdinv;
                Scalar rmd6inv = rmd2inv * rmd2inv * rmd2inv;
                force_divr= rinv * rmdinv * rmd6inv * (Scalar(12.0)*lj1*rmd6inv - Scalar(6.0)*lj2);

                pair_eng = rmd6inv * (lj1*rmd6inv - lj2);

                /*
                 * Energy shift requires another sqrt call. If this pair becomes
                 * performance limiting, it could be replaced by caching the shift
                 * into a Scalar4.
                 */
                if (energy_shift)
                    {
                    Scalar rcutinv = fast::rsqrt(rcutsq);
                    Scalar rcut = Scalar(1.0) / rcutinv;

                    Scalar rcutmd = rcut - delta;
                    Scalar rcutmdinv = Scalar(1.0) / rcutmd;
                    Scalar rcutmd2inv = rcutmdinv * rcutmdinv;
                    Scalar rcutmd6inv = rcutmd2inv * rcutmd2inv * rcutmd2inv;
                    pair_eng -= rcutmd6inv * (lj1*rcutmd6inv - lj2);
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Return the name of this potential
        static std::string getName()
            {
            return std::string("slj");
            }
        #endif

    protected:
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
        Scalar delta;   //!< shift parameter
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_SHIFTED_LJ_H_
