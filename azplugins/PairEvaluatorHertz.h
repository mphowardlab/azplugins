// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: arjunsg2

/*!
 * \file PairEvaluatorHertz.h
 * \brief Defines the pair force evaluator class for Hertz potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_HERTZ_H_
#define AZPLUGINS_PAIR_EVALUATOR_HERTZ_H_

#include "PairEvaluator.h"
#include <math.h>

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace azplugins
{

namespace detail
{
//! Class for evaluating the Hertz pair potential
/*!
 * PairEvaluatorHertz evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & \varepsilon (1-\frac{r}{r_{\mathrm{cut}}})^\frac{5}{2} & r < r_{\mathrm{cut}} \\
 *            = & 0 r \ge r_{\mathrm{cut}}
 *      \f}
 *
 * This potential is implemented as given in
 * <a href="https://doi.org/10.1063/1.3186742">Pàmies, J. C., Cacciuto, A., & Frenkel, D., Journal of Chemical Physics, 131(4) (2009)</a>.
 *
 * The Hertz potential does not need diameter or charge. One parameter is specified and stored in a
 * Scalar2.
 */
class PairEvaluatorHertz : public PairEvaluator
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructor
        /*!
         * \param _rsq Squared distance between particles
         * \param _rcutsq Cutoff radius squared
         * \param _params Pair potential parameters, given by typedef above
         *
         * The functor initializes its members from \a _params.
         */
        DEVICE PairEvaluatorHertz(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
          : PairEvaluator(_rsq,_rcutsq), epsilon(_params.x)
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
            if (rsq < rcutsq && epsilon != 0)
                {
                Scalar r = sqrt(rsq);
                Scalar rcut = sqrt(rcutsq);
                Scalar x = (Scalar(1.0) - (r / rcut));
                force_divr = (Scalar(1.0) / r) * ((Scalar(5.0) * epsilon) / (Scalar(2.0) * rcut)) * pow(x, Scalar(1.5));
                pair_eng = epsilon * pow(x, Scalar(2.5));
                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Return the name of this potential
        static std::string getName()
            {
            return std::string("hertz");
            }
        #endif

    protected:
        Scalar epsilon;     //!< epsilon parameter extracted from the params passed to the constructor
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_HERTZ_H_
