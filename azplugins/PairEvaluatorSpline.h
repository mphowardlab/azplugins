// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file PairEvaluatorSpline.h
 * \brief Defines the pair force evaluator class for a spline pair potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_SPLINE_H_
#define AZPLUGINS_PAIR_EVALUATOR_SPLINE_H_

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

//! Class for evaluating a spline pair potential
/*! This evaluator calculates the following function:
 *      \begin{eqnarray*}
 *       V(r) = & a & r < r_{\rm s}\\
 *       = & a*(r_{\rm on}**2-r**2)^m * (r_{\rm cut}^2 + m*r**2 - (m+1)*r_{\rm on}^2)
 *          / (r_{\rm cut}^2-r_{\rm on}**2)**(m+1) & r_{\rm on} <r < r_{\rm cut} \\
 *             = & 0 & r \ge r_{\rm cut}
 *       \end{eqnarray*}
 *
 * The three  needed parameters are specified and stored in a
 * Scalar3.
 */
class PairEvaluatorSpline : public PairEvaluator
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
        DEVICE PairEvaluatorSpline(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : PairEvaluator(_rsq,_rcutsq), a(_params.x), m(_params.y), ron_sq(_params.z)
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
            if (rsq < rcutsq && a != 0)
                {
                if (rsq <= ron_sq)
                    {
                    force_divr = 0;
                    pair_eng   = a;
                    }
                else
                    {
                    Scalar A = pow(rcutsq-rsq,m-1);
                    Scalar B = A*(rcutsq-rsq);
                    Scalar numerator = a*B*(rcutsq + m*rsq - (m+1)*ron_sq);
                    Scalar denominator = pow(rcutsq-ron_sq,m+1);
                    pair_eng = numerator/denominator;
                    force_divr = -a*2*m*(m+1)*A*(ron_sq-rsq)/denominator;
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
            return std::string("spline");
            }
        #endif

    protected:
        Scalar a;     //!< a - amplitude parameter extracted from the params passed to the constructor
        Scalar m;     //!< m - exponent parameter extracted from the params passed to the constructor
        Scalar ron_sq;    //!< r_on**2 - squared distance for where the potential reaches the plateau value a
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_SPLINE_H_
