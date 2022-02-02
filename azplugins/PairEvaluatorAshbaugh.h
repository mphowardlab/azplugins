// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file PairEvaluatorAshbaugh.h
 * \brief Defines the pair force evaluator class for Ashbaugh-Hatch potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_ASHBAUGH_H_
#define AZPLUGINS_PAIR_EVALUATOR_ASHBAUGH_H_

#include "PairEvaluator.h"

#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace azplugins
{

namespace detail
{
//! Ashbaugh-Hatch parameters
/*!
 * \sa PairEvaluatorAshbaugh
 */
struct ashbaugh_params
    {
    Scalar lj1; //!< The coefficient for 1/r^12
    Scalar lj2; //!< The coefficient for 1/r^6
    Scalar lambda; //!< Controls the attractive tail, between 0 and 1
    Scalar rwcasq; //!< The square of the location of the LJ potential minimum
    Scalar wca_shift; //!< The amount to shift the repulsive part by
    };

//! Convenience function for making ashbaugh_params in python
HOSTDEVICE inline ashbaugh_params make_ashbaugh_params(Scalar lj1,
                                                       Scalar lj2,
                                                       Scalar lambda,
                                                       Scalar rwcasq,
                                                       Scalar wca_shift)
    {
    ashbaugh_params p;
    p.lj1 = lj1;
    p.lj2 = lj2;
    p.lambda = lambda;
    p.rwcasq = rwcasq;
    p.wca_shift = wca_shift;
    return p;
    }

//! Class for evaluating the Ashbaugh-Hatch pair potential
/*!
 * PairEvaluatorAshbaugh evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma, \alpha) + (1-\lambda)\varepsilon & r < (2/\alpha)^{1/6}\sigma \\
 *            = & \lambda V_{\mathrm{LJ}}(r, \varepsilon, \sigma, \alpha) & (2/\alpha)^{1/6}\sigma \ge r < r_{\mathrm{cut}} \\
 *            = & 0 & r \ge r_{\mathrm{cut}}
 *      \f}
 *
 * where \f$V_{\mathrm{LJ}}(r,\varepsilon,\sigma,\alpha)\f$ is the standard Lennard-Jones potential (see EvaluatorPairLJ)
 * with parameters \f$\varepsilon\f$, \f$\sigma\f$, and \f$\alpha\f$ (default: 1.0). This potential is implemented
 * as given in
 * <a href="http://dx.doi.org/10.1021/ja802124e">H.S. Ashbaugh and H.W. Hatch, J. Am. Chem. Soc., 130, 9536 (2008)</a>.
 *
 * The Ashbaugh-Hatch potential does not need diameter or charge. Five parameters are specified and stored in a
 * ashbaugh_params. These are related to the standard Lennard-Jones and Ashbaugh-Hatch parameters by:
 * - \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
 * - \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);
 * - \a lambda is the scale factor for the attraction (0 = WCA, 1 = LJ)
 * - \a rwcasq is the square of the location of the potential minimum (WCA cutoff), pow(2.0/alpha,1./3.) * sigma * sigma
 * - \a wca_shift is the amount needed to shift the energy of the repulsive part to match the attractive energy.
 */
class PairEvaluatorAshbaugh : public PairEvaluator
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef ashbaugh_params param_type;

        //! Constructor
        /*!
         * \param _rsq Squared distance between particles
         * \param _rcutsq Cutoff radius squared
         * \param _params Pair potential parameters, given by typedef above
         *
         * The functor initializes its members from \a _params.
         */
        DEVICE PairEvaluatorAshbaugh(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : PairEvaluator(_rsq, _rcutsq), lj1(_params.lj1), lj2(_params.lj2), lambda(_params.lambda),
              rwcasq(_params.rwcasq), wca_shift(_params.wca_shift)
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
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;
                force_divr= r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);

                pair_eng = r6inv * (lj1*r6inv - lj2);
                if (rsq < rwcasq)
                    {
                    pair_eng += wca_shift;
                    }
                else
                    {
                    force_divr *= lambda;
                    pair_eng *= lambda;
                    }

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= lambda * rcut6inv * (lj1*rcut6inv - lj2);
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
            return std::string("ashbaugh");
            }
        #endif

    protected:
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
        Scalar lambda;  //!< lambda parameter
        Scalar rwcasq;  //!< WCA cutoff radius squared
        Scalar wca_shift; //!< Energy shift for WCA part of the potential
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_ASHBAUGH_H_
