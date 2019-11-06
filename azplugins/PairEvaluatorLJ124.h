// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: sjiao

/*!
 * \file PairEvaluatorLJ124.h
 * \brief Defines the pair force evaluator class for LJ 12-4 potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_LJ124_H_
#define AZPLUGINS_PAIR_EVALUATOR_LJ124_H_

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
//! Class for evaluating the LJ 12-4 pair potential
/*!
 * PairEvaluatorLJ124 evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & \frac{3 \sqrt{3}}{2} \varepsilon \left(\left(\frac{\sigma}{r}\right)^{12} - \alpha \left(\frac{\sigma}{r}\right)^4\right) & r < r_{\mathrm{cut}} \\
 *            = & 0 r \ge r_{\mathrm{cut}}
 *      \f}
 *
 * This potential is implemented as given in
 * <a href="http://dx.doi.org/10.1080/08927020601054050">W. Shinoda, R. DeVane, and M. L. Klein, Molecular Simulation, 33, 27-36 (2007)</a>.
 *
 * The LJ 12-4 potential does not need diameter or charge. Two parameters are specified and stored in a
 * Scalar2. These are related to the standard Lennard-Jones and LJ 12-4 parameters by:
 * - \a lj1 = 1.5 * sqrt(3.0) * epsilon * pow(sigma,12.0)
 * - \a lj2 = alpha * 1.5 * sqrt(3.0) * epsilon * pow(sigma,4.0);
 */
class PairEvaluatorLJ124 : public PairEvaluator
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
        DEVICE PairEvaluatorLJ124(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
          : PairEvaluator(_rsq,_rcutsq), lj1(_params.x), lj2(_params.y)
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
                Scalar r4inv = r2inv * r2inv;
                Scalar r8inv = r4inv * r4inv;
                force_divr= r2inv * r4inv * (Scalar(12.0)*lj1*r8inv - Scalar(4.0)*lj2);

                pair_eng = r4inv * (lj1*r8inv - lj2);

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut4inv = rcut2inv * rcut2inv;
                    Scalar rcut8inv = rcut4inv * rcut4inv;
                    pair_eng -= rcut4inv * (lj1*rcut8inv - lj2);
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
            return std::string("lj124");
            }
        #endif

    protected:
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_LJ124_H_
