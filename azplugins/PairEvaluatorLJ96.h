// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file PairEvaluatorLJ96.h
 * \brief Defines the pair force evaluator class for LJ 9-6 potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_LJ96_H_
#define AZPLUGINS_PAIR_EVALUATOR_LJ96_H_

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
//! Class for evaluating the LJ 9-6 pair potential
/*!
 * PairEvaluatorLJ96 evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & \frac{27}{4} \varepsilon \left(\left(\frac{\sigma}{r}\right)^9 - \alpha \left(\frac{\sigma}{r}\right)^6\right) & r < r_{\mathrm{cut}} \\
 *            = & 0 r \ge r_{\mathrm{cut}}
 *      \f}
 *
 * This potential is implemented as given in
 * <a href="http://dx.doi.org/10.1080/08927020601054050">W. Shinoda, R. DeVane, and M. L. Klein, Molecular Simulation, 33, 27-36 (2007)</a>.
 *
 * The LJ 9-6 potential does not need diameter or charge. Two parameters are specified and stored in a
 * Scalar2. These are related to the standard Lennard-Jones and LJ 9-6 parameters by:
 * - \a lj1 = 27 / 4 * epsilon * pow(sigma,9.0)
 * - \a lj2 = alpha * 27 / 4 * epsilon * pow(sigma,6.0);
 */
class PairEvaluatorLJ96 : public PairEvaluator
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
        DEVICE PairEvaluatorLJ96(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
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
                Scalar r3inv = r2inv * fast::sqrt(r2inv);
                Scalar r6inv = r3inv * r3inv;
                force_divr= r2inv * r6inv * (Scalar(9.0)*lj1*r3inv - Scalar(6.0)*lj2);

                pair_eng = r6inv * (lj1*r3inv - lj2);

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut3inv = rcut2inv * fast::sqrt(rcut2inv);
                    Scalar rcut6inv = rcut3inv * rcut3inv;
                    pair_eng -= rcut6inv * (lj1*rcut3inv - lj2);
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
            return std::string("lj96");
            }
        #endif

    protected:
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_LJ96_H_
