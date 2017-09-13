// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file PairEvaluatorAshbaugh24.h
 * \brief Defines the pair force evaluator class for a generalized Ashbaugh-Hatch 48-24 potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_ASHBAUGH24_H_
#define AZPLUGINS_PAIR_EVALUATOR_ASHBAUGH24_H_

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "PairEvaluatorAshbaugh.h"
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

//! Class for evaluating a generalized LJ 48-24 pair potential
/*!
 * PairEvaluatorAshbaugh24 evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) + (1-\lambda)\varepsilon & r < 2^{1/24}\sigma \\
 *            = & \lambda V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) & 2^{1/24}\sigma \ge r < r_{\mathrm{cut}} \\
 *            = & 0 & r \ge r_{\mathrm{cut}}
 *      \f}
 *
 * where \f$V_{\mathrm{LJ,48-24}}(r,\varepsilon,\sigma)\f$ is a generalized Lennard-Jones potential
 * with parameters \f$\varepsilon\f$ and \f$\sigma\f$:
 *
 *   \f{eqnarray*}{
 *     V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) = \left( \left(\frac{\sigma}{r}\right)^{48} - \left(\frac{\sigma}{r}\right)^{24} \right)
 *      \f}
 *
 * This potential is implemented as given in
 * <a href="http://dx.doi.org/10.1039/C5NR04661K">L. Rovigatti,B. Capone, C. Likos, Nanoscale, 8 (2016) </a>.
 * It is essentially the same as the Ashbaugh-Hatch potential (see PairEvaluatorAshbaugh) but
 * with exponents 48 and 24 instead of the regular LJ exponents.
 * The parameter \f$\alpha\f$ for the original Ashbaugh-Hatch potential is set to 1.0.
 *
 * The Ashbaugh24 potential does not need diameter or charge.
 * Five parameters are specified and stored in a ashbaugh_params.
 * These are related to the  Lennard-Jones parameters by:
 * - \a lj1 = 4.0 * epsilon * pow(sigma,48.0)
 * - \a lj2 = 4.0 * epsilon * pow(sigma,24.0);
 * - \a lambda is the scale factor for the attraction (0 = purely repulsive, 1 = full LJ 48-24)
 * - \a rwcasq is the square of the location of the potential minimum (repulsive cutoff), pow(2.0,1./12.) * sigma * sigma
 * - \a wca_shift is the amount needed to shift the energy of the repulsive
 *      part to match the attractive energy, (1-lambda) * epsilon.
 *
 * Here, WCA means the repulsive part of the Lennard-Jones 48-24 potential.
 */
class PairEvaluatorAshbaugh24
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
        DEVICE PairEvaluatorAshbaugh24(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
          : rsq(_rsq), rcutsq(_rcutsq),lj1(_params.lj1), lj2(_params.lj2), lambda(_params.lambda),
              rwcasq(_params.rwcasq), wca_shift(_params.wca_shift)
            {
            }

        //! LJ 48-24 potential does not need diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*!
         * \param di Diameter of particle i
         * \param dj Diameter of particle j
         */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ 48-24 potential does not need charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional charge values
        /*!
         * \param qi Charge of particle i
         * \param qj Charge of particle j
         */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

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
                const Scalar r4inv = r2inv*r2inv;
                const Scalar r6inv = r2inv*r4inv;
                const Scalar r12inv = r6inv*r6inv;
                const Scalar r24inv = r12inv * r12inv;
                force_divr= r2inv * r24inv * (Scalar(48.0)*lj1*r24inv - Scalar(24.0)*lj2);

                pair_eng = r24inv * (lj1*r24inv - lj2);
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
                    const Scalar rcut4inv = rcut2inv*rcut2inv;
                    const Scalar rcut6inv = rcut2inv*rcut4inv;
                    const Scalar rcut12inv = rcut6inv*rcut6inv;
                    const Scalar rcut24inv = rcut12inv * rcut12inv;
                    pair_eng -= lambda * rcut24inv * (lj1*rcut24inv - lj2);
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
            return std::string("ashbaugh24");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
        Scalar lambda;  //!< lambda parameter
        Scalar rwcasq;  //!< WCA cutoff radius squared
        Scalar wca_shift; //!< Energy shift for WCA part of the potential
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_ASHBAUGH24_H_
