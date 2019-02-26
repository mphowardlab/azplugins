// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*! \file BondEvaluatorFENEAsh24.h
    \brief Defines the bond evaluator class for a FENE-24-48 potential
*/

#ifndef AZPLUGINS_BOND_EVALUATOR_FENE_ASH24_H_
#define AZPLUGINS_BOND_EVALUATOR_FENE_ASH24_H_

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

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
struct ashbaugh_bond_params
    {
    Scalar lj1;      //!< The coefficient for 1/r^12
    Scalar lj2;      //!< The coefficient for 1/r^6
    Scalar lambda;   //!< Controls the attractive tail, between 0 and 1
    Scalar rwcasq;   //!< The square of the location of the LJ potential minimum
    Scalar wca_shift;//!< The amount to shift the repulsive part by
    Scalar K;        //!< Stiffness parameter for the bond
    Scalar r_0;      //!< maximum bond length
    };

//! Convenience function for making ashbaugh_bond_params in python
HOSTDEVICE inline ashbaugh_bond_params make_ashbaugh_bond_params(Scalar lj1,
                                                       Scalar lj2,
                                                       Scalar lambda,
                                                       Scalar rwcasq,
                                                       Scalar wca_shift,
                                                       Scalar K,
                                                       Scalar r_0)
    {
    ashbaugh_bond_params p;
    p.lj1 = lj1;
    p.lj2 = lj2;
    p.lambda = lambda;
    p.rwcasq = rwcasq;
    p.wca_shift = wca_shift;
    p.K = K;
    p.r_0 = r_0;
    return p;
    }


/*!
 *  Class for evaluating the FENE-48-24 bond potential
 *  The parameters are:
 *   - \a K (params.x) Stiffness parameter for the force computation
 *   - \a r_0 (params.y) maximum bond length for the force computation
 *   - \a lj1 (params.z) Value of lj1 = 4.0*epsilon*pow(sigma,48.0)
 *      of the WCA potential in the force calculation
 *   - \a lj2 (params.w) Value of lj2 = 4.0*epsilon*pow(sigma,24.0)
 *      of the WCA potential in the force calculation
 *   - \a lambda - attractive tail value for the pair potential part  (0 = purely repulsive, 1 = full LJ 48-24)
 *   - \a wca_shift is the amount needed to shift the energy of the repulsive
 *      part to match the attractive energy, (1-lambda) * epsilon.
 *
 *  The pair potential part for this FENE bond is given by (see PairEvaluatorAshbaugh24)
 *
 *      \f{eqnarray*}{
 *      V(r)  = & V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) + (1-\lambda)\varepsilon & r < 2^{1/24}\sigma \\
 *            = & \lambda V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) & 2^{1/24}\sigma \ge r < r_{\mathrm{cut}} \\
 *            = & 0 & r \ge r_{\mathrm{cut}}
 *      \f}
 *
 *  where \f$V_{\mathrm{LJ,48-24}}(r,\varepsilon,\sigma)\f$ is a generalized Lennard-Jones potential
 *  with parameters \f$\varepsilon\f$ and \f$\sigma\f$:
 *
 *   \f{eqnarray*}{
 *     V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) = \left( \left(\frac{\sigma}{r}\right)^{48} - \left(\frac{\sigma}{r}\right)^{24} \right)
 *      \f}
 *
 * This potential is implemented as given in
 * <a href="http://dx.doi.org/10.1039/C5NR04661K">L. Rovigatti,B. Capone, C. Likos, Nanoscale, 8 (2016) </a>.
 * The Ashbaugh24 potential does not need diameter or charge, so the bond potential also doesn't need charge or
 * diameter.
 */
class BondEvaluatorFENEAsh24
    {
    public:
        //! Define the parameter type used by this bond potential evaluator
        typedef ashbaugh_bond_params param_type;

        /*!
         * Constructs the pair potential evaluator
         * \param _rsq Squared distance beteen the particles
         * \param _params Per type bond parameters of this potential as given above
         */
        DEVICE BondEvaluatorFENEAsh24(Scalar _rsq, const param_type& _params)
            : rsq(_rsq), K(_params.K), r_0(_params.r_0), lj1(_params.lj1), lj2(_params.lj2), lambda(_params.lambda),
              rwcasq(_params.rwcasq), wca_shift(_params.wca_shift)
            {
            }

        //! This evaluator doesn't use diameter information
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*!
         * \param da Diameter of particle a
         * \param db Diameter of particle b
         */
        DEVICE void setDiameter(Scalar da, Scalar db) {  }

        //! FENE-24 doesn't use charge
        DEVICE static bool needsCharge() { return false; }

        //! Accept the optional charge values
        /*!
         * \param qa Charge of particle a
         * \param qb Charge of particle b
         */
        DEVICE void setCharge(Scalar qa, Scalar qb) { }

        //! Evaluate the force and energy
        /*!
         * \param force_divr Output parameter to write the computed force divided by r.
         * \param bond_eng Output parameter to write the computed bond energy
         *
         *  \return True if they are evaluated or false if the bond energy is not defined.
         */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
            {
            // check for invalid parameters
            if (lj1 == Scalar(0.0) || r_0 == Scalar(0.0) || K == Scalar(0.0) ) return false;

            // Check if bond length restriction is violated
            if (rsq >= r_0*r_0) return false;

            Scalar r2inv = Scalar(1.0)/rsq;

            const Scalar r4inv = r2inv*r2inv;
            const Scalar r6inv = r2inv*r4inv;
            const Scalar r12inv = r6inv*r6inv;
            const Scalar r24inv = r12inv * r12inv;

            if (rsq < rwcasq)
                {
                force_divr = r2inv * r24inv * (Scalar(48.0)*lj1*r24inv - Scalar(24.0)*lj2);
                bond_eng = r24inv * (lj1*r24inv - lj2)+ wca_shift;
                }
            else
                {
                force_divr = lambda * r2inv * r24inv * (Scalar(48.0)*lj1*r24inv - Scalar(24.0)*lj2);
                bond_eng = lambda * r24inv * (lj1*r24inv - lj2);
                }

            force_divr += -K / (Scalar(1.0) - rsq/(r_0*r_0));
            bond_eng += -Scalar(0.5)*K*(r_0*r_0)*log(Scalar(1.0) - rsq/(r_0*r_0));

            return true;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*!
         * \returns The potential name. Must be short and all lowercase, as this is the name energies
         * will be logged as via analyze.log.
         */
        static std::string getName()
            {
            return std::string("fene24");
            }
        #endif

    protected:
        Scalar rsq;       //!< Stored rsq from the constructor
        Scalar K;         //!< K parameter from params passed to the constructor
        Scalar r_0;       //!< r_0 parameter from params passed to the constructor
        Scalar lj1;       //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;       //!< lj2 parameter extracted from the params passed to the constructor
        Scalar lambda;    //!< lambda parameter from params passed to the constructor
        Scalar rwcasq;    //!< WCA cutoff radius squared from params passed to the constructor
        Scalar wca_shift; //!< Energy shift for WCA part of the potential from params passed to the constructor
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE
#undef HOSTDEVICE
#endif // AZPLUGINS_BOND_EVALUATOR_FENE_ASH24_H_
