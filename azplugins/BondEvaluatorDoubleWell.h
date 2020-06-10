// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*! \file BondEvaluatorDoubleWell.h
    \brief Defines the bond evaluator class for a double well potential
*/

#ifndef AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
#define AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace azplugins
{
namespace detail
{

//! Class for evaluating the double well bond potential
/*! The parameters are:
    - \a V_max (params.x) maximum potential value between the two minima
    - \a a (params.y) shift for the location of the V_max, maximun is at 1/2 a
    - \a b (params.z) scaling for the distance of the two minima at 1/2(a +/- 2b)
*/
class BondEvaluatorDoubleWell
    {
    public:
        //! Define the parameter type used by this bond potential evaluator
        typedef Scalar3 param_type;

        //! Constructs the pair potential evaluator
        /*!
         * \param _rsq Squared distance beteen the particles
         * \param _params Per type bond parameters of this potential as given above
         */
        DEVICE BondEvaluatorDoubleWell(Scalar _rsq, const param_type& _params)
            : rsq(_rsq), V_max(_params.x), a(_params.y), b(_params.z)
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

        //! FENE doesn't use charge
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
            if (V_max == Scalar(0.0) || a == Scalar(0.0) || b == Scalar(0.0) ) return false;

            Scalar r2inv = Scalar(1.0)/rsq;

            //force_divr += -K / (Scalar(1.0) - rsq/(r_0*r_0));
            //bond_eng += -Scalar(0.5)*K*(r_0*r_0)*log(Scalar(1.0) - rsq/(r_0*r_0));

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
            return std::string("doublewell");
            }
        #endif

    protected:
        Scalar rsq;       //!< Stored rsq from the constructor
        Scalar V_max;     //!< V_max parameter
        Scalar a;         //!< a parameter
        Scalar b;         //!< b parameter
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
