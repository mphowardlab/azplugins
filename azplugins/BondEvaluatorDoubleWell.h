// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

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
/*!
This bond potential follows the functional form
\f{eqnarray*}

V_{\rm{DW}}(r)  =   \frac{V_{max}-c/2}{b^4} \left[ \left( r - a/2 \right)^2 -b^2 \right]^2 + \frac{c}{2b}\left(r-a/2\right)+c/2

\f}
which has two minima at r = (a/2 +/- b), seperated by a maximum at a/2 of height V_max when c is set to zero.

The parameter a tunes the location of the maximal value and the parameter b tunes the distance of the
two maxima from each other.  This potential is useful to model bonds which can be either mechanically or
thermally "activated" into a effectively longer state. The value of V_max can be used to tune the height of the
energy barrier in between the two states.

If c is non zero, the relative energy of the minima can be tuned, where c is the energy of the second minima,
the first minima value is at zero. This  causes a small shift in the location of the minima and the maxima,
because of the added linear term.

The parameters are:
    - \a V_max (params.x) potential difference between the the first minima and maxima
    - \a a (params.y) shift for the location of the V_max, the maximum is at approx. a/2
    - \a b (params.z) scaling for the distance of the two minima at approx. (a/2 +/- b)
    - \a c (params.w) potential difference between the two minima, default is c=0

*/
class BondEvaluatorDoubleWell
    {
    public:
        //! Define the parameter type used by this bond potential evaluator

        typedef Scalar4 param_type;

        //! Constructs the pair potential evaluator
        /*!
         * \param _rsq Squared distance beteen the particles
         * \param _params Per type bond parameters of this potential as given above
         */
        DEVICE BondEvaluatorDoubleWell(Scalar _rsq, const param_type& _params)
            : rsq(_rsq), V_max(_params.x), a(_params.y), b(_params.z), c(_params.w)
            { }

        //! This evaluator doesn't use diameter information
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*!
         * \param da Diameter of particle a
         * \param db Diameter of particle b
         */
        DEVICE void setDiameter(Scalar da, Scalar db) {  }

        //! This evaluator doesn't use charge
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
            bond_eng = 0;
            force_divr = 0;

            // check for invalid parameters
            if (b == Scalar(0.0)) return false;

            Scalar c_half = Scalar(0.5)*c;
            Scalar r = fast::sqrt(rsq);
            Scalar r_min_half_a = r-Scalar(0.5)*a;
            Scalar b_sq = b*b;
            Scalar d = r_min_half_a*r_min_half_a - b_sq;

            bond_eng = ((V_max-c_half)/(b_sq*b_sq))*d*d + c_half/b*r_min_half_a + c_half;
            force_divr = - (4*(V_max-c_half)/(b_sq*b_sq)*d*r_min_half_a+c_half/b)/r;

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
        Scalar c;         //!< c parameter
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
