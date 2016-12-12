// Maintainer: mphoward

/*!
 * \file PairEvaluatorColloid.h
 * \brief Defines the pair force evaluator class for colloid (integrated Lennard-Jones) potential
 */

#ifndef AZPLUGINS_PAIR_EVALUATOR_COLLOID_H_
#define AZPLUGINS_PAIR_EVALUATOR_COLLOID_H_

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

//! Class for evaluating the colloid pair potential
class PairEvaluatorColloid
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar4 param_type;

        //! Different forms for the (i,j) interaction
        enum interaction_type {SOLVENT_SOLVENT=0, COLLOID_SOLVENT, COLLOID_COLLOID};

        DEVICE PairEvaluatorColloid(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq)
            {
            A = _params.x;
            sigma_3 = _params.y;
            sigma_6 = _params.z;
            form = static_cast<interaction_type>(__scalar_as_int(_params.w));
            }

        DEVICE static bool needsDiameter() { return true; }
        DEVICE void setDiameter(Scalar di, Scalar dj)
            {
            ai = Scalar(0.5) * di;
            aj = Scalar(0.5) * dj;
            }

        DEVICE static bool needsCharge() { return false; }
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        template<bool force>
        DEVICE inline Scalar computeSolventSolvent(Scalar& force_divr, Scalar _rsq)
            {
            const Scalar r2inv = Scalar(1.0)/_rsq;
            const Scalar r6inv = r2inv * r2inv * r2inv;
            const Scalar c1 = A * sigma_6 / Scalar(36.0);
            if (force)
                {
                force_divr = Scalar(6.0) * c1 * r2inv * r6inv * (Scalar(2.0)*sigma_6*r6inv - Scalar(1.0));
                }

            return c1 * r6inv * (sigma_6 * r6inv - Scalar(1.0));
            }

        template<bool force>
        DEVICE inline Scalar computeColloidSolvent(Scalar& force_divr, Scalar _rsq)
            {
            // we require the NP to have the larger radius
            const Scalar a = (ai > aj) ? ai : aj;

            const Scalar asq = a*a; // asq
            Scalar asq_minus_rsq = asq - _rsq;
            Scalar rsqsq = _rsq*_rsq;
            Scalar asq_minus_rsq_3 = asq_minus_rsq*asq_minus_rsq*asq_minus_rsq;
            Scalar asq_minus_rsq_6 = asq_minus_rsq_3*asq_minus_rsq_3;

            Scalar fR = sigma_3 * A * a * asq / asq_minus_rsq_3;
            if (force)
                {
                force_divr = Scalar(4.0/15.0)*fR*(2.0*(asq+_rsq)*(asq*(Scalar(5.0)*asq+Scalar(22.0)*_rsq)+5.0*rsqsq)*
                                     sigma_6/asq_minus_rsq_6 - Scalar(5.0))/asq_minus_rsq;
                }

            return Scalar(2.0/9.0)*fR*(Scalar(1.0)-(asq*(asq*(asq/Scalar(3.0) + Scalar(3.0)*_rsq)+Scalar(4.2)*rsqsq)+_rsq*rsqsq)*sigma_6/asq_minus_rsq_6);
            }

        template<bool force>
        DEVICE inline Scalar computeColloidColloid(Scalar& force_divr, Scalar _rsq)
            {
            const Scalar r = sqrt(_rsq);
            const Scalar k0 = ai*aj;
            const Scalar k1 = ai+aj;
            const Scalar k2 = ai-aj;
            const Scalar k3 = k1 + r;
            const Scalar k4 = k1 - r;
            const Scalar k5 = k2 + r;
            const Scalar k6 = k2 - r;
            const Scalar k7 = Scalar(1.0)/(k3*k4);
            const Scalar k8 = Scalar(1.0)/(k5*k6);

            const Scalar k3inv = Scalar(1.0)/k3;
            Scalar g0 = k3inv*k3inv;
            g0 *= g0*g0;
            g0 *= k3inv;

            const Scalar k4inv = Scalar(1.0)/k4;
            Scalar g1 = k4inv*k4inv;
            g1 *= g1*g1;
            g1 *= k4inv;

            const Scalar k5inv = Scalar(1.0)/k5;
            Scalar g2 = k5inv*k5inv;
            g2 *= g2*g2;
            g2 *= k5inv;

            const Scalar k6inv = Scalar(1.0)/k6;
            Scalar g3 = k6inv*k6inv;
            g3 *= g3*g3;
            g3 *= k6inv;

            const Scalar h0 = ((k3+Scalar(5.0)*k1)*k3+Scalar(30.0)*k0)*g0;
            const Scalar h1 = ((k4+Scalar(5.0)*k1)*k4+Scalar(30.0)*k0)*g1;
            const Scalar h2 = ((k5+Scalar(5.0)*k2)*k5-Scalar(30.0)*k0)*g2;
            const Scalar h3 = ((k6+Scalar(5.0)*k2)*k6-Scalar(30.0)*k0)*g3;

            g0 *= Scalar(42.0)*k0/k3 + Scalar(6.0)*k1 + k3;
            g1 *= Scalar(42.0)*k0/k4 + Scalar(6.0)*k1 + k4;
            g2 *= Scalar(-42.0)*k0/k5 + Scalar(6.0)*k2 + k5;
            g3 *= Scalar(-42.0)*k0/k6 + Scalar(6.0)*k2 + k6;

            const Scalar fR = A*sigma_6/r/Scalar(37800.0);
            Scalar pair_eng = fR * (h0 - h1 - h2 + h3);
            if (force)
                {
                const Scalar dUR = pair_eng/r + Scalar(5.0)*fR*(g0+g1-g2-g3);
                const Scalar dUA = -A/Scalar(3.0)*r*((Scalar(2.0)*k0*k7+Scalar(1.0))*k7 + (Scalar(2.0)*k0*k8-Scalar(1.0))*Scalar(k8));

                force_divr = (dUR+dUA)/r;
                }
            pair_eng += A/Scalar(6.0)*(Scalar(2.0)*k0*(k7+k8)-log(k8/k7));
            return pair_eng;
            }

        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && A != 0)
                {
                if (form == SOLVENT_SOLVENT)
                    {
                    pair_eng = computeSolventSolvent<true>(force_divr, rsq);
                    if (energy_shift)
                        {
                        pair_eng -= computeSolventSolvent<false>(force_divr, rcutsq);
                        }
                    }
                else if (form == COLLOID_SOLVENT)
                    {
                    pair_eng = computeColloidSolvent<true>(force_divr, rsq);
                    if (energy_shift)
                        {
                        pair_eng -= computeColloidSolvent<false>(force_divr, rcutsq);
                        }
                    }
                else if (form == COLLOID_COLLOID)
                    {
                    pair_eng = computeColloidColloid<true>(force_divr, rsq);
                    if (energy_shift)
                        {
                        pair_eng -= computeColloidColloid<false>(force_divr, rcutsq);
                        }
                    }
                else
                    {
                    return false;
                    }

                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("np");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor

        Scalar A;       //!< Hamaker constant
        Scalar sigma_3; //!< Sigma^3
        Scalar sigma_6; //!< Sigma^6
        interaction_type form;    //!< Form of the interaction

        Scalar ai;      //!< radius 1
        Scalar aj;      //!< radius 2
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_COLLOID_H_
