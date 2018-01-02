// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file DPDEvaluatorGeneralWeight.h
 * \brief Defines the dpd force evaluator using generalized weight functions.
 */

#ifndef AZPLUGINS_DPD_EVALUATOR_GENERAL_WEIGHT_H_
#define AZPLUGINS_DPD_EVALUATOR_GENERAL_WEIGHT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/Saru.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <string>
#endif

namespace azplugins
{
namespace detail
{

//! Evaluator for DPD generalized dissipative / random weight functions
/*!
 * Implements the generalized weight functions described in;
 *  X. Fan et al. Phys. Fluids 18, 063102 (2006).
 *  <a href="https://doi.org/10.1063/1.2206595">https://doi.org/10.1063/1.2206595</a>
 *
 * The conservative force is the standard DPD function:
 *  \f{eqnarray*}{
 *  \mathbf{F}_{\rm C} =& A (1- r_{ij}/r_{\rm cut}) & r \le r_{\rm cut} \\
 *                     =& 0 & r > r_{\rm cut}
 *  \f}
 *
 * and the dissipative and random forces are
 *  \f{eqnarray*}{
 *  \mathbf{F}_{\rm D} =& -\gamma \omega_{\rm D}(r_{ij}) (\mathbf{v}_{ij} \cdot \mathbf{\hat r}_{ij}) \mathbf{\hat r}_{ij} \\
 *  \mathbf{F}_{\rm R} =& \sigma \omega_{\rm R}(r_{ij}) \xi_{ij} \mathbf{\hat r}_{ij}
 *  \f}
 *
 * with
 *
 *  \f{eqnarray*}{
 *  w_{\rm D}(r) = \left[ \omega_{\rm R} \right]^2 = &\left( 1 - r/r_{\mathrm{cut}} \right)^s  & r \le r_{\mathrm{cut}} \\
 *                                                 = & 0 & r > r_{\mathrm{cut}} \\
 *  \f}
 *
 * where \a s is usually 2 for the "standard" DPD method. Refer to the original paper for more details.
 */
class DPDEvaluatorGeneralWeight
    {
    public:
        //! Three parameters are used by this DPD potential evaluator
        typedef Scalar3 param_type;

        //! Constructs the DPD potential evaluator
        /*!
         * \param _rsq Squared distance beteen the particles
         * \param _rcutsq Sqauared distance at which the potential goes to 0
         * \param _params Per type pair parameters of this potential
         */
        DEVICE DPDEvaluatorGeneralWeight(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), a(_params.x), gamma(_params.y), s(_params.z)
            {
            }

        //! Set seed, i and j (particle tags), and the timestep
        /*!
         * \param seed Seed to the random number generator
         * \param i Tag of first particle
         * \param j Tag of second particle
         * \param timestep Timestep to evaluate forces at
         */
        DEVICE void set_seed_ij_timestep(unsigned int seed, unsigned int i, unsigned int j, unsigned int timestep)
            {
            m_seed = seed;
            m_i = i;
            m_j = j;
            m_timestep = timestep;
            }

        //! Set the timestep size
        /*!
         * \param dt Current step size
         */
        DEVICE void setDeltaT(Scalar dt)
            {
            m_deltaT = dt;
            }

        //! Set the velocity term
        /*!
         * \param dot The dot product \f$r_{ij} \cdot v_{ij}\f$
         */
        DEVICE void setRDotV(Scalar dot)
            {
            m_dot = dot;
            }

        //! Set the temperature
        /*!
         * \param Temp Temperature in energy units
         */
        DEVICE void setT(Scalar Temp)
            {
            m_T = Temp;
            }

        //! DPD does not use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*!
         * \param di Diameter of particle i
         * \param dj Diameter of particle j
         */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! DPD doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*!
         * \param qi Charge of particle i
         * \param qj Charge of particle j
         */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy using the conservative force only
        /*!
         * \param force_divr Output parameter to write the computed force divided by r.
         * \param pair_eng Output parameter to write the computed pair energy
         * \param energy_shift Ignored.
         *
         * \return True if they are evaluated or false if they are not because we are beyond the cuttoff
         */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar rcutinv = fast::rsqrt(rcutsq);
                Scalar rcut = Scalar(1.0) / rcutinv;

                // force is easy to calculate
                force_divr = a*(rinv - rcutinv);
                pair_eng = a * (rcut - r) - Scalar(0.5) * a * rcutinv * (rcutsq - rsq);

                return true;
                }
            else
                return false;
            }

        //! Evaluate the force and energy using the thermostat
        /*!
         * \param force_divr Output parameter to write the computed force divided by r.
         * \param force_divr_cons Output parameter to write the computed conservative force divided by r.
         * \param pair_eng Output parameter to write the computed pair energy
         * \param energy_shift Ignored. DPD always goes to 0 at the cutoff.
         *
         * \note The conservative part \b only must be output to \a force_divr_cons so that the virial may be
                 computed correctly.
         *
         * \return True if they are evaluated or false if they are not because we are beyond the cuttoff
         */
        DEVICE bool evalForceEnergyThermo(Scalar& force_divr, Scalar& force_divr_cons, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar rcutinv = fast::rsqrt(rcutsq);
                Scalar rcut = Scalar(1.0) / rcutinv;

                // force calculation

                unsigned int m_oi, m_oj;
                // initialize the RNG
                if (m_i > m_j)
                   {
                   m_oi = m_j;
                   m_oj = m_i;
                   }
                else
                   {
                   m_oi = m_i;
                   m_oj = m_j;
                   }

                hoomd::detail::Saru rng(m_oi, m_oj, m_seed + m_timestep);

                // Generate a single random number
                Scalar alpha = rng.s<Scalar>(-1,1);

                // conservative dpd
                force_divr = a*(rinv - rcutinv);

                //  conservative force only
                force_divr_cons = force_divr;

                //  Drag Term
                const Scalar wR = fast::pow(Scalar(1.)-r*rcutinv, Scalar(0.5)*s) * rinv;
                force_divr -=  gamma*wR*wR*m_dot;

                //  Random Force
                force_divr += fast::rsqrt(m_deltaT/(m_T*gamma*Scalar(6.0)))*wR*alpha;

                //conservative energy only
                pair_eng = a * (rcut - r) - Scalar(0.5) * a * rcutinv * (rcutsq - rsq);

                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        static std::string getName()
            {
            return std::string("dpd_gen");
            }
        #endif

    protected:
        Scalar rsq;             //!< Squared distance between particles
        Scalar rcutsq;          //!< Squared cutoff between particles

        Scalar a;               //!< Strength of repulsion
        Scalar gamma;           //!< Drag term
        Scalar s;               //!< Exponent for the dissipative weight function

        unsigned int m_seed;    //!< User set seed for PRNG
        unsigned int m_i;       //!< Tag of first particle for PRNG
        unsigned int m_j;       //!< Tag of second particle for PRNG
        unsigned int m_timestep;//!< timestep for use in PRNG

        Scalar m_T;             //!< Temperature in energy units
        Scalar m_dot;           //!< Velocity difference dotted with displacement vector
        Scalar m_deltaT;        //!< Step size
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_DPD_EVALUATOR_GENERAL_WEIGHT_H_
