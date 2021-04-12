// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file AnisoPairEvaluatorTwoPatchMorse.h
 * \brief Defines the aniostropic pair force evaluator class for Two-patch Morse potential
 */

#ifndef AZPLUGINS_ANISO_PAIR_EVALUATOR_TWO_PATCH_MORSE_H_
#define AZPLUGINS_ANISO_PAIR_EVALUATOR_TWO_PATCH_MORSE_H_

#include "AnisoPairEvaluator.h"

#include "hoomd/VectorMath.h"

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
//! Two-patch Morse parameters
/*!
 * \sa AnisoPairEvaluatorTwoPatchMorse
 */
struct two_patch_morse_params : public AnisoPairParams
    {
    Scalar Mdeps;     //<! Controls the well depth
    Scalar Mrinv;     //<! Controls the well steepness
    Scalar req;       //<! Controls the well position
    Scalar omega;     //<! Controls the patch steepness
    Scalar alpha;     //<! Controls the patch width (lower is greater coverage)
    bool   repulsion; //<! Whether to include Morse repulsion
    };

//! Convenience function for making two_patch_morse_params in python
HOSTDEVICE inline two_patch_morse_params make_two_patch_morse_params(Scalar Mdeps,
                                                                     Scalar Mrinv,
                                                                     Scalar req,
                                                                     Scalar omega,
                                                                     Scalar alpha,
                                                                     bool   repulsion)
    {
    two_patch_morse_params retval;
    retval.Mdeps       = Mdeps;
    retval.Mrinv       = Mrinv;
    retval.req         = req;
    retval.omega       = omega;
    retval.alpha       = alpha;
    retval.repulsion   = repulsion;
    return retval;
    }

//! Class for evaluating the Two-patch Morse anisotropic pair potential
/*!
 * PairEvaluatorTwoPatchMorse evaluates the function:
 *      \f{eqnarray*}{
 *      V_{M2P} (\vec{r}_{ij}, \hat{n}_i, \hat{n}_j) = & V_M(|\vec{r}_{ij}|) \Omega(\hat{r}_{ij} \cdot \hat{n}_i) \Omega(\hat{r}_{ij} \cdot \hat{n}_j)
 *      V_M(r) = &\left\{ \begin{matrix}
 *      -M_d,
 *      &
 *      r < r_{\rm eq} \text{ and !repulsion}
 *      \\
 *      M_d \left( \left[ 1 - \exp\left( -\frac{r-r_{\rm eq}}{M_r}\right) \right]^2 - 1 \right),
 *      &
 *      \text{otherwise}
 *      \end{matrix}
 *      \right.
 *      \Omega(\gamma) = & \frac{1}{1+\exp[-\omega (\gamma^2 - \alpha)]}
 *      \f}
 *
 * Here, \f$vec{r}_{ij}\f$ is the displacement vector between particles \f$i\f$ and \f$j\f$,
 * \f$|\vec{r}_{ij}|\f$ is the magnitude of that displacement, and \f$\hat{n}\f$ is the normalized
 * orientation vector of the particle. The parameters \f$M_d\f$, \f$M_r\f$, and \f$r_{\rm eq}\f$
 * control the depth, width, and position of the potential well. The parameters \f$\alpha\f$ and
 * \f$\omega\f$ control the width and steepness of the orientation dependence.
 *
 * The Two-patch Morse potential does not need diameter or charge. Five parameters are specified and stored in a
 * two_patch_morse_params:
 * - \a Mdeps = epsilon * Md
 * - \a Mrinv = 1 / Mr
 * - \a req
 * - \a omega
 * - \a alpha
 * - \a repulsion
 */
class AnisoPairEvaluatorTwoPatchMorse : public AnisoPairEvaluator
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef two_patch_morse_params param_type;
        typedef AnisoPairEvaluator::shape_param_type shape_param_type;

        //! Constructor
        /*!
         * \param _dr Displacement vector between particle centres of mass
         * \param _q_i Quaterion of i^th particle
         * \param _q_j Quaterion of j^th particle
         * \param _rsq Squared distance between particles
         * \param _rcutsq Cutoff radius squared
         * \param _params Pair potential parameters, given by typedef above
         *
         * The functor initializes its members from \a _params.
         */
        DEVICE AnisoPairEvaluatorTwoPatchMorse(Scalar3& _dr,
                                               Scalar4& _quat_i,
                                               Scalar4& _quat_j,
                                               Scalar _rcutsq,
                                               const param_type& _params)
            : AnisoPairEvaluator(_dr, _quat_i, _quat_j, _rcutsq),
              Mdeps(_params.Mdeps), Mrinv(_params.Mrinv), req(_params.req),
              omega(_params.omega), alpha(_params.alpha), repulsion(_params.repulsion)
            {
            }

        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
         *  \param pair_eng Output parameter to write the computed pair energy.
         *  \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
         *  \param torque_i The torque exterted on the i^th particle.
         *  \param torque_j The torque exterted on the j^th particle.
         *
         *  \returns True if they are evaluated or false if they are not because we are beyond the cutoff.
         *
        */
        DEVICE  bool
        evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            Scalar rsq = dot(dr,dr);

            if(rsq > rcutsq)
                return false;

            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.)/rinv;

            vec3<Scalar> rvec(dr);
            vec3<Scalar> unitr = rvec * rinv;

            //! Convert patch vector in the body frame of each particle to space frame
            vec3<Scalar> n_i = rotate(quat<Scalar>(quat_i), vec3<Scalar>(1.0, 0, 0));
            vec3<Scalar> n_j = rotate(quat<Scalar>(quat_j), vec3<Scalar>(1.0, 0, 0));

            vec3<Scalar> f;
            vec3<Scalar> t_i;
            vec3<Scalar> t_j;
            Scalar e = Scalar(0.0);

            //! Morse potential
            Scalar UMorse = Scalar(-1.0) * Mdeps;
            Scalar dUMorse_dr = Scalar(0.0);
            //! Purely attractive when r > req
            if(r > req || repulsion)
                {
                Scalar Morse_exp = fast::exp(-(r - req) * Mrinv);
                Scalar one_minus_exp = Scalar(1.0) - Morse_exp;
                UMorse = Mdeps * ( one_minus_exp * one_minus_exp - Scalar(1.0) );
                dUMorse_dr = Scalar(2.0) * Mdeps * Mrinv * Morse_exp * one_minus_exp;
                }

            //! Patch orientation for particle i
            Scalar gamma_i = dot(unitr,n_i);
            Scalar gamma_i_exp = fast::exp(-omega * (gamma_i * gamma_i - alpha));
            Scalar Omega_i = Scalar(1.0) / ( Scalar(1.0) + gamma_i_exp );

            //! Patch orientation for particle j
            Scalar gamma_j = dot(unitr,n_j);
            Scalar gamma_j_exp = fast::exp(-omega * (gamma_j * gamma_j - alpha));
            Scalar Omega_j = Scalar(1.0) / ( Scalar(1.0) + gamma_j_exp );

            //! Modulate the Morse potential according to orientations
            e += UMorse * Omega_i * Omega_j;

            //! Compute scalar values for force and torque
            Scalar dU_dr  = dUMorse_dr * Omega_i * Omega_j;
            Scalar dOmegai_dgi = Scalar(2.0) * omega * gamma_i * gamma_i_exp * Omega_i * Omega_i;
            Scalar dOmegaj_dgj = Scalar(2.0) * omega * gamma_j * gamma_j_exp * Omega_j * Omega_j;
            Scalar dU_dgi = dOmegai_dgi * UMorse * Omega_j;
            Scalar dU_dgj = dOmegaj_dgj * UMorse * Omega_i;

            //! Compute vector directions for forces
            vec3<Scalar> n_i_perp = cross(-unitr,cross(unitr,n_i));
            vec3<Scalar> n_j_perp = cross(-unitr,cross(unitr,n_j));

            //! Compute force and torque
            f   = -dU_dr * unitr - rinv * ( dU_dgi * n_i_perp + dU_dgj * n_j_perp );
            t_i = dU_dgi * cross(unitr, n_i);
            t_j = dU_dgj * cross(unitr, n_j);

            if (energy_shift)
              {
              //! Preprocess rcut parameter
              Scalar rcut = fast::sqrt(rcutsq);

              //! Preprocess Morse parameters
              Scalar Morse_exp_shift = fast::exp(-(rcut - req) * Mrinv);
              Scalar one_minus_exp_shift = Scalar(1.0) - Morse_exp_shift;

              //! Shift by UMorse
              Scalar UMorse_shift = Mdeps * ( one_minus_exp_shift * one_minus_exp_shift - Scalar(1.0) );

              e -= UMorse_shift * Omega_i * Omega_j;
              }

            //! Return vector force and torque
            force = vec_to_scalar3(f);
            torque_i = vec_to_scalar3(t_i);
            torque_j = vec_to_scalar3(t_j);
            pair_eng = e;

            return true;
            }

        #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("two_patch_morse");
            }
        #endif

    protected:
        Scalar Mdeps;     //<! Controls the well depth
        Scalar Mrinv;     //<! Controls the well steepness
        Scalar req;       //<! Controls the well position
        Scalar omega;     //<! Controls the patch steepness
        Scalar alpha;     //<! Controls the patch width (lower is greater coverage)
        bool   repulsion; //<! Whether to include Morse repulsion
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_ANISO_PAIR_EVALUATOR_TWO_PATCH_MORSE_H_
