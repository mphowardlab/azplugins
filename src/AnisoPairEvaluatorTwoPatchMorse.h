// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_ANISO_PAIR_EVALUATOR_TWO_PATCH_MORSE_H_
#define AZPLUGINS_ANISO_PAIR_EVALUATOR_TWO_PATCH_MORSE_H_

#ifndef __HIPCC__
#include <string>
#endif // __HIPCC__

#include "AnisoPairEvaluator.h"

#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

//! Two-patch Morse parameters
struct AnisoPairParametersTwoPatchMorse : public AnisoPairParameters
    {
#ifndef __HIPCC__
    AnisoPairParametersTwoPatchMorse()
        : M_d(0), M_rinv(1.0), r_eq(0), omega(0), alpha(0), repulsion(false)
        {
        }

    AnisoPairParametersTwoPatchMorse(pybind11::dict v, bool managed = false)
        {
        M_d = v["M_d"].cast<Scalar>();
        M_rinv = Scalar(1.0) / v["M_r"].cast<Scalar>();
        r_eq = v["r_eq"].cast<Scalar>();
        omega = v["omega"].cast<Scalar>();
        alpha = v["alpha"].cast<Scalar>();
        repulsion = v["repulsion"].cast<bool>();
        }

    pybind11::dict toPython()
        {
        pybind11::dict v;
        v["M_d"] = M_d;
        v["M_r"] = Scalar(1.0) / M_rinv;
        v["r_eq"] = r_eq;
        v["omega"] = omega;
        v["alpha"] = alpha;
        v["repulsion"] = repulsion;
        return v;
        }
#endif // __HIPCC__

    Scalar M_d;     //<! Controls the well depth
    Scalar M_rinv;  //<! Controls the well steepness
    Scalar r_eq;    //<! Controls the well position
    Scalar omega;   //<! Controls the patch steepness
    Scalar alpha;   //<! Controls the patch width (lower is greater coverage)
    bool repulsion; //<! Whether to include Morse repulsion
    };

//! Class for evaluating the Two-patch Morse anisotropic pair potential
/*!
 * PairEvaluatorTwoPatchMorse evaluates the function:
 *      \f{eqnarray*}{
 *      V_{M2P} (\vec{r}_{ij}, \hat{n}_i, \hat{n}_j) = & V_M(|\vec{r}_{ij}|) \Omega(\hat{r}_{ij}
 * \cdot \hat{n}_i) \Omega(\hat{r}_{ij} \cdot \hat{n}_j) V_M(r) = &\left\{ \begin{matrix} -M_d,
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
 */
class AnisoPairEvaluatorTwoPatchMorse : public AnisoPairEvaluator
    {
    public:
    typedef AnisoPairParametersTwoPatchMorse param_type;

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
        : AnisoPairEvaluator(_dr, _quat_i, _quat_j, _rcutsq), M_d(_params.M_d),
          M_rinv(_params.M_rinv), r_eq(_params.r_eq), omega(_params.omega), alpha(_params.alpha),
          repulsion(_params.repulsion)
        {
        }

    //! Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return true;
        }

    DEVICE bool evaluate(Scalar3& force,
                         Scalar& pair_eng,
                         bool energy_shift,
                         Scalar3& torque_i,
                         Scalar3& torque_j)
        {
        Scalar rsq = dot(dr, dr);

        if (rsq > rcutsq)
            return false;

        Scalar rinv = fast::rsqrt(rsq);
        Scalar r = Scalar(1.) / rinv;

        vec3<Scalar> rvec(dr);
        vec3<Scalar> unitr = rvec * rinv;

        // Convert patch vector in the body frame of each particle to space frame
        vec3<Scalar> n_i = rotate(quat<Scalar>(quat_i), vec3<Scalar>(1.0, 0, 0));
        vec3<Scalar> n_j = rotate(quat<Scalar>(quat_j), vec3<Scalar>(1.0, 0, 0));

        vec3<Scalar> f;
        vec3<Scalar> t_i;
        vec3<Scalar> t_j;
        Scalar e = Scalar(0.0);

        // Morse potential
        Scalar UMorse = Scalar(-1.0) * M_d;
        Scalar dUMorse_dr = Scalar(0.0);
        // Purely attractive when r > r_eq
        if (r > r_eq || repulsion)
            {
            Scalar Morse_exp = fast::exp(-(r - r_eq) * M_rinv);
            Scalar one_minus_exp = Scalar(1.0) - Morse_exp;
            UMorse = M_d * (one_minus_exp * one_minus_exp - Scalar(1.0));
            dUMorse_dr = Scalar(2.0) * M_d * M_rinv * Morse_exp * one_minus_exp;
            }

        // Patch orientation for particle i
        Scalar gamma_i = dot(unitr, n_i);
        Scalar gamma_i_exp = fast::exp(-omega * (gamma_i * gamma_i - alpha));
        Scalar Omega_i = Scalar(1.0) / (Scalar(1.0) + gamma_i_exp);

        // Patch orientation for particle j
        Scalar gamma_j = dot(unitr, n_j);
        Scalar gamma_j_exp = fast::exp(-omega * (gamma_j * gamma_j - alpha));
        Scalar Omega_j = Scalar(1.0) / (Scalar(1.0) + gamma_j_exp);

        // Modulate the Morse potential according to orientations
        e += UMorse * Omega_i * Omega_j;

        // Compute scalar values for force and torque
        Scalar dU_dr = dUMorse_dr * Omega_i * Omega_j;
        Scalar dOmegai_dgi = Scalar(2.0) * omega * gamma_i * gamma_i_exp * Omega_i * Omega_i;
        Scalar dOmegaj_dgj = Scalar(2.0) * omega * gamma_j * gamma_j_exp * Omega_j * Omega_j;
        Scalar dU_dgi = dOmegai_dgi * UMorse * Omega_j;
        Scalar dU_dgj = dOmegaj_dgj * UMorse * Omega_i;

        // Compute vector directions for forces
        vec3<Scalar> n_i_perp = cross(-unitr, cross(unitr, n_i));
        vec3<Scalar> n_j_perp = cross(-unitr, cross(unitr, n_j));

        // Compute force and torque
        f = -dU_dr * unitr - rinv * (dU_dgi * n_i_perp + dU_dgj * n_j_perp);
        t_i = dU_dgi * cross(unitr, n_i);
        t_j = dU_dgj * cross(unitr, n_j);

        if (energy_shift)
            {
            // Preprocess rcut parameter
            Scalar rcut = fast::sqrt(rcutsq);

            // Preprocess Morse parameters
            Scalar Morse_exp_shift = fast::exp(-(rcut - r_eq) * M_rinv);
            Scalar one_minus_exp_shift = Scalar(1.0) - Morse_exp_shift;

            // Shift by UMorse
            Scalar UMorse_shift = M_d * (one_minus_exp_shift * one_minus_exp_shift - Scalar(1.0));

            e -= UMorse_shift * Omega_i * Omega_j;
            }

        // Return vector force and torque
        force = vec_to_scalar3(f);
        torque_i = vec_to_scalar3(t_i);
        torque_j = vec_to_scalar3(t_j);
        pair_eng = e;

        return true;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return std::string("TwoPatchMorse");
        }
#endif

    private:
    Scalar M_d;     //<! Controls the well depth
    Scalar M_rinv;  //<! Controls the well steepness
    Scalar r_eq;    //<! Controls the well position
    Scalar omega;   //<! Controls the patch steepness
    Scalar alpha;   //<! Controls the patch width (lower is greater coverage)
    bool repulsion; //<! Whether to include Morse repulsion
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_ANISO_PAIR_EVALUATOR_TWO_PATCH_MORSE_H_
