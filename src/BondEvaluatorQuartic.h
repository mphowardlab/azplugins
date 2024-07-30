// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_BOND_EVALUATOR_QUARTIC_H_
#define AZPLUGINS_BOND_EVALUATOR_QUARTIC_H_

#ifndef __HIPCC__
#include <string>
#endif

#include "BondEvaluator.h"

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

//! Parameters of quartic bond potential
struct BondParametersQuartic : public BondParameters
    {
#ifndef __HIPCC__
    BondParametersQuartic()
        : k(0), r_0(0), b_1(0), b_2(0), U_0(0), sigma_6(0), epsilon_x_4(0), delta(0)
        {
        }

    BondParametersQuartic(pybind11::dict v)
        {
        k = v["k"].cast<Scalar>();
        r_0 = v["r_0"].cast<Scalar>();
        b_1 = v["b_1"].cast<Scalar>();
        b_2 = v["b_2"].cast<Scalar>();
        U_0 = v["U_0"].cast<Scalar>();
        delta = v["delta"].cast<Scalar>();

        auto sigma(v["sigma"].cast<Scalar>());
        auto epsilon(v["epsilon"].cast<Scalar>());

        const Scalar sigma_2 = sigma * sigma;
        const Scalar sigma_4 = sigma_2 * sigma_2;
        sigma_6 = sigma_2 * sigma_4;
        epsilon_x_4 = Scalar(4.0) * epsilon;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["r_0"] = r_0;
        v["b_1"] = b_1;
        v["b_2"] = b_2;
        v["U_0"] = U_0;
        v["sigma"] = pow(sigma_6, 1. / 6.);
        v["epsilon"] = epsilon_x_4 / Scalar(4.);
        v["delta"] = delta;
        return v;
        }
#endif

    Scalar k;           //!< k parameter for quartic potential strength
    Scalar r_0;         //!< energy barrier breaking distance
    Scalar b_1;         //!< quartic tuning parameter #1
    Scalar b_2;         //!< quartic tuning parameter #2
    Scalar U_0;         //!< quartic energy barrier to "breaking"
    Scalar sigma_6;     //!< sigma raised to the power of 6 for WCA
    Scalar epsilon_x_4; //!< epsilon * 4 for WCA
    Scalar delta;       //!< delta parameter bond
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the quartic bond potential
/*!
 * This bond potential follows the functional form
 *
 * \begin{eqnarray*}
 *      U(r) = & k4 ((r - r_{0}) - b_{1})((r - r_{0}) - b_{2})(r - r_{0})^2
 *                 + U_{0} + U_{\rm WCA}(r)      & r < r_0\\
 *            = & U_0                            & r \ge r_0
 * \end{eqnarray*}
 *
 * which has a WCA repulsive force at close range, and a quartic energy well. On the far side of
 * this energy well, there is an energy barrier, to a plateau energy, U_0. It reaches this energy
 * at r_0. Beyond r_0, the energy is constant, and there is no force from the bond potential.
 *
 * Since this potential has a finite range, beyond which no force is applied, this potential is
 * useful for modeling bonds with scissile functionality. r_0 defines the bond length at which
 * the bond can be considered broken, and U_0 defines the height of the energy barrier to breaking.
 *
 * The parameters b_1 and b_2 tune the location of the energy minimum, as well as the well width.
 * They do not cleanly translate to meaningful locations on the potential energy surface, but serve
 * as helpful tuning parameters to mimic other bond topologies. For example, making a previously
 * FENE or harmonic bond breakable. Using linear regression or some other fitting algorithm may do
 * this.
 *
 * The delta parameter shifts the whole potential with respect to r. The default value is 0.
 * Positive values shift to higher r values.
 */
class BondEvaluatorQuartic : public BondEvaluator
    {
    public:
    typedef BondParametersQuartic param_type;

    DEVICE BondEvaluatorQuartic(Scalar _rsq, const param_type& _params) : BondEvaluator(_rsq)
        {
        k = _params.k;
        r_0 = _params.r_0;
        b_1 = _params.b_1;
        b_2 = _params.b_2;
        U_0 = _params.U_0;
        delta = _params.delta;
        lj1 = _params.epsilon_x_4 * _params.sigma_6 * _params.sigma_6;
        lj2 = _params.epsilon_x_4 * _params.sigma_6;
        epsilon = _params.epsilon_x_4 / Scalar(4.0);
        }

    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
        {
        bond_eng = 0;
        force_divr = 0;
        // check for invalid parameters
        if (r_0 == Scalar(0.0))
            return false;

        Scalar r_red(1.0);

        if (delta == Scalar(0.0)) // case when delta = 0, no square root is needed for WCA
            {
            const Scalar r2inv = Scalar(1.0) / rsq;
            const Scalar r6inv = r2inv * r2inv * r2inv;
            const Scalar sigma6inv = lj2 / lj1;

            // WCA component without delta
            // WCA cutoff: r < 2^(1/6)*sigma
            // if epsilon or sigma is zero OR r is beyond cutoff, the force and energy are zero
            if (lj1 != Scalar(0) && r6inv > sigma6inv / Scalar(2.0))
                {
                Scalar epsilon = lj2 * lj2 / Scalar(4.0) / lj1;
                force_divr += r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);
                bond_eng += r6inv * (lj1 * r6inv - lj2) + epsilon;
                }

            // Quartic component prep
            // If the distance is less than the quartic cutoff distance, calculate as normal
            if (rsq < r_0 * r_0)
                {
                r_red = fast::sqrt(rsq) - r_0;
                }
            }
        else // case when delta != 0, a square root needs to be taken
            {
            Scalar r = fast::sqrt(rsq) - delta;
            const Scalar r2inv = Scalar(1.0) / r / r;
            const Scalar r6inv = r2inv * r2inv * r2inv;
            const Scalar sigma6inv = lj2 / lj1;

            // WCA component with delta
            // wca cutoff: r < 2^(1/6)*sigma
            // if epsilon or sigma is zero OR r is beyond cutoff, the WCA force and energy are zero
            if (lj1 != Scalar(0) && r6inv > sigma6inv / Scalar(2.0))
                {
                force_divr
                    += r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2) / r / (r + delta);
                bond_eng += r6inv * (lj1 * r6inv - lj2) + epsilon;
                }

            // Quartic component prep
            if (r < r_0)
                {
                r_red = r - r_0;
                }
            }

        // Quartic bond potential is on when r_red < 0
        if (r_red < Scalar(0))
            {
            force_divr += Scalar(-1) * k * r_red
                          * (4 * r_red * r_red - 3 * (b_1 + b_2) * r_red + 2 * b_1 * b_2)
                          / (r_red + r_0 + delta);
            bond_eng += k * (r_red - b_1) * (r_red - b_2) * r_red * r_red + U_0;
            }
        else
            {
            force_divr += 0;
            bond_eng += U_0;
            }

        return true;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return std::string("Quartic");
        }
#endif

    private:
    Scalar k;       //!< k parameter for quartic potential strength
    Scalar r_0;     //!< energy barrier breaking distance
    Scalar b_1;     //!< primary quartic potential tuning parameter
    Scalar b_2;     //!< secondary quartic potential tuning parameter
    Scalar U_0;     //!< quartic energy barrier to "breaking"
    Scalar delta;   //!< delta parameter bond
    Scalar lj1;     //!< lj1 parameter used in WCA calculation
    Scalar lj2;     //!< lj2 parameter used in WCA calculation
    Scalar epsilon; //!< epsilon parameter used in WCA calculation
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_QUARTIC_H_
