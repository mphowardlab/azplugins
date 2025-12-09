// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file WallEvaluatorColloid.h
 * \brief Defines the wall potential evaluator class for the colloid (integrated Lennard-Jones)
 * potential
 */

#ifndef AZPLUGINS_WALL_EVALUATOR_COLLOID_H_
#define AZPLUGINS_WALL_EVALUATOR_COLLOID_H_

#include "PairEvaluator.h"

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
//! Define the parameter type used by this wall potential evaluator
struct WallParametersColloid : public PairParameters
    {
#ifndef __HIPCC__
    WallParametersColloid() : A(0), a(0), sigma_3(0) { }

    WallParametersColloid(pybind11::dict v, bool managed = false)
        {
        A = v["A"].cast<Scalar>();
        a = v["a"].cast<Scalar>();
        auto sigma(v["sigma"].cast<Scalar>());
        sigma_3 = sigma * sigma * sigma;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["A"] = A;
        v["a"] = a;
        v["sigma"] = std::cbrt(sigma_3);
        return v;
        }
#endif // __HIPCC__

    Scalar A;       //!< Hamaker constant
    Scalar a;       //!< particle radius
    Scalar sigma_3; //!< Lennard-Jones sigma
    }

#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Class for evaluating the Lennard-Jones colloid wall force
/*!
 * The Lennard-Jones colloid wall potential is derived from integrating the standard Lennard-Jones
 * potential between a spherical particle of radius \f$ a \f$ and a half plane, and it takes the
 * form:
 *
 * \f[ V(r) = C_1 \left( \frac{7a-z}{(z-a)^7} + \frac{7a+z}{(z+a)^7} \right)
 *          - C_2 \left( \frac{2 a z}{z^2-a^2} + \ln\left(\frac{z-a}{z+a}\right) \right) \f]
 *
 * with force
 * \f[ F(r)/r = 6 C_1 \left( \frac{8(a/r) - 1}{(z-a)^8} + \frac{8(a/r) + 1}{(z+a)^8} \right)
 *            - C_2 \left( \frac{4 a^2(a/z)}{(z^2-a^2)^2} \right ) \f]
 *
 * where \f$ C_1 = A \sigma^6 / 7560 \f$ and \f$ C_2 = A/6 \f$ are constants, \f$ \sigma \f$ is the
 * Lennard-Jones diameter for particles in the sphere and the wall, and \f$A\f$ is the effective
 * Hamaker constant, which properly takes the form
 *
 * \f[ A = 4 \pi^2 \rho_{\rm w} \rho_{\rm c} \epsilon \sigma^6 \f]
 *
 * where \f$ \rho_{\rm w} \f$ and \f$ \rho_{\rm c} \f$ are the wall and colloid densities and \f$
 * \epsilon \f$ is the Lennard-Jones interaction energy.
 */
class WallEvaluatorColloid : public PairEvaluator
    {
    public:
    //! Define the parameter type used by this wall potential evaluator
    typedef WallParametersColloid param_type;

    //! Constructor
    /*!
     * \param _rsq Squared distance between particles
     * \param _rcutsq Cutoff radius squared
     * \param _params Pair potential parameters, given by typedef above
     *
     * The functor initializes its members from \a _params.
     */
    DEVICE WallEvaluatorColloid(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : PairEvaluator(_rsq, _rcutsq)
        {
        a = _params.a;
        lj1 = _params.A * _params.sigma_3 * _params.sigma_3 / Scalar(7560);
        lj2 = _params.A / Scalar(6);
        }

    //! Computes the colloid-wall interaction potential
    /*!
     * \param force_divr Force divided by r
     * \param _rsq Distance between particle centers, squared
     * \tparam force If true, compute the force
     * \returns energy
     *
     * The force is only computed if \a force is true. This is useful for
     * performing the energy shift, where the potential needs to be reevaluated at rcut.
     */
    template<bool force> DEVICE inline Scalar computePotential(Scalar& force_divr, Scalar _rsq)
        {
        Scalar r = sqrt(_rsq);
        Scalar arinv = a / r;

        // 1/(r-a) and 1/(r+a)
        Scalar r_minus_a_inv = Scalar(1.0) / (r - a);
        Scalar r_plus_a_inv = Scalar(1.0) / (r + a);

        // 1/(r^2-a^2)
        Scalar r2_minus_a2_inv = r_minus_a_inv * r_plus_a_inv;

        // 1/(r-a)^2 and 1/(r-a)^6
        Scalar r_minus_a_inv2 = r_minus_a_inv * r_minus_a_inv;
        Scalar r_minus_a_inv6 = r_minus_a_inv2 * r_minus_a_inv2 * r_minus_a_inv2;

        // 1/(r+a)^2 and 1/(r+a)^6
        Scalar r_plus_a_inv2 = r_plus_a_inv * r_plus_a_inv;
        Scalar r_plus_a_inv6 = r_plus_a_inv2 * r_plus_a_inv2 * r_plus_a_inv2;

        // force
        if (force)
            {
            Scalar arinv8 = Scalar(8.0) * arinv;
            force_divr = Scalar(6.0) * lj1
                         * ((arinv8 - Scalar(1.0)) * r_minus_a_inv2 * r_minus_a_inv6
                            + (arinv8 + Scalar(1.0)) * r_plus_a_inv2 * r_plus_a_inv6);
            force_divr -= lj2 * (Scalar(4.0) * a * a * arinv * r2_minus_a2_inv * r2_minus_a2_inv);
            }

        // energy
        Scalar a7 = Scalar(7.0) * a;
        Scalar energy = lj1
                        * ((a7 - r) * r_minus_a_inv * r_minus_a_inv6
                           + (a7 + r) * r_plus_a_inv * r_plus_a_inv6);
        energy -= lj2 * (Scalar(2.0) * a * r * r2_minus_a2_inv + log(r_plus_a_inv / r_minus_a_inv));
        return energy;
        }

    //! Evaluate the force and energy
    /*!
     * \param force_divr Holds the computed force divided by r
     * \param energy Holds the computed pair energy
     * \param energy_shift If true, the potential is shifted to zero at the cutoff
     *
     * \returns True if the energy calculation occurs
     *
     * The calculation does not occur if the pair distance is greater than the cutoff
     * or if the potential is scaled to zero.
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& energy, bool energy_shift)
        {
        if (rsq < rcutsq && lj1 != 0 && a > 0)
            {
            energy = computePotential<true>(force_divr, rsq);
            if (energy_shift)
                {
                energy -= computePotential<false>(force_divr, rcutsq);
                }
            return true;
            }
        else
            return false;
        }

#ifndef __HIPCC__
    //! Return the name of this potential
    static std::string getName()
        {
        return std::string("colloid");
        }
#endif

    private:
    Scalar a;   //!< The particle radius
    Scalar lj1; //!< Prefactor for first term (includes Hamaker constant)
    Scalar lj2; //!< Prefactor for second term (includes Hamaker constant)
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE
#endif // AZPLUGINS_WALL_EVALUATOR_COLLOID_H_
