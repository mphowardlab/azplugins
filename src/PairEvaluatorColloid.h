// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PAIR_EVALUATOR_COLLOID_H_
#define AZPLUGINS_PAIR_EVALUATOR_COLLOID_H_

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

struct PairParametersColloid : public PairParameters
    {
#ifndef __HIPCC__
    PairParametersColloid() : A(0), a_1(0), a_2(0), sigma_3(0) { }

    PairParametersColloid(pybind11::dict v, bool managed = false)
        {
        A = v["A"].cast<Scalar>();
        a_1 = v["a_1"].cast<Scalar>();
        a_2 = v["a_2"].cast<Scalar>();
        const Scalar sigma = v["sigma"].cast<Scalar>();
        sigma_3 = sigma * sigma * sigma;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["A"] = A;
        v["a_1"] = a_1;
        v["a_2"] = a_2;
        v["sigma"] = std::cbrt(sigma_3);
        return v;
        }
#endif // __HIPCC__

    Scalar A;       //!< Hamaker constant
    Scalar a_1;     //!< particle 1 radius
    Scalar a_2;     //!< particle 2 radius
    Scalar sigma_3; //!< Lennard-Jones sigma
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the colloid pair potential
/*!
 * An effective Lennard-Jones potential obtained by integrating the Lennard-Jones potential
 * between a point and a sphere or a sphere and a sphere. The attractive part of the
 * colloid-colloid pair potential was derived originally by Hamaker, and the full potential
 * by <a href="http://doi.org/10.1103/PhysRevE.67.041710">Everaers and Ejtehadi</a>.
 * A discussion of the application of these potentials to colloidal suspensions can be found
 * in <a href="http://dx.doi.org/10.1063/1.3578181">Grest et al.</a>
 */
class PairEvaluatorColloid : public PairEvaluator
    {
    public:
    typedef PairParametersColloid param_type;

    //! Constructor
    /*!
     * \param _rsq Squared distance between particles
     * \param _rcutsq Cutoff radius squared
     * \param _params Pair potential parameters, given by typedef above
     *
     * The functor initializes its members from \a _params.
     */
    DEVICE PairEvaluatorColloid(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : PairEvaluator(_rsq, _rcutsq)
        {
        A = _params.A;
        sigma_3 = _params.sigma_3;
        sigma_6 = sigma_3 * sigma_3;
        ai = _params.a_1;
        aj = _params.a_2;
        }

    //! Computes the solvent-solvent interaction
    /*!
     * \param force_divr Force divided by r
     * \param _rsq Distance between particle centers, squared
     * \tparam force If true, compute the force
     * \returns energy
     *
     * The force is only computed if \a force is true. This is useful for
     * performing the energy shift, where the potential needs to be reevaluated at rcut.
     */
    template<bool force> DEVICE inline Scalar computeSolventSolvent(Scalar& force_divr, Scalar _rsq)
        {
        const Scalar r2inv = Scalar(1.0) / _rsq;
        const Scalar r6inv = r2inv * r2inv * r2inv;
        const Scalar c1 = A * sigma_6 / Scalar(36.0);
        if (force)
            {
            force_divr
                = Scalar(6.0) * c1 * r2inv * r6inv * (Scalar(2.0) * sigma_6 * r6inv - Scalar(1.0));
            }

        return c1 * r6inv * (sigma_6 * r6inv - Scalar(1.0));
        }

    //! Computes the colloid-solvent interaction
    /*!
     * \param force_divr Force divided by r
     * \param _rsq Distance between particle centers, squared
     * \tparam force If true, compute the force
     * \returns energy
     *
     * The force is only computed if \a force is true. This is useful for
     * performing the energy shift, where the potential needs to be reevaluated at rcut.
     */
    template<bool force> DEVICE inline Scalar computeColloidSolvent(Scalar& force_divr, Scalar _rsq)
        {
        // we require the NP to have the larger radius
        const Scalar a = (ai > aj) ? ai : aj;

        const Scalar asq = a * a; // asq
        Scalar asq_minus_rsq = asq - _rsq;
        Scalar rsqsq = _rsq * _rsq;
        Scalar asq_minus_rsq_3 = asq_minus_rsq * asq_minus_rsq * asq_minus_rsq;
        Scalar asq_minus_rsq_6 = asq_minus_rsq_3 * asq_minus_rsq_3;

        Scalar fR = sigma_3 * A * a * asq / asq_minus_rsq_3;
        if (force)
            {
            force_divr = Scalar(4.0 / 15.0) * fR
                         * (2.0 * (asq + _rsq)
                                * (asq * (Scalar(5.0) * asq + Scalar(22.0) * _rsq) + 5.0 * rsqsq)
                                * sigma_6 / asq_minus_rsq_6
                            - Scalar(5.0))
                         / asq_minus_rsq;
            }

        return Scalar(2.0 / 9.0) * fR
               * (Scalar(1.0)
                  - (asq * (asq * (asq / Scalar(3.0) + Scalar(3.0) * _rsq) + Scalar(4.2) * rsqsq)
                     + _rsq * rsqsq)
                        * sigma_6 / asq_minus_rsq_6);
        }

    //! Computes the colloid-colloid interaction
    /*!
     * \param force_divr Force divided by r
     * \param _rsq Distance between particle centers, squared
     * \tparam force If true, compute the force
     * \returns energy
     *
     * The force is only computed if \a force is true. This is useful for
     * performing the energy shift, where the potential needs to be reevaluated at rcut.
     */
    template<bool force> DEVICE inline Scalar computeColloidColloid(Scalar& force_divr, Scalar _rsq)
        {
        const Scalar r = sqrt(_rsq);
        const Scalar k0 = ai * aj;
        const Scalar k1 = ai + aj;
        const Scalar k2 = ai - aj;
        const Scalar k3 = k1 + r;
        const Scalar k4 = k1 - r;
        const Scalar k5 = k2 + r;
        const Scalar k6 = k2 - r;
        const Scalar k7 = Scalar(1.0) / (k3 * k4);
        const Scalar k8 = Scalar(1.0) / (k5 * k6);

        const Scalar k3inv = Scalar(1.0) / k3;
        Scalar g0 = k3inv * k3inv;
        g0 *= g0 * g0;
        g0 *= k3inv;

        const Scalar k4inv = Scalar(1.0) / k4;
        Scalar g1 = k4inv * k4inv;
        g1 *= g1 * g1;
        g1 *= k4inv;

        const Scalar k5inv = Scalar(1.0) / k5;
        Scalar g2 = k5inv * k5inv;
        g2 *= g2 * g2;
        g2 *= k5inv;

        const Scalar k6inv = Scalar(1.0) / k6;
        Scalar g3 = k6inv * k6inv;
        g3 *= g3 * g3;
        g3 *= k6inv;

        const Scalar h0 = ((k3 + Scalar(5.0) * k1) * k3 + Scalar(30.0) * k0) * g0;
        const Scalar h1 = ((k4 + Scalar(5.0) * k1) * k4 + Scalar(30.0) * k0) * g1;
        const Scalar h2 = ((k5 + Scalar(5.0) * k2) * k5 - Scalar(30.0) * k0) * g2;
        const Scalar h3 = ((k6 + Scalar(5.0) * k2) * k6 - Scalar(30.0) * k0) * g3;

        g0 *= Scalar(42.0) * k0 / k3 + Scalar(6.0) * k1 + k3;
        g1 *= Scalar(42.0) * k0 / k4 + Scalar(6.0) * k1 + k4;
        g2 *= Scalar(-42.0) * k0 / k5 + Scalar(6.0) * k2 + k5;
        g3 *= Scalar(-42.0) * k0 / k6 + Scalar(6.0) * k2 + k6;

        const Scalar fR = A * sigma_6 / r / Scalar(37800.0);
        Scalar pair_eng = fR * (h0 - h1 - h2 + h3);
        if (force)
            {
            const Scalar dUR = pair_eng / r + Scalar(5.0) * fR * (g0 + g1 - g2 - g3);
            const Scalar dUA = -A / Scalar(3.0) * r
                               * ((Scalar(2.0) * k0 * k7 + Scalar(1.0)) * k7
                                  + (Scalar(2.0) * k0 * k8 - Scalar(1.0)) * Scalar(k8));

            force_divr = (dUR + dUA) / r;
            }
        pair_eng += A / Scalar(6.0) * (Scalar(2.0) * k0 * (k7 + k8) - log(k8 / k7));
        return pair_eng;
        }

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
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && A != Scalar(0))
            {
            // Solvent-Solvent interaction
            if (ai == Scalar(0) && aj == Scalar(0))
                {
                pair_eng = computeSolventSolvent<true>(force_divr, rsq);
                if (energy_shift)
                    {
                    pair_eng -= computeSolventSolvent<false>(force_divr, rcutsq);
                    }
                }
            // Colloid-Colloid interaction
            else if (ai != Scalar(0) && aj != Scalar(0))
                {
                pair_eng = computeColloidColloid<true>(force_divr, rsq);
                if (energy_shift)
                    {
                    pair_eng -= computeColloidColloid<false>(force_divr, rcutsq);
                    }
                }
            // Colloid-Solvent interaction
            else
                {
                pair_eng = computeColloidSolvent<true>(force_divr, rsq);
                if (energy_shift)
                    {
                    pair_eng -= computeColloidSolvent<false>(force_divr, rcutsq);
                    }
                }
            return true;
            }
        else
            return false;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name. Must be short and all lowercase, as this is the name energies
       will be logged as via analyze.log.
    */
    static std::string getName()
        {
        return std::string("colloid");
        }
#endif

    protected:
    Scalar A;       //!< Hamaker constant
    Scalar sigma_3; //!< Sigma^3
    Scalar sigma_6; //!< Sigma^6
    Scalar ai;      //!< radius 1
    Scalar aj;      //!< radius 2
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_COLLOID_H_
