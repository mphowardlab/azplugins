// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PAIR_EVALUATOR_PERTURBED_LENNARD_JONES_H_
#define AZPLUGINS_PAIR_EVALUATOR_PERTURBED_LENNARD_JONES_H_

#include "PairEvaluator.h"

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
//! Define the parameter type used by this pair potential evaluator
struct PairParametersPerturbedLennardJones : public PairParameters
    {
#ifndef __HIPCC__
    PairParametersPerturbedLennardJones() : sigma_6(0), epsilon_x_4(0), lam(0), rwcasq(0) { }

    PairParametersPerturbedLennardJones(pybind11::dict v, bool managed = false)
        {
        auto sigma(v["sigma"].cast<Scalar>());
        auto epsilon(v["epsilon"].cast<Scalar>());

        const Scalar sigma_2 = sigma * sigma;
        const Scalar sigma_4 = sigma_2 * sigma_2;
        sigma_6 = sigma_2 * sigma_4;

        epsilon_x_4 = Scalar(4.0) * epsilon;
        lam = v["lam"].cast<Scalar>();
        rwcasq = pow(Scalar(2.), 1. / 3.) * sigma_2;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["sigma"] = pow(sigma_6, 1. / 6.);
        v["epsilon"] = epsilon_x_4 / Scalar(4.);
        v["lam"] = lam;
        return v;
        }
#endif // __HIPCC__

    Scalar sigma_6;     //!< The coefficient for 1/r^12
    Scalar epsilon_x_4; //!< The coefficient for 1/r^6
    Scalar lam;         //!< Controls the attractive tail, between 0 and 1
    Scalar rwcasq;      //!< The square of the location of the LJ potential minimum
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the perturbed Lennard-Jones pair potential
/*!
 * This class evaluates the function:
 *      \f{eqnarray*}{
 *      V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma)
 *              + (1-\lambda)\varepsilon & r < 2^{1/6}\sigma \\
 *            = & \lambda V_{\mathrm{LJ}}(r, \varepsilon, \sigma) &
 *              2^{1/6}\sigma \ge r < r_{\mathrm{cut}} \\
 *            = & 0 & r \ge r_{\mathrm{cut}}
 *      \f}
 *
 * where \f$V_{\mathrm{LJ}}(r,\varepsilon,\sigma)\f$ is the standard
 * Lennard-Jones potential (see EvaluatorPairLJ) with parameters
 * \f$\varepsilon\f$ and \f$\sigma\f$.
 */
class PairEvaluatorPerturbedLennardJones : public PairEvaluator
    {
    public:
    typedef PairParametersPerturbedLennardJones param_type;

    //! Constructor
    /*!
     * \param _rsq Squared distance between particles
     * \param _rcutsq Cutoff radius squared
     * \param _params Pair potential parameters, given by typedef above
     *
     * The functor initializes its members from \a _params.
     */
    DEVICE
    PairEvaluatorPerturbedLennardJones(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : PairEvaluator(_rsq, _rcutsq),
          lj1(_params.epsilon_x_4 * _params.sigma_6 * _params.sigma_6),
          lj2(_params.epsilon_x_4 * _params.sigma_6), lam(_params.lam), rwcasq(_params.rwcasq)
        {
        wca_shift = _params.epsilon_x_4 * (Scalar(1.0) - lam) / Scalar(4.0);
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
        if (rsq < rcutsq && lj1 != 0)
            {
            const Scalar r2inv = Scalar(1.0) / rsq;
            const Scalar r6inv = r2inv * r2inv * r2inv;
            force_divr = r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);

            pair_eng = r6inv * (lj1 * r6inv - lj2);
            if (rsq < rwcasq)
                {
                pair_eng += wca_shift;
                }
            else
                {
                force_divr *= lam;
                pair_eng *= lam;
                }

            if (energy_shift)
                {
                const Scalar rcut2inv = Scalar(1.0) / rcutsq;
                const Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                Scalar pair_eng_shift = rcut6inv * (lj1 * rcut6inv - lj2);
                if (rcutsq < rwcasq)
                    {
                    pair_eng_shift += wca_shift;
                    }
                else
                    {
                    pair_eng_shift *= lam;
                    }
                pair_eng -= pair_eng_shift;
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
        return std::string("PerturbedLennardJones");
        }
#endif

    private:
    Scalar lj1;       //!< lj1 parameter - 4 epsilon sigma^12
    Scalar lj2;       //!< lj2 parameter - 4 epsilon sigma^6
    Scalar lam;       //!< lambda parameter
    Scalar rwcasq;    //!< WCA cutoff radius squared
    Scalar wca_shift; //!< Energy shift for WCA part of the potential
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_PERTURBED_LENNARD_JONES_H_
