// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PAIR_EVALUATOR_EXPANDEDYUKAWA_H_
#define AZPLUGINS_PAIR_EVALUATOR_EXPANDEDYUKAWA_H_

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

struct PairParametersExpandedYukawa : public PairParameters
    {
#ifndef __HIPCC__
    PairParametersExpandedYukawa() : epsilon(0), kappa(0), delta(0) { }

    PairParametersExpandedYukawa(pybind11::dict v, bool managed = false)
        {
        epsilon = v["epsilon"].cast<Scalar>();
        kappa = v["kappa"].cast<Scalar>();
        delta = v["delta"].cast<Scalar>();
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["epsilon"] = epsilon;
        v["kappa"] = kappa;
        v["delta"] = delta;
        return v;
        }
#endif // __HIPCC__

    Scalar epsilon; //!< energy parameter [energy]
    Scalar kappa;   //!< scaling parameter [length]^-1
    Scalar delta;   //!< minimum interaction distance [length]
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the expanded yukawa pair potential
/*!
 * This is the typical Yukawa potential modified to shift the potential
 * to account for particle diameters not equal to 1.
 */
class PairEvaluatorExpandedYukawa : public PairEvaluator
    {
    public:
    typedef PairParametersExpandedYukawa param_type;

    //! Constructor
    /*!
     * \param _rsq Squared distance between particles
     * \param _rcutsq Cutoff radius squared
     * \param _params Pair potential parameters, given by typedef above
     *
     * The functor initializes its members from \a _params.
     */
    DEVICE PairEvaluatorExpandedYukawa(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : PairEvaluator(_rsq, _rcutsq)
        {
        epsilon = _params.epsilon;
        kappa = _params.kappa;
        delta = _params.delta;
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
        if (rsq < rcutsq && epsilon != Scalar(0))
            {
            Scalar r = fast::sqrt(rsq);
            Scalar rinv = 1 / r;
            Scalar delta_dist = r - delta;
            Scalar rinv_delt = 1 / (r - delta);
            Scalar kappa_delt = kappa * delta_dist;
            Scalar exponent = exp(-kappa_delt);

            force_divr = epsilon * exponent * (1 + kappa_delt) * rinv_delt * rinv_delt * rinv;
            pair_eng = epsilon * exponent * rinv_delt;

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
        return std::string("exyuk");
        }
#endif

    protected:
    Scalar epsilon; //!< Energy parameter
    Scalar kappa;   //!< Scaling parameter
    Scalar delta;   //!< Minimum interaction distance
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_EXPANDEDYUKAWA_H_