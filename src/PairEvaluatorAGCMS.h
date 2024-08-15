// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PAIR_EVALUATOR_AGCMS_H_
#define AZPLUGINS_PAIR_EVALUATOR_AGCMS_H_

#include "PairEvaluator.h"
#include <stdio.h>

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

struct PairParametersAGCMS : public PairParameters
    {
#ifndef __HIPCC__
    PairParametersAGCMS() : w(0), sigma(0), a(0), q(0) { }

    PairParametersAGCMS(pybind11::dict v, bool managed = false)
        {
        w = v["w"].cast<Scalar>();
        sigma = v["sigma"].cast<Scalar>();
        a = v["a"].cast<Scalar>();
        q = v["q"].cast<Scalar>();
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["w"] = w;
        v["sigma"] = sigma;
        v["a"] = a;
        v["q"] = q;
        return v;
        }
#endif // __HIPCC__

    Scalar w;     //!< well width [length]
    Scalar sigma; //!< Hard core repulsion distance [length]
    Scalar a;     //!< Well depth [energy]
    Scalar q;     //!< steepness
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the adjusted generalized continuous multiple step pair potential
/*!
 * This is the generalized continuous multiple step pair potential
 * modified such that there is only one sigma and one minimum present
 * with any combination of well depth and width causing the potential
 * to have a value of zero at sigma.
 */
class PairEvaluatorAGCMS : public PairEvaluator
    {
    public:
    typedef PairParametersAGCMS param_type;

    //! Constructor
    /*!
     * \param _rsq Squared distance between particles
     * \param _rcutsq Cutoff radius squared
     * \param _params Pair potential parameters, given by typedef above
     *
     * The functor initializes its members from \a _params.
     */
    DEVICE PairEvaluatorAGCMS(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : PairEvaluator(_rsq, _rcutsq)
        {
        w = _params.w;
        sigma = _params.sigma;
        a = _params.a;
        q = _params.q;
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
        if (rsq < rcutsq && a != Scalar(0))
            {
            Scalar r = fast::sqrt(rsq);
            Scalar rinv = 1 / r;
            Scalar w_rinv_shifted = w / (r - sigma + w);
            Scalar core_repuls = pow(w_rinv_shifted, q);
            Scalar exponent = exp(q * (r - sigma - w) / w);
            force_divr = -a * rinv
                         * (q * core_repuls * w_rinv_shifted
                            - q * exponent / (w * pow(1 + exponent, 2.0)));
            pair_eng = a * (1 / (1 + exponent) - core_repuls);
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
        return std::string("agcms");
        }
#endif

    protected:
    Scalar w;     //!< well width
    Scalar sigma; //!< Hard core repulsion distance
    Scalar a;     //!< well depth
    Scalar q;     //!< Steepness
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_AGCMS_H_
