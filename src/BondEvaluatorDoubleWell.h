// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
#define AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_

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

//! Parameters of double well bond potential
struct BondParametersDoubleWell : public BondParameters
    {
#ifndef __HIPCC__
    BondParametersDoubleWell() : V_max(0), a(0), b(0), c(0) { }

    BondParametersDoubleWell(pybind11::dict v)
        {
        V_max = v["V_max"].cast<Scalar>();
        a = v["a"].cast<Scalar>();
        b = v["b"].cast<Scalar>();
        c = v["c"].cast<Scalar>();
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["V_max"] = V_max;
        v["a"] = a;
        v["b"] = b;
        v["c"] = c;
        return v;
        }
#endif

    Scalar V_max; //!< Potential difference between the the first minima and maxima
    Scalar a;     //!< Shift for the location of V_max (to approx. a/2)
    Scalar b;     //!< Scaling for distance of the two minima (to approx. a/2 +/- b)
    Scalar c;     //!< Potential difference between the two minima
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(16)));
#else
    __attribute__((aligned(32)));
#endif

//! Class for evaluating the double well bond potential
/*!
 * This bond potential follows the functional form
 * \f{eqnarray*}
 *
 * V_{\rm{DW}}(r) = \frac{V_{max}-c/2}{b^4} \left[ \left( r - a/2 \right)^2 -b^2 \right]^2 +
 * \frac{c}{2b}\left(r-a/2\right)+c/2
 *
 * \f}
 * which has two minima at r = (a/2 +/- b), seperated by a maximum at a/2 of height V_max when c is
 * set to zero.
 *
 * The parameter a tunes the location of the maximal value and the parameter b tunes the distance of
 * the two maxima from each other.  This potential is useful to model bonds which can be either
 * mechanically or thermally "activated" into a effectively longer state. The value of V_max can be
 * used to tune the height of the energy barrier in between the two states.
 *
 * If c is non zero, the relative energy of the minima can be tuned, where c is the energy of the
 * second minima, the first minima value is at zero. This  causes a small shift in the location of
 * the minima and the maxima, because of the added linear term.
 */
class BondEvaluatorDoubleWell : public BondEvaluator
    {
    public:
    typedef BondParametersDoubleWell param_type;

    DEVICE BondEvaluatorDoubleWell(Scalar _rsq, const param_type& _params)
        : BondEvaluator(_rsq), V_max(_params.V_max), a(_params.a), b(_params.b), c(_params.c)
        {
        }

    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
        {
        bond_eng = 0;
        force_divr = 0;

        // check for invalid parameters
        if (b == Scalar(0.0))
            return false;

        Scalar c_half = Scalar(0.5) * c;
        Scalar r = fast::sqrt(rsq);
        Scalar r_min_half_a = r - Scalar(0.5) * a;
        Scalar b_sq = b * b;
        Scalar d = r_min_half_a * r_min_half_a - b_sq;

        bond_eng = ((V_max - c_half) / (b_sq * b_sq)) * d * d + c_half / b * r_min_half_a + c_half;
        force_divr = -(4 * (V_max - c_half) / (b_sq * b_sq) * d * r_min_half_a + c_half / b) / r;

        return true;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return std::string("DoubleWell");
        }
#endif

    private:
    Scalar V_max; //!< V_max parameter
    Scalar a;     //!< a parameter
    Scalar b;     //!< b parameter
    Scalar c;     //!< c parameter
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
