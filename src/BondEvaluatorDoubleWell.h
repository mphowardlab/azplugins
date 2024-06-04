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
    BondParametersDoubleWell() : r_0(0), r_1(0), U_1(0), U_tilt(0) { }

    BondParametersDoubleWell(pybind11::dict v)
        {
        r_0 = v["r_0"].cast<Scalar>();
        r_1 = v["r_1"].cast<Scalar>();
        U_1 = v["U_1"].cast<Scalar>();
        U_tilt = v["U_tilt"].cast<Scalar>();

        const Scalar r_diff = r_1 - r_0;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["r_0"] = r_0;
        v["r_1"] = r_1;
        v["U_1"] = U_1;
        v["U_tilt"] = U_tilt;
        return v;
        }
#endif

    Scalar r_0;    //!< Potential difference between the the first minima and maxima
    Scalar r_1;    //!< Shift for the location of U_1 (to approx. a/2)
    Scalar U_1;    //!< Scaling for distance of the two minima (to approx. a/2 +/- b)
    Scalar U_tilt; //!< Potential difference between the two minima
    Scalar r_diff;
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
 * which has two minima at r = (a/2 +/- b), seperated by a maximum at a/2 of height U_1 when c is
 * set to zero.
 *
 * The parameter a tunes the location of the maximal value and the parameter b tunes the distance of
 * the two maxima from each other.  This potential is useful to model bonds which can be either
 * mechanically or thermally "activated" into a effectively longer state. The value of U_1 can be
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
        : BondEvaluator(_rsq), r_0(_params.r_0), r_1(_params.r_1), U_1(_params.U_1),
          U_tilt(_params.U_tilt)
        {
        }

    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
        {
        bond_eng = 0;
        force_divr = 0;

        // check for invalid parameters r0 = r1
        if (r_diff == Scalar(0.0))
            return false;

        Scalar r = fast::sqrt(rsq);
        Scalar x = (r_1 - r) / r_diff;
        Scalar x2 = x * x;
        Scalar y = Scalar(1.0) - x2;
        Scalar y2 = y * y;
        Scalar w = (x * y) / r_diff;

        bond_eng = (U_1 * y2 + U_tilt * (Scalar(1.0) - x - y2));
        force_divr
            = (-U_1 * x * y / r_diff - U_tilt * (Scalar(1.0) / r_diff - Scalar(4.0) * w / r_diff))
              / r;
        return true;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return std::string("DoubleWell");
        }
#endif

    private:
    Scalar r_0;    //!< U_1 parameter
    Scalar r_1;    //!< a parameter
    Scalar U_1;    //!< b parameter
    Scalar U_tilt; //!< c parameter
    Scalar r_diff;
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
