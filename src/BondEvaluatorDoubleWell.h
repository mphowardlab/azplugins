// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
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
    BondParametersDoubleWell() : r_1(0), r_diff(1.0), U_1(0), U_tilt(0) { }

    BondParametersDoubleWell(pybind11::dict v)
        {
        r_1 = v["r_1"].cast<Scalar>();
        r_diff = r_1 - v["r_0"].cast<Scalar>();
        U_1 = v["U_1"].cast<Scalar>();
        U_tilt = v["U_tilt"].cast<Scalar>();
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["r_0"] = r_1 - r_diff;
        v["r_1"] = r_1;
        v["U_1"] = U_1;
        v["U_tilt"] = U_tilt;
        return v;
        }
#endif

    Scalar r_1;    //!< location of the potential local maximum
    Scalar r_diff; //!< difference between r_1 and r_0 (location of first minimum)
    Scalar U_1;    //!< Potential Potential maximum energy barrier between minima
    Scalar U_tilt; //!< tunes the energy offset (tilt) between minima
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
 * U(r)  =  U_1\left[\frac{\left((r-r_1)^2-(r_1-r_0)^2\right)^2}{\left(r_1-r_0\right)^4}\right]
 *  +
 * U_{\rm{tilt}}\left[1+\frac{r-r_1}{r_1-r_0}-\frac{\left((r-r_1)^2-(r_1-r_0)^2\right)^2}{\left(r_1-r_0\right)^4}\right]
 *
 * \f}
 * which has two minima at r = r_0 and r = 2 * r_1 - r_0, seperated by a maximum
 * at r_1 of height U_1 when U_tilt is set to zero.
 *
 * The parameter r_1 tunes the location of the maximal value and the parameter r_0 tunes the
 * distance of the two minima from each other.  This potential is useful to model bonds which can be
 * either mechanically or thermally "activated" into a effectively longer state. The value of U_1
 * can be used to tune the height of the energy barrier in between the two states.
 *
 * If U_tilt is non zero, the relative energy of the minima can be tuned, where 2 * U_tilt
 * is the energy of the second minima, the first minima value is at zero. This causes a
 * small shift in the location of the minima and the maxima, because of the added linear term.
 */
class BondEvaluatorDoubleWell : public BondEvaluator
    {
    public:
    typedef BondParametersDoubleWell param_type;

    DEVICE BondEvaluatorDoubleWell(Scalar _rsq, const param_type& _params)
        : BondEvaluator(_rsq), r_1(_params.r_1), r_diff(_params.r_diff), U_1(_params.U_1),
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

        const Scalar r = fast::sqrt(rsq);
        const Scalar x = (r_1 - r) / r_diff;
        const Scalar x2 = x * x;
        const Scalar y = Scalar(1.0) - x2;
        const Scalar y2 = y * y;

        bond_eng = U_1 * y2 + U_tilt * (Scalar(1.0) - x - y2);
        force_divr = (Scalar(4.0) * x * y * (U_tilt - U_1) - U_tilt) / (r_diff * r);
        return true;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return std::string("DoubleWell");
        }
#endif

    private:
    Scalar r_1;    //!< r_1 parameter
    Scalar r_diff; //!< r_diff parameter
    Scalar U_1;    //!< U_1 parameter
    Scalar U_tilt; //!< U_tilt parameter
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_DOUBLE_WELL_H_
