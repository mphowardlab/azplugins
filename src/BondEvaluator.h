// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_BOND_EVALUATOR_H_
#define AZPLUGINS_BOND_EVALUATOR_H_

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <string>
#endif // __HIPCC__

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

//! Base class for isotropic pair potential parameters
/*!
 * This class covers the default case of a simple potential that doesn't do
 * anything special loading its parameters. This class can then be used as
 * a \a param_type for the evaluator.
 *
 * Deriving classes \b must implement the constructors below. They should also
 * set the aligned attribute based on the size of the object.
 */
struct BondParameters
    {
#ifndef __HIPCC__
    BondParameters() { }

    BondParameters(pybind11::dict v) { }

    pybind11::dict asDict()
        {
        return pybind11::dict();
        }
#endif //__HIPCC__
    };

//! Base class for bond potential evaluator
/*!
 * This class covers the default case of a simple potential that doesn't require
 * additional data. Deriving classes \b must define a \a param_type . They must
 * also give the potential a name and implement the evalForceAndEnergy() method.
 */
class BondEvaluator
    {
    public:
    typedef BondParameters param_type;

    DEVICE BondEvaluator(const Scalar _rsq) : rsq(_rsq) { }

    //! Base potential does not need charge
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values
    /*!
     * \param qi Charge of particle i
     * \param qj Charge of particle j
     */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*!
     * \param force_divr Computed force divided by r
     * \param bond_eng Computed bond energy
     *
     * \returns True if the energy calculation occurs
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
        {
        return false;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*!
     * This method must be overridden by deriving classes.
     */
    static std::string getName()
        {
        throw std::runtime_error("Name not defined for this pair potential.");
        }
#endif // __HIPCC__

    protected:
    Scalar rsq; //!< Squared distance between particles
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_H_
