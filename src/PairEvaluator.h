// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_PAIR_EVALUATOR_H_
#define AZPLUGINS_PAIR_EVALUATOR_H_

#ifndef __HIPCC__
#include <string>
#endif // __HIPCC__

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
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
struct PairParameters
    {
#ifndef __HIPCC__
    PairParameters() { }

    PairParameters(pybind11::dict v, bool managed = false) { }

    pybind11::dict asDict()
        {
        return pybind11::dict();
        }
#endif //__HIPCC__

    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    };

//! Base class for isotropic pair potential evaluator
/*!
 * This class covers the default case of a simple potential that doesn't require
 * additional data. Its constructor stores the common variables. The class can
 * be overridden for more complex potentials.
 *
 * Deriving classes \b must define a \a param_type . They must also
 * give the potential a name and implement the evaluate() method.
 */
class PairEvaluator
    {
    public:
    typedef PairParameters param_type;

    DEVICE PairEvaluator(const Scalar _rsq, const Scalar _rcutsq) : rsq(_rsq), rcutsq(_rcutsq) { }

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
        return false;
        }

    //! Evaluate the long-ranged correction to the pressure.
    /*!
     * \returns Default value of 0 (no correction).
     */
    DEVICE Scalar evalPressureLRCIntegral()
        {
        return Scalar(0.0);
        }

    //! Evaluate the long-ranged correction to the energy.
    /*!
     * \returns Default value of 0 (no correction).
     */
    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return Scalar(0.0);
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

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif // __HIPCC__

    protected:
    Scalar rsq;    //!< Squared distance between particles
    Scalar rcutsq; //!< Squared cutoff distance
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_PAIR_EVALUATOR_H_
