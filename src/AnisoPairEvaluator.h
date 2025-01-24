// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_ANISO_PAIR_EVALUATOR_H_
#define AZPLUGINS_ANISO_PAIR_EVALUATOR_H_

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <string>
#endif // __HIPCC__

#include "hoomd/HOOMDMath.h"

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

//! Base class for anisotropic pair potential parameters
/*!
 * This class covers the default case of a simple potential that doesn't do
 * anything special loading its parameters. This class can then be used as
 * a \a param_type for the evaluator.
 *
 * Deriving classes \b must implement the constructors below. They should also
 * set the aligned attribute based on the size of the object.
 */
struct AnisoPairParameters
    {
#ifndef __HIPCC__
    AnisoPairParameters() { }

    AnisoPairParameters(pybind11::dict v, bool managed = false) { }

    pybind11::dict toPython()
        {
        return pybind11::dict();
        }
#endif // __HIPCC__

    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    };

//! Base class for anisotropic pair potential shape parameters
/*!
 * These types of parameters do nothing by default.
 */
struct AnisoPairShapeParameters
    {
#ifndef __HIPCC__
    AnisoPairShapeParameters() { }

    AnisoPairShapeParameters(pybind11::object shape_params, bool managed) { }

    pybind11::object toPython()
        {
        return pybind11::none();
        }
#endif // __HIPCC__

    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    };

//! Base class for anisotropic pair potential evaluator
/*!
 * This class covers the default case of a simple potential that doesn't require
 * additional data. Its constructor stores the common variables. The class can
 * be overridden for more complex potentials.
 *
 * Deriving classes \b must define a \a param_type and a \a shape_param_type
 * (which can simply be a redeclaration of the one here). They must also
 * give the potential a name and implement the evaluate() method.
 */
class AnisoPairEvaluator
    {
    public:
    typedef AnisoPairParameters param_type;
    typedef AnisoPairShapeParameters shape_type;

    DEVICE AnisoPairEvaluator(const Scalar3& _dr,
                              const Scalar4& _quat_i,
                              const Scalar4& _quat_j,
                              const Scalar _rcutsq)
        : dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j)
        {
        }

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

    //! Base potential does not need shape
    DEVICE static bool needsShape()
        {
        return false;
        }

    //! Accept the optional shape values
    /*!
     * \param shapei Shape of particle i
     * \param shapej Shape of particle j
     */
    DEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }

    //! Base potential does not need tags
    DEVICE static bool needsTags()
        {
        return false;
        }

    //! Accept the optional tags
    /*!
     * \param tagi Tag of particle i
     * \param tagj Tag of particle j
     */
    DEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
     *  \param pair_eng Output parameter to write the computed pair energy.
     *  \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
     * cutoff. \param torque_i The torque exterted on the i^th particle. \param torque_j The torque
     * exterted on the j^th particle.
     *
     *  \returns True if they are evaluated or false if they are not because we are beyond the
     * cutoff.
     */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& pair_eng,
                         bool energy_shift,
                         Scalar3& torque_i,
                         Scalar3& torque_j)
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
    static std::string getName()
        {
        throw std::runtime_error("Name not defined for this pair potential.");
        }

    static std::string getShapeParamName()
        {
        throw std::runtime_error("Shape name not defined for this pair potential.");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif // __HIPCC__

    protected:
    Scalar3 dr;     //!< Stored dr from the constructor
    Scalar rcutsq;  //!< Stored rcutsq from the constructor
    Scalar4 quat_i; //!< Orientation quaternion for particle i
    Scalar4 quat_j; //!< Orientation quaternion for particle j
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_ANISO_PAIR_EVALUATOR_H_
