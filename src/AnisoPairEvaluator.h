// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file AnisoPairEvaluator.h
 * \brief Base class for anisotropic pair evaluators.
 */

#ifndef AZPLUGINS_ANISO_PAIR_EVALUATOR_H_
#define AZPLUGINS_ANISO_PAIR_EVALUATOR_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <string>
#endif

namespace azplugins
{
namespace detail
{

//! Base class for anisotropic pair parameters
/*!
 * These types of parameters do nothing by default.
 */
struct AnisoPairParams
    {
    HOSTDEVICE AnisoPairParams() {}

    //! Load dynamic data members into shared memory and increase pointer
    /*!
     * \param ptr Pointer to load data to (will be incremented)
     * \param available_bytes Size of remaining shared memory allocation
     *
     * This does nothing for this struct.
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const {}
    };

//! Base class for anisotropic shape parameters
/*!
 * These types of parameters do nothing by default.
 */
struct AnisoShapeParams
    {
    HOSTDEVICE AnisoShapeParams() {}

    //! Load dynamic data members into shared memory and increase pointer
    /*!
     * \param ptr Pointer to load data to (will be incremented)
     * \param available_bytes Size of remaining shared memory allocation
     *
     * This does nothing for this struct.
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const {}

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const {}
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
        typedef AnisoShapeParams shape_param_type;

        DEVICE AnisoPairEvaluator(const Scalar3& _dr,
                                  const Scalar4& _quat_i,
                                  const Scalar4& _quat_j,
                                  const Scalar _rcutsq)
        : dr(_dr), rcutsq(_rcutsq), quat_i(_quat_i), quat_j(_quat_j)
        {}

        //! Base potential does not need diameter
        DEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Accept the optional diameter values
        /*!
         * \param di Diameter of particle i
         * \param dj Diameter of particle j
         */
        DEVICE void setDiameter(Scalar di, Scalar dj) {}

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
        DEVICE void setCharge(Scalar qi, Scalar qj) {}

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
        DEVICE void setShape(const shape_param_type *shapei, const shape_param_type *shapej) {}

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
        DEVICE void setTags(unsigned int tagi, unsigned int tagj) {}

        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
         *  \param pair_eng Output parameter to write the computed pair energy.
         *  \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
         *  \param torque_i The torque exterted on the i^th particle.
         *  \param torque_j The torque exterted on the j^th particle.
         *
         *  \returns True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        DEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
            {
            return false;
            }

    #ifndef NVCC
        static std::string getName()
            {
            throw std::runtime_error("Name not defined for this pair potential.");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
    #endif // NVCC

    protected:
        Scalar3 dr;       //!< Stored dr from the constructor
        Scalar rcutsq;    //!< Stored rcutsq from the constructor
        Scalar4 quat_i;   //!< Orientation quaternion for particle i
        Scalar4 quat_j;   //!< Orientation quaternion for particle j
    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE
#undef HOSTDEVICE

#endif // AZPLUGINS_ANISO_PAIR_EVALUATOR_H_
