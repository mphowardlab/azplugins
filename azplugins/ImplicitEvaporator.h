// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitEvaporator.h
 * \brief Declaration of ImplicitEvaporator
 */

#ifndef AZPLUGINS_IMPLICIT_EVAPORATOR_H_
#define AZPLUGINS_IMPLICIT_EVAPORATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/Variant.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Implicit solvent evaporator
/*!
 * Implicitly models the effect of solvent evaporation as a moving interface.
 * The moving interface compute acts on particles along the inward normal.
 * The interface potential is harmonic. It does not include an attractive
 * part (i.e., it is truncated at its minimum, and is zero for any negative
 * displacements relative to the minimum). The position of the minimum can be
 * adjusted with an offset, which controls an effective contact angle.
 * The potential is cutoff at a certain distance above the interface, at which
 * point it switches to a linear potential with a fixed force constant. This models
 * the gravitational force experienced by particles once the interface has moved
 * past the particles. In practice, this cutoff should be roughly the particle radius.
 *
 * The specific form of the potential is:
 *
 *      \f{eqnarray*}{
 *      V(z) = & 0 & z < H \\
 *             & \frac{\kappa}{2} (z-H)^2 & H \le z < H_{\rm c} \\
 *             & \frac{\kappa}{2} (H_{\rm c} - H)^2 - F_g (z - H_{\rm c}) & z \ge H_{\rm c}
 *      \f}
 *
 * with the following parameters:
 *
 *  - \f$\kappa\f$ - \a k (energy per distance squared) - spring constant
 *  - \a offset (distance) - per-particle-type amount to shift \a H, default: 0.0
 *  - \f$F_g\f$ - \a g (force) - force to apply above \f$H_{\rm c}\f$
 *  - \f$\Delta\f$ - \a cutoff (distance) - sets cutoff at \f$H_{\rm c} = H + \Delta\f$
 *
 * The meaning of \f$z\f$ is the distance from some origin, and \f$H\f$ is the distance of
 * the interface (e.g., a plane, sphere, etc.) from that same origin. This class doesn't
 * actually compute anything on its own. Deriving classes should implement this geometry.
 *
 * \warning The temperature reported by ComputeThermo will likely not be accurate
 *          during evaporation because the system is out-of-equilibrium,
 *          and so may experience a net convective drift. This ForceCompute should
 *          only be used with a Langevin thermostat, which does not rely on
 *          computing the temperature of the system.
 *
 * \warning The virial is not computed for this external potential, and a warning
 *          will be raised the first time it is requested.
 *
 */
class PYBIND11_EXPORT ImplicitEvaporator : public ForceCompute
    {
    public:
        //! Constructor
        ImplicitEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Variant> interf);

        //! Destructor
        virtual ~ImplicitEvaporator();

        //! Set the per-type potential parameters
        /*!
         * \param type Particle type id
         * \param k Spring constant
         * \param offset Distance to shift potential minimum from interface
         * \param g Linear potential force constant
         * \param cutoff Distance from potential minimum to cutoff harmonic potential and switch to linear
         */
        void setParams(unsigned int type, Scalar k, Scalar offset, Scalar g, Scalar cutoff)
            {
            assert(type < m_pdata->getNTypes());
            ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
            h_params.data[type] = make_scalar4(k, offset, g, cutoff);
            }

    protected:
        std::shared_ptr<Variant> m_interf;      //!< Current location of the interface
        GPUArray<Scalar4> m_params;             //!< Per-type array of parameters for the potential

        //! Method to compute the forces
        virtual void computeForces(unsigned int timestep);

    private:
        bool m_has_warned;  //!< Flag if a warning has been issued about the virial

        //! Reallocate the per-type parameter arrays when the number of types changes
        void reallocateParams()
            {
            m_params.resize(m_pdata->getNTypes());
            }
    };

namespace detail
{
//! Exports the ImplicitEvaporator to python
void export_ImplicitEvaporator(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_EVAPORATOR_H_
