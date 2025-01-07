// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file HarmonicBarrier.h
 * \brief Declaration of HarmonicBarrier
 */

#ifndef AZPLUGINS_HARMONIC_BARRIER_H_
#define AZPLUGINS_HARMONIC_BARRIER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/Variant.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {

namespace azplugins
    {

//! Harmonic Barrier
/*!
 * Models moving interface with harmonic potential
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
class PYBIND11_EXPORT HarmonicBarrier : public ForceCompute
    {
    public:
    //! Constructor
    HarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Variant> interf);

    //! Destructor
    virtual ~HarmonicBarrier();

    struct param_type
        {
        Scalar k;
        Scalar offset;
        Scalar g;
        Scalar cutoff;

        param_type() : k(0), offset(0), g(0), cutoff(0) { }

        param_type(pybind11::dict params)
            {
            k = pybind11::cast<Scalar>(params["k"]);
            offset = pybind11::cast<Scalar>(params["offset"]);
            g = pybind11::cast<Scalar>(params["g"]);
            cutoff = pybind11::cast<Scalar>(params["cutoff"]);
            }

        pybind11::dict toPython()
            {
            pybind11::dict d;
            d["k"] = pybind11::cast(k);
            d["offset"] = pybind11::cast(offset);
            d["g"] = pybind11::cast(g);
            d["cutoff"] = pybind11::cast(cutoff);
            return d;
            }
        } __attribute__((aligned(16)));

    //! Set the per-type potential parameters
    /*!
     * \param type Particle type id
     * \param k Spring constant
     * \param offset Distance to shift potential minimum from interface
     * \param g Linear potential force constant
     * \param cutoff Distance from potential minimum to cutoff harmonic potential and switch to
     * linear
     */
    void setParams(unsigned int type, const param_type& params)
        {
        assert(type < m_pdata->getNTypes());
        ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
        h_params.data[type] = make_scalar4(params.k, params.offset, params.g, params.cutoff);
        }

    void setParamsPython(std::string type_name, pybind11::dict params)
        {
        unsigned int type_idx = m_pdata->getTypeByName(type_name);
        param_type h_params(params);
        setParams(type_idx, h_params);
        }

    pybind11::dict getParams(std::string type_name)
        {
        unsigned int type_idx = m_pdata->getTypeByName(type_name);
        assert(type_idx < m_pdata->getNTypes());

        ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);
        Scalar4 param_values = h_params.data[type_idx];

        param_type h_params_obj;
        h_params_obj.k = param_values.x;
        h_params_obj.offset = param_values.y;
        h_params_obj.g = param_values.z;
        h_params_obj.cutoff = param_values.w;

        return h_params_obj.toPython();
        }

    protected:
    std::shared_ptr<Variant> m_interf; //!< Current location of the interface
    GPUArray<Scalar4> m_params;        //!< Per-type array of parameters for the potential

    //! Method to compute the forces
    virtual void computeForces(uint64_t timestep);

    private:
    bool m_has_warned; //!< Flag if a warning has been issued about the virial

    //! Reallocate the per-type parameter arrays when the number of types changes
    void reallocateParams()
        {
        m_params.resize(m_pdata->getNTypes());
        }
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_HARMONIC_BARRIER_H_
