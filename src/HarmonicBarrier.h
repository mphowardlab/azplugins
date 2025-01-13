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
 *
 * The specific form of the potential is:
 *
 *      \f{eqnarray*}{
 *      V(z) = & 0 & z < H \\
 *             & \frac{\kappa}{2} (z-H)^2 & z > H \\
 *      \f}
 *
 * with the following parameters:
 *
 *  - \f$\kappa\f$ - \a k (energy per distance squared) - spring constant
 *  - \a offset (distance) - per-particle-type amount to shift \a H, default: 0.0
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

        param_type() : k(0), offset(0) { }

        param_type(pybind11::dict params)
            {
            k = pybind11::cast<Scalar>(params["k"]);
            offset = pybind11::cast<Scalar>(params["offset"]);
            }

        pybind11::dict toPython()
            {
            pybind11::dict d;
            d["k"] = pybind11::cast(k);
            d["offset"] = pybind11::cast(offset);
            return d;
            }
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Set the per-type potential parameters
    /*!
     * \param type Particle type id
     * \param k Spring constant
     * \param offset Distance to shift potential minimum from interface
     */
    void setParams(unsigned int type, const param_type& params)
        {
        assert(type < m_pdata->getNTypes());
        ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
        h_params.data[type] = make_scalar2(params.k, params.offset);
        }

    void setParamsPython(std::string type_name, pybind11::dict params)
        {
        unsigned int type_idx = m_pdata->getTypeByName(type_name);
        param_type h_params(params);
        setParams(type_idx, h_params);
        }

    param_type getParams(std::string type_name)
        {
        unsigned int type_idx = m_pdata->getTypeByName(type_name);
        assert(type_idx < m_pdata->getNTypes());

        ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
        Scalar2 param_values = h_params.data[type_idx];

        param_type h_params_obj;
        h_params_obj.k = param_values.x;
        h_params_obj.offset = param_values.y;

        return h_params_obj;
        }

    pybind11::dict getParamsPython(std::string type_name)
        {
        param_type params = getParams(type_name);

        pybind11::dict params_dict;
        params_dict["k"] = params.k;
        params_dict["offset"] = params.offset;

        return params_dict;
        }

    protected:
    std::shared_ptr<Variant> m_interf; //!< Current location of the interface
    GPUArray<Scalar2> m_params;        //!< Per-type array of parameters for the potential

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
