// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file TypeUpdater.h
 * \brief Declaration of TypeUpdater
 */

#ifndef AZPLUGINS_TYPE_UPDATER_H_
#define AZPLUGINS_TYPE_UPDATER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Updater.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
    {

//! Particle type updater
/*!
 * Flips particle types based on their z height. Particles are classified as
 * either inside or outside of the region, and can be flipped between these two
 * types. Particles that are of neither the inside nor outside type are ignored.
 *
 * The region is defined by a slab along z. This could be easily extended to
 * accommodate a generic region criteria, but for now, the planar slab in z is
 * all that is necessary.
 */
class PYBIND11_EXPORT TypeUpdater : public Updater
    {
    public:
    //! Simple constructor
    TypeUpdater(std::shared_ptr<SystemDefinition> sysdef);

    //! Constructor with parameters
    TypeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                unsigned int inside_type,
                unsigned int outside_type,
                Scalar z_lo,
                Scalar z_hi);

    //! Destructor
    virtual ~TypeUpdater();

    //! Evaporate particles
    virtual void update(unsigned int timestep);

    //! Get the inside particle type
    unsigned int getInsideType() const
        {
        return m_inside_type;
        }

    //! Set the inside particle type
    void setInsideType(unsigned int inside_type)
        {
        m_inside_type = inside_type;
        requestCheckTypes();
        }

    //! Get the outside particle type
    unsigned int getOutsideType() const
        {
        return m_outside_type;
        }

    //! Set the outside particle type
    void setOutsideType(unsigned int outside_type)
        {
        m_outside_type = outside_type;
        requestCheckTypes();
        }

    //! Get region lower bound
    Scalar getRegionLo() const
        {
        return m_z_lo;
        }

    //! Get region upper bound
    Scalar getRegionHi() const
        {
        return m_z_hi;
        }

    //! Set region lower bound
    void setRegionLo(Scalar z_lo)
        {
        m_z_lo = z_lo;
        requestCheckRegion();
        }

    //! Set region upper bound
    void setRegionHi(Scalar z_hi)
        {
        m_z_hi = z_hi;
        requestCheckRegion();
        }

    protected:
    unsigned int m_inside_type;  //!< Type id of particles in region
    unsigned int m_outside_type; //!< Type id of particles outside region

    Scalar m_z_lo; //!< Minimum bound of region in z
    Scalar m_z_hi; //!< Maximum bound of region in z

    //! Changes the particle types according to an update rule
    virtual void changeTypes(unsigned int timestep);

    private:
    bool m_check_types; //!< Flag if type check is necessary
    //! Request to check types on next update
    void requestCheckTypes()
        {
        m_check_types = true;
        }
    //! Check that the particle types are valid
    void checkTypes() const;

    bool m_check_region; //!< Flag if region check is necessary
    //! Request to check region on next update
    void requestCheckRegion()
        {
        m_check_region = true;
        }
    //! Check that the particle region is valid
    void checkRegion() const;
    };

namespace detail
    {
//! Export the Evaporator to python
void export_TypeUpdater(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_H_
