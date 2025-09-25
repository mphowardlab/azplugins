// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// File modified from HOOMD-blue
// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/MeshDefinition.h"
#include "hoomd/md/PotentialBond.h"
#include <memory>

#include <vector>

/*! \file ImagePotentialBond.h
    \brief Declares ImagePotentialBond
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef AZPLUGINS_IMAGE_POTENTIAL_BOND_H_
#define AZPLUGINS_IMAGE_POTENTIAL_BOND_H_

namespace hoomd
    {
namespace azplugins
    {
/*! Bond potential using unwrapped coordinates

    \ingroup computes
*/
template<class evaluator, class Bonds>
class ImagePotentialBond : public md::PotentialBond<evaluator, Bonds>
    {
    public:
    //! Inherit constructors from base class
    using md::PotentialBond<evaluator, Bonds>::PotentialBond;

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    CommFlags getRequestedCommFlags(uint64_t timestep) override;
#endif

    protected:
    void computeForces(uint64_t timestep) override;
    };

/*! Actually perform the force computation
    \param timestep Current time step
 */
template<class evaluator, class Bonds>
void ImagePotentialBond<evaluator, Bonds>::computeForces(uint64_t timestep)
    {
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(this->m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(),
                                 access_location::host,
                                 access_mode::read);

    ArrayHandle<Scalar4> h_force(this->m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(this->m_virial, access_location::host, access_mode::readwrite);

    // access the parameters
    ArrayHandle<typename evaluator::param_type> h_params(this->m_params,
                                                         access_location::host,
                                                         access_mode::read);

    // Zero data for force calculation
    this->m_force.zeroFill();
    this->m_virial.zeroFill();

    const BoxDim box = this->m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar bond_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        bond_virial[i] = Scalar(0.0);

    ArrayHandle<typename Bonds::members_t> h_bonds(this->m_bond_data->getMembersArray(),
                                                   access_location::host,
                                                   access_mode::read);
    ArrayHandle<typeval_t> h_typeval(this->m_bond_data->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    unsigned int max_local = this->m_pdata->getN() + this->m_pdata->getNGhosts();

    // for each of the bonds
    const unsigned int size = (unsigned int)this->m_bond_data->getN();

    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename Bonds::members_t& bond = h_bonds.data[i];

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.tag[0]];
        unsigned int idx_b = h_rtag.data[bond.tag[1]];

        // throw an error if this bond is incomplete
        if (idx_a >= max_local || idx_b >= max_local)
            {
            std::ostringstream stream;
            stream << "Error: bond " << bond.tag[0] << " " << bond.tag[1] << " is incomplete.";
            throw std::runtime_error(stream.str());
            }

        // calculate d\vec{r}
        // (MEM TRANSFER: 6 Scalars / FLOPS: 3)
        Scalar3 posa = make_scalar3(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        Scalar3 posb = make_scalar3(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);

        // access charge (if needed)
        Scalar charge_a = Scalar(0.0);
        Scalar charge_b = Scalar(0.0);
        if (evaluator::needsCharge())
            {
            charge_a = h_charge.data[idx_a];
            charge_b = h_charge.data[idx_b];
            }

        // get images of particles
        const int3 img_a = h_image.data[idx_a];
        const int3 img_b = h_image.data[idx_b];

        // get relative vector between particles
        const Scalar3 dx = box.shift(posb - posa, img_b - img_a);

        // calculate r_ab squared
        Scalar rsq = dot(dx, dx);

        // compute the force and potential energy
        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);
        evaluator eval(rsq, h_params.data[h_typeval.data[i].type]);
        if (evaluator::needsCharge())
            eval.setCharge(charge_a, charge_b);

        bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

        // Bond energy must be halved
        bond_eng *= Scalar(0.5);

        if (evaluated)
            {
            // calculate virial
            if (compute_virial)
                {
                Scalar force_div2r = Scalar(1.0 / 2.0) * force_divr;
                bond_virial[0] = dx.x * dx.x * force_div2r; // xx
                bond_virial[1] = dx.x * dx.y * force_div2r; // xy
                bond_virial[2] = dx.x * dx.z * force_div2r; // xz
                bond_virial[3] = dx.y * dx.y * force_div2r; // yy
                bond_virial[4] = dx.y * dx.z * force_div2r; // yz
                bond_virial[5] = dx.z * dx.z * force_div2r; // zz
                }

            // add the force to the particles (only for non-ghost particles)
            if (idx_b < this->m_pdata->getN())
                {
                h_force.data[idx_b].x += force_divr * dx.x;
                h_force.data[idx_b].y += force_divr * dx.y;
                h_force.data[idx_b].z += force_divr * dx.z;
                h_force.data[idx_b].w += bond_eng;
                if (compute_virial)
                    for (unsigned int i = 0; i < 6; i++)
                        h_virial.data[i * this->m_virial_pitch + idx_b] += bond_virial[i];
                }

            if (idx_a < this->m_pdata->getN())
                {
                h_force.data[idx_a].x -= force_divr * dx.x;
                h_force.data[idx_a].y -= force_divr * dx.y;
                h_force.data[idx_a].z -= force_divr * dx.z;
                h_force.data[idx_a].w += bond_eng;
                if (compute_virial)
                    for (unsigned int i = 0; i < 6; i++)
                        h_virial.data[i * this->m_virial_pitch + idx_a] += bond_virial[i];
                }
            }
        else
            {
            this->m_exec_conf->msg->error()
                << "bond." << evaluator::getName() << ": bond out of bounds" << std::endl
                << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator, class Bonds>
CommFlags ImagePotentialBond<evaluator, Bonds>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = md::PotentialBond<evaluator, Bonds>::getRequestedCommFlags(timestep);
    flags[comm_flag::image] = 1;

    return flags;
    }
#endif

namespace detail
    {
//! Exports the ImagePotentialBond class to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_ImagePotentialBond(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ImagePotentialBond<T, BondData>,
                     md::PotentialBond<T, BondData>,
                     std::shared_ptr<ImagePotentialBond<T, BondData>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#endif
