// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_GPU_H_
#define AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_GPU_H_

#include <memory>
#ifdef ENABLE_HIP

#include "ChebyshevAnisotropicPairPotential.h"
#include "ChebyshevAnisotropicPairPotentialGPU.cuh"

/*!\file ChebyshevAnisotropicPairPotentialGPU.h
   \brief Defines a GPU shell for the Chebyshev anisotropic pair potential.
   \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace azplugins
    {

//! Chebyshev anisotropic pair potential, templated on a symmetry reducer.
/*!
 * \tparam ShapeSymmetryT A class providing a static \c domain_upper[5] and domain_lower[5] array
 *         and a static \c reduce(theta, phi, alpha, beta, gamma) method that
 *         maps the angles into a fundamental domain and returns the applied
 *         rotation as a quaternion.  See \c ShapeSymmetry.h.
 */
template<class ShapeSymmetryT>
class PYBIND11_EXPORT ChebyshevAnisotropicPairPotentialGPU
    : public ChebyshevAnisotropicPairPotential<ShapeSymmetryT>
    {
    public:
    using Base = ChebyshevAnisotropicPairPotential<ShapeSymmetryT>;

    ChebyshevAnisotropicPairPotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<hoomd::md::NeighborList> nlist,
                                         const Scalar r_cut,
                                         const unsigned int* terms,
                                         const Scalar* coeffs,
                                         unsigned int Nterms,
                                         const Scalar* r0_data,
                                         const unsigned int* r0_shape);

    virtual ~ChebyshevAnisotropicPairPotentialGPU() { }

    protected:
    //! Override the CPU computeForces with a GPU kernel.
    void computeForces(uint64_t timestep) override;
    };

template<class ShapeSymmetryT>
ChebyshevAnisotropicPairPotentialGPU<ShapeSymmetryT>::ChebyshevAnisotropicPairPotentialGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar r_cut,
    const unsigned int* terms,
    const Scalar* coeffs,
    unsigned int Nterms,
    const Scalar* r0_data,
    const unsigned int* r0_shape)
    : Base(sysdef, nlist, r_cut, terms, coeffs, Nterms, r0_data, r0_shape)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a ChebyshevAnisotropicPairPotentialGPU with no GPU in the "
            << "execution configuration" << std::endl;
        throw std::runtime_error("Error initializing ChebyshevAnisotropicPairPotentialGPU");
        }
    }

template<class ShapeSymmetryT>
void ChebyshevAnisotropicPairPotentialGPU<ShapeSymmetryT>::computeForces(uint64_t timestep)
    {
    this->m_nlist->compute(timestep);

    this->m_force.zeroFill();
    this->m_torque.zeroFill();

    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar> d_r0_data(this->m_r0_data, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_r0_shape(this->m_r0_shape,
                                         access_location::device,
                                         access_mode::read);
    ArrayHandle<unsigned int> d_terms(this->m_terms, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_coeffs(this->m_coeffs, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(this->m_torque, access_location::device, access_mode::overwrite);

    Scalar domain_lower[5];
    Scalar domain_upper[5];
    for (unsigned int d = 0; d < 5; ++d)
        {
        domain_lower[d] = ShapeSymmetryT::domain_lower(d);
        domain_upper[d] = ShapeSymmetryT::domain_upper(d);
        }

    this->m_exec_conf->setDevice();

    kernel::gpu_compute_chebyshev_pair_forces<ShapeSymmetryT>(
        kernel::chebyshev_pair_args_t(d_force.data,
                                      d_torque.data,
                                      this->m_pdata->getN(),
                                      d_pos.data,
                                      d_orientation.data,
                                      d_tag.data,
                                      this->m_pdata->getGlobalBox(),
                                      d_n_neigh.data,
                                      d_nlist.data,
                                      d_head_list.data,
                                      d_r0_data.data,
                                      d_r0_shape.data,
                                      d_terms.data,
                                      d_coeffs.data,
                                      this->m_Nterms,
                                      this->m_r_cut,
                                      this->m_nlist_r_cut,
                                      domain_lower,
                                      domain_upper,
                                      256));

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

namespace detail
    {

//! Export one GPU subclass of ChebyshevAnisotropicPairPotential to python.
template<class ShapeSymmetryT>
void export_ChebyshevAnisotropicPairPotentialGPU(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    using NL = hoomd::md::NeighborList;
    using Pot = ChebyshevAnisotropicPairPotentialGPU<ShapeSymmetryT>;
    using Base = ChebyshevAnisotropicPairPotential<ShapeSymmetryT>;

    py::class_<Pot, Base, std::shared_ptr<Pot>>(m, name.c_str())
        .def(py::init(
            [](std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<NL> nlist,
               Scalar r_cut,
               py::array_t<unsigned int, py::array::c_style | py::array::forcecast> terms,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> coeffs,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> r0_data)
            {
                if (terms.ndim() != 2 || terms.shape(1) != Base::num_coordinates)
                    {
                    throw std::runtime_error("terms must have shape (Nterms,6).");
                    }

                const unsigned int Nterms = static_cast<unsigned int>(terms.shape(0));

                if (coeffs.ndim() != 1 || static_cast<unsigned int>(coeffs.shape(0)) != Nterms)
                    {
                    throw std::runtime_error("coeffs must have shape (Nterms,).");
                    }

                if (r0_data.ndim() != Base::num_angle_coordinates)
                    {
                    throw std::runtime_error("r0_data must be a 5D array.");
                    }

                std::array<unsigned int, Base::num_angle_coordinates> r0_shape;
                for (unsigned int k = 0; k < Base::num_angle_coordinates; ++k)
                    {
                    const auto dim = r0_data.shape(k);
                    if (dim < 2)
                        {
                        throw std::runtime_error("r0_data has invalid dimension(s).");
                        }
                    r0_shape[k] = static_cast<unsigned int>(dim);
                    }

                return std::make_shared<Pot>(sysdef,
                                             nlist,
                                             r_cut,
                                             terms.data(),
                                             coeffs.data(),
                                             Nterms,
                                             r0_data.data(),
                                             r0_shape.data());
            }));
    }

    } // end namespace detail
    } // namespace azplugins
    } // namespace hoomd

#endif // ENABLE_HIP
#endif // AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_GPU_H_
