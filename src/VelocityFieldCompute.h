// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VELOCITY_FIELD_COMPUTE_H_
#define AZPLUGINS_VELOCITY_FIELD_COMPUTE_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ParticleDataLoader.h"

#include "hoomd/Compute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/SystemDefinition.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace azplugins
    {

//! Compute a velocity field in a region of space using histograms
template<class BinOpT> class PYBIND11_EXPORT VelocityFieldCompute : public Compute
    {
    public:
    //! Constructor
    VelocityFieldCompute(std::shared_ptr<SystemDefinition> sysdef,
                         uint3 num_bins,
                         Scalar3 lower_bounds,
                         Scalar3 upper_bounds,
                         std::shared_ptr<ParticleGroup> group,
                         bool include_mpcd_particles)
        : Compute(sysdef), m_num_bins(num_bins), m_lower_bounds(lower_bounds),
          m_upper_bounds(upper_bounds), m_group(group),
          m_include_mpcd_particles(include_mpcd_particles), m_request_compute(false)
        {
#ifdef ENABLE_MPI
        MPI_Op_create(&VelocityFieldCompute<BinOpT>::reduceScalar3, true, &m_reduce_scalar3);
#endif // ENABLE_MPI
        }

    //! Destructor
    virtual ~VelocityFieldCompute()
        {
#ifdef ENABLE_MPI
        MPI_Op_free(&m_reduce_scalar3);
#endif // ENABLE_MPI
        }

    //! Compute center-of-mass velocity of group
    void compute(uint64_t timestep) override;

    // Get number of bins
    uint3 getNumBins() const
        {
        return m_num_bins;
        }

    //! Set number of bins
    void setNumBins(const uint3& num_bins)
        {
        m_num_bins = num_bins;
        m_binning_op.reset();
        m_request_compute = true;
        }

    //! Get lower bounds for binning
    Scalar3 getLowerBounds() const
        {
        return m_lower_bounds;
        }

    //! Set lower bounds for binning
    void setLowerBounds(const Scalar3& lower_bounds)
        {
        m_lower_bounds = lower_bounds;
        m_binning_op.reset();
        m_request_compute = true;
        }

    //! Get upper bounds for binning
    Scalar3 getUpperBounds() const
        {
        return m_upper_bounds;
        }

    //! Set upper bounds for binning
    void setUpperBounds(const Scalar3& upper_bounds)
        {
        m_upper_bounds = upper_bounds;
        m_binning_op.reset();
        m_request_compute = true;
        }

    //! Get group of HOOMD particles in calculation (may be nullptr)
    std::shared_ptr<ParticleGroup> getGroup() const
        {
        return m_group;
        }

    //! Whether to include MPCD particles in calculation
    bool includeMPCDParticles() const
        {
        return m_include_mpcd_particles;
        }

    const GPUArray<Scalar>& getMasses() const
        {
        return m_mass;
        }

    const GPUArray<Scalar3>& getMomenta() const
        {
        return m_momentum;
        }

    const std::vector<Scalar3>& getVelocities() const
        {
        return m_velocity;
        }

    //! Compact number of bins to remove zeros
    std::vector<unsigned int> getCompactShape() const
        {
        std::vector<unsigned int> compact_num_bins;
        if (m_num_bins.x > 0)
            {
            compact_num_bins.push_back(m_num_bins.x);
            }
        if (m_num_bins.y > 0)
            {
            compact_num_bins.push_back(m_num_bins.y);
            }
        if (m_num_bins.z > 0)
            {
            compact_num_bins.push_back(m_num_bins.z);
            }
        return compact_num_bins;
        }

    protected:
    uint3 m_num_bins;                       //!< Number of bins (0 allowed)
    Scalar3 m_lower_bounds;                 //!< Lower bounds for binning
    Scalar3 m_upper_bounds;                 //!< Upper bounds for binning
    std::shared_ptr<ParticleGroup> m_group; //!< Particle group
    bool m_include_mpcd_particles;          //!< If true, include the MPCD particles

    std::shared_ptr<BinOpT> m_binning_op; //!< Binning operation
    GPUArray<Scalar> m_mass;              //!< Total mass in bin
    GPUArray<Scalar3> m_momentum;         //!< Total momentum in bin
    std::vector<Scalar3> m_velocity;      //!< Mass averaged velocity in bin

    bool shouldCompute(uint64_t timestep) override;

    //! Bin particles
    virtual void binParticles();

    private:
    bool m_request_compute;

    template<class LoadOpT>
    static void addParticleToBin(Scalar* masses,
                                 Scalar3* momenta,
                                 const LoadOpT& load_op,
                                 const BinOpT& bin_op,
                                 const BoxDim& global_box,
                                 unsigned int idx)
        {
        Scalar3 position, momentum;
        Scalar mass;
        load_op(position, momentum, mass, idx);
        momentum *= mass;

        // ensure particle is inside global box
        int3 img;
        global_box.wrap(position, img);

        uint3 bin = make_uint3(0, 0, 0);
        Scalar3 transformed_momentum = make_scalar3(0, 0, 0);
        const bool binned = bin_op.bin(bin, transformed_momentum, position, momentum);
        if (!binned)
            {
            return;
            }
        const auto bin_idx = bin_op.ravelBin(bin);

        masses[bin_idx] += mass;
        momenta[bin_idx] += transformed_momentum;
        }

#ifdef ENABLE_MPI
    MPI_Op m_reduce_scalar3; //!< MPI operation to reduce a vector
    static void reduceScalar3(void* in, void* out, int* len, MPI_Datatype* datatype)
        {
        Scalar3* a = static_cast<Scalar3*>(in);
        Scalar3* b = static_cast<Scalar3*>(out);
        for (int i = 0; i < *len; ++i)
            {
            b[i] += a[i];
            }
        }
#endif // ENABLE_MPI
    };

template<class BinOpT> void VelocityFieldCompute<BinOpT>::compute(uint64_t timestep)
    {
    if (!shouldCompute(timestep))
        return;

    // create binning operation and memory (triggers on any resize)
    if (!m_binning_op)
        {
        m_binning_op = std::make_shared<BinOpT>(m_num_bins, m_lower_bounds, m_upper_bounds);

        const size_t total_num_bins = m_binning_op->getTotalNumBins();

        GPUArray<Scalar> mass(total_num_bins, m_exec_conf);
        m_mass.swap(mass);

        GPUArray<Scalar3> momentum(total_num_bins, m_exec_conf);
        m_momentum.swap(momentum);

        m_velocity.resize(total_num_bins);
        }

    // assign particle quantities to bins
    binParticles();

    // reduce the sums in MPI
#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        const size_t total_num_bins = m_binning_op->getTotalNumBins();
        if (total_num_bins > std::numeric_limits<int>::max())
            {
            throw std::runtime_error("Number of bins exceeds maximum value of integer");
            }

        ArrayHandle<Scalar> h_mass(m_mass, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_momentum(m_momentum, access_location::host, access_mode::readwrite);

        const auto mpi_comm = m_exec_conf->getMPICommunicator();
        MPI_Allreduce(MPI_IN_PLACE,
                      h_mass.data,
                      static_cast<int>(total_num_bins),
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_momentum.data,
                      static_cast<int>(total_num_bins),
                      m_exec_conf->getMPIConfig()->getScalar3Datatype(),
                      m_reduce_scalar3,
                      mpi_comm);
        }
#endif // ENABLE_MPI

        // normalize total momentum of bin by mass to get mass-averaged velocity
        {
        ArrayHandle<Scalar> h_mass(m_mass, access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_momentum(m_momentum, access_location::host, access_mode::read);
        for (size_t idx = 0; idx < m_binning_op->getTotalNumBins(); ++idx)
            {
            const Scalar bin_mass = h_mass.data[idx];
            if (bin_mass > Scalar(0))
                {
                m_velocity[idx] = h_momentum.data[idx] / bin_mass;
                }
            else
                {
                m_velocity[idx] = make_scalar3(0, 0, 0);
                }
            }
        }
    }

template<class BinOpT> bool VelocityFieldCompute<BinOpT>::shouldCompute(uint64_t timestep)
    {
    bool result = Compute::shouldCompute(timestep);

    if (result)
        {
        // request has been satisfed already
        m_request_compute = false;
        }
    else if (m_request_compute)
        {
        m_request_compute = false;
        m_last_computed = timestep;
        result = true;
        }

    return result;
    }

template<class BinOpT> void VelocityFieldCompute<BinOpT>::binParticles()
    {
    const BinOpT& bin_op = *m_binning_op;
    ArrayHandle<Scalar> h_mass(m_mass, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_momentum(m_momentum, access_location::host, access_mode::overwrite);

    const size_t total_num_bins = bin_op.getTotalNumBins();
    memset(h_mass.data, 0, sizeof(Scalar) * total_num_bins);
    memset(h_momentum.data, 0, sizeof(Scalar3) * total_num_bins);

    if (m_group)
        {
        const unsigned int N = m_group->getNumMembers();
        ArrayHandle<unsigned int> h_index(m_group->getIndexArray(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        detail::LoadHOOMDGroupPositionVelocityMass load_op(h_pos.data, h_vel.data, h_index.data);
        const BoxDim& global_box = m_pdata->getGlobalBox();

        for (unsigned int idx = 0; idx < N; ++idx)
            {
            addParticleToBin(h_mass.data, h_momentum.data, load_op, bin_op, global_box, idx);
            }
        }

#ifdef BUILD_MPCD
    if (m_include_mpcd_particles)
        {
        auto mpcd_pdata = m_sysdef->getMPCDParticleData();
        const unsigned int N = mpcd_pdata->getN();
        ArrayHandle<Scalar4> h_pos(mpcd_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<Scalar4> h_vel(mpcd_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        detail::LoadMPCDPositionVelocityMass load_op(h_pos.data, h_vel.data, mpcd_pdata->getMass());
        const BoxDim& global_box = m_pdata->getGlobalBox();

        for (unsigned int idx = 0; idx < N; ++idx)
            {
            addParticleToBin(h_mass.data, h_momentum.data, load_op, bin_op, global_box, idx);
            }
        }
#endif // BUILD_MPCD
    }

namespace detail
    {
template<class BinOpT>
void export_VelocityFieldCompute(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<VelocityFieldCompute<BinOpT>,
                     Compute,
                     std::shared_ptr<VelocityFieldCompute<BinOpT>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            uint3,
                            Scalar3,
                            Scalar3,
                            std::shared_ptr<ParticleGroup>,
                            bool>())
        .def_property(
            "num_bins",
            [](const VelocityFieldCompute<BinOpT>& self)
            {
                const auto num_bins = self.getNumBins();
                return pybind11::make_tuple(num_bins.x, num_bins.y, num_bins.z);
            },
            [](VelocityFieldCompute<BinOpT>& self, const pybind11::tuple& num_bins)
            {
                self.setNumBins(make_uint3(pybind11::cast<unsigned int>(num_bins[0]),
                                           pybind11::cast<unsigned int>(num_bins[1]),
                                           pybind11::cast<unsigned int>(num_bins[2])));
            })
        .def_property(
            "lower_bounds",
            [](const VelocityFieldCompute<BinOpT>& self)
            {
                const auto lower_bounds = self.getLowerBounds();
                return pybind11::make_tuple(lower_bounds.x, lower_bounds.y, lower_bounds.z);
            },
            [](VelocityFieldCompute<BinOpT>& self, const pybind11::tuple& lower_bounds)
            {
                self.setLowerBounds(make_scalar3(pybind11::cast<Scalar>(lower_bounds[0]),
                                                 pybind11::cast<Scalar>(lower_bounds[1]),
                                                 pybind11::cast<Scalar>(lower_bounds[2])));
            })
        .def_property(
            "upper_bounds",
            [](const VelocityFieldCompute<BinOpT>& self)
            {
                const auto upper_bounds = self.getUpperBounds();
                return pybind11::make_tuple(upper_bounds.x, upper_bounds.y, upper_bounds.z);
            },
            [](VelocityFieldCompute<BinOpT>& self, const pybind11::tuple& upper_bounds)
            {
                self.setUpperBounds(make_scalar3(pybind11::cast<Scalar>(upper_bounds[0]),
                                                 pybind11::cast<Scalar>(upper_bounds[1]),
                                                 pybind11::cast<Scalar>(upper_bounds[2])));
            })
        .def_property_readonly("filter",
                               [](const VelocityFieldCompute<BinOpT>& self)
                               {
                                   auto group = self.getGroup();
                                   return (group) ? group->getFilter()
                                                  : std::shared_ptr<hoomd::ParticleFilter>();
                               })
        .def_property_readonly("include_mpcd_particles",
                               &VelocityFieldCompute<BinOpT>::includeMPCDParticles)
        .def_property_readonly("velocities",
                               [](pybind11::object& obj)
                               {
                                   auto self = obj.cast<VelocityFieldCompute<BinOpT>*>();

                                   // shape is the number of bins with last dim 3 for the vector
                                   const auto num_bins = self->getCompactShape();
                                   std::vector<ssize_t> shape(num_bins.size() + 1);
                                   std::copy(num_bins.begin(), num_bins.end(), shape.begin());
                                   shape[shape.size() - 1] = 3;

                                   return pybind11::array(shape,
                                                          reinterpret_cast<const Scalar*>(
                                                              self->getVelocities().data()),
                                                          obj);
                               });
    }
    } // namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_VELOCITY_FIELD_COMPUTE_H_
