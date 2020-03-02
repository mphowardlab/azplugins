// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file MPCDReversePertubationFlow.cc
 * \brief Definition of reverse pertubation flow updater
 */

#include "MPCDReversePerturbationFlow.h"
#include "ReversePerturbationUtilities.h"

namespace azplugins
{
/*!
 * \param sysdata MPCD system data this updater will act on
 * \param num_swap Max number of swaps
 * \param slab_width Slab thickness
 * \param slab_distance Slab distance
 * \param p_target target momentum in x-direction for the two slabs
 */
MPCDReversePerturbationFlow::MPCDReversePerturbationFlow(std::shared_ptr<mpcd::SystemData> sysdata,
                                                         unsigned int num_swap,
                                                         Scalar slab_width,
                                                         Scalar slab_distance,
                                                         Scalar p_target)
    : Updater(sysdata->getSystemDefinition()), m_mpcd_sys(sysdata), m_mpcd_pdata(sysdata->getParticleData()), m_num_swap(num_swap), m_slab_width(slab_width),m_slab_distance(slab_distance), m_momentum_exchange(0),
      m_num_lo(0), m_num_hi(0), m_p_target(p_target), m_update_slabs(true)
    {
    unsigned int n_mpcd = m_mpcd_pdata->getN();
    if (n_mpcd == 0)
        {
        m_exec_conf->msg->warning() << "Creating a MPCDReversePerturbationFlow with no mpcd particles" << std::endl;
        }
    if (p_target < 0)
        {
        m_exec_conf->msg->warning() << "Creating a MPCDReversePerturbationFlow with negative target velocity" << std::endl;
        }

    m_exec_conf->msg->notice(5) << "Constructing MPCDReversePerturbationFlow" << std::endl;

    // subscribe to box change signal, to ensure geometry is still valid
    m_mpcd_sys->getSystemDefinition()->getParticleData()->getBoxChangeSignal().connect<MPCDReversePerturbationFlow, &MPCDReversePerturbationFlow::requestUpdateSlabs>(this);

    // allocate memory
    GPUArray<Scalar2> lo_idx(m_num_swap,m_exec_conf);
    m_layer_lo.swap(lo_idx);
    GPUArray<Scalar2> hi_idx(m_num_swap, m_exec_conf);
    m_layer_hi.swap(hi_idx);
    }

MPCDReversePerturbationFlow::~MPCDReversePerturbationFlow()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCDReversePerturbationFlow" << std::endl;
    m_mpcd_sys->getSystemDefinition()->getParticleData()->getBoxChangeSignal().disconnect<MPCDReversePerturbationFlow, &MPCDReversePerturbationFlow::requestUpdateSlabs>(this);
    }

/*!
 * \param new_num_swap Max number of swaps
 */
void MPCDReversePerturbationFlow::setNswap(unsigned int new_num_swap)
    {
    if (new_num_swap > m_num_swap)
        {
        // allocate memory if needed
        GPUArray<Scalar2> lo_idx(new_num_swap, m_exec_conf);
        m_layer_lo.swap(lo_idx);
        GPUArray<Scalar2> hi_idx(new_num_swap, m_exec_conf);
        m_layer_hi.swap(hi_idx);
        }
    m_num_swap = new_num_swap;
    }

//! Set the target velocity
void MPCDReversePerturbationFlow::setTargetMomentum(Scalar p_target)
    {
    m_p_target = p_target;
    }


/*!
 * \param slab_width Slab thickness
 */
void MPCDReversePerturbationFlow::setSlabWidth(Scalar slab_width)
    {
    // check that delta is bigger than zero
    if (slab_width <= Scalar(0))
        {
        m_exec_conf->msg->error() << "MPCDReversePerturbationFlow: slab thickness " << slab_width
                                  << " needs to be bigger than zero." << std::endl;
        throw std::runtime_error("invalid slab thickness in MPCDReversePerturbationFlow");
        }

    m_slab_width = slab_width;
    requestUpdateSlabs();
    }

/*!
 * \param slab_distance Slab distance
 */
void MPCDReversePerturbationFlow::setSlabDistance(Scalar slab_distance)
    {
    // check that delta is bigger than zero
    if (slab_distance <= Scalar(0))
        {
        m_exec_conf->msg->error() << "MPCDReversePerturbationFlow: slab distance " << slab_distance
                                  << " needs to be bigger than zero." << std::endl;
        throw std::runtime_error("invalid slab thickness in MPCDReversePerturbationFlow");
        }

    m_slab_distance = slab_distance;
    requestUpdateSlabs();
    }

void MPCDReversePerturbationFlow::setSlabs()
    {
    const BoxDim& global_box = m_mpcd_sys->getGlobalBox();
    const Scalar3 global_lo = global_box.getLo();
    const Scalar3 global_hi = global_box.getHi();

    const Scalar lo_pos = -m_slab_distance;
    const Scalar hi_pos = m_slab_distance;
    const Scalar delta_half =  Scalar(0.5)*m_slab_width;
    m_hi_pos = make_scalar2(hi_pos - delta_half, hi_pos + delta_half);
    m_lo_pos = make_scalar2(lo_pos - delta_half, lo_pos + delta_half);

    // explicit check that the two regions for velocity swapping are inside
    // of the box entirely and don't overlap
    if (m_hi_pos.y >= global_hi.z)
        {
        m_exec_conf->msg->error() << "MPCDReversePerturbationFlow: hi slab " << m_hi_pos.y
                                  << "is outside of box: " << global_hi.z<< "." << std::endl;
        throw std::runtime_error("slab outside simulation box in MPCDReversePerturbationFlow");
        }
    if (m_lo_pos.x < global_lo.z)
        {
        m_exec_conf->msg->error() << "MPCDReversePerturbationFlow: max slab " << m_lo_pos.y
                                  << "is outside of box: " << global_lo.z<< "." << std::endl;
        throw std::runtime_error("slab outside simulation box in MPCDReversePerturbationFlow");
        }
    if (m_lo_pos.y > m_hi_pos.x)
        {
        m_exec_conf->msg->error() << "MPCDReversePerturbationFlow: hi slab "
                                  << m_hi_pos.x << " and lo slab "
                                  << m_lo_pos.y << " are overlapping." << std::endl;
        throw std::runtime_error("slab outside simulation box in MPCDReversePerturbationFlow");
        }
    }

/*!
 * \param timestep Current time step of the simulation
 */
void MPCDReversePerturbationFlow::update(unsigned int timestep)
    {
    // reset momentum exchange for this step
    m_momentum_exchange = Scalar(0);

    // don't do anything if this is an empty group
    const unsigned int n_mpcd = m_mpcd_pdata->getN();
    if(n_mpcd == 0)
        return;

    // if slabs have changed, update them
    if(m_update_slabs)
        {
        setSlabs();
        m_update_slabs = false;
        }

    findSwapParticles();
    swapPairMomentum();
    }

std::vector<std::string> MPCDReversePerturbationFlow::getProvidedLogQuantities()
    {
    std::vector<std::string> ret {"rp_momentum"};
    return ret;
    }

/*!
 * \param quantity Name of the log quantity to get
 * \param timestep Current time step of the simulation
 */
Scalar MPCDReversePerturbationFlow::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "rp_momentum")
        {
        return m_momentum_exchange;
        }
    else
        {
        m_exec_conf->msg->error() << "azplugins.mpcd.reverse_pertubation: " << quantity
                                  << " is not a valid log quantity." << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    return Scalar(0.0);
    }

/*!
 * Finds all particles in the "min" and "max" slab in z direction and
 * puts them into two GPUArrays, sorted by their momentum closest to p_target in x direction.
 */
void MPCDReversePerturbationFlow::findSwapParticles()
    {
    // get needed array handles. All in read mode, no particle is changed in this search function
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);

    // sort pairs according to their sign and momentum
    // temporary pairs to hold particles in slab
    // first index is momentum in x dir - second index is particle tag
    detail::ReversePerturbationSorter cmp(m_p_target);
    std::map<Scalar,unsigned int,detail::ReversePerturbationSorter> particles_in_lo_slab(cmp);
    std::map<Scalar,unsigned int,detail::ReversePerturbationSorter> particles_in_hi_slab(cmp);

    // sort particles into their slab in z-direction and record momentum in x-direction and index
    // injecting into std::map will sort them from lowest to highest momentum
    for(unsigned int j=0; j < m_mpcd_pdata->getN(); j++)
        {
        const Scalar4 vel = h_vel.data[j];
        const Scalar momentum = vel.x * m_mpcd_pdata->getMass();
        const Scalar z = h_pos.data[j].z;
        if (m_lo_pos.x <= z && z < m_lo_pos.y && momentum > 0) // lower slab, search for max momentum
            {
            particles_in_lo_slab.insert(std::make_pair(momentum,j));
            }
        else if (m_hi_pos.x <= z && z < m_hi_pos.y && momentum < 0) // higher slab, search for min momentum
            {
            particles_in_hi_slab.insert(std::make_pair(momentum,j));
            }
        }

    // find the fastest particles (in +x) in the lo slab
    ArrayHandle<Scalar2> h_layer_lo(m_layer_lo, access_location::host, access_mode::overwrite);
    m_num_lo = 0;
    for (auto iter = particles_in_lo_slab.rbegin(); iter != particles_in_lo_slab.rend(); ++iter)
        {
        if (m_num_lo < m_num_swap)
            {
            // (x,y) = (index, momentum)
            h_layer_lo.data[m_num_lo++] = make_scalar2(__int_as_scalar(iter->second), iter->first);

            }
        }

    // find the slowest particles (in -x) in the high slab
    ArrayHandle<Scalar2> h_layer_hi(m_layer_hi, access_location::host, access_mode::overwrite);
    m_num_hi = 0;
    for (auto iter = particles_in_hi_slab.begin(); iter != particles_in_hi_slab.end(); ++iter)
        {
        if (m_num_hi < m_num_swap)
            {
            h_layer_hi.data[m_num_hi++] = make_scalar2(__int_as_scalar(iter->second), iter->first);
            }
        }
    }

/*!
 * Takes the up to m_num_swap pairs from the top and bottom slab and swaps
 * their velocities.
 */
void MPCDReversePerturbationFlow::swapPairMomentum()
    {
    // get all the needed array handles. Only velocities needs to be readwrite
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),access_location::host,access_mode::readwrite);
    ArrayHandle<Scalar2> h_layer_lo(m_layer_lo, access_location::host, access_mode::read);
    ArrayHandle<Scalar2> h_layer_hi(m_layer_hi, access_location::host, access_mode::read);

    // find out how many pairs there are
    unsigned int num_pairs = std::min(m_num_swap, std::min(m_num_lo, m_num_hi));

    // swap num_pairs (up to m_num_swap)
    m_momentum_exchange = 0;
    for (unsigned int i = 0; i < num_pairs; i++)
        {
        // load data from the lo slab
        const Scalar2 lo = h_layer_lo.data[i];
        const unsigned int lo_idx = __scalar_as_int(lo.x);
        const Scalar lo_momentum = lo.y;

        // load data from the hi slab
        const Scalar2 hi = h_layer_hi.data[i];
        const unsigned int hi_idx = __scalar_as_int(hi.x);
        const Scalar hi_momentum = hi.y;
        // swap velocities & calculate momentum exchange
        m_momentum_exchange += (lo_momentum - hi_momentum);
        h_vel.data[lo_idx].x =  hi_momentum / m_mpcd_pdata->getMass();
        h_vel.data[hi_idx].x =  lo_momentum / m_mpcd_pdata->getMass();
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_MPCDReversePerturbationFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< MPCDReversePerturbationFlow, std::shared_ptr<MPCDReversePerturbationFlow> >(m, "MPCDReversePerturbationFlow", py::base<Updater>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, Scalar,Scalar,Scalar>())
        .def_property("Nswap", &MPCDReversePerturbationFlow::getNswap, &MPCDReversePerturbationFlow::setNswap)
        .def_property("width", &MPCDReversePerturbationFlow::getSlabWidth, &MPCDReversePerturbationFlow::setSlabWidth)
        .def_property("distance", &MPCDReversePerturbationFlow::getSlabDistance, &MPCDReversePerturbationFlow::setSlabDistance)
        .def_property("target_momentum", &MPCDReversePerturbationFlow::getTargetMomentum, &MPCDReversePerturbationFlow::setTargetMomentum);
    }
}

} // end namespace azplugins
