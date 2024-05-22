// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ReversePerturbationFlow.cc
 * \brief Definition of reverse perturbation flow updater
 */

#include "ReversePerturbationFlow.h"
#include "ReversePerturbationUtilities.h"

namespace azplugins
{
/*!
 * \param sysdef SystemDefinition this updater will act on
 * \param group Group to operate on
 * \param num_swap Max number of swaps
 * \param slab_width Slab thickness
 * \param p_target target momentum for particle selection
 */
ReversePerturbationFlow::ReversePerturbationFlow(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group,
                                                 unsigned int num_swap,
                                                 Scalar slab_width,
                                                 Scalar p_target)
    : Updater(sysdef), m_group(group), m_num_swap(num_swap), m_slab_width(slab_width), m_p_target(p_target), m_momentum_exchange(0),
      m_num_lo(0), m_num_hi(0), m_update_slabs(true)
    {

    unsigned int group_size = m_group->getNumMembersGlobal();
    if (group_size == 0)
        {
        m_exec_conf->msg->warning() << "Creating a ReversePerturbationFlow with an empty group" << std::endl;
        }
    if (p_target < 0)
        {
        m_exec_conf->msg->warning() << "Creating a ReversePerturbationFlow with a negative target momentum" << std::endl;
        }
    m_exec_conf->msg->notice(5) << "Constructing ReversePerturbationFlow" << std::endl;

    // subscribe to box change signal, to ensure geometry is still valid
    m_pdata->getBoxChangeSignal().connect<ReversePerturbationFlow, &ReversePerturbationFlow::requestUpdateSlabs>(this);

    // allocate memory
    GPUArray<Scalar2> lo_idx(m_num_swap, m_pdata->getExecConf());
    m_layer_lo.swap(lo_idx);
    GPUArray<Scalar2> hi_idx(m_num_swap, m_pdata->getExecConf());
    m_layer_hi.swap(hi_idx);
    }

ReversePerturbationFlow::~ReversePerturbationFlow()
    {
    m_exec_conf->msg->notice(5) << "Destroying ReversePerturbationFlow" << std::endl;
    m_pdata->getBoxChangeSignal().disconnect<ReversePerturbationFlow, &ReversePerturbationFlow::requestUpdateSlabs>(this);
    }

/*!
 * \param new_num_swap Max number of swaps
 */
void ReversePerturbationFlow::setNswap(unsigned int new_num_swap)
    {
    if (new_num_swap > m_num_swap)
        {
        // allocate memory if needed
        GPUArray<Scalar2> lo_idx(new_num_swap, m_pdata->getExecConf());
        m_layer_lo.swap(lo_idx);
        GPUArray<Scalar2> hi_idx(new_num_swap, m_pdata->getExecConf());
        m_layer_hi.swap(hi_idx);
        }
    m_num_swap = new_num_swap;
    }

/*!
 * \param slab_width Slab thickness
 */
void ReversePerturbationFlow::setSlabWidth(Scalar slab_width)
    {
    // check that delta is bigger than zero
    if (slab_width <= Scalar(0))
        {
        m_exec_conf->msg->error() << "ReversePerturbationFlow: slab thickness " << slab_width
                                  << " needs to be bigger than zero." << std::endl;
        throw std::runtime_error("invalid slab thickness in ReversePerturbationFlow");
        }

    m_slab_width = slab_width;
    requestUpdateSlabs();
    }

void ReversePerturbationFlow::setSlabs()
    {
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 global_lo = global_box.getLo();
    const Scalar3 global_hi = global_box.getHi();

    const Scalar lo_pos = global_box.makeCoordinates(make_scalar3(0,0,0.25)).z;
    const Scalar hi_pos = global_box.makeCoordinates(make_scalar3(0,0,0.75)).z;
    const Scalar delta_half =  Scalar(0.5)*m_slab_width;
    m_hi_pos = make_scalar2(hi_pos - delta_half, hi_pos + delta_half);
    m_lo_pos = make_scalar2(lo_pos - delta_half, lo_pos + delta_half);

    // explicit check that the two regions for velocity swapping are inside
    // of the box entirely and don't overlap
    if (m_hi_pos.y >= global_hi.z)
        {
        m_exec_conf->msg->error() << "ReversePerturbationFlow: hi slab " << m_hi_pos.y
                                  << "is outside of box: " << global_hi.z<< "." << std::endl;
        throw std::runtime_error("slab outside simulation box in ReversePerturbationFlow");
        }
    if (m_lo_pos.x < global_lo.z)
        {
        m_exec_conf->msg->error() << "ReversePerturbationFlow: max slab " << m_lo_pos.y
                                  << "is outside of box: " << global_lo.z<< "." << std::endl;
        throw std::runtime_error("slab outside simulation box in ReversePerturbationFlow");
        }
    if (m_lo_pos.y > m_hi_pos.x)
        {
        m_exec_conf->msg->error() << "ReversePerturbationFlow: hi slab "
                                  << m_hi_pos.x << " and lo slab " <<
                                  m_lo_pos.y << " are overlapping." << std::endl;
        throw std::runtime_error("slab outside simulation box in ReversePerturbationFlow");
        }
    }

/*!
 * \param timestep Current time step of the simulation
 */
void ReversePerturbationFlow::update(unsigned int timestep)
    {
    // reset momentum exchange for this step
    m_momentum_exchange = Scalar(0);

    // don't do anything if this is an empty group
    const unsigned int group_size = m_group->getNumMembers();
    if(group_size == 0)
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

std::vector<std::string> ReversePerturbationFlow::getProvidedLogQuantities()
    {
    std::vector<std::string> ret {"rp_momentum"};
    return ret;
    }

/*!
 * \param quantity Name of the log quantity to get
 * \param timestep Current time step of the simulation
 */
Scalar ReversePerturbationFlow::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "rp_momentum")
        {
        return m_momentum_exchange;
        }
    else
        {
        m_exec_conf->msg->error() << "azplugins.flow.reverse_perturbation: " << quantity
                                  << " is not a valid log quantity." << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    return Scalar(0.0);
    }


/*!
 * Finds all particles in the "min" and "max" slab in z direction and
 * puts them into two GPUArrays, sorted by their momentum closest to p_targt in x direction.
 */
void ReversePerturbationFlow::findSwapParticles()
    {
    // get needed array handles. All in read mode, no particle is changed in this search function
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_idx(m_group->getIndexArray(), access_location::host, access_mode::read);

    // temporary pairs to hold particles in slab
    // first index is momentum in x dir - second index is particle tag
    detail::ReversePerturbationSorter cmp(m_p_target);
    std::map<Scalar,unsigned int,detail::ReversePerturbationSorter> particles_in_lo_slab(cmp);
    std::map<Scalar,unsigned int,detail::ReversePerturbationSorter> particles_in_hi_slab(cmp);

    // sort particles into their slab in z-direction and record momentum in x-direction and index
    // injecting into std::map will sort them from lowest to highest momentum
    for(unsigned int group_idx=0; group_idx < m_group->getNumMembers(); group_idx++)
        {
        unsigned int j = h_member_idx.data[group_idx];
        assert(j < m_pdata->getN());

        const Scalar4 vel = h_vel.data[j];
        const Scalar momentum = vel.x * vel.w;
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
void ReversePerturbationFlow::swapPairMomentum()
    {
    // get all the needed array handles. Only velocities needs to be readwrite
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),access_location::host,access_mode::readwrite);
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
        h_vel.data[lo_idx].x =  hi_momentum / h_vel.data[lo_idx].w;
        h_vel.data[hi_idx].x = lo_momentum / h_vel.data[hi_idx].w;
        }
    }

namespace detail
{

/*!
 * \param m Python module to export to
 */
void export_ReversePerturbationFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< ReversePerturbationFlow, std::shared_ptr<ReversePerturbationFlow> >(m, "ReversePerturbationFlow", py::base<Updater>())
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>, unsigned int, Scalar,Scalar>())
        .def_property("group", &ReversePerturbationFlow::getGroup, &ReversePerturbationFlow::setGroup)
        .def_property("Nswap", &ReversePerturbationFlow::getNswap, &ReversePerturbationFlow::setNswap)
        .def_property("target_momentum", &ReversePerturbationFlow::getTargetMomentum, &ReversePerturbationFlow::setTargetMomentum)
        .def_property("width", &ReversePerturbationFlow::getSlabWidth, &ReversePerturbationFlow::setSlabWidth);
    }
}

} // end namespace azplugins
