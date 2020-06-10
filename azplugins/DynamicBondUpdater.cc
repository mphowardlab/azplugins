// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file DynamicBondUpdater.cc
 * \brief Definition of DynamicBondUpdater
 */

#include "DynamicBondUpdater.h"

namespace azplugins
{


/*!
 * \param sysdef System definition
 * \param inside_type Type id of particles inside region
 * \param outside_type Type id of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 */
DynamicBondUpdater::DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<NeighborList> nlist,
                         std::shared_ptr<ParticleGroup> group_1,
                         std::shared_ptr<ParticleGroup> group_2,
                         const Scalar r_cutsq,
                         unsigned int bond_type,
                         unsigned int bond_reservoir_type,
                         unsigned int max_bonds_group_1,
                         unsigned int max_bonds_group_2)
        : Updater(sysdef), m_nlist(nlist), m_group_1(group_1), m_group_2(group_2), m_r_cutsq(r_cutsq),
         m_bond_type(bond_type),m_bond_reservoir_type(bond_reservoir_type),m_max_bonds_group_1(max_bonds_group_1),m_max_bonds_group_2(max_bonds_group_2)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;

    assert(m_nlist);

    m_bond_data = m_sysdef->getBondData();
    // allocate memory for the number of current bonds array
    GPUArray<unsigned int> counts((int)m_pdata->getN(), m_exec_conf);
    m_curr_num_bonds.swap(counts);

    // allocate a max size for all possible pairs - is there a better way to do this?
    const unsigned int size = m_group_1->getNumMembers()*m_max_bonds_group_1;
    GPUArray<Scalar2> possible_bonds(size, m_exec_conf);
    m_possible_bonds.swap(possible_bonds);

    calculateCurrentBonds();
    checkSystemSetup();
    }

DynamicBondUpdater::~DynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdater" << std::endl;

    }

void DynamicBondUpdater::calculateCurrentBonds()
{

    ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::readwrite);

    m_reservoir_size=0;

    // this should make the simulation also restartable, already existing bonds of m_bond_type will be registered
    const unsigned int size = (unsigned int)m_bond_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        //lookup current bond type
        unsigned int type = m_bond_data->getTypeByIndex(i);

        if(type==m_bond_reservoir_type)
            {
            ++m_reservoir_size;
            }
        if (type==m_bond_type)
            {
            // lookup the tag of each of the particles participating in the bond
            const BondData::members_t bond = m_bond_data->getMembersByIndex(i);

            unsigned int tag_i = bond.tag[0];
            unsigned int tag_j = bond.tag[1];

            // add this bond to the book keeping arrays and the map of all exisitng bonds
            ++h_curr_num_bonds.data[tag_i];
            ++h_curr_num_bonds.data[tag_j];
            // saving exisitng bonds in both directions. This should make the bond finding algorithm safe regardless of
            // the storage mode of the neighbour list. 
            m_all_existing_bonds[{tag_i,tag_j}] = 1;
            m_all_existing_bonds[{tag_i,tag_j}] = 1;
            }
        }
}

void DynamicBondUpdater::checkSystemSetup()
{

    if (m_bond_type >= m_bond_data -> getNTypes())
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: bond type id " << m_bond_type
                                  << " is not a valid bond type." << std::endl;
        throw std::runtime_error("Invalid bond type for DynamicBondUpdater");
        }


    if (m_bond_reservoir_type >= m_bond_data -> getNTypes())
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: bond type id " << m_bond_reservoir_type
                                  << " is not a valid bond type." << std::endl;
        throw std::runtime_error("Invalid bond type for DynamicBondUpdater");
        }

    //ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::readwrite);

    if (m_reservoir_size==0)
        {
        m_exec_conf->msg->error() << "DynamicBondUpdater: Bond reservoir size is zero." << std::endl;
        throw std::runtime_error("DynamicBondUpdater: Bond reservoir size must be larger than zero.");
        }

}

/*!
 * \param timestep Timestep update is called
 */
void DynamicBondUpdater::update(unsigned int timestep)
    {

    //calculateCurrentBonds();
    findPotentialBondPairs(timestep);

    formBondPairs(timestep);

    }

/*!
 * \param timestep Timestep update is called
 */
void DynamicBondUpdater::findPotentialBondPairs(unsigned int timestep)
    {

    // start by updating the neighborlist
    //m_nlist->addExclusionsFromBonds();
    m_nlist->compute(timestep);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // neighbour list information
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);
    // box
    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::read);

    // temp vector storage of possible bonds found
    std::vector<Scalar2> possible_bonds;

    // for each particle in group_1 - parallelize the outer loop on the GPU?
    for (unsigned int i = 0; i < m_group_1->getNumMembers(); ++i)
        {
        // get particle index
        const unsigned int idx_i = m_group_1->getMemberIndex(i);
        const unsigned int tag_i = h_tag.data[idx_i];
        //const unsigned int idx_i = h_group_1.data[i];

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[idx_i];
        const unsigned int size = (unsigned int)h_n_neigh.data[idx_i];
        unsigned int num_possible_bonds_i = 0;

        // this way of finding neighbours introduces some artifacts if the particle index and spatial positions are
        // correlated, because the neighbor list returns neighbours ordered by index, so the particles with lower index
        // get bonded first if the number of possible bonds exceedst the bond limit. if there is spatial correlation
        // this could lead to artifacts in the spatial configuration as well. Is it worth it to shuffle the order?
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor
            const unsigned int idx_j = h_nlist.data[myHead + k];
            const unsigned int tag_j = h_tag.data[idx_j];

            bool is_in_group_2 = m_group_2->isMember(idx_j); // needs to be replaced with something else on the GPU

            const unsigned int current_bonds_on_j = h_curr_num_bonds.data[tag_j];
            const unsigned int current_bonds_on_i = h_curr_num_bonds.data[tag_i];

            // check that this bond doesn't already exists, second particle is in second group, and max number of bonds is not reached for both
            if (is_in_group_2
                && m_all_existing_bonds.count({tag_i, tag_j}) ==0
                && m_all_existing_bonds.count({tag_j, tag_i}) ==0
                && current_bonds_on_j < m_max_bonds_group_2
                && num_possible_bonds_i < m_max_bonds_group_1-current_bonds_on_i )
                {
                //  caclulate distance squared
                const Scalar3 pi = make_scalar3(h_pos.data[idx_i].x, h_pos.data[idx_i].y, h_pos.data[idx_i].z);
                const Scalar3 pj = make_scalar3(h_pos.data[idx_j].x, h_pos.data[idx_j].y, h_pos.data[idx_j].z);
                Scalar3 dx = pi - pj;
                dx = box.minImage(dx);
                const Scalar rsq = dot(dx, dx);

                if (rsq < m_r_cutsq)
                    {
                    possible_bonds.push_back(make_scalar2(tag_i,tag_j));
                    //std::cout<< "added bond to possible list "<< idx_i << " "<< idx_j << "tag "<< tag_i << " "<< tag_j<< std::endl;
                    ++num_possible_bonds_i;
                    }

                }
            }


        }

    // reset possible bond list
    ArrayHandle<Scalar2> h_possible_bonds(m_possible_bonds, access_location::host, access_mode::overwrite);
    const unsigned int size = m_group_1->getNumMembers()*m_max_bonds_group_1;
    memset((void*)h_possible_bonds.data,-1.0,sizeof(Scalar2)*size);

    // Before we copy the possible_bonds vector content to the h_possible_bonds array,  we need count number of bonds
    // formed towards particles in group_2 (second entry in possible bonds) because there could be too many.
    // The group_1 bonds should be okay because we are able to check in the for loop above.

     // a temp map which holds count of each encountered particle tag in group_2
    std::unordered_map<int, size_t> count_group_2_possible_bonds;

    // iterate over all possible bonds and use the unordered_map to count occurences, if occurences is larger than max_bonds_group_2
    // then don't copy that entry into the h_possible_bonds Array
    unsigned int current = 0;
    for (auto i = possible_bonds.begin(); i != possible_bonds.end(); ++i)
        {
        unsigned int tag_j = i->y;
        ++count_group_2_possible_bonds[tag_j];
        const unsigned int current_bonds_on_j = h_curr_num_bonds.data[tag_j];
        if(count_group_2_possible_bonds[tag_j] + current_bonds_on_j < m_max_bonds_group_2)
            {
            h_possible_bonds.data[current]=make_scalar2(i->x,i->y);
            ++current;
            }
        }

     m_curr_bonds_to_form = current;

    }


void DynamicBondUpdater::formBondPairs(unsigned int timestep)
    {


    ArrayHandle<Scalar2> h_possible_bonds(m_possible_bonds, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_curr_num_bonds(m_curr_num_bonds, access_location::host, access_mode::readwrite);

    ArrayHandle<typename BondData::members_t> h_bonds(m_bond_data->getMembersArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<typeval_t> h_typeval(m_bond_data->getTypeValArray(), access_location::host, access_mode::readwrite);
//    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // only do something if there are bonds to form and there are blank bonds left
    if (m_curr_bonds_to_form>0 && m_reservoir_size>0)
        {

        const unsigned int size = (unsigned int)m_bond_data->getN();
        unsigned int current = 0;
        for (unsigned int i = 0; i < size; i++)
            {

            unsigned int type = h_typeval.data[i].type;

            if (type == m_bond_reservoir_type && current < m_curr_bonds_to_form)
                {
                h_typeval.data[i].type = m_bond_type;
                unsigned int tag_i = h_possible_bonds.data[current].x;
                unsigned int tag_j = h_possible_bonds.data[current].y;

                h_bonds.data[i].tag[0] =  tag_i;
                h_bonds.data[i].tag[1] =  tag_j;

                //add new bond to the book keeping arrays and the map
                ++h_curr_num_bonds.data[tag_i];
                ++h_curr_num_bonds.data[tag_j];
                m_all_existing_bonds[{tag_i,tag_j}]=1;
                m_all_existing_bonds[{tag_i,tag_j}]=1;

                ++current;
                --m_reservoir_size;
                }
            }
        }
        m_curr_bonds_to_form=0;

    }


namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_DynamicBondUpdater(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< DynamicBondUpdater, std::shared_ptr<DynamicBondUpdater> >(m, "DynamicBondUpdater", py::base<Updater>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, std::shared_ptr<ParticleGroup>,
             std::shared_ptr<ParticleGroup>, const Scalar, unsigned int, unsigned int, unsigned int, unsigned int>());

        //.def_property("inside", &DynamicBondUpdater::getInsideType, &DynamicBondUpdater::setInsideType)
        //.def_property("outside", &DynamicBondUpdater::getOutsideType, &DynamicBondUpdater::setOutsideType)
        //.def_property("lo", &DynamicBondUpdater::getRegionLo, &DynamicBondUpdater::setRegionLo)
        //.def_property("hi", &DynamicBondUpdater::getRegionHi, &DynamicBondUpdater::setRegionHi);
    }
}

} // end namespace azplugins
