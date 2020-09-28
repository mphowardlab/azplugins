// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward


#include "TwoStepSLLODLangevinFlow.h"
#include "hoomd/VectorMath.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

/*! \file TwoStepSLLODLangevinFlow.h
    \brief Contains code for the TwoStepSLLODLangevinFlow class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param noiseless If set true, there will be no translational noise (random force)
    \param suffix Suffix to attach to the end of log quantity names

*/
namespace azplugins
{
TwoStepSLLODLangevinFlow::TwoStepSLLODLangevinFlow(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
                           Scalar shear_rate,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless,
                           const std::string& suffix)
    : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda), m_shear_rate(shear_rate), m_reservoir_energy(0),  m_extra_energy_overdeltaT(0),
      m_tally(false), m_noiseless(noiseless)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepSLLODLangevinFlow" << std::endl;

    m_log_name = std::string("langevin_reservoir_energy") + suffix;
    }

TwoStepSLLODLangevinFlow::~TwoStepSLLODLangevinFlow()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepSLLODLangevinFlow" << std::endl;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepSLLODLangevinFlow::getProvidedLogQuantities()
    {
    std::vector<std::string> result;
    if (m_tally)
        result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quantity logged here
*/

Scalar TwoStepSLLODLangevinFlow::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (m_tally && quantity == m_log_name)
        {
        my_quantity_flag = true;
        return m_reservoir_energy+m_extra_energy_overdeltaT*m_deltaT;
        }
    else
        return Scalar(0);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepSLLODLangevinFlow::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("SLLOD Langevin step 1");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    // box deformation: update tilt factor of global box
    bool flipped = deformGlobalBox();

    BoxDim global_box = m_pdata->getGlobalBox();
    const Scalar3 global_hi = global_box.getHi();
    const Scalar3 global_lo = global_box.getLo();

    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar dx = h_vel.data[j].x*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT*m_deltaT;
        Scalar dy = h_vel.data[j].y*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT*m_deltaT;
        Scalar dz = h_vel.data[j].z*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT*m_deltaT;

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;
        // particles may have been moved slightly outside the box by the above steps, wrap them back into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepSLLODLangevinFlow::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("SLLOD Langevin step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    const unsigned int D = Scalar(m_sysdef->getNDimensions());

    // energy transferred over this time step
    Scalar bd_energy_transfer = 0;

    // a(t+deltaT) gets modified with the bd forces
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        hoomd::RandomGenerator rng(RNGIdentifier::TwoStepSLLODLangevinFlow, m_seed, ptag, timestep);

        // first, calculate the BD forces
        // Generate three random numbers
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }

        // compute the bd force
        Scalar coeff = fast::sqrt(Scalar(6.0) *gamma*currentTemp/m_deltaT);
        if (m_noiseless)
            coeff = Scalar(0.0);
        Scalar bd_fx = rx*coeff - gamma*h_vel.data[j].x;
        Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
        Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

        if (D < 3)
            bd_fz = Scalar(0.0);

        // then, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
        h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
        h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;

        // tally the energy transfer from the bd thermal reservoir to the particles
        if (m_tally) bd_energy_transfer += bd_fx * h_vel.data[j].x + bd_fy * h_vel.data[j].y + bd_fz * h_vel.data[j].z;

        }

    // update energy reservoir
    if (m_tally)
        {
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &bd_energy_transfer, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        #endif
        m_reservoir_energy -= bd_energy_transfer*m_deltaT;
        m_extra_energy_overdeltaT = 0.5*bd_energy_transfer;
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

bool TwoStepSLLODLangevinFlow::deformGlobalBox()
{
  // box deformation: update tilt factor of global box
  BoxDim global_box = m_pdata->getGlobalBox();

  Scalar xy = global_box.getTiltFactorXY();
  Scalar yz = global_box.getTiltFactorYZ();
  Scalar xz = global_box.getTiltFactorXZ();

  xy += m_shear_rate * m_deltaT;
  bool flipped = false;
  if (xy > 1){
      xy = -1;
      flipped = true;
  }
  global_box.setTiltFactors(xy, xz, yz);
  m_pdata->setGlobalBox(global_box);
  return flipped;
}

namespace detail
{
void export_TwoStepSLLODLangevinFlow(pybind11::module& m)
    {
    pybind11::class_<TwoStepSLLODLangevinFlow, std::shared_ptr<TwoStepSLLODLangevinFlow> >(m, "TwoStepSLLODLangevinFlow", pybind11::base<TwoStepLangevinBase>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            Scalar,
                            unsigned int,
                            bool,
                            Scalar,
                            bool,
                            const std::string&>())
        .def("setTally", &TwoStepSLLODLangevinFlow::setTally)
        .def("setNoiseless", &TwoStepSLLODLangevinFlow::setNoiseless);
        ;
    }

} // end namespace detail
} // end namespace azplugins
