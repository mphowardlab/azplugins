// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file TwoStepSLLODNVTFlow.cc
 * \brief Declaration of SLLOD equation of motion with NVT Nosé-Hoover thermostat
 */


#include "TwoStepSLLODNVTFlow.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

namespace azplugins
{

/*! \file TwoStepSLLODNVTFlow.h
    \brief Contains code for the TwoStepSLLODNVTFlow class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepSLLODNVTFlow::TwoStepSLLODNVTFlow(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       std::shared_ptr<ComputeThermoSLLOD> thermo,
                       Scalar tau,
                       std::shared_ptr<Variant> T,
                       Scalar shear_rate,
                       const std::string& suffix)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_tau(tau), m_T(T), m_exp_thermo_fac(1.0), m_shear_rate(shear_rate)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepSLLODNVTFlow" << std::endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "azplugins.sllod_nvt(): tau set less than 0.0 in SLLODNVTUpdater" << std::endl;

    // set initial state
    if (!restartInfoTestValid(getIntegratorVariables(), "sllod_nvt", 4))
        {
        initializeIntegratorVariables();
        setValidRestart(false);
        }
    else
        {
        setValidRestart(true);
        }

    m_log_name = std::string("sllod_nvt_reservoir_energy") + suffix;

    setShearRate(m_shear_rate);

    }

TwoStepSLLODNVTFlow::~TwoStepSLLODNVTFlow()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepSLLODNVTFlow" << std::endl;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepSLLODNVTFlow::getProvidedLogQuantities()
    {
    std::vector<std::string> result;
    result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quantity logged here
*/

Scalar TwoStepSLLODNVTFlow::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {

    if (quantity == m_log_name)
        {
        my_quantity_flag = true;
        Scalar g = m_thermo->getNDOF();
        IntegratorVariables v = getIntegratorVariables();
        Scalar& xi = v.variable[0];
        Scalar& eta = v.variable[1];
        Scalar thermostat_energy = (Scalar) g * m_T->getValue(timestep) * (xi*xi*m_tau*m_tau / Scalar(2.0) + eta);

        return thermostat_energy;
        }
    else
        return Scalar(0);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepSLLODNVTFlow::integrateStepOne(unsigned int timestep)
    {

    if (m_group->getNumMembersGlobal() == 0)
        {
        m_exec_conf->msg->error() << "azplugins.sllod_nvt(): Integration group empty." << std::endl;
        throw std::runtime_error("Error during NVT integration.");
        }

    // box deformation: update tilt factor of global box
    bool flipped = deformGlobalBox();

    BoxDim global_box = m_pdata->getGlobalBox();
    const Scalar3 global_hi = global_box.getHi();
    const Scalar3 global_lo = global_box.getLo();

    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("SLLOD NVT step 1");

    // scope array handles for proper releasing before calling the thermo compute
    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load variables
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 pos = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 accel = h_accel.data[j];


        // remove flow field
        v.x -= m_shear_rate*pos.y;

        // rescale velocity
        v *= m_exp_thermo_fac;

        // apply sllod velocity correction
        v.x -= Scalar(0.5)*m_shear_rate*v.y*m_deltaT;

        // add flow field
        v.x += m_shear_rate*pos.y;

        // update velocity
        v += Scalar(0.5)*accel*m_deltaT;

        // update position
        pos += m_deltaT * v;

        // if box deformation caused a flip, wrap positions back into box
        if (flipped){
            pos.x *= -1;
        }

        // Periodic boundary correction to velocity:
        // if particle leaves from (+/-) y boundary it gets (-/+) velocity at boundary
        // note carefully that pair potentials dependent on differences in
        // velocities (e.g. DPD) are not yet explicitly supported.

        if (pos.y > global_hi.y) // crossed pbc in +y
        {
          v.x -= m_boundary_shear_velocity;//Scalar(2.0)*m_shear_rate*global_hi.y;
        }
        else if (pos.y < global_lo.y) // crossed pbc in -y
        {
          v.x += m_boundary_shear_velocity;//-= Scalar(2.0)*m_shear_rate*global_lo.y;
        }

        // store updated variables
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        h_pos.data[j].x = pos.x;
        h_pos.data[j].y = pos.y;
        h_pos.data[j].z = pos.z;
        }

    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // wrap the particles around the box
        box.wrap(h_pos.data[j], h_image.data[j]);
        }
    }

    // get temperature and advance thermostat
    advanceThermostat(timestep);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepSLLODNVTFlow::integrateStepTwo(unsigned int timestep)
    {

    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("SLLOD NVT step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // perform second half step of Nose-Hoover integration

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load velocity
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 accel = h_accel.data[j];
        Scalar3 net_force = make_scalar3(h_net_force.data[j].x,h_net_force.data[j].y,h_net_force.data[j].z);

        // first, calculate acceleration from the net force
        Scalar m = h_vel.data[j].w;
        Scalar minv = Scalar(1.0) / m;
        accel = net_force*minv;

        // update velocity
        v += Scalar(0.5)*accel*m_deltaT;

        // remove flow field
        v.x -= m_shear_rate*h_pos.data[j].y;

        // rescale velocity
        v *= m_exp_thermo_fac;

        // apply sllod velocity correction
        v.x -= Scalar(0.5)*m_shear_rate*v.y*m_deltaT;

        // add flow field
        v.x += m_shear_rate*h_pos.data[j].y;


        // store velocity
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        // store acceleration
        h_accel.data[j] = accel;
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

bool TwoStepSLLODNVTFlow::deformGlobalBox()
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


void TwoStepSLLODNVTFlow::advanceThermostat(unsigned int timestep, bool broadcast)
    {


    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    // compute the current thermodynamic properties
    m_thermo->compute(timestep+1);

    Scalar curr_T_trans = m_thermo->getTranslationalTemperature();

    // update the state variables Xi and eta
    Scalar xi_prime = xi + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*(curr_T_trans/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*(curr_T_trans/m_T->getValue(timestep) - Scalar(1.0));
    eta += xi_prime*m_deltaT;

    // update loop-invariant quantity
    m_exp_thermo_fac = exp(-Scalar(1.0/2.0)*xi*m_deltaT);

    #ifdef ENABLE_MPI
    if (m_comm && broadcast)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    setIntegratorVariables(v);

    }

void TwoStepSLLODNVTFlow::randomizeVelocities(unsigned int timestep)
    {
    if (m_shouldRandomize == false)
        {
        return;
        }

    m_exec_conf->msg->notice(6) << "TwoStepSLLODNVTFlow randomizing velocities" << std::endl;

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];

    unsigned int g = m_thermo->getNDOF();
    Scalar sigmasq_t = Scalar(1.0)/((Scalar) g*m_tau*m_tau);

    bool master = m_exec_conf->getRank() == 0;
    hoomd::RandomGenerator rng(azplugins::RNGIdentifier::TwoStepSLLODNVTFlow, m_seed_randomize, timestep);

    if (master)
        {
        // draw a random Gaussian thermostat variable on rank 0
        xi = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_t))(rng);
        }

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    setIntegratorVariables(v);

    // call base class method
    IntegrationMethodTwoStep::randomizeVelocities(timestep);
    }

namespace detail
{

void export_TwoStepSLLODNVTFlow(pybind11::module& m)
    {
    pybind11::class_<TwoStepSLLODNVTFlow, std::shared_ptr<TwoStepSLLODNVTFlow> >(m, "TwoStepSLLODNVTFlow", pybind11::base<IntegrationMethodTwoStep>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                       std::shared_ptr<ParticleGroup>,
                       std::shared_ptr<ComputeThermoSLLOD>,
                       Scalar,
                       std::shared_ptr<Variant>,
                       Scalar,
                       const std::string&
                       >())
        .def("setT", &TwoStepSLLODNVTFlow::setT)
        .def("setShearRate", &TwoStepSLLODNVTFlow::setShearRate)
        .def("setTau", &TwoStepSLLODNVTFlow::setTau)
        ;
    }

} //end namespace detail
} //end namespace azplugins
