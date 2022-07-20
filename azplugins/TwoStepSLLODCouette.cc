// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: arjunsg2

/*!
 * \file TwoStepSLLODCouette.h
 * \brief Declaration of TwoStepSLLODCouette
 */

 #include "TwoStepSLLODCouette.h"

namespace azplugins
{

//! Constructor
TwoStepSLLODCouette::TwoStepSLLODCouette(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    bool use_lambda,
                    Scalar lambda,
                    Scalar gamma_dot,
                    bool noiseless)
: IntegrationMethodTwoStep(sysdef, group), m_T(T), m_seed(seed),
    m_use_lambda(use_lambda), m_lambda(lambda), m_gamma_dot(gamma_dot), m_noiseless(noiseless)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepSLLODCouette" << std::endl;

    if (use_lambda)
        m_exec_conf->msg->notice(2) << "flow.sllod is determining gamma from particle diameters" << std::endl;
    else
        m_exec_conf->msg->notice(2) << "flow.sllod is using specified gamma values" << std::endl;

    // In case of MPI run, every rank should be initialized with the same seed.
    // For simplicity we broadcast the seed of rank 0 to all ranks.

    #ifdef ENABLE_MPI
    if( this->m_pdata->getDomainDecomposition() )
        bcast(m_seed,0,this->m_exec_conf->getMPICommunicator());
    #endif

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GlobalVector<Scalar> gamma(m_pdata->getNTypes(), m_exec_conf);
    m_gamma.swap(gamma);
    TAG_ALLOCATION(m_gamma);

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.size(); i++)
        h_gamma.data[i] = Scalar(1.0);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_gamma.get(), sizeof(Scalar)*m_gamma.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        }
    #endif
    }


 void TwoStepSLLODCouette::integrateStepOne(unsigned int timestep)
     {
     if (m_aniso)
         {
         m_exec_conf->msg->error() << "azplugins.flow: anisotropic particles are not supported with couette flow integrators." << std::endl;
         throw std::runtime_error("Anisotropic integration not supported with couette flow");
         }
     if (m_prof) m_prof->push("SLLOD-Couette step 1");

     ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
     ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
     ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
     ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);

     // Deform box
     BoxDim newBox = m_pdata->getGlobalBox();
     Scalar3 y = newBox.getLatticeVector(1);
     Scalar xy = newBox.getTiltFactorXY();
     Scalar yz = newBox.getTiltFactorYZ();
     Scalar xz = newBox.getTiltFactorXZ();
     const Scalar boundary_shear = y.y * m_gamma_dot;
     xy += m_gamma_dot * m_deltaT;
     bool flipped = false;
     if (xy > 0.5){
         xy = -0.5;
         flipped = true;
     }
     newBox.setTiltFactors(xy, xz, yz);
     m_pdata->setGlobalBox(newBox);
     const BoxDim& box = m_pdata->getBox();

     // perform the first half step of velocity verlet
     unsigned int group_size = m_group->getNumMembers();
     for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
         {
         unsigned int idx = m_group->getMemberIndex(group_idx);

         const Scalar4 postype = h_pos.data[idx];
         Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
         unsigned int type = __scalar_as_int(postype.w);

         // velocity
         const Scalar4 velmass = h_vel.data[idx];
         Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
         Scalar mass = velmass.w;

         // acceleration
         const Scalar3 accel = h_accel.data[idx];

         // shear rate tensor dotted with velocity
         const Scalar3 v_del_u = make_scalar3(m_gamma_dot * velmass.y, 0.0, 0.0);

         // update velocity
         vel += Scalar(0.5) * m_deltaT * (accel - v_del_u);

         // update position
         pos += (vel + Scalar(0.5) * m_deltaT * accel) * m_deltaT;

         // if particle leaves from (+/-) y boundary it gets (-/+) shear_rate
         // note carefully that pair potentials dependent on dv (e.g. DPD)
         // not yet explicitly supported due to minimum image convention
         if (pos.y > y.y/2){
             vel.x -= boundary_shear;
         }
         if (pos.y < -y.y/2){
             vel.x += boundary_shear;
         }

         // Wrap back into box
         if (flipped){
             // TODO check this
             h_images.data[j].x += h_images.data[j].y;
            // pos.x *= -1;
         }
         box.wrap(pos,h_image.data[idx]);

         h_pos.data[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
         h_vel.data[idx] = make_scalar4(vel.x, vel.y, vel.z, mass);
         }

     if (m_prof) m_prof->pop();
     }

 void TwoStepSLLODCouette::integrateStepTwo(unsigned int timestep)
     {
     if (m_prof) m_prof->push("SLLOD-Couette step 2");

     unsigned int group_size = m_group->getNumMembers();

     const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
     ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
     ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
     ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
     ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
     ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
     ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
     ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

     Scalar3 binLabel[group_size];
     const Scalar currentTemp = m_T->getValue(timestep);
     int period = 500;

     // assigning particles to bins
     BoxDim currentBox = m_pdata->getGlobalBox();
     Scalar3 x = currentBox.getLatticeVector(0);
     Scalar3 y = currentBox.getLatticeVector(1);
     Scalar3 z = currentBox.getLatticeVector(2);
     Scalar xy = currentBox.getTiltFactorXY();

     // number of bins in each direction
     int numBinsX = 10;
     int numBinsY = 10;
     int numBinsZ = 10;

     // end positions of the bins in each direction
     Scalar binPosX[numBinsX];
     Scalar binPosY[numBinsY];
     Scalar binPosZ[numBinsZ];

     for (int binIdxX = 0; binIdxX < numBinsX; binIdxX++)
     {
         binPosX[binIdxX] = -Scalar(0.5) * x.x + (binIdxX + 1) * x.x / numBinsX;
     }

     for (int binIdxY = 0; binIdxY < numBinsY; binIdxY++)
     {
         binPosY[binIdxY] = -Scalar(0.5) * y.y + (binIdxY + 1) * y.y / numBinsY;
     }

     for (int binIdxZ = 0; binIdxZ < numBinsZ; binIdxZ++)
     {
         binPosZ[binIdxZ] = -Scalar(0.5) * z.z + (binIdxZ + 1) * z.z / numBinsX;
     }

     std::vector<Scalar> vBinPosX(binPosX, binPosX + numBinsX);
     std::vector<Scalar> vBinPosY(binPosY, binPosY + numBinsY);
     std::vector<Scalar> vBinPosZ(binPosZ, binPosZ + numBinsZ);

     // Finding which bin each particle belongs to
     if (timestep % period == 0){
         for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
         {
             unsigned int idx = m_group->getMemberIndex(group_idx);

             const Scalar4 postype = h_pos.data[idx];
             Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

             std::vector<Scalar>::iterator itX, itY, itZ;

             itX = std::lower_bound(vBinPosX.begin(), vBinPosX.end(), pos.x - xy * pos.y);
             binLabel[idx].x = Scalar(int(itX - vBinPosX.begin()));

             itY = std::lower_bound(vBinPosY.begin(), vBinPosY.end(), pos.y);
             binLabel[idx].y = Scalar(int(itY - vBinPosY.begin()));

             itZ = std::lower_bound(vBinPosZ.begin(), vBinPosZ.end(), pos.z);
             binLabel[idx].z = Scalar(int(itZ - vBinPosZ.begin()));
         }
     }

     // computing the bin center-of-mass velocities
     Scalar3 Bin_COM_Vel[numBinsX][numBinsY][numBinsZ];

     if (timestep % period == 0){
         for (int binIdxX = 0; binIdxX < numBinsX; binIdxX++)
         {
             for (int binIdxY = 0; binIdxY < numBinsY; binIdxY++)
             {
                 for (int binIdxZ = 0; binIdxZ < numBinsZ; binIdxZ++)
                 {
                     Scalar3 v_sum = make_scalar3(0.0, 0.0, 0.0);
                     int num = 0;
                     for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                     {
                         unsigned int idx = m_group->getMemberIndex(group_idx);
                         if (   binLabel[idx].x == Scalar(binIdxX)
                             && binLabel[idx].y == Scalar(binIdxY)
                             && binLabel[idx].z == Scalar(binIdxZ) ){
                                 Scalar4 velmass_idx = h_vel.data[idx];
                                 Scalar3 vel = make_scalar3(velmass_idx.x, velmass_idx.y, velmass_idx.z);
                                 v_sum += vel;
                                 num += 1;
                             }
                     }
                     if (num != 0){
                         Bin_COM_Vel[binIdxX][binIdxY][binIdxZ] = v_sum / Scalar(num);
                     }
                     else {
                         Bin_COM_Vel[binIdxX][binIdxY][binIdxZ] = v_sum;
                     }
                 }
             }
         }
     }

     // v(t+deltaT) = v(t+deltaT/2) + 1/2 * deltaT * (a(t+deltaT) - v(t+deltaT/2)*del_u)
     for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
         {
         unsigned int j = m_group->getMemberIndex(group_idx);
         unsigned int ptag = h_tag.data[j];

         // Scalar gamma;
         // if (m_use_lambda)
         //     gamma = m_lambda*h_diameter.data[j];
         // else
         //     {
         //     unsigned int type = __scalar_as_int(h_pos.data[j].w);
         //     gamma = h_gamma.data[type];
         //     }
         //
         // hoomd::RandomGenerator rng(azplugins::RNGIdentifier::TwoStepLangevinFlow, m_seed, ptag, timestep);
         // hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
         // Scalar rx = uniform(rng);
         // Scalar ry = uniform(rng);
         // Scalar rz = uniform(rng);
         // Scalar coeff = fast::sqrt(Scalar(6.0)*gamma*currentTemp/m_deltaT);
         // if (m_noiseless)
         //     coeff = Scalar(0.0);
         // Scalar bd_fx = rx*coeff - gamma*(h_vel.data[j].x - h_pos.data[j].y * m_gamma_dot);
         // Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
         // Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;
         //
         // // first, calculate acceleration from the net force
         // Scalar minv = Scalar(1.0) / h_vel.data[j].w;
         // h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
         // h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
         // h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

         if (timestep % period == 0) {

             // randomly draw from Maxwell-Boltzmann-style distribution of
             // velocities at given temperature, with mean 0

             hoomd::RandomGenerator rng(azplugins::RNGIdentifier::TwoStepLangevinFlow, m_seed, ptag, timestep);
             hoomd::NormalDistribution<Scalar> gen(fast::sqrt(currentTemp / h_vel.data[j].w), 0.0);
             Scalar3 vel;
             gen(vel.x, vel.y, rng);
             vel.z = gen(rng);

             // Update velocity of particle to this randomly drawn velocity
             // plus bin center-of-mass velocity

             Scalar3 v_com = Bin_COM_Vel[int(binLabel[j].x)][int(binLabel[j].y)][int(binLabel[j].z)];
             h_vel.data[j].x = vel.x + v_com.x;
             h_vel.data[j].y = vel.y + v_com.y;
             h_vel.data[j].z = vel.z + v_com.z;
         }
         else {
             // first, calculate acceleration from the net force
             Scalar minv = Scalar(1.0) / h_vel.data[j].w;
             h_accel.data[j].x = h_net_force.data[j].x * minv;
             h_accel.data[j].y = h_net_force.data[j].y * minv;
             h_accel.data[j].z = h_net_force.data[j].z * minv;

             // then, update the velocity
             h_vel.data[j].x += Scalar(1.0/2.0)*(h_accel.data[j].x - h_vel.data[j].y * m_gamma_dot)*m_deltaT;
             h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
             h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
         }
         }

     if (m_prof) m_prof->pop();
     }

namespace detail
{
//! Export TwoStepSLLODCouette to python
void export_TwoStepSLLODCouette(pybind11::module& m)
   {
   namespace py = pybind11;
   py::class_<TwoStepSLLODCouette, std::shared_ptr<TwoStepSLLODCouette> >(m, "TwoStepSLLODCouette", py::base<IntegrationMethodTwoStep>())
       .def(py::init<std::shared_ptr<SystemDefinition>,
                     std::shared_ptr<ParticleGroup>,
                     std::shared_ptr<Variant>,
                     unsigned int,
                     bool,
                     Scalar,
                     Scalar,
                     bool>())
       .def("setT", &TwoStepSLLODCouette::setT)
       .def("set_gamma_dot", &TwoStepSLLODCouette::set_gamma_dot)
       .def("setNoiseless", &TwoStepSLLODCouette::setNoiseless)
       .def("setGamma", &TwoStepSLLODCouette::setGamma)
       ;
   }

} // end namespace detail
} //end namespace azplugins
