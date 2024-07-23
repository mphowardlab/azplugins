// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_WALL_EVALUATOR_LJ_93_H_
#define AZPLUGINS_WALL_EVALUATOR_LJ_93_H_

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairLJ.h
    \brief Defines the pair evaluator class for LJ potentials
    \details As the prototypical example of a MD pair potential, this also serves as the primary
   documentation and base reference for the implementation of pair evaluators.
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace azplugins
    {
namespace detail
{



class WallEvaluatorLJ93
{
public:
  struct param_type
      {

      DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

      HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifndef ENABLE_HIP
      //! set CUDA memory hints
      void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
      param_type() : sigma(0), epsilon(0), lj1(0), lj2(0) { }

      param_type(pybind11::dict v, bool managed = false)
          {
          sigma = v["sigma"].cast<Scalar>();
          epsilon = v["epsilon"].cast<Scalar>();
          Scalar sigma_3 = sigma * sigma * sigma;
          Scalar sigma_9 = sigma_3 * sigma_3 * sigma_3;
          lj1 = (Scalar(2.0)/Scalar(15.0)) * epsilon * sigma_9;
          lj2 = epsilon * sigma_3;
          }

      pybind11::dict asDict()
          {
          pybind11::dict v;
          v["sigma"] = sigma;
          v["epsilon"] = epsilon;
          return v;
          }
#endif
      Scalar sigma;
      Scalar epsilon;
      Scalar lj1;
      Scalar lj2;
      }
#if HOOMD_LONGREAL_SIZE == 32
      __attribute__((aligned(8)));
#else
      __attribute__((aligned(16)));
#endif

DEVICE WallEvaluatorLJ93(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
    : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.lj1), lj2(_params.lj2)
    {
    }

    DEVICE static bool needsCharge()
        {
        return false;
        }

  DEVICE void setCharge(Scalar qi, Scalar qj) { }


  DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& energy, bool energy_shift)
      {
      if (rsq < rcutsq && lj1 != 0)
          {
          Scalar r2inv = Scalar(1.0) / rsq;
          Scalar r3inv
              = r2inv
                * sqrt(
                    r2inv); // we need to have the odd power to get the energy, so must take sqrt
          Scalar r6inv = r3inv * r3inv;

          force_divr = r2inv * r3inv * (Scalar(9.0) * lj1 * r6inv - Scalar(3.0) * lj2);
          energy = r3inv * (lj1 * r6inv - lj2);

          if (energy_shift)
              {
              // this could be cached once per type as a parameter to save flops and a sqrt
              Scalar rcut2inv = Scalar(1.0) / rcutsq;
              Scalar rcut3inv = rcut2inv * sqrt(rcut2inv);
              Scalar rcut6inv = rcut3inv * rcut3inv;
              energy -= rcut3inv * (lj1 * rcut6inv - lj2);
              }
          return true;
          }
      else
          return false;
      }

      DEVICE Scalar evalPressureLRCIntegral()
          {
          return 0;
          }

      //! Example doesn't eval LRC integrals
      DEVICE Scalar evalEnergyLRCIntegral()
          {
          return 0;
          }


#ifndef __HIPCC__
//! Get the name of this potential
/*! \returns The potential name.
 */
static std::string getName()
    {
    return std::string("LJ93");
    }

#endif

protected:
Scalar rsq;    //!< Stored rsq from the constructor
Scalar rcutsq; //!< Stored rcutsq from the constructor
Scalar lj1;    //!< lj1 parameter extracted from the params passed to the constructor
Scalar lj2;    //!< lj2 parameter extracted from the params passed to the constructor


};



    } // end namespace azplugins
    } // end namespace hoomd
  }

#endif // AZPLUGINS_WALL_EVALUATOR_LJ_93_H_
