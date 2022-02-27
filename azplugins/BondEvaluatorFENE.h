// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*! \file BondEvaluatorFENE.h
    \brief Defines the bond evaluator class for a FENE potential
*/

#ifndef AZPLUGINS_BOND_EVALUATOR_FENE_H_
#define AZPLUGINS_BOND_EVALUATOR_FENE_H_

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace azplugins
{
namespace detail
{
  struct fene_bond_params
      {
      Scalar lj1;      //!< The coefficient for 1/r^12
      Scalar lj2;      //!< The coefficient for 1/r^6
      Scalar K;        //!< Stiffness parameter for the bond
      Scalar r_0;      //!< maximum bond length
      Scalar delta;    //!< extra shift parameter
      };

  //! Convenience function for makingfene_bond_params in python
  HOSTDEVICE inline fene_bond_params make_fene_bond_params(Scalar lj1,
                                                         Scalar lj2,
                                                         Scalar K,
                                                         Scalar r_0,
                                                         Scalar delta)
      {
      fene_bond_params p;
      p.lj1 = lj1;
      p.lj2 = lj2;
      p.K = K;
      p.r_0 = r_0;
      p.delta = delta;
      return p;
      }




//! Class for evaluating the FENE bond potential
/*! The parameters are:
    - \a K (params.x) Stiffness parameter for the force computation
    - \a r_0 (params.y) maximum bond length for the force computation
    - \a lj1 (params.z) Value of lj1 = 4.0*epsilon*pow(sigma,12.0)
       of the WCA potential in the force calculation
    - \a lj2 (params.w) Value of lj2 = 4.0*epsilon*pow(sigma,6.0)
       of the WCA potential in the force calculation
*/
class BondEvaluatorFENE
    {
    public:
        //! Define the parameter type used by this bond potential evaluator
        typedef fene_bond_params param_type;


        //! Constructs the pair potential evaluator
        /*!
         * \param _rsq Squared distance beteen the particles
         * \param _params Per type bond parameters of this potential as given above
         */
        DEVICE BondEvaluatorFENE(Scalar _rsq, const param_type& _params)
            : rsq(_rsq), K(_params.K), r_0(_params.r0), lj1(_params.lj1), lj2(_params.lj2),
              delta(_params.delta)
            {
            }

        //! This evaluator doesn't use diameter information
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*!
         * \param da Diameter of particle a
         * \param db Diameter of particle b
         */
        DEVICE void setDiameter(Scalar da, Scalar db) {  }

        //! FENE doesn't use charge
        DEVICE static bool needsCharge() { return false; }

        //! Accept the optional charge values
        /*!
         * \param qa Charge of particle a
         * \param qb Charge of particle b
         */
        DEVICE void setCharge(Scalar qa, Scalar qb) { }

        //! Evaluate the force and energy
        /*!
         * \param force_divr Output parameter to write the computed force divided by r.
         * \param bond_eng Output parameter to write the computed bond energy
         *
         *  \return True if they are evaluated or false if the bond energy is not defined.
         */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
            {
            // check for invalid parameters
            if (lj1 == Scalar(0.0) || r_0 == Scalar(0.0) || K == Scalar(0.0) ) return false;

            // Check if bond length restriction is violated
            if (rsq >= r_0*r_0) return false;

            Scalar r2inv = Scalar(1.0)/rsq;
            const Scalar r6inv = r2inv*r2inv*r2inv;
            Scalar sigma6inv = lj2/lj1;

            // wca cutoff: r < 2^(1/6)*sigma
            // wca cutoff: 1 / r^6 > 1/((2^(1/6))^6 sigma^6)
            if (r6inv > sigma6inv/Scalar(2.0))
                {
                Scalar epsilon = lj2*lj2/Scalar(4.0)/lj1;
                force_divr = r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);
                bond_eng = r6inv * (lj1*r6inv - lj2) + epsilon;
                }
            // Check if delta is non -zero,if non-zero sqrt of rsq is calculated
            if (delta!=Scalar(0))
                {
                const Scalar r = fast::sqrt(rsq);
                const Scalar r_red = r - delta;

                force_divr += -K*r_red / (Scalar(1.0) - r_red*r_red/(r_0*r_0))/r;
                bond_eng += -Scalar(0.5)*K*(r_0*r_0)*fast::log(Scalar(1.0) - r_red*r_red/(r_0*r_0));
                }
            else
                {
                force_divr += -K / (Scalar(1.0) -rsq/(r_0*r_0));
                bond_eng += -Scalar(0.5)*K*(r_0*r_0)*fast::log(Scalar(1.0) - rsq/(r_0*r_0));
                }
            return true;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*!
         * \returns The potential name. Must be short and all lowercase, as this is the name energies
         * will be logged as via analyze.log.
         */
        static std::string getName()
            {
            return std::string("fene");
            }
        #endif

    protected:
        Scalar rsq;        //!< Stored rsq from the constructor
        Scalar K;          //!< K parameter
        Scalar r_0;        //!< r_0 parameter
        Scalar lj1;        //!< lj1 parameter
        Scalar lj2;        //!< lj2 parameter
        Scalar delta;      //!< delta parameter in FENE bond

    };

} // end namespace detail
} // end namespace azplugins

#undef DEVICE

#endif // AZPLUGINS_BOND_EVALUATOR_FENE_H_
