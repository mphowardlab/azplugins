// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file WallEvaluatorLJ93.h
 * \brief Defines the wall potential evaluator class for the LJ 9-3 potential
 */

#ifndef AZPLUGINS_WALL_EVALUATOR_LJ_93_H_
#define AZPLUGINS_WALL_EVALUATOR_LJ_93_H_

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
//! Evaluates the Lennard-Jones 9-3 wall force
/*!
 * WallEvaluatorLJ93 computes the Lennard-Jones 9-3 wall potential, which is derived from integrating the standard
 * Lennard-Jones potential between a point particle and a half plane:
 *
 * \f[ V(r) = \varepsilon \left( \frac{2}{15}\left(\frac{\sigma}{r}\right)^9 - \left(\frac{\sigma}{r}\right)^3 \right) \f]
 *
 * where \f$\sigma\f$ is the diameter of Lennard-Jones particles in the wall, and \f$ \varepsilon \f$ is the effective Hamaker
 * constant \f$ \varepsilon = (2/3) \pi \varepsilon_{\rm LJ} \rho_{\rm w} \sigma^3 \f$ with \f$\varepsilon_{\rm LJ}\f$
 * the energy of interaction and \f$\rho_{\rm w}\f$ the density of particles in the wall. Evaluation of this energy
 * is simplified into the following parameters:
 *
 * - \verbatim lj1 = (2.0/15.0) * epsilon * pow(sigma,9.0) \endverbatim
 * - \verbatim lj2 = epsilon * pow(sigma,3.0) \endverbatim
 *
 * The force acting on the particle is then
 * \f[ F(r)/r = \frac{\varepsilon}{r^2} \left ( \frac{6}{5}\left(\frac{\sigma}{r}\right)^9 - 3 \left(\frac{\sigma}{r}\right)^3 \right) \f]
 */
class WallEvaluatorLJ93
    {
    public:
        //! Define the parameter type used by this wall potential evaluator
        typedef Scalar2 param_type;

        //! Constructor
        /*!
         * \param _rsq Squared distance between particles
         * \param _rcutsq Cutoff radius squared
         * \param _params Pair potential parameters, given by typedef above
         *
         * The functor initializes its members from \a _params.
         */
        DEVICE WallEvaluatorLJ93(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.x), lj2(_params.y)
            {
            }

        //! LJ 9-3 doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*!
         * \param di Diameter of particle
         * \param dj Dummy diameter
         *
         * \note The way HOOMD computes wall forces by recycling evaluators requires that we give
         *       a second diameter, even though this is meaningless for the potential.
         */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ 9-3 doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional charge values
        /*!
         * \param qi Charge of particle
         * \param qj Dummy charge
         *
         * \note The way HOOMD computes wall forces by recycling evaluators requires that we give
         *       a second charge, even though this is meaningless for the potential.
         */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*!
         * \param force_divr Holds the computed force divided by r
         * \param energy Holds the computed pair energy
         * \param energy_shift If true, the potential is shifted to zero at the cutoff
         *
         * \returns True if the energy calculation occurs
         *
         * The calculation does not occur if the pair distance is greater than the cutoff
         * or if the potential is scaled to zero.
         */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& energy, bool energy_shift)
            {
            if (rsq < rcutsq && lj1 != 0)
                {
                Scalar r2inv = Scalar(1.0) / rsq;
                Scalar r3inv = r2inv * sqrt(r2inv); // we need to have the odd power to get the energy, so must take sqrt
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

        #ifndef NVCC
        //! Return the name of this potential
        static std::string getName()
            {
            return std::string("lj93");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
    };
} // end namespace detail
} // end namespace azplugins

#undef DEVICE
#endif // AZPLUGINS_WALL_EVALUATOR_LJ_93_H_
