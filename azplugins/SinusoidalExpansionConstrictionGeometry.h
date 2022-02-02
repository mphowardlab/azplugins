// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


/*!
 * \file SinusoidalExpansionConstrictionGeometry.h
 * \brief Definition of the MPCD symmetric cosine channel geometry
 */

#ifndef AZPLUGINS_SINUSOIDAL_EXPANSION_CONSTRICTION_GEOMETRY_H_
#define AZPLUGINS_SINUSOIDAL_EXPANSION_CONSTRICTION_GEOMETRY_H_

#include "hoomd/mpcd/BoundaryCondition.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // NVCC

namespace azplugins
{
namespace detail
{

//! Sinusoidal expansion constriction channel geometry
/*!
 * This class defines a channel with a series of expansions and constrictions. Symmetric cosines
 * given by the equations +/-(A cos(x*2*pi*p/Lx) + A + H_narrow) are used for the walls.
 * A = 0.5*(H_wide-H_narrow) is the amplitude and p is the period of the wall cosine.
 * H_wide is the half height of the channel at its widest point, H_narrow is the half height of the channel at its
 * narrowest point. The cosine wall wavelength/frenquency needs to be consumable with the periodic boundary conditions in x,
 * therefore the period p is specified and the wavelength 2*pi*p/Lx is calculated.
 *
 * Below is an example how a cosine channel looks like in a 30x30x30 box with H_wide=10, H_narrow=1, and p=1.
 * The wall cosine period p determines how many repetitions of the geometry are in the simulation cell and
 * there will be p wide sections, centered at the origin of the simulation box.
 *
 *
 * 15 +-------------------------------------------------+
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *  10 |XXXXXXXXXXXXXXXXXXX===========XXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXX====           ====XXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXX===                 ===XXXXXXXXXXXXX|
 *   5 |XXXXXXXXXXX==                       ==XXXXXXXXXXX|
 *     |XXXXXXXX===                           ===XXXXXXXX|
 *     |XXXXX====                               ====XXXXX|
 *     |=====                                       =====|
 * z 0 |                                                 |
 *     |=====                                       =====|
 *     |XXXXX====                               ====XXXXX|
 *     |XXXXXXXX===                           ===XXXXXXXX|
 *  -5 |XXXXXXXXXXX==                       ==XXXXXXXXXXX|
 *     |XXXXXXXXXXXXX===                 ===XXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXX====           ====XXXXXXXXXXXXXXX|
 * -10 |XXXXXXXXXXXXXXXXXXX===========XXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 * -15 +-------------------------------------------------+
 *    -15     -10      -5       0       5        10      15
 *                              x
 *
 * The wall boundary conditions can optionally be changed to slip conditions.
 *
 */
class __attribute__((visibility("default"))) SinusoidalExpansionConstriction
    {
    public:
        //! Constructor
        /*!
         * \param L Channel length (Simulation box length in x)
           \param H_wide Channel half-width at widest point
           \param H_narrow Channel half-width at narrowest point
           \param Period Channel cosine period (integer >0)
         * \param bc Boundary condition at the wall (slip or no-slip)
         */
        HOSTDEVICE SinusoidalExpansionConstriction(Scalar L, Scalar H_wide,Scalar H_narrow, unsigned int Repetitions, mpcd::detail::boundary bc)
            : m_pi_period_div_L(2*M_PI*Repetitions/L), m_H_wide(H_wide), m_H_narrow(H_narrow), m_Repetitions(Repetitions), m_bc(bc)
            {
            }

        //! Detect collision between the particle and the boundary
        /*!
         * \param pos Proposed particle position
         * \param vel Proposed particle velocity
         * \param dt Integration time remaining
         *
         * \returns True if a collision occurred, and false otherwise
         *
         * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel is updated
         *       according to the appropriate bounce back rule, and the integration time \a dt is decreased to the
         *       amount of time remaining.
         */
        HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
            {
            /*
             * Detect if particle has left the box. The sign used
             * in calculations is +1 if the particle is out-of-bounds at the top wall, -1 if the particle is
             * out-of-bounds at the bottom wall, and 0 otherwise.
             *
             * We intentionally use > / < rather than >= / <= to make sure that spurious collisions do not get detected
             * when a particle is reset to the boundary location. A particle landing exactly on the boundary from the bulk
             * can be immediately reflected on the next streaming step, and so the motion is essentially equivalent up to
             * an epsilon of difference in the channel width.
             */
            Scalar A = 0.5*(m_H_wide-m_H_narrow);
            Scalar a = A*fast::cos(pos.x*m_pi_period_div_L) + A + m_H_narrow;
            const signed char sign = (pos.z > a) - (pos.z < -a);

            // exit immediately if no collision is found
            if (sign == 0)
                {
                dt = Scalar(0);
                return false;
                }

            /* Calculate position (x0,y0,z0) of collision with wall:
            *  Because there is no analythical solution for equations like f(x) = cos(x)-x = 0, we use Newtons's method
            *  or Bisection (if Newton fails) to nummerically estimate the
            *  x positon of the intersection first. It is convinient to use the halfway point between the last particle
            *  position outside the wall (at time t-dt) and the current position inside the wall (at time t) as initial
            *  guess for the intersection.
            *
            *  We limit the number of iterations (max_iteration) and the desired presicion (target_precision) for
            *  performance reasons. These values have been tested in python code seperately and gave satisfactory results.
            *
            */
            const unsigned int max_iteration = 6;
            const Scalar target_precision = 1e-5;

            Scalar x0 = pos.x - 0.5*dt*vel.x;
            Scalar y0;
            Scalar z0;
            Scalar n,n2;
            Scalar s,c;
            Scalar delta;

            // excatly horizontal z-collision, has a solution:
            if (vel.z == 0)
                {
                x0 = 1/m_pi_period_div_L*fast::acos((pos.z-A-m_H_narrow)/sign*A);
                z0 = pos.z;
                y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);
                }
            /* chatch the case where a particle collides exactly vertically (v_x=0 -> old x pos = new x pos)
            * In this case in Newton's method one would get: y0 = -(0)*0/0 + (y-dt*v_y) == nan, should be y0 =(y-dt*v_y)
            */
            else if (vel.x == 0)
                {
                x0 = pos.x;
                y0 = (pos.y-dt*vel.y);
                z0 = sign*(A*fast::cos(x0*m_pi_period_div_L)+A+m_H_narrow);
                }
            else // not horizontal or vertical collision - do Newthon's method
                {
                delta = fabs(0 - (sign*(A*fast::cos(x0*m_pi_period_div_L)+ A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z));

                unsigned int counter = 0;
                while( delta > target_precision && counter < max_iteration)
                    {
                    fast::sincos(x0*m_pi_period_div_L,s,c);
                    n = sign*(A*c + A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z;  // f
                    n2 = -sign*m_pi_period_div_L*A*s - vel.z/vel.x;                      // df
                    x0 = x0 - n/n2;                                                      // x = x - f/df
                    delta = fabs(0-(sign*(A*fast::cos(x0*m_pi_period_div_L)+A+m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z));
                    ++counter;
                    }
                /* The new z position is calculated from the wall equation to guarantee that the new particle positon is exactly at the wall
                 * and not accidentally slightly inside of the wall because of nummerical presicion.
                 */
                z0 = sign*(A*fast::cos(x0*m_pi_period_div_L)+A+m_H_narrow);

                /* The new y position can be calculated from the fact that the last position outside of the wall, the current position inside
                 * of the  wall, and the new position exactly at the wall are on a straight line.
                 */
                y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);

                // Newton's method sometimes failes to converge (close to saddle points, df'==0, overshoot, bad initial guess,..)
                // catch all of them here and do bisection if Newthon's method didn't work
                Scalar lower_x = fmin(pos.x - dt*vel.x,pos.x);
                Scalar upper_x = fmax(pos.x - dt*vel.x,pos.x);

                // found intersection is NOT in between old and new point, ie intersection is wrong/inaccurate.
                // do bisection to find intersection - slower but more robust than Newton's method
                if (x0 < lower_x || x0 > upper_x)
                    {
                    counter = 0;
                    Scalar3 point1 = pos;  // final position at t+dt
                    Scalar3 point2 = pos-dt*vel; // initial position
                    Scalar3 point3 = 0.5*(point1+point2); // halfway point
                    Scalar fpoint3 = (sign*(A*fast::cos(point3.x*m_pi_period_div_L)+ A + m_H_narrow) - point3.z); // value at halfway point, f(x)
                    // Note: technically, the presicion of Newton's method and bisection is slightly different, with
                    // bisection being less precise and slower convergence.
                    while (fabs(fpoint3) > target_precision && counter < max_iteration)
                        {
                        fpoint3 = (sign*(A*fast::cos(point3.x*m_pi_period_div_L)+ A + m_H_narrow) - point3.z);
                        // because we know that point1 outside of the channel and point2 is inside of the channel, we
                        // only need to check the halfway point3 - if it is inside, replace point2, if it is outside, replace point1
                        if (isOutside(point3) == false)
                            {
                            point2 = point3;
                            }
                        else
                            {
                            point1 = point3;
                            }
                        point3 = 0.5*(point1+point2);
                        ++counter;
                        }
                    // final point3 == intersection
                    x0 = point3.x;
                    z0 = sign*(A*fast::cos(x0*m_pi_period_div_L)+A+m_H_narrow);
                    y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);
                    }
                }

            // Remaining integration time dt is amount of time spent traveling distance out of bounds.
            Scalar3 pos_new = make_scalar3(x0,y0,z0);
            dt = fast::sqrt(dot((pos - pos_new),(pos - pos_new))/dot(vel,vel));
            pos = pos_new;

            /* update velocity according to boundary conditions.
             *
             * A upwards normal of the surface is given by (-df/dx,-df/dy,1) with f = sign*(A*cos(x*2*pi*p/L)+A+h), so
             * normal  = (sign*A*2*pi*p/L*sin(x*2*pi*p/L),0,1)/|length|
             * We define B = sign*A*2*pi*p/L*sin(x*2*pi*p/L), so then the normal is given by (B,0,1)/|length|
             * The direction of the normal is not important for the reflection.
             */
            Scalar3 vel_new;
            if (m_bc ==  mpcd::detail::boundary::no_slip) // No-slip requires reflection of both tangential and normal components:
                {
                vel_new = -vel;
                }
            else // Slip conditions require only tangential components to be reflected:
                {
                Scalar B = sign*A*m_pi_period_div_L*fast::sin(x0*m_pi_period_div_L);
                // The reflected vector is given by v_reflected = -2*(v_normal*v_incoming)*v_normal + v_incoming
                // Calculate components by hand to avoid sqrt in normalization of the normal of the surface.
                vel_new.x = vel.x - 2*B*(B*vel.x + vel.z)/(B*B+1);
                vel_new.y = vel.y;
                vel_new.z = vel.z - 2*(B*vel.x + vel.z)/(B*B+1);
                }
            vel = vel_new;

            return true;
            }

        //! Check if a particle is out of bounds
        /*!
         * \param pos Current particle position
         * \returns True if particle is out of bounds, and false otherwise
         */
        HOSTDEVICE bool isOutside(const Scalar3& pos) const
            {
            const Scalar a = 0.5*(m_H_wide-m_H_narrow)*fast::cos(pos.x*m_pi_period_div_L)+0.5*(m_H_wide-m_H_narrow)+m_H_narrow;
            return (pos.z > a || pos.z < -a);
            }

        //! Validate that the simulation box is large enough for the geometry
        /*!
         * \param box Global simulation box
         * \param cell_size Size of MPCD cell
         *
         * The box is large enough for the cosine if it is padded along the z direction so that
         * the cells just outside the highest point of the cosine would not interact with each
         * other through the boundary.
         */
        HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
            {
            const Scalar hi = box.getHi().z;
            const Scalar lo = box.getLo().z;
            return ((hi-m_H_wide) >= cell_size && (-m_H_wide-lo) >= cell_size);
            }

        //! Get channel half width at widest point
        /*!
         * \returns Channel half width at widest point
         */
        HOSTDEVICE Scalar getHwide() const
            {
            return m_H_wide;
            }
        //! Get channel half width at narrowest point
        /*!
         * \returns Channel half width at narrowest point
         */
        HOSTDEVICE Scalar getHnarrow() const
            {
            return m_H_narrow;
            }

        //! Get channel cosine wall repetitions
        /*!
         * \returns Channel cosine wall repetitions
         */
        HOSTDEVICE Scalar getRepetitions() const
            {
            return m_Repetitions;
            }

        //! Get the wall boundary condition
        /*!
         * \returns Boundary condition at wall
         */
        HOSTDEVICE  mpcd::detail::boundary getBoundaryCondition() const
            {
            return m_bc;
            }

        #ifndef NVCC
        //! Get the unique name of this geometry
        static std::string getName()
            {
            return std::string("SinusoidalExpansionConstriction");
            }
        #endif // NVCC

    private:
        const Scalar m_pi_period_div_L;     //!< Argument of the wall cosine (pi*period/Lx = 2*pi*repetitions/Lx)
        const Scalar m_H_wide;              //!< Half of the channel widest width
        const Scalar m_H_narrow;            //!< Half of the channel narrowest width
        const unsigned int m_Repetitions;         //!< Number of repetitions of the wide sections in the channel =  period
        const mpcd::detail::boundary m_bc; //!< Boundary condition
    };

} // end namespace detail
} // end namespace azplugins
#undef HOSTDEVICE

#endif // AZPLUGINS_SINUSOIDAL_EXPANSION_CONSTRICTION_GEOMETRY_H_
