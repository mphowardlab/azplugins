// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


/*!
 * \file MPCDSineGeometry.h
 * \brief Definition of the MPCD sine channel geometry
 */

#ifndef AZPLUGINS_MPCD_SINE_GEOMETRY_H_
#define AZPLUGINS_MPCD_SINE_GEOMETRY_H_

#include "hoomd/mpcd/BoundaryCondition.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#include <cstdio>
#include <iostream>

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

//! Sine channel geometry
/*!
 * This class defines a channel with sine walls given by the equations +/-(A cos(x*2*pi*p/Lx) + A + H_narrow).
 * A = 0.5*(H_wide-H_narrow) is the amplitude and p is the period of the wall sine.
 * H_wide is the half height of the channel at its widest point, H_narrow is the half height of the channel at its
 * narrowest point. The sine wall wavelength/frenquency needs to be consumable with the periodic boundary conditions in x,
 * therefore the period p is specified and the wavelength 2*pi*p/Lx is calculated.
 *
 * Below is an example how a sine channel looks like in a 30x30x30 box with H_wide=10, H_narrow=1, and p=1.
 * The wall sine period p determines how many repetitions of the geometry are in the simulation cell and
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
 *
 *
 *
 * The geometry enforces boundary conditions \b only on the MPCD solvent particles. Additional interactions
 * are required with any embedded particles using appropriate wall potentials.
 *
 * The wall boundary conditions can optionally be changed to slip conditions.
 */
class __attribute__((visibility("default"))) SineGeometry
    {
    public:
        //! Constructor
        /*!
         * \param L Channel length (Simulation box length in x)
           \param H_wide Channel half-width at widest point
           \param H_narrow Channel half-width at narrowest point
           \param Period Channel sine period (integer >0)
         * \param V Velocity of the wall
         * \param bc Boundary condition at the wall (slip or no-slip)
         */
        HOSTDEVICE SineGeometry(Scalar L, Scalar H_wide,Scalar H_narrow, unsigned int Repetitions, Scalar V, mpcd::detail::boundary bc)
            : m_pi_period_div_L(2*M_PI*Repetitions/L), m_H_wide(H_wide), m_H_narrow(H_narrow), m_Repetitions(Repetitions), m_V(V), m_bc(bc)
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
             * Detect if particle has left the box, and try to avoid branching or absolute value calls. The sign used
             * in calculations is +1 if the particle is out-of-bounds in the +z direction, -1 if the particle is
             * out-of-bounds in the -z direction, and 0 otherwise.
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
            *  Because there is no analythical solution for f(x) = cos(x)-x = 0, we use Newtons's method to nummerically estimate the
            *  x positon of the intersection first. It is convinient to use the halfway point between the last particle
            *  position outside the wall (at time t-dt) and the current position inside the wall (at time t) as initial
            *  guess for the intersection.
            *
            *  We limit the number of iterations (max_iteration) and the desired presicion (target_presicion) for performance reasons.
            */
            Scalar max_iteration = 5;
            Scalar counter = 0;
            Scalar target_presicion = 0.00001;
            Scalar x0 = pos.x - 0.5*dt*vel.x;

            // delta =  abs(0-f(x))
            Scalar delta = abs(0 - (sign*(A*fast::cos(x0*m_pi_period_div_L)+ A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z));

            Scalar n,n2;
            Scalar s,c;
            while( delta > target_presicion && counter < max_iteration)
                {
                fast::sincos(x0*m_pi_period_div_L,s,c);
                n  =  sign*(A*c + A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z;  // f
                n2 = -sign*m_pi_period_div_L*A*s - vel.z/vel.x;                       // df
                x0 = x0 - n/n2;                                                                      // x = x - f/df
                delta = abs(0-(sign*(A*fast::cos(x0*m_pi_period_div_L)+A+m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z));
                counter +=1;
                }

            /* The new z position is calculated from the wall equation to guarantee that the new particle positon is exactly at the wall
             * and not accidentally slightly inside of the wall because of nummerical presicion.
             */
            Scalar z0 = sign*(A*fast::cos(x0*m_pi_period_div_L)+A+m_H_narrow);

            /* The new y position can be calculated from the fact that the last position outside of the wall, the current position inside
             * of the  wall, and the new position exactly at the wall are on a straight line.
             */
            Scalar y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);

            /* chatch the case where a particle collides exactly vertically (v_x=0 -> old x pos = new x pos)
             * In this case, y0 = -(0)*0/0 + (y-dt*v_y) == nan, should be y0 =(y-dt*v_y)
             */
            if (vel.x==0. && pos.x==x0)
                {
                y0 = (pos.y-dt*vel.y);
                }

            // Remaining integration time dt is amount of time spent traveling distance out of bounds.
            dt = fast::sqrt(((pos.x - x0)*(pos.x - x0) + (pos.y - y0)*(pos.y -y0) + (pos.z - z0)*(pos.z - z0))/(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z));

            //  positions are updated
            pos.x = x0;
            pos.y = y0;
            pos.z = z0;

            /* update velocity according to boundary conditions.
             *
             * A upwards normal of the surface is given by (-df/dx,-df/dy,1) with f = sign*(A*cos(x*2*pi*p/L)+A+h), so
             * normal  = (sign*A*2*pi*p/L*sin(x*2*pi*p/L),0,1)/|length|
             * The direction of the normal is not important for the reflection.
             * Calculate components by hand to avoid sqrt in normalization of the normal of the surface.
             *
             * TO DO: do moving boundaries (velocity m_V) in opposite directions even make sense for the curved sine geometry?
             */
            Scalar3 vel_new;
            if (m_bc ==  mpcd::detail::boundary::no_slip) // No-slip requires reflection of both tangential and normal components:
                {

                vel_new.x = -vel.x + Scalar(sign * 2) * m_V;
                vel_new.y = -vel.y;
                vel_new.z = -vel.z;

                }
            else // Slip conditions require only tangential components to be reflected:
                {
                Scalar B = sign*A*m_pi_period_div_L*fast::sin(x0*m_pi_period_div_L);

                vel_new.x = vel.x - 2*B*(B*vel.x + vel.z)/(B*B+1);
                vel_new.y = vel.y;
                vel_new.z = vel.z -   2*(B*vel.x + vel.z)/(B*B+1);
              //  printf("%f %f %f %f %f %f\n",vel.x,vel.y,vel.z,vel_new.x,vel_new.y,vel_new.z );
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
            Scalar a = 0.5*(m_H_wide-m_H_narrow)*fast::cos(pos.x*m_pi_period_div_L)+0.5*(m_H_wide-m_H_narrow)+m_H_narrow;

            return (pos.z > a || pos.z < -a);
            }

        //! Validate that the simulation box is large enough for the geometry
        /*!
         * \param box Global simulation box
         * \param cell_size Size of MPCD cell
         *
         * The box is large enough for the sine if it is padded along the z direction so that
         * the cells just outside the highest point of the sine + the filler thinckness
         * would not interact with each other through the boundary.
         *
         */
        HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
            {
            const Scalar hi = box.getHi().z;
            const Scalar lo = box.getLo().z;
            // TO DO: get max_shift from  mpcd , can't use function argument because hoomd/mpcd/ConfinedStreamingMethod.h complains
            // if arguments of validateBox() change
            const Scalar max_shift = 0.5*cell_size;

            const Scalar filler_thickness = cell_size +  0.5*(m_H_wide-m_H_narrow)*fast::sin((cell_size+max_shift)*m_pi_period_div_L);
            return (hi >= m_H_wide+filler_thickness && lo <= -m_H_wide-filler_thickness );
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

        //! Get channel sine wall repetitions
        /*!
         * \returns Channel sine wall repetitions
         */
        HOSTDEVICE Scalar getRepetitions() const
            {
            return m_Repetitions;
            }

        //! Get the wall velocity
        /*!
         * \returns Wall velocity
         */
        HOSTDEVICE Scalar getVelocity() const
            {
            return m_V;
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
            return std::string("Sine");
            }
        #endif // NVCC

    private:
        const Scalar m_pi_period_div_L;     //!< Argument of the wall sine (pi*period/Lx = 2*pi*repetitions/Lx)
        const Scalar m_H_wide;              //!< Half of the channel widest width
        const Scalar m_H_narrow;            //!< Half of the channel narrowest width
        const unsigned int m_Repetitions;         //!< Number of repetitions of the wide sections in the channel =  period
        const Scalar m_V;                   //!< Velocity of the wall
        const mpcd::detail::boundary m_bc; //!< Boundary condition
    };

} // end namespace detail
} // end namespace azplugins
#undef HOSTDEVICE

#endif // AZPLUGINS_MPCD_SINE_GEOMETRY_H_
