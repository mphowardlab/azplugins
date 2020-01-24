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
 * This class defines a channel with sine walls given by the equations +/-(A cos(x*2pi*p/Lx) + A + H_narrow).
 * A = 0.5*(H_wide-H_narrow) is the amplitude and p is the period of the wall sine,
 * where H_wide is the half height of the channel at its widest point, H_narrow is the half height of the channel at its
 * narrowest point. The sine wall period/ number of repetitions has to be an integer larger than 0 to be consumable
 * with the periodic boundary conditions in x.
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
           \param Period Channel sine period (even integer >0)
         * \param V Velocity of the wall
         * \param bc Boundary condition at the wall (slip or no-slip)
         */
        HOSTDEVICE SineGeometry(Scalar L, Scalar H_wide,Scalar H_narrow, Scalar Repetitions, Scalar V, mpcd::detail::boundary bc)
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

            Scalar a = (m_H_wide-m_H_narrow)*cos(pos.x*m_pi_period_div_L)+m_H_wide;
            const signed char sign = (pos.z > a) - (pos.z < -a);

            // exit immediately if no collision is found
            if (sign == 0)
                {
                dt = Scalar(0);
                return false;
                }


            /* Calculate position (x0,y0,z0) of collision with wall:
            *  Because there is no analythical solution for cos(x)-x = 0, we use Newtons's method to nummerically estimate the
            *  x positon of the intersection first. It is convinient to use the halfway point between the last particle
            *  position outside the wall (at time t-dt) and the current position inside the wall (at time t) as initial
            *  guess for the intersection.
            *
            *  We limit the number of iterations (max_iteration) and the desired presicion (target_presicion) for performance reasons.
            */

            Scalar max_iteration = 5;
            Scalar counter = 0;
            Scalar target_presicion = 0.0001;
            Scalar x0 = pos.x - 0.5*dt*vel.x;
            Scalar A = 0.5*(m_H_wide-m_H_narrow);
            Scalar delta = abs(0 - sign*(A*fast::cos(x0*m_pi_period_div_L)+ A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z);
            
            Scalar n,n2;
            Scalar s,c;
            
            while( delta > target_presicion && counter < max_iteration)
                {
                fast::sincos(x0*m_pi_period_div_L,s,c);
                n  =  sign*(A*c + A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z;  // f
                n2 = -sign*(A*s + A + m_H_narrow) - vel.z/vel.x;                       // df
                x0 = x0 - n/n2;                                                        // x = x - f/df
                delta = abs(0 - sign*(A*fast::cos(x0*m_pi_period_div_L)+ A + m_H_narrow) - vel.z/vel.x*(x0 - pos.x) - pos.z);
                counter +=1;
                }

            /* The new z position is calculated from the wall equation to guarantee that the new particle positon is exactly at the wall
             * and not accidentally slightly inside of the wall because of nummerical presicion.
             */
            fast::sincos(x0*m_pi_period_div_L,s,c);
            Scalar z0 = sign*(A*c+A+m_H_narrow);

            /* The new y position can be calculated from the fact that the last position outside of the wall, the current position inside
             * of the  wall, and the new position exactly at the wall are on a straight line.
             */
            Scalar y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);

            // Remaining integration time dt is amount of time spent traveling distance out of bounds.
            dt = fast::sqrt(((pos.x-x0)*(pos.x - x0) + (pos.y-y0)*(pos.y -y0) + (pos.z-z0)*(pos.z - z0))/(vel.x*vel.x+vel.z*vel.z+vel.y*vel.y));


            /* update velocity according to boundary conditions. No-slip requires reflection of the tangential components:
             * velocity_new = velocity -2*dot(velocity,normal)*normal
             * A upwards normal of the surface is given by (-df/dx,-df/dy,1) with f = sign*((H-h)*cos(x*pi*p/L)+H), so
             * normal  = (sign*(H-h)*pi*p/L*sin(x*pi*p/L),0,1)/|normal|
             * The direction of the normal is not important for the reflection.
             * Calculate components by hand to avoid sqrt in normalization of the normal of the surface.
             */
            
            if (m_bc ==  mpcd::detail::boundary::no_slip)
                {
                Scalar B = sign*A*m_pi_period_div_L*s;
                vel.x = vel.x- 2*(B*B*vel.x +B*vel.z)/(B*B+1) + Scalar(sign * 2) * m_V;
                vel.z = vel.z - 2*(vel.z + B*vel.x)/(B*B+1);
                }
            else // Slip conditions require both normal and tangential components to be reflected:
                {
                vel.x = -vel.x + Scalar(sign * 2) * m_V;
                vel.y = -vel.y;
                vel.z = -vel.z;
                }

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
	    const Scalar max_shift = 0.5; // todo get this from  mpcd , can't use function argument because hoomd/mpcd/ConfinedStreamingMethod.h complains
	    // if arguments of validateBox () change 
	    //
            const Scalar filler_thickness = cell_size +  0.5*(m_H_wide-m_H_narrow)*fast::sin((1+max_shift)*cell_size*m_pi_period_div_L);
	     
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
        const Scalar m_Repetitions;         //!< Nubmer of repetitions of the wide sections in the channel
        const Scalar m_V;                   //!< Velocity of the wall
        const mpcd::detail::boundary m_bc; //!< Boundary condition
    };

} // end namespace detail
} // end namespace azplugins
#undef HOSTDEVICE

#endif // AZPLUGINS_MPCD_SINE_GEOMETRY_H_
