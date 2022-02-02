// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file SinusoidalChannelGeometry.h
 * \brief Definition of the MPCD symmetric cosine channel geometry
 */

#ifndef AZPLUGINS_SINUSOIDAL_CHANNEL_GEOMETRY_H_
#define AZPLUGINS_SINUSOIDAL_CHANNEL_GEOMETRY_H_

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

//! Sinusoidal channel geometry
/*!
 * This class defines a channel with anti-symmetric cosine walls given by the
 * equations (A cos(x*2*pi*p/Lx) +/- H_narrow), creating a sinusoidal channel.
 * A is the amplitude and p is the period of the wall cosine.
 * H_narrow is the half height of the channel. This creates a wavy channel.
 * The cosine wall wavelength/frenquency needs to be consumable with the
 * periodic boundary conditions in x, therefore the period p is specified and
 * the wavelength 2*pi*p/Lx is calculated.
 *
 * Below is what the channel looks like with A=5, h=2, p=1 in a box of 10x10x18:
 *
 *    8 +------------------------------------------------+
 *      |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXXXXXXXXX========XXXXXXXXXXXXXXXXXXXX|
 *    6 |XXXXXXXXXXXXXXXXXX==        ==XXXXXXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXXXXXX==          ==XXXXXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXXXX==              ==XXXXXXXXXXXXXXX|
 *    4 |XXXXXXXXXXXXXX==                ==XXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXX==                  ==XXXXXXXXXXXXX|
 *      |XXXXXXXXXXXX==      ========      ==XXXXXXXXXXXX|
 *    2 |XXXXXXXXXXX==     ===XXXXXX===     ==XXXXXXXXXXX|
 *      |XXXXXXXXXX==     ==XXXXXXXXXX==     ==XXXXXXXXXX|
 *      |XXXXXXXXX==     ==XXXXXXXXXXXX==     ==XXXXXXXXX|
 *    0 |XXXXXXXX==     =XXXXXXXXXXXXXXXX=     ==XXXXXXXX|
 * z    |XXXXXXX==     =XXXXXXXXXXXXXXXXXX=     ==XXXXXXX|
 *      |XXXXXX=      =XXXXXXXXXXXXXXXXXXXX=      =XXXXXX|
 *      |XXXX==     ==XXXXXXXXXXXXXXXXXXXXXX==     ==XXXX|
 *   -2 |XX===     ==XXXXXXXXXXXXXXXXXXXXXXXX==     ===XX|
 *      |===      ==XXXXXXXXXXXXXXXXXXXXXXXXXX==      ===|
 *      |        ==XXXXXXXXXXXXXXXXXXXXXXXXXXXX==        |
 *   -4 |       ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==       |
 *      |      ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==      |
 *      |     ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==     |
 *   -6 |   ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==   |
 *      |===XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX===|
 *      |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *   -8 +------------------------------------------------+
 *          -4        -2         0        2         4
 *                               x
 *
 *
 * The wall boundary conditions can optionally be changed to slip conditions.
 */
class __attribute__((visibility("default"))) SinusoidalChannel
    {
    public:
        //! Constructor
        /*!
         * \param L Channel length (Simulation box length in x)
           \param Amplitude Channel Cosine Amplitude
           \param H_narrow Channel half-width
           \param Period Channel cosine period (integer >0)
         * \param bc Boundary condition at the wall (slip or no-slip)
         */
        HOSTDEVICE SinusoidalChannel(Scalar L, Scalar Amplitude, Scalar h, unsigned int Repetitions, mpcd::detail::boundary bc)
            : m_pi_period_div_L(2*M_PI*Repetitions/L), m_Amplitude(Amplitude), m_h(h), m_Repetitions(Repetitions), m_bc(bc)
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

            Scalar a = pos.z - m_Amplitude*fast::cos(pos.x*m_pi_period_div_L);
            const signed char sign = (a > m_h) - (a < -m_h);
            // exit immediately if no collision is found
            if (sign == 0)
                {
                dt = Scalar(0);
                return false;
                }

            /* Calculate position (x0,y0,z0) of collision with wall:
            *  Because there is no analytical solution for f(x) = cos(x)-x = 0, we use Newton's method to numerically estimate the
            *  x positon of the intersection first. It is convinient to use the halfway point between the last particle
            *  position outside the wall (at time t-dt) and the current position inside the wall (at time t) as initial
            *  guess for the intersection.
            *
            *  We limit the number of iterations (max_iteration) and the desired presicion (target_precision) for performance reasons.
            */
            const unsigned int max_iteration = 6;
            const Scalar target_precision = 1e-5;

            Scalar x0 = pos.x - 0.5*dt*vel.x;
            Scalar y0;
            Scalar z0;


            /* catch the case where a particle collides exactly vertically (v_x=0 -> old x pos = new x pos)
             * In this case, y0 = -(0)*0/0 + (y-dt*v_y) == nan, should be y0 =(y-dt*v_y)
             */
            if (vel.x == 0) // exactly vertical x-collision
                {
                x0 = pos.x;
                y0 = (pos.y-dt*vel.y);
                z0 = (m_Amplitude*fast::cos(x0*m_pi_period_div_L)+sign*m_h);

                }
            else if (vel.z == 0) // exactly horizontal z-collision
                {
                x0 = 1/m_pi_period_div_L*fast::acos((pos.z-sign*m_h)/m_Amplitude);
                y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);
                z0 = pos.z;
                }
            else
                {
                Scalar delta = fabs(0-((m_Amplitude*fast::cos(x0*m_pi_period_div_L)+ sign*m_h) - vel.z/vel.x*(x0 - pos.x) - pos.z));

                Scalar n,n2;
                Scalar s,c;
                unsigned int counter = 0;
                while(delta > target_precision && counter < max_iteration)
                    {
                    fast::sincos(x0*m_pi_period_div_L,s,c);
                    n = (m_Amplitude*c + sign*m_h) - vel.z/vel.x*(x0 - pos.x) - pos.z;  // f
                    n2 = -m_pi_period_div_L*m_Amplitude*s - vel.z/vel.x;                // df
                    x0 = x0 - n/n2;                                                     // x = x - f/df
                    delta = fabs(0-((m_Amplitude*fast::cos(x0*m_pi_period_div_L)+sign*m_h) - vel.z/vel.x*(x0 - pos.x) - pos.z));
                    ++counter;
                    }

                /* The new z position is calculated from the wall equation to guarantee that the new particle positon is exactly at the wall
                 * and not accidentally slightly inside of the wall because of nummerical presicion.
                 */
                z0 = (m_Amplitude*fast::cos(x0*m_pi_period_div_L)+sign*m_h);

                /* The new y position can be calculated from the fact that the last position outside of the wall, the current position inside
                 * of the  wall, and the new position exactly at the wall are on a straight line.
                 */
                y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);

                // Newton's method sometimes failes to converge (close to saddle points, df'==0, bad initial guess,overshoot,..)
                // catch all of them here and do bisection if Newthon's method didn't work
                Scalar lower_x = fmin(pos.x - dt*vel.x,pos.x);
                Scalar upper_x = fmax(pos.x - dt*vel.x,pos.x);

                // found intersection is NOT in between old and new point, ie intersection is wrong/inaccurate.
                // do bisection to find intersection - slower but more robust than Newton's method
                if (x0 < lower_x || x0 > upper_x)
                    {
                    counter = 0;
                    Scalar3 point1 = pos;  // final position at t+dt, outside of channel
                    Scalar3 point2 = pos-dt*vel; // initial position, inside of channel
                    Scalar3 point3 = 0.5*(point1+point2); // halfway point
                    Scalar fpoint3 = (m_Amplitude*fast::cos(point3.x*m_pi_period_div_L) + sign*m_h) - point3.z; // value at halfway point, f(x)
                    // Note: technically, the presicion of Newton's method and bisection is slightly different, with
                    // bisection being less precise and slower convergence.
                    while (fabs(fpoint3) > target_precision && counter < max_iteration)
                        {
                        fpoint3 = (m_Amplitude*fast::cos(point3.x*m_pi_period_div_L) + sign*m_h) - point3.z;
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
                    x0 =  point3.x;
                    z0 = (m_Amplitude*fast::cos(x0*m_pi_period_div_L)+sign*m_h);
                    y0 = -(pos.x-dt*vel.x - x0)*vel.y/vel.x + (pos.y-dt*vel.y);
                    }
                }


            // Remaining integration time dt is amount of time spent traveling distance out of bounds.
            Scalar3 pos_new = make_scalar3(x0,y0,z0);
            dt = fast::sqrt(dot((pos-pos_new),(pos-pos_new))/dot(vel,vel));
            pos = pos_new;

            /* update velocity according to boundary conditions.
             *
             * A upwards normal of the surface is given by (-df/dx,-df/dy,1) with f = (A*cos(x*2*pi*p/L) +/- sign*h), so
             * normal  = (A*2*pi*p/L*sin(x*2*pi*p/L),0,1)/|length|
             * We define B = A*2*pi*p/L*sin(x*2*pi*p/L), so then the normal is given by (B,0,1)/sqrt(B^2+1)
             * The direction of the normal is not important for the reflection.
             */
            Scalar3 vel_new;
            if (m_bc ==  mpcd::detail::boundary::no_slip) // No-slip requires reflection of both tangential and normal components:
                {
                vel_new = -vel;
                }
            else // Slip conditions require only tangential components to be reflected:
                {
                Scalar B = m_Amplitude*m_pi_period_div_L*fast::sin(x0*m_pi_period_div_L);
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
            Scalar a = pos.z - m_Amplitude*fast::cos(pos.x*m_pi_period_div_L);
            return (a > m_h || a < -m_h);
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
            return ((hi-m_Amplitude) >= cell_size && (-m_Amplitude-lo) >= cell_size);
            }

        //! Get channel amplitude
        /*!
         * \returns Channel amplitude
         */
        HOSTDEVICE Scalar getAmplitude() const
            {
            return m_Amplitude;
            }

        //! Get channel half width at narrowest point
        /*!
         * \returns Channel half width at narrowest point
         */
        HOSTDEVICE Scalar getHnarrow() const
            {
            return m_h;
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
            return std::string("SinusoidalChannel");
            }
        #endif // NVCC

    private:
        const Scalar m_pi_period_div_L;     //!< Argument of the wall cosine (pi*period/Lx = 2*pi*repetitions/Lx)
        const Scalar m_Amplitude;           //!< Amplitude of the channel
        const Scalar m_h;                   //!< Half of the channel width
        const unsigned int m_Repetitions;   //!< Number of repetitions of the wide sections in the channel =  period
        const mpcd::detail::boundary m_bc;  //!< Boundary condition
    };

} // end namespace detail
} // end namespace azplugins
#undef HOSTDEVICE

#endif // AZPLUGINS_SINUSOIDAL_CHANNEL_GEOMETRY_H_
