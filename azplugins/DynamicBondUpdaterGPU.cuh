// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cuh
 * \brief Declaration of kernel drivers for DynamicBondUpdaterGPU
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/ParticleData.cuh"
#include <iostream>

#include "hoomd/extern/neighbor/neighbor/BoundingVolumes.h"
#include "hoomd/extern/neighbor/neighbor/InsertOps.h"
#include "hoomd/extern/neighbor/neighbor/TransformOps.h"

#ifdef NVCC
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace azplugins
{
namespace gpu
{

cudaError_t sort_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                     const unsigned int size);

cudaError_t filter_existing_bonds(Scalar3 *d_all_possible_bonds,
                             unsigned int *d_n_existing_bonds,
                             const unsigned int *d_existing_bonds_list,
                             const Index2D& exli,
                             const unsigned int size,
                             const unsigned int block_size);

cudaError_t sort_and_remove_zeros_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                            const unsigned int size,
                                            int *d_max_non_zero_bonds);

cudaError_t remove_zeros_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                             const unsigned int size,
                                             int *d_max_non_zero_bonds);

cudaError_t make_sorted_index_array(unsigned int *d_sorted_indexes,
                                unsigned int *d_indexes_group_1,
                                unsigned int *d_indexes_group_2,
                                const unsigned int size_group_1,
                                const unsigned int size_group_2,
                                const unsigned int block_size);

cudaError_t nlist_copy_nlist_possible_bonds(Scalar3 *d_all_possible_bonds,
                          const Scalar4 *d_postype,
                          const unsigned int * d_tag,
                          const unsigned int * d_sorted_indexes,
                          const unsigned int * d_n_neigh,
                          const unsigned int * d_nlist,
                          const BoxDim box,
                          const unsigned int max_bonds,
                          const  Scalar r_cut,
                          const unsigned int size,
                          const unsigned int block_size);

//! Insert operation for a point under a mapping.
/*!
 * Extends the base neighbor::PointInsertOp to insert a point primitive
 * subject to a mapping of the indexes. This is useful for reading from
 * the array of particles that is pre-sorted by type so that the original
 * particle data does not need to be shuffled.
 */
struct PointMapInsertOp : public neighbor::PointInsertOp
    {
    //! Constructor
    /*!
     * \param points_ List of points to insert (w entry is unused).
     * \param map_ Map of the nominal index to the index in \a points_.
     * \param N_ Number of primitives to insert.
     */
    PointMapInsertOp(const Scalar4 *points_, const unsigned int *map_, unsigned int N_)
        : neighbor::PointInsertOp(points_, N_), map(map_)
        {}

    #ifdef NVCC
    //! Construct bounding box
    /*!
     * \param idx Nominal index of the primitive [0,N).
     * \returns A neighbor::BoundingBox corresponding to the point at map[idx].
     */
    DEVICE neighbor::BoundingBox get(const unsigned int idx) const
        {
        const Scalar4 point = points[map[idx]];
        const Scalar3 p = make_scalar3(point.x, point.y, point.z);
        // construct the bounding box for a point
        return neighbor::BoundingBox(p,p);
        }
    #endif

    const unsigned int *map;    //!< Map of particle indexes.
    };

//! Neighbor list particle query operation.
/*!
 * \tparam use_body If true, use the body fields during query.
 * \tparam use_diam If true, use the diameter fields during query.
 *
 * This operation specifies the neighbor list traversal scheme. The
 * query is between a SkippableBoundingSphere and the bounding boxes in
 * the LBVH. The template parameters can be activated to engage body-filtering
 * or diameter-shifting, which are defined elsewhere in HOOMD.
 *
 * All spheres in the traversal are given the same search radius. This is compatible
 * with a traversal-per-type-per-type scheme. It was found that passing this radius
 * as a constant to the traversal program decreased register pressure in the kernel
 * from a traversal-per-type scheme.
 *
 * The particles are traversed using a \a map. Ghost particles can be included
 * in this map, and they will be neglected during traversal.
 */
struct ParticleQueryOp
    {
    //! Constructor
    /*!
     * \param positions_ Particle positions.
     * \param map_ Map of the particle indexes to traverse.
     * \param N_ Number of particles (total).
     * \param Nown_ Number of locally owned particles.
     * \param rcut_ Cutoff radius for the spheres.
     */
    ParticleQueryOp(const Scalar4 *positions_,
                    const unsigned int* map_,
                    unsigned int N_,
                    unsigned int Nown_,
                    const Scalar rcut_,
                    const BoxDim& box_)
        : positions(positions_), map(map_),
          N(N_), Nown(Nown_), rcut(rcut_), box(box_)
          {}

    #ifdef NVCC
    //! Data stored per thread for traversal
    /*!
     * The body tag and diameter are only actually set if these are specified
     * by the template parameters. The compiler might be able to optimize them
     * out if they are unused.
     */
    struct ThreadData
        {
        HOSTDEVICE ThreadData(Scalar3 position_,
                              int idx_)
            : position(position_), idx(idx_)
            {}

        Scalar3 position;   //!< Particle position
        int idx;            //!< True particle index
        };

    // specify that the traversal Volume is a bounding sphere
    typedef neighbor::BoundingSphere Volume;

    //! Loads the per-thread data
    /*!
     * \param idx Nominal primitive index.
     * \returns The ThreadData required for traversal.
     *
     * The ThreadData is loaded subject to a mapping. The particle position
     * is always loaded. The body and diameter are only loaded if the template
     * parameter requires it.
     */
    DEVICE ThreadData setup(const unsigned int idx) const
        {
        const unsigned int pidx = map[idx];

        const Scalar4 position = positions[pidx];
        const Scalar3 r = make_scalar3(position.x, position.y, position.z);

        return ThreadData(r, pidx);
        }

    //! Return the traversal volume subject to a translation
    /*!
     * \param q The current thread data.
     * \param image The image vector for traversal.
     * \returns The traversal bounding volume.
     *
     * The ThreadData is converted to a search volume. The search sphere is
     * made to be skipped if this is a ghost particle.
     */
    DEVICE Volume get(const ThreadData& q, const Scalar3& image) const
        {
        return Volume(q.position+image, (q.idx < Nown) ? rcut : -1.0);
        }

    //! Perform the overlap test with the LBVH
    /*!
     * \param v Traversal volume.
     * \param box Box in LBVH to intersect with.
     * \returns True if the volume and box overlap.
     *
     * The overlap test is implemented by the sphere.
     */
    DEVICE bool overlap(const Volume& v, const neighbor::BoundingBox& box) const
        {
        return v.overlap(box);
        }

    //! Refine the rough overlap test with a primitive
    /*!
     * \param q The current thread data.
     * \param primitive Index of the intersected primitive.
     * \returns True If the volumes still overlap after refinement.
     *
     */
    DEVICE bool refine(const ThreadData& q, const int primitive) const
        {
        bool exclude = (q.idx == primitive);
        return !exclude;
        }
    #endif

    //! Get the number of primitives
    HOSTDEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4 *positions;   //!< Particle positions
    const unsigned int *map;    //!< Mapping of particles to read
    unsigned int N;             //!< Total number of particles in map
    unsigned int Nown;          //!< Number of particles owned by the local rank
    Scalar rcut;                //!< True cutoff radius + buffer
    const BoxDim box;           //!< Box dimensions
    };

//! Operation to write the neighbor list
/*!
 * The neighbor list is assumed to be aligned to multiples of 4. This enables
 * coalescing writes into packets of 4 neighbors without adding much register pressure.
 * This object maintains an internal stack to do this, and it can restart from a previous
 * traversal without losing information.
 */
struct NeighborListOp
    {
    //! Constructor
    /*!
     * \param neigh_list_ Neighbor list (aligned to multiple of 4)
     * \param nneigh_ Neighbor of neighbors per particle
     * \param new_max_neigh_ Maximum number of neighbors to allocate if overflow occurs.
     * \param max_neigh_ Maximum number of neighbors to allow per particle.
     *
     * The \a neigh_list_ pointer is internally cast into a uint4 for coalescing.
     */
    NeighborListOp(unsigned int* neigh_list_,
                   unsigned int* nneigh_,
                   unsigned int* new_max_neigh_,
                   unsigned int max_neigh_)
        : nneigh(nneigh_), new_max_neigh(new_max_neigh_), max_neigh(max_neigh_)
        {
        neigh_list = reinterpret_cast<uint4*>(neigh_list_);
        }

    #ifdef NVCC
    //! Thread-local data
    /*!
     * The thread-local data constitutes a stack of neighbors to write, the index of the current
     * primitive, the first index to write into, and the current number of neighbors found for this thread.
     */
    struct ThreadData
        {
        //! Constructor
        /*!
         * \param idx_ The index of this particle.
         * \param first_ The first neighbor index of this particle.
         * \param num_neigh_ The current number of neighbors of this particle.
         * \param stack_ The initial values for the stack (can be all 0s if \a num_neigh_ is aligned to 4).
         */
        DEVICE ThreadData(const unsigned int idx_,
                          const unsigned int first_,
                          const unsigned int num_neigh_,
                          const uint4 stack_)
            : idx(idx_), first(first_), num_neigh(num_neigh_)
            {
            stack[0] = stack_.x;
            stack[1] = stack_.y;
            stack[2] = stack_.z;
            stack[3] = stack_.w;
            }

        unsigned int idx;       //!< Index of primitive
        unsigned int first;     //!< First index to use for writing neighbors
        unsigned int num_neigh; //!< Number of neighbors for this thread
        unsigned int stack[4];  //!< Internal stack of neighbors
        };

    //! Setup the thread data
    /*!
     * \param idx Index of this thread.
     * \param q Thread-local query data.
     * \returns The ThreadData for output.
     *
     * \tparam Type of QueryData.
     *
     * This setup function can poach data from the query data in order to save loads.
     * In this case, it makes use of the particle index mapping.
     */
    template<class QueryDataT>
    DEVICE ThreadData setup(const unsigned int idx, const QueryDataT& q) const
        {
        //const unsigned int first = __ldg(first_neigh + q.idx);
        const unsigned int first = q.idx+q.idx*(max_neigh-1);
        const unsigned int num_neigh = nneigh[q.idx]; // no __ldg, since this is writeable

        // prefetch from the stack if current number of neighbors does not align with a boundary
        /* NOTE: There seemed to be a compiler error/bug when stack was declared outside this if
                 statement, initialized with zeros, and then assigned inside (so that only
                 one return statement was needed). It went away using:
                 uint4 tmp = neigh_list[...];
                 stack = tmp;
                 But this looked funny, so the structure below seems more human readable.
         */
        if (num_neigh % 4 != 0)
            {
            uint4 stack = neigh_list[(first+num_neigh-1)/4];
            return ThreadData(q.idx, first, num_neigh, stack);
            }
        else
            {
            return ThreadData(q.idx, first, num_neigh, make_uint4(0,0,0,0));
            }
        }

    //! Processes a newly intersected primitive.
    /*!
     * \param t My output thread data.
     * \param primitive The index of the primitive to process.
     *
     * If the neighbor will fit into the allocated memory, it is pushed onto the stack.
     * The stack is written to memory if it is full. The number of neighbors found for this
     * thread is incremented, regardless.
     */
    DEVICE void process(ThreadData& t, const int primitive) const
        {
        if (t.num_neigh < max_neigh)
            {
            // push primitive into the stack of 4, pre-increment
            const unsigned int offset = t.num_neigh % 4;
            t.stack[offset] = primitive;
            // coalesce writes into chunks of 4
            if (offset == 3)
                {
                neigh_list[(t.first+t.num_neigh)/4] = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
                }
            }
        ++t.num_neigh;
        }

    //! Finish the output job once the thread is ready to terminate.
    /*!
     * \param t My output thread data
     *
     * The number of neighbors found for this thread is written. If this value
     * exceeds the current allocation, this value is atomically maximized for
     * reallocation. Any values remaining on the stack are written to ensure the
     * list is complete.
     */
    DEVICE void finalize(const ThreadData& t) const
        {
        nneigh[t.idx] = t.num_neigh;
      //  printf("max %d %d %d\n",new_max_neigh,max_neigh,t.num_neigh);
        if (t.num_neigh > max_neigh)
            {
            atomicMax(new_max_neigh, t.num_neigh);
            }
        else if (t.num_neigh % 4 != 0)
            {
            // write partial (leftover) stack, counting is now post-increment so need to shift by 1
            // only need to do this if didn't overflow, since all neighbors were already written due to alignment of max
            neigh_list[(t.first+t.num_neigh-1)/4] = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
            }
        }
    #endif

    uint4* neigh_list;                  //!< Neighbors of each sphere
    unsigned int* nneigh;               //!< Number of neighbors per search sphere
    unsigned int* new_max_neigh;        //!< New maximum number of neighbors
    unsigned int max_neigh;             //!< Maximum number of neighbors allocated
    };

//! Sentinel for an invalid particle (e.g., ghost)
const unsigned int NeighborListTypeSentinel = 0xffffffff;


} // end namespace gpu
} // end namespace azplugins



#undef DEVICE
#undef HOSTDEVICE


#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
