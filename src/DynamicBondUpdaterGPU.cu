// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cu
 * \brief Definition of kernel drivers and kernels for DynamicBondUpdaterGPU
 */

#include "DynamicBondUpdaterGPU.cuh"
#include "NeighborListGPUTree.cuh"
#include "hoomd/HOOMDMath.h"
// #include <thrust/sort.h>
#include <neighbor/neighbor.h>
#include <thrust/device_vector.h>

namespace hoomd
    {
namespace azplugins
    {

// todo: migrate to separate file/class.
// this sorts according distance, then first tag, then second tag
struct SortBondsGPU
    {
    __host__ __device__ bool operator()(const Scalar3& i, const Scalar3& j)
        {
        const Scalar r_sq_1 = i.z;
        const Scalar r_sq_2 = j.z;
        if (r_sq_1 == r_sq_2)
            {
            const unsigned int tag_11 = __scalar_as_int(i.x);
            const unsigned int tag_21 = __scalar_as_int(j.x);
            if (tag_11 == tag_21)
                {
                const unsigned int tag_12 = __scalar_as_int(i.y);
                const unsigned int tag_22 = __scalar_as_int(j.y);
                return tag_22 > tag_12;
                }
            else
                {
                return tag_21 > tag_11;
                }
            }
        else
            {
            return r_sq_2 > r_sq_1;
            }
        }
    };

// returns true if given possible bond is zero, e.g. (0,0,0.0)
// possible bonds are ordered, such that tag_a < tag_b in (tag_a,tag_b,rsq)
// meaning we only need to check tag_b == 0
struct isZeroBondGPU
    {
    __host__ __device__ bool operator()(const Scalar3& i)
        {
        const unsigned int tag_1 = __scalar_as_int(i.y);
        return !(bool)tag_1;
        }
    };

struct CompareBondsGPU
    {
    __host__ __device__ bool operator()(const Scalar3& i, const Scalar3& j)
        {
        const unsigned int tag_11 = __scalar_as_int(i.x);
        const unsigned int tag_12 = __scalar_as_int(i.y);
        const unsigned int tag_21 = __scalar_as_int(j.x);
        const unsigned int tag_22 = __scalar_as_int(j.y);

        if ((tag_11 == tag_21 && tag_12 == tag_22)) // should work because pairs are ordered
            {
            return true;
            }
        else
            {
            return false;
            }
        }
    };

namespace gpu
    {

//! Number of elements of the exclusion list to process in each batch
const unsigned int FILTER_BATCH_SIZE = 4;

namespace kernel
    {

__global__ void copy_nlist_possible_bonds(Scalar3* d_all_possible_bonds,
                                          const Scalar4* d_postype,
                                          const unsigned int* d_tag,
                                          const unsigned int* d_sorted_indexes,
                                          const unsigned int* d_n_neigh,
                                          const unsigned int* d_nlist,
                                          const BoxDim box,
                                          const unsigned int max_bonds,
                                          const Scalar r_cut,
                                          const bool groups_identical,
                                          const unsigned int N)
    {
    // one thread per particle in group_1
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;

    // idx = group index , pidx = actual particle index
    const unsigned int pidx_i = d_sorted_indexes[idx];
    unsigned int n_curr_bond = 0;
    const Scalar r_cutsq = r_cut * r_cut;

    // get all information for this particle
    Scalar4 postype_i = d_postype[pidx_i];
    const unsigned int tag_i = d_tag[pidx_i];
    const unsigned int n_neigh = d_n_neigh[idx];

    // loop over all neighbors of this particle
    for (unsigned int j = 0; j < n_neigh; ++j)
        {
        // get index of neighbor from neigh_list
        const unsigned int pidx_j = d_nlist[idx * max_bonds + j];
        Scalar4 postype_j = d_postype[pidx_j];
        const unsigned int tag_j = d_tag[pidx_j];

        Scalar3 drij = make_scalar3(postype_j.x, postype_j.y, postype_j.z)
                       - make_scalar3(postype_i.x, postype_i.y, postype_i.z);

        // apply periodic boundary conditions (FLOPS: 12)
        drij = box.minImage(drij);

        // same as on the cpu, just not during the tree traversal
        Scalar dr_sq = dot(drij, drij);

        if (dr_sq < r_cutsq)
            {
            if (n_curr_bond < max_bonds)
                {
                Scalar3 d;
                if (groups_identical)
                    {
                    // sort the two tags in this possible bond pair if groups are identical
                    const unsigned int tag_a = tag_j > tag_i ? tag_i : tag_j;
                    const unsigned int tag_b = tag_j > tag_i ? tag_j : tag_i;
                    d = make_scalar3(__int_as_scalar(tag_a), __int_as_scalar(tag_b), dr_sq);
                    }
                else
                    {
                    d = make_scalar3(__int_as_scalar(tag_i), __int_as_scalar(tag_j), dr_sq);
                    }
                d_all_possible_bonds[idx * max_bonds + n_curr_bond] = d;
                }
            ++n_curr_bond;
            }
        }
    }

/*! \param d_all_possible_bonds all possible bonds list
    \param d_n_existing_bonds Number of existing for each particle
    \param d_existing_bonds_list List of exitsting for each particle
    \param exli Indexer for indexing into d_existing_bonds_list
    \param size Length of d_all_possible_bonds
    \param ex_start Start filtering  d_all_possible_bonds from existing bond number \a ex_start

    the kernel filter_existing_bonds() processes the all possible bonds list \a d_nlist and removes
   any entries that already exist. To allow for an arbitrary large number of existing bonds, these
   are processed in batch sizes of FILTER_BATCH_SIZE. The kernel must be called multiple times in
   order to fully remove all exclusions from the nlist.

    \note The driver filter_existing_bonds properly makes as many calls as are necessary, it only
   needs to be called once.

    \b Implementation

    One thread is run for each particle. Existing bonds \a ex_start, \a ex_start + 1, ... are loaded
   in for that particle (or the thread returns if there are no exclusions past that point). The
   thread then loops over the neighbor list, comparing each entry to the list of exclusions. If the
   entry is not excluded, it is written back out. \a d_n_neigh is updated to reflect the current
   number of particles in the list at the end of the kernel call.
*/
__global__ void filter_existing_bonds(Scalar3* d_all_possible_bonds,
                                      const unsigned int* d_n_existing_bonds,
                                      const unsigned int* d_existing_bonds_list,
                                      const Index2D exli,
                                      const unsigned int size,
                                      const unsigned int ex_start)
    {
    // compute the bond index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // quit now if this thread is processing past the end of the list of all possible bonds
    if (idx >= size)
        return;

    Scalar3 current_bond = d_all_possible_bonds[idx];
    unsigned int tag_1 = __scalar_as_int(current_bond.x);
    unsigned int tag_2 = __scalar_as_int(current_bond.y);

    if (tag_1 == 0 && tag_2 == 0)
        return;

    // const unsigned int n_neigh = d_n_neigh[idx];
    const unsigned int n_ex = d_n_existing_bonds[tag_1];

    // quit now if the ex_start flag is past the end of n_ex
    if (ex_start >= n_ex)
        return;

    // count the number of existing bonds to process in this thread
    const unsigned int n_ex_process = n_ex - ex_start;

    // load the existing bond list into "local" memory - fully unrolled loops should dump this into
    // registers
    unsigned int l_existing_bonds_list[FILTER_BATCH_SIZE];
#pragma unroll
    for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
        {
        if (cur_ex_idx < n_ex_process)
            {
            l_existing_bonds_list[cur_ex_idx]
                = d_existing_bonds_list[exli(tag_1, cur_ex_idx + ex_start)];
            }
        else
            {
            l_existing_bonds_list[cur_ex_idx] = 0xffffffff;
            }
        }

    // test if excluded
    bool excluded = false;
#pragma unroll
    for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
        {
        if (tag_2 == l_existing_bonds_list[cur_ex_idx])
            excluded = true;
        }
    // if it is excluded, overwrite that entry with (0,0,0).
    if (excluded)
        {
        d_all_possible_bonds[idx] = make_scalar3(__int_as_scalar(0), __int_as_scalar(0), 0.0);
        }
    }

    } // end namespace kernel

cudaError_t remove_zeros_and_sort_possible_bond_array(Scalar3* d_all_possible_bonds,
                                                      const unsigned int size,
                                                      int* d_max_non_zero_bonds)
    {
    if (size == 0)
        return cudaSuccess;
    // wrapper for pointer needed for thrust
    HOOMD_THRUST::device_ptr<Scalar3> d_all_possible_bonds_wrap(d_all_possible_bonds);

    isZeroBondGPU zero;
    HOOMD_THRUST::device_ptr<Scalar3> last0
        = HOOMD_THRUST::remove_if(d_all_possible_bonds_wrap,
                                  d_all_possible_bonds_wrap + size,
                                  zero);
    unsigned int l0 = HOOMD_THRUST::distance(d_all_possible_bonds_wrap, last0);

    // sort remainder by distance, should make all identical bonds consequtive
    SortBondsGPU sort;
    HOOMD_THRUST::sort(d_all_possible_bonds_wrap, d_all_possible_bonds_wrap + l0, sort);

    // thrust::unique only removes identical consequtive elements, so sort above is needed.
    CompareBondsGPU comp;
    HOOMD_THRUST::device_ptr<Scalar3> last1
        = HOOMD_THRUST::unique(d_all_possible_bonds_wrap, d_all_possible_bonds_wrap + l0, comp);
    unsigned int l1 = HOOMD_THRUST::distance(d_all_possible_bonds_wrap, last1);

    *d_max_non_zero_bonds = l1;

    return cudaSuccess;
    }

cudaError_t filter_existing_bonds(Scalar3* d_all_possible_bonds,
                                  unsigned int* d_n_existing_bonds,
                                  const unsigned int* d_existing_bonds_list,
                                  const Index2D& exli,
                                  const unsigned int size,
                                  const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::filter_existing_bonds);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // determine parameters for kernel launch
    int n_blocks = size / run_block_size + 1;

    // split the processing of the full exclusion list up into a number of batches
    unsigned int n_batches = (unsigned int)ceil(double(exli.getH()) / double(FILTER_BATCH_SIZE));
    unsigned int ex_start = 0;
    for (unsigned int batch = 0; batch < n_batches; batch++)
        {
        kernel::filter_existing_bonds<<<n_blocks, run_block_size>>>(d_all_possible_bonds,
                                                                    d_n_existing_bonds,
                                                                    d_existing_bonds_list,
                                                                    exli,
                                                                    size,
                                                                    ex_start);

        ex_start += FILTER_BATCH_SIZE;
        }

    return cudaSuccess;
    }

cudaError_t copy_possible_bonds(Scalar3* d_all_possible_bonds,
                                const Scalar4* d_postype,
                                const unsigned int* d_tag,
                                const unsigned int* d_sorted_indexes,
                                const unsigned int* d_n_neigh,
                                const unsigned int* d_nlist,
                                const BoxDim box,
                                const unsigned int max_bonds,
                                const Scalar r_cut,
                                const bool groups_identical,
                                const unsigned int N,
                                const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;

    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::copy_nlist_possible_bonds);
        max_block_size = attr.maxThreadsPerBlock;
        }
    unsigned int run_block_size = min(block_size, max_block_size);

    kernel::copy_nlist_possible_bonds<<<N / run_block_size + 1, run_block_size>>>(
        d_all_possible_bonds,
        d_postype,
        d_tag,
        d_sorted_indexes,
        d_n_neigh,
        d_nlist,
        box,
        max_bonds,
        r_cut,
        groups_identical,
        N);
    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace azplugins

    } // end namespace hoomd

// explicit templates for neighbor::LBVH with PointMapInsertOp
template void neighbor::gpu::lbvh_gen_codes(unsigned int*,
                                            unsigned int*,
                                            const azplugins::gpu::PointMapInsertOp&,
                                            const Scalar3,
                                            const Scalar3,
                                            const unsigned int,
                                            const unsigned int,
                                            cudaStream_t);
template void neighbor::gpu::lbvh_bubble_aabbs(const neighbor::gpu::LBVHData,
                                               const azplugins::gpu::PointMapInsertOp&,
                                               unsigned int*,
                                               const unsigned int,
                                               const unsigned int,
                                               cudaStream_t);
template void neighbor::gpu::lbvh_one_primitive(const neighbor::gpu::LBVHData,
                                                const azplugins::gpu::PointMapInsertOp&,
                                                cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(azplugins::gpu::NeighborListOp&,
                                                 const neighbor::gpu::LBVHCompressedData&,
                                                 const azplugins::gpu::ParticleQueryOp&,
                                                 const Scalar3*,
                                                 unsigned int,
                                                 unsigned int,
                                                 cudaStream_t);

/////////////////////////////////////
// neighbor program and wrappers
/////////////////////////////////////

#define DEVICE __device__ __forceinline__

//! Insert operation for a point under a mapping.
/*!
 * Extends the base neighbor::PointInsertOp to insert a point primitive
 * subject to a mapping of the indexes. This is useful for reading from
 * the array of particles that is pre-sorted by type so that the original
 * particle data does not need to be shuffled.
 */
struct PointMapInsertOp
    {
    //! Constructor
    /*!
     * \param points_ List of points to insert (w entry is unused).
     * \param map_ Map of the nominal index to the index in \a points_.
     * \param N_ Number of primitives to insert.
     */
    PointMapInsertOp(const Scalar4* points_, const unsigned int* map_, unsigned int N_)
        : points(points_), map(map_), N(N_)
        {
        }

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
        return neighbor::BoundingBox(p, p);
        }

    __host__ DEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4* points;
    const unsigned int* map; //!< Map of particle indexes.
    const unsigned int N;
    };

//! Neighbor list particle query operation.
/*!
 * \tparam use_body If true, use the body fields during query.
 *
 * This operation specifies the neighbor list traversal scheme. The
 * query is between a SkippableBoundingSphere and the bounding boxes in
 * the LBVH. The template parameters can be activated to engage body-filtering
 * which is defined elsewhere in HOOMD.
 *
 * All spheres in the traversal are given the same search radius. This is compatible
 * with a traversal-per-type-per-type scheme. It was found that passing this radius
 * as a constant to the traversal program decreased register pressure in the kernel
 * from a traversal-per-type scheme.
 *
 * The particles are traversed using a \a map. Ghost particles can be included
 * in this map, and they will be neglected during traversal.
 */
template<bool use_body> struct ParticleQueryOp
    {
    //! Constructor
    /*!
     * \param positions_ Particle positions.
     * \param bodies_ Particle body tags.
     * \param map_ Map of the particle indexes to traverse.
     * \param N_ Number of particles (total).
     * \param Nown_ Number of locally owned particles.
     * \param rcut_ Cutoff radius for the spheres.
     * \param rlist_ Total search radius for the spheres (differs under shifting).
     */
    ParticleQueryOp(const Scalar4* positions_,
                    const unsigned int* bodies_,
                    const unsigned int* map_,
                    unsigned int N_,
                    unsigned int Nown_,
                    const Scalar rcut_,
                    const Scalar rlist_,
                    const BoxDim& box_)
        : positions(positions_), bodies(bodies_), map(map_), N(N_), Nown(Nown_), rcut(rcut_),
          rlist(rlist_), box(box_)
        {
        }

    //! Data stored per thread for traversal
    /*!
     * The body tags are only actually set if these are specified
     * by the template parameters. The compiler might be able to optimize them
     * out if they are unused.
     */
    struct ThreadData
        {
        DEVICE ThreadData(Scalar3 position_, int idx_, unsigned int body_)
            : position(position_), idx(idx_), body(body_)
            {
            }

        Scalar3 position;  //!< Particle position
        int idx;           //!< True particle index
        unsigned int body; //!< Particle body tag (may be invalid)
        };

    // specify that the traversal Volume is a bounding sphere
    typedef SkippableBoundingSphere Volume;

    //! Loads the per-thread data
    /*!
     * \param idx Nominal primitive index.
     * \returns The ThreadData required for traversal.
     *
     * The ThreadData is loaded subject to a mapping. The particle position
     * is always loaded. The body is only loaded if the template
     * parameter requires it.
     */
    DEVICE ThreadData setup(const unsigned int idx) const
        {
        const unsigned int pidx = map[idx];

        const Scalar4 position = positions[pidx];
        const Scalar3 r = make_scalar3(position.x, position.y, position.z);

        unsigned int body(0xffffffff);
        if (use_body)
            {
            body = __ldg(bodies + pidx);
            }

        return ThreadData(r, pidx, body);
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
        return Volume(q.position + image, (q.idx < Nown) ? rlist : -1.0);
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
     * HOOMD's neighbor lists require additional filtering. This first ensures
     * that the overlap is not with itself. If body filtering is enabled,
     * particles in the same body do not overlap.
     */
    DEVICE bool refine(const ThreadData& q, const int primitive) const
        {
        bool exclude = (q.idx == primitive);

        // body exclusion
        if (use_body && !exclude && q.body != 0xffffffff)
            {
            const unsigned int body = __ldg(bodies + primitive);
            exclude |= (q.body == body);
            }

        return !exclude;
        }

    //! Get the number of primitives
    __host__ DEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4* positions;   //!< Particle positions
    const unsigned int* bodies; //!< Particle bodies
    const unsigned int* map;    //!< Mapping of particles to read
    unsigned int N;             //!< Total number of particles in map
    unsigned int Nown;          //!< Number of particles owned by the local rank
    Scalar rcut;                //!< True cutoff radius + buffer
    Scalar rlist;               //!< Maximum cutoff (may include shifting)
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
     * \param first_neigh_ First index for the current particle index in the neighbor list.
     * \param max_neigh_ Maximum number of neighbors to allow per particle.
     *
     * The \a neigh_list_ pointer is internally cast into a uint4 for coalescing.
     */
    NeighborListOp(unsigned int* neigh_list_,
                   unsigned int* nneigh_,
                   unsigned int* new_max_neigh_,
                   const size_t* first_neigh_,
                   unsigned int max_neigh_)
        : nneigh(nneigh_), new_max_neigh(new_max_neigh_), first_neigh(first_neigh_),
          max_neigh(max_neigh_)
        {
        neigh_list = reinterpret_cast<uint4*>(neigh_list_);
        }

    //! Thread-local data
    /*!
     * The thread-local data constitutes a stack of neighbors to write, the index of the current
     * primitive, the first index to write into, and the current number of neighbors found for this
     * thread.
     */
    struct ThreadData
        {
        //! Constructor
        /*!
         * \param idx_ The index of this particle.
         * \param first_ The first neighbor index of this particle.
         * \param num_neigh_ The current number of neighbors of this particle.
         * \param stack_ The initial values for the stack (can be all 0s if \a num_neigh_ is aligned
         * to 4).
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
        size_t first;           //!< First index to use for writing neighbors
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
        const size_t first = __ldg(first_neigh + q.idx);
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
            uint4 stack = neigh_list[(first + num_neigh - 1) / 4];
            return ThreadData(q.idx, first, num_neigh, stack);
            }
        else
            {
            return ThreadData(q.idx, first, num_neigh, make_uint4(0, 0, 0, 0));
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
                neigh_list[(t.first + t.num_neigh) / 4]
                    = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
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
        if (t.num_neigh > max_neigh)
            {
            atomicMax(new_max_neigh, t.num_neigh);
            }
        else if (t.num_neigh % 4 != 0)
            {
            // write partial (leftover) stack, counting is now post-increment so need to shift by 1
            // only need to do this if didn't overflow, since all neighbors were already written due
            // to alignment of max
            neigh_list[(t.first + t.num_neigh - 1) / 4]
                = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
            }
        }

    uint4* neigh_list;           //!< Neighbors of each sphere
    unsigned int* nneigh;        //!< Number of neighbors per search sphere
    unsigned int* new_max_neigh; //!< New maximum number of neighbors
    const size_t* first_neigh;   //!< Index of first neighbor
    unsigned int max_neigh;      //!< Maximum number of neighbors allocated
    };

//! Host function to convert a double to a float in round-down mode
float double2float_rd(double x)
    {
    float xf = static_cast<float>(x);
    if (static_cast<double>(xf) > x)
        {
        xf = std::nextafterf(xf, -std::numeric_limits<float>::infinity());
        }
    return xf;
    }