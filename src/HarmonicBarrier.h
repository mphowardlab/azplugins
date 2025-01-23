// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_HARMONIC_BARRIER_H_
#define AZPLUGINS_HARMONIC_BARRIER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/Variant.h"

#include <pybind11/pybind11.h>
#include <string>

namespace hoomd
    {

namespace azplugins
    {

//! Harmonic barrier
template<class BarrierEvaluatorT> class PYBIND11_EXPORT HarmonicBarrier : public ForceCompute
    {
    public:
    //! Constructor
    HarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Variant> location)
        : ForceCompute(sysdef), m_location(location), m_has_warned(false)
        {
        // allocate memory per type for parameters
        GPUArray<Scalar2> params(m_pdata->getNTypes(), m_exec_conf);
        m_params.swap(params);
        }

    //! Destructor
    virtual ~HarmonicBarrier() { }

    //! Harmonic potential parameters
    struct param_type
        {
        Scalar k;
        Scalar offset;

        param_type() : k(0), offset(0) { }

        param_type(Scalar k_, Scalar offset_) : k(k_), offset(offset_) { }

        param_type(const Scalar2& params) : k(params.x), offset(params.y) { }

        param_type(pybind11::dict params)
            {
            k = pybind11::cast<Scalar>(params["k"]);
            offset = pybind11::cast<Scalar>(params["offset"]);
            }

        pybind11::dict toPython()
            {
            pybind11::dict d;
            d["k"] = pybind11::cast(k);
            d["offset"] = pybind11::cast(offset);
            return d;
            }
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Get location of barrier
    std::shared_ptr<Variant> getLocation() const
        {
        return m_location;
        }

    //! Set location of barrier
    void setLocation(std::shared_ptr<Variant> location)
        {
        m_location = location;
        }

    //! Get the per-type potential parameters
    param_type getParams(std::string type_name)
        {
        unsigned int type_idx = m_pdata->getTypeByName(type_name);
        ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
        return param_type(h_params.data[type_idx]);
        }

    //! Set the per-type potential parameters
    void setParams(unsigned int type, const param_type& params)
        {
        ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
        h_params.data[type] = make_scalar2(params.k, params.offset);
        }

    //! Get the per-type potential parameters in Python
    pybind11::dict getParamsPython(std::string type_name)
        {
        return getParams(type_name).toPython();
        }

    //! Set the per-type potential parameters in Python
    void setParamsPython(std::string type_name, pybind11::dict params)
        {
        setParams(m_pdata->getTypeByName(type_name), param_type(params));
        }

    protected:
    std::shared_ptr<Variant> m_location; //!< Current location of the barrier
    GPUArray<Scalar2> m_params;          //!< Per-type array of parameters for the potential

    //! Method to compute the forces
    void computeForces(uint64_t timestep) override;

    //! Make barrier evaluator
    BarrierEvaluatorT makeEvaluator(uint64_t timestep) const
        {
        // make evaluator and check box
        const Scalar location = m_location->operator()(timestep);
        BarrierEvaluatorT evaluator(location);
        if (!evaluator.valid(m_pdata->getGlobalBox()))
            {
            throw std::runtime_error("Barrier position is invalid");
            }
        return evaluator;
        }

    //! Warn about not computing the virial
    void warnVirialOnce()
        {
        PDataFlags flags = m_pdata->getFlags();
        if (!m_has_warned && flags[pdata_flag::pressure_tensor])
            {
            m_exec_conf->msg->warning() << "HarmonicBarrier does not compute its virial "
                                           "contribution, pressure may be inaccurate"
                                        << std::endl;
            m_has_warned = true;
            }
        }

    private:
    bool m_has_warned; //!< Flag if a warning has been issued about the virial
    };

template<class BarrierEvaluatorT>
void HarmonicBarrier<BarrierEvaluatorT>::computeForces(uint64_t timestep)
    {
    const BarrierEvaluatorT evaluator = makeEvaluator(timestep);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar4 postype = h_pos.data[idx];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        const unsigned int type = __scalar_as_int(postype.w);

        const Scalar2 params = h_params.data[type];

        h_force.data[idx] = evaluator(pos, params.x, params.y);
        }

    // virial is not computed, set to zeros
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());
    warnVirialOnce();
    }

namespace detail
    {
template<class BarrierEvaluatorT>
void export_HarmonicBarrier(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    py::class_<HarmonicBarrier<BarrierEvaluatorT>,
               ForceCompute,
               std::shared_ptr<HarmonicBarrier<BarrierEvaluatorT>>>(m, name.c_str())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>())
        .def_property("location",
                      &HarmonicBarrier<BarrierEvaluatorT>::getLocation,
                      &HarmonicBarrier<BarrierEvaluatorT>::setLocation)
        .def("getParams", &HarmonicBarrier<BarrierEvaluatorT>::getParamsPython)
        .def("setParams", &HarmonicBarrier<BarrierEvaluatorT>::setParamsPython);
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_HARMONIC_BARRIER_H_
