#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Emitter<Float, Spectrum>::Emitter(const Properties &props) : Base(props) { 
    m_min_bounds = props.vector3f("min_bounds", 0.0f);
    m_range_bounds = props.vector3f("range_bounds", 1.0f);
    m_num_parameters = props.int_("num_parameters", 3);
}
MTS_VARIANT Emitter<Float, Spectrum>::~Emitter() { }

MTS_IMPLEMENT_CLASS_VARIANT(Emitter, Endpoint, "emitter")
MTS_INSTANTIATE_CLASS(Emitter)
NAMESPACE_END(mitsuba)
