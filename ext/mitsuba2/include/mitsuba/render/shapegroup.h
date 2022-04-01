#include <mitsuba/core/fwd.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/kdtree.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
#endif
#if defined(MTS_ENABLE_OPTIX)
    #include <mitsuba/render/optix/shapes.h>
#endif

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER ShapeGroup : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_id, m_to_world, m_original_to_world, m_to_object, m_num_parameters, m_min_bounds, m_range_bounds, m_rotation_axis, m_range_angle, m_min_angle)
    MTS_IMPORT_TYPES(ShapeKDTree)

    using typename Base::ScalarSize;

    ShapeGroup(const Properties &props);
    ~ShapeGroup();

#if defined(MTS_ENABLE_EMBREE)
    RTCGeometry embree_geometry(RTCDevice device) override;
#else
    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                        Mask active) const override;

    Mask ray_test(const Ray3f &ray, Mask active) const override;
#endif

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override;

    ScalarSize primitive_count() const override;

    ScalarBoundingBox3f bbox() const override{ return m_bbox; }

    ScalarFloat surface_area() const override { return 0.f; }

    void traverse(TraversalCallback* callback) override {
        for (auto shape : m_shapes) {
            shape->traverse(callback);
        }
    }
    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {}

    bool has_emitter() {
        for (auto shape : m_shapes) {
            if (shape->is_emitter()) return true;
        }

        return false;
    }

    Base* get_emitter() const {
        for (auto shape : m_shapes) {
            if (shape->is_emitter()) return shape.get();
        }
    }

    int num_alternatives() const override { return m_shapes.size(); }

    void set_alternative(int index) override {
        for (int i = 0; i < m_shapes.size(); ++i)
        {
            if(i == index) m_shapes[i]->set_to_world(ScalarTransform4f::translate(ScalarVector3f(0, 0, 0)));
            else  m_shapes[i]->set_to_world(ScalarTransform4f::translate(ScalarVector3f(0, -1000, 0)));
        }
    }

    MTS_INLINE ScalarSize effective_primitive_count() const override { return 0; }

    std::string to_string() const override;

#if defined(MTS_ENABLE_OPTIX)
    void optix_prepare_ias(const OptixDeviceContext& context,
                           std::vector<OptixInstance>& instances,
                           uint32_t instance_id,
                           const ScalarTransform4f& transf) override;

    void optix_fill_hitgroup_records(std::vector<HitGroupSbtRecord> &hitgroup_records,
                                     const OptixProgramGroup *program_groups) override;

    /// Build OptiX geometry acceleration structures
    void optix_build_gas(const OptixDeviceContext& context) {
        if (!optix_accel_ready) {
            build_gas(context, m_shapes, m_accel);
            optix_accel_ready = true;
        }
    }
#endif

    MTS_DECLARE_CLASS()
private:
    ScalarBoundingBox3f m_bbox;

#if defined(MTS_ENABLE_EMBREE) || defined(MTS_ENABLE_OPTIX)
    std::vector<ref<Base>> m_shapes;
#endif

#if defined(MTS_ENABLE_EMBREE)
    RTCScene m_embree_scene = nullptr;
#else
    ref<ShapeKDTree> m_kdtree;
#endif

#if defined(MTS_ENABLE_OPTIX)
    OptixAccelData m_accel;
    bool optix_accel_ready = false;
    /// OptiX hitgroup sbt offset
    uint32_t m_sbt_offset;
#endif
};

MTS_EXTERN_CLASS_RENDER(ShapeGroup)
NAMESPACE_END(mitsuba)
