// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "accel.h"

namespace embree
{
  /*! merges N acceleration structures together, by processing them in order */
  class AccelN : public Accel
  {
  public:
    AccelN ();
    ~AccelN();

  public:
    void accels_add(Accel* accel);
    void accels_init();

  public:
    static bool pointQuery (Accel::Intersectors* This, PointQuery* query, PointQueryContext* context);

  public:
    static void intersect (Accel::Intersectors* This, RTCRayHit& ray, IntersectContext* context);
    static void intersect4 (const void* valid, Accel::Intersectors* This, RTCRayHit4& ray, IntersectContext* context);
    static void intersect8 (const void* valid, Accel::Intersectors* This, RTCRayHit8& ray, IntersectContext* context);
    static void intersect16 (const void* valid, Accel::Intersectors* This, RTCRayHit16& ray, IntersectContext* context);
    static void intersectN (Accel::Intersectors* This, RTCRayHitN** ray, const size_t N, IntersectContext* context);

  public:
    static void occluded (Accel::Intersectors* This, RTCRay& ray, IntersectContext* context);
    static void occluded4 (const void* valid, Accel::Intersectors* This, RTCRay4& ray, IntersectContext* context);
    static void occluded8 (const void* valid, Accel::Intersectors* This, RTCRay8& ray, IntersectContext* context);
    static void occluded16 (const void* valid, Accel::Intersectors* This, RTCRay16& ray, IntersectContext* context);
    static void occludedN (Accel::Intersectors* This, RTCRayN** ray, const size_t N, IntersectContext* context);

  public:
    void accels_print(size_t ident);
    void accels_immutable();
    void accels_build ();
    void accels_select(bool filter);
    void accels_deleteGeometry(size_t geomID);
    void accels_clear ();

  public:
    std::vector<Accel*> accels;
  };
}
