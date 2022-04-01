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

#include "../common/default.h"
#include "../common/scene.h"
#include "../../common/simd/simd.h"
#include "../common/primref.h"
#include "../common/primref_mb.h"

namespace embree
{
  struct PrimitiveType
  {
    /*! returns name of this primitive type */
    virtual const char* name() const = 0;
    
    /*! Returns the number of stored active primitives in a block. */
    virtual size_t sizeActive(const char* This) const = 0;

    /*! Returns the number of stored active and inactive primitives in a block. */
    virtual size_t sizeTotal(const char* This) const = 0;

    /*! Returns the number of bytes of block. */
    virtual size_t getBytes(const char* This) const = 0;
  };
  
  template<typename Primitive>
  struct PrimitivePointQuery1
  {
    static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& prim)
    {
      bool changed = false;
      for (size_t i = 0; i < Primitive::max_size(); i++)
      {
        if (!prim.valid(i)) break;
        STAT3(point_query.trav_prims,1,1,1);
        AccelSet* accel = (AccelSet*)context->scene->get(prim.geomID(i));
        context->geomID = prim.geomID(i);
        context->primID = prim.primID(i);
        changed |= accel->pointQuery(query, context);
      }
      return changed;
    }
    
    static __forceinline void pointQueryNoop(PointQuery* query, PointQueryContext* context, const Primitive& prim) { }
  };
}
