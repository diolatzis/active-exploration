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

#include <stddef.h>
#include "../common/math/vec.h"

namespace embree { namespace collide2 {

using vec_t = Vec3fa;

struct ClothModel;

class Constraint {
public:

    Constraint (size_t const numConstainedBodies) 
    : numConstrainedBodies_ (numConstainedBodies)
    {
        bodyIDs_ = new size_t[numConstrainedBodies_];
    }

    virtual ~Constraint () { delete[] bodyIDs_; }

    virtual void solvePositionConstraint (ClothModel & model, float timeStep, size_t iter) = 0;

private:
    Constraint (const Constraint& other) DELETED; // do not implement
    Constraint& operator= (const Constraint& other) DELETED; // do not implement

protected:
    size_t  numConstrainedBodies_ = 0;
    size_t* bodyIDs_ = nullptr;
};

class DistanceConstraint : public Constraint {
public:

    DistanceConstraint ()
    :
        Constraint (2)
    {}

    virtual void initConstraint             (ClothModel const & model, size_t p0ID, size_t p1ID);
    virtual void solvePositionConstraint    (ClothModel & model, float timeStep, size_t iter);

protected:
    float rl_ {0.f};
    float lambda_old_0_ {0.f};
    float lambda_old_1_ {0.f};
};

class CollisionConstraint : public Constraint {
  ALIGNED_CLASS_(16);
public:

    CollisionConstraint ()
      : Constraint (1) {}

    virtual void initConstraint             (size_t qID, const vec_t& x0, const vec_t& n, float d);
    virtual void solvePositionConstraint    (ClothModel & model, float timeStep, size_t iter);

protected:
    vec_t x0_   {0.f, 0.f, 0.f, 0.f};
    vec_t n_    {0.f, 0.f, 0.f, 0.f};
    float d_    {1.e5f};
};

} // namespace collide2
} // namespace embree
