#pragma once

#include "Triangle.hpp"
#include "utils/Array.hpp"
#include "utils/SceneObject.hpp"
#include "TRSMatrix.hpp"
#include "Transformations.hpp"

class Mesh : public Array<Triangle>, public SceneObject {
    public:
        Mesh() {
            setDefaultsTransforms();
            setDefaultsOrientations();
        };
        Mesh(const uint num_triangle) : Array<Triangle>(num_triangle) {
            setDefaultsTransforms();
            setDefaultsOrientations();
        };
        Mesh(const Triangle& triangle) : Array<Triangle>(triangle) {
            setDefaultsTransforms();
            setDefaultsOrientations();
        };

        __host__ void cuda() override {
            Array<Triangle>::cuda();
            SceneObject::cuda();
        }

        __host__ void cpu() override {
            Array<Triangle>::cpu();
            SceneObject::cpu();
        }

        __host__ void free() override {
            Array<Triangle>::free();
            SceneObject::free();
        }

};

using Meshes = Array<Mesh>;