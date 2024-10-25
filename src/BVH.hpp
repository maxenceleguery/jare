#pragma once

#include <tuple>

#include "Vector.hpp"
#include "Triangle.hpp"
#include "Mesh.hpp"

#include "utils/CudaReady.hpp"

class BoundingBox {
    private:
        Vector<float> min = Vector<float>(1, 1, 1)*INFINITY;
        Vector<float> max = Vector<float>(1, 1, 1)*-INFINITY;

    public:
        __host__ __device__ BoundingBox() {};

        __host__ __device__ BoundingBox(const BoundingBox& other) {
            min = other.min;
            max = other.max;
        };

        __host__ __device__ BoundingBox operator=(const BoundingBox& other) {
            if (this != &other) {
                min = other.min;
                max = other.max;
            }
            return *this;
        }

        __host__ __device__ void growToInclude(const Vector<float> vec) {
            min = min.min(vec);
            max = max.max(vec);
        }

        __host__ __device__ void growToInclude(const Vector<float> mini, const Vector<float> maxi) {
            min = min.min(mini);
            max = max.max(maxi);
        }

        __host__ __device__ void growToInclude(const Triangle& triangle) {
            growToInclude(triangle.getMin(), triangle.getMax());
        }

        __host__ __device__ void growToInclude(const Mesh& mesh) {
            for (int i=0; i<mesh.size(); i++) {
                growToInclude(mesh[i]);
            }
        }

        __host__ __device__ Vector<float> getMin() const {
            return min;
        }

        __host__ __device__ Vector<float> getMax() const {
            return max;
        }

        __host__ __device__ Vector<float> getCenter() const {
            return (min + max) * 0.5;
        }

        __host__ __device__ Vector<float> getSize() const {
            return max - min;
        }
};

class Node {
    private:
        BoundingBox bounds;
        uint triangleIndex = 0;
        uint triangleCount = 0;
        uint childIndex = 0;
        
    public:

        __host__ __device__ Node() {};
        __host__ __device__ Node(const uint triangleIndex) : triangleIndex(triangleIndex) {};
        __host__ __device__ Node(BoundingBox bounds) : bounds(bounds) {};
        __host__ __device__ Node(const Node& other) {
            bounds = other.bounds;
            triangleIndex = other.triangleIndex;
            triangleCount = other.triangleCount;
            childIndex = other.childIndex;
        };

        __host__ __device__ ~Node() {};

        __host__ __device__ Node operator=(const Node& other) {
            if (this != &other) {
                bounds = other.bounds;
                triangleIndex = other.triangleIndex;
                triangleCount = other.triangleCount;
                childIndex = other.childIndex;
            }
            return *this;
        }

        __host__ __device__ void addToBoundingBox(const Triangle& tri) {
            bounds.growToInclude(tri);
        }

        __host__ __device__ BoundingBox getBoundingBox() const {
            return bounds;
        }

        __host__ __device__ void setChildIndex(const uint idx) {
            childIndex = idx;
        }

        __host__ __device__ uint getChildIndex() const {
            return childIndex;
        }

        __host__ __device__ void setTriangleIndex(const uint idx) {
            triangleIndex = idx;
        }

        __host__ __device__ uint getTriangleIndex() const {
            return triangleIndex;
        }

        __host__ __device__ void setTriangleCount(const uint count) {
            triangleCount = count;
        }

        __host__ __device__ uint getTriangleCount() const {
            return triangleCount;
        }
};

class BVH : public CudaReady, public SceneObject {
    private:
        uint maxDepth = 7;
    
    public:
        Array<Node> allNodes;
        Mesh allTriangles;
    
        __host__ BVH() {}; 
        __host__ BVH(const Mesh mesh) : allTriangles(mesh) {
            BoundingBox bounds;
            bounds.growToInclude(mesh);

            allNodes.push_back(Node(bounds));
            split(0, 0, allTriangles.size(), 0);
        };

         __host__ BVH(const Mesh mesh, const uint _maxDepth) : BVH(mesh) {
            maxDepth = _maxDepth;
         };

        __host__ static float NodeCost(const Vector<float>& size, const int numTriangles) {
            float halfArea = size.getX() * size.getY() + size.getX() * size.getZ() + size.getY() * size.getZ();
            return halfArea * numTriangles;
        }

        __host__ float evaluateSplit(const uint splitAxis, const float splitPos, const uint start, const uint count) {
        BoundingBox boundsLeft;
        BoundingBox boundsRight;
        uint numOnLeft = 0;
        uint numOnRight = 0;

            for (int i = start; i < start + count; i++) {
            Triangle tri = allTriangles[i];
                if (tri.getBarycenter()[splitAxis] < splitPos) {
                boundsLeft.growToInclude(tri);
                numOnLeft++;
                } else {
                boundsRight.growToInclude(tri);
                numOnRight++;
            }
        }
        float costA = NodeCost(boundsLeft.getSize(), numOnLeft);
        float costB = NodeCost(boundsRight.getSize(), numOnRight);
        return costA + costB;
    }

        __host__ std::tuple<uint, float, float> chooseSplit(const Node& node, const uint start, const uint count) {
            if (count <= 1) return std::make_tuple<uint, float, float>(0, 0, INFINITY);

            float bestSplitPos = 0;
            uint bestSplitAxis = 0;
            const uint numSplitTests = 5;

            float bestCost = INFINITY;

            // Estimate best split pos
            for (int axis = 0; axis < 3; axis++) {
                for (int i = 0; i < numSplitTests; i++) {
                    float splitT = (i + 1) / (numSplitTests + 1.);
                    float splitPos = Utils::smoothStep(node.getBoundingBox().getMin()[axis], node.getBoundingBox().getMax()[axis], splitT);
                    float cost = evaluateSplit(axis, splitPos, start, count);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestSplitPos = splitPos;
                        bestSplitAxis = axis;
                    }
                }
            }

            return std::make_tuple(bestSplitAxis, bestSplitPos, bestCost);
        }

        __host__ void split(const uint parentIndex, const uint triGlobalStart, const uint triNum, const uint depth = 0) {
            const Vector<float> size = allNodes[parentIndex].getBoundingBox().getSize();
            const float parentCost = NodeCost(size, triNum);

            std::tuple<uint, float, float> splitting = chooseSplit(allNodes[parentIndex], triGlobalStart, triNum);
            const uint splitAxis = std::get<0>(splitting);
            const float splitPos = std::get<1>(splitting);
            const float cost = std::get<2>(splitting);
            
            if (depth < maxDepth && cost < parentCost) {
                Node childA = Node();
                Node childB = Node();

                uint numOnLeft = 0;

                for (int i = triGlobalStart; i < triGlobalStart + triNum; i++) {
                    const Triangle tri = allTriangles[i];
                    if (tri.getBarycenter()[splitAxis] < splitPos) {
                        childA.addToBoundingBox(tri);

                        const Triangle swap = allTriangles[triGlobalStart + numOnLeft];
                        allTriangles[triGlobalStart + numOnLeft] = tri;
                        allTriangles[i] = swap;
                        numOnLeft++;

                    } else {
                        childB.addToBoundingBox(tri);
                    }
                }

                const uint numOnRight = triNum - numOnLeft;
                const uint triStartLeft = triGlobalStart + 0;
                const uint triStartRight = triGlobalStart + numOnLeft;

                childA.setTriangleIndex(triStartLeft);
                childB.setTriangleIndex(triStartRight);
                const uint childIndexLeft = allNodes.push_back(childA);
                const uint childIndexRight = allNodes.push_back(childB);

                allNodes[parentIndex].setChildIndex(childIndexLeft);

                split(childIndexLeft, triGlobalStart, numOnLeft, depth + 1);
                split(childIndexRight, triGlobalStart + numOnLeft, numOnRight, depth + 1);
            } else {
                allNodes[parentIndex].setTriangleIndex(triGlobalStart);
                allNodes[parentIndex].setTriangleCount(triNum);
            }
        }

        __host__ void cuda() override {
            allNodes.cuda();
            allTriangles.cuda();
        }

        __host__ void cpu() override {
            allNodes.cpu();
            allTriangles.cpu();
        }

        __host__ void free() override {
            allNodes.free();
            allTriangles.free();
        }
};




