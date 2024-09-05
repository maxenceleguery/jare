#pragma once

#include <tuple>

#include "Vector.hpp"
#include "Triangle.hpp"
#include "Mesh.hpp"

class BoundingBox {
    private:
        Vector<double> min = Vector<double>(1, 1, 1)*INFINITY;
        Vector<double> max = Vector<double>(1, 1, 1)*-INFINITY;

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

        __host__ __device__ void growToInclude(const Vector<double> vec) {
            min = min.min(vec);
            max = max.max(vec);
        }

        __host__ __device__ void growToInclude(const Vector<double> mini, const Vector<double> maxi) {
            min = min.min(mini);
            max = max.max(maxi);
        }

        __host__ __device__ void growToInclude(const Triangle& triangle) {
            growToInclude(triangle.getMin(), triangle.getMax());
        }

        __host__ __device__ void growToInclude(const Mesh& mesh) {
            for (uint i=0; i<mesh.size(); i++) {
                growToInclude(mesh[i]);
            }
        }

        __host__ __device__ Vector<double> getMin() const {
            return min;
        }

        __host__ __device__ Vector<double> getMax() const {
            return max;
        }

        __host__ __device__ Vector<double> getCenter() const {
            return (min + max) * 0.5;
        }

        __host__ __device__ Vector<double> getSize() const {
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

class BVH {
    private:
        uint maxDepth = 5;
    
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

        __host__ static double NodeCost(const Vector<double>& size, const int numTriangles) {
            double halfArea = size.getX() * size.getY() + size.getX() * size.getZ() + size.getY() * size.getZ();
            return halfArea * numTriangles;
        }

        __host__ double evaluateSplit(const uint splitAxis, const double splitPos, const uint start, const uint count)
    {
        BoundingBox boundsLeft;
        BoundingBox boundsRight;
        uint numOnLeft = 0;
        uint numOnRight = 0;

        for (uint i = start; i < start + count; i++)
        {
            Triangle tri = allTriangles[i];
            if (tri.getBarycenter()[splitAxis] < splitPos)
            {
                boundsLeft.growToInclude(tri);
                numOnLeft++;
            }
            else
            {
                boundsRight.growToInclude(tri);
                numOnRight++;
            }
        }

        double costA = NodeCost(boundsLeft.getSize(), numOnLeft);
        double costB = NodeCost(boundsRight.getSize(), numOnRight);
        return costA + costB;
    }

        __host__ std::tuple<uint, double, double> chooseSplit(const Node& node, const uint start, const uint count) {
            if (count <= 1) return std::make_tuple<uint, double, double>(0, 0, INFINITY);

            double bestSplitPos = 0;
            uint bestSplitAxis = 0;
            const uint numSplitTests = 5;

            double bestCost = INFINITY;

            // Estimate best split pos
            for (uint axis = 0; axis < 3; axis++)
            {
                for (uint i = 0; i < numSplitTests; i++)
                {
                    double splitT = (i + 1) / (numSplitTests + 1.);
                    double splitPos = Utils::smoothStep(node.getBoundingBox().getMin()[axis], node.getBoundingBox().getMax()[axis], splitT);
                    double cost = evaluateSplit(axis, splitPos, start, count);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestSplitPos = splitPos;
                        bestSplitAxis = axis;
                    }
                }
            }

            return std::make_tuple(bestSplitAxis, bestSplitPos, bestCost);
        }

        __host__ void split(const uint parentIndex, const uint triGlobalStart, const uint triNum, const uint depth = 0) {
            const Vector<double> size = allNodes[parentIndex].getBoundingBox().getSize();
            const double parentCost = NodeCost(size, triNum);

            std::tuple<uint, double, double> splitting = chooseSplit(allNodes[parentIndex], triGlobalStart, triNum);
            const uint splitAxis = std::get<0>(splitting);
            const double splitPos = std::get<1>(splitting);
            const double cost = std::get<2>(splitting);
            
            if (depth < maxDepth && cost < parentCost) {
                Node childA = Node();
                Node childB = Node();

                uint numOnLeft = 0;

                for (uint i = triGlobalStart; i < triGlobalStart + triNum; i++) {
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

        __host__ void cuda() {
            allNodes.cuda();
            allTriangles.cuda();
        }

        __host__ void cpu() {
            allNodes.cpu();
            allTriangles.cpu();
        }

        __host__ void free() {
            allNodes.free();
            allTriangles.free();
        }
};




