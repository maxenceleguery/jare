#pragma once

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

        __host__ __device__ void growToInclude(const Triangle& triangle) {
            growToInclude(triangle.getVertex(0));
            growToInclude(triangle.getVertex(1));
            growToInclude(triangle.getVertex(2));
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
            //incrementCount();
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

        __host__ __device__ void incrementCount() {
            triangleCount++;
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
    
        __host__ __device__ BVH() {}; 
        __host__ __device__ BVH(const Mesh& mesh, const uint maxDepth = 5) : maxDepth(maxDepth), allTriangles(mesh) {
            BoundingBox bounds;
            bounds.growToInclude(mesh);

            Node root = Node(bounds);
            //root.setTriangleCount(allTriangles.size());
            //root.setChildIndex(1);
            allNodes.push_back(root);
            split(0, 0, allTriangles.size(), 0);
        };

        __host__ __device__ ~BVH() {
            //delete root;
        }

        __host__ __device__ static double NodeCost(const Vector<double>& size, const int numTriangles) {
            double halfArea = size.getX() * size.getY() + size.getX() * size.getZ() + size.getY() * size.getZ();
            return halfArea * numTriangles;
        }

        __host__ __device__ void split(const uint parentIndex, const uint triGlobalStart, const uint triNum, const uint depth = 0) {

            const Vector<double> size = allNodes[parentIndex].getBoundingBox().getSize();
            const uint splitAxis = size.getX() > Utils::max(size.getY(), size.getZ()) ? 0 : size.getY() > size.getZ() ? 1 : 2;
            const double splitPos = allNodes[parentIndex].getBoundingBox().getCenter()[splitAxis];
            //const double parentCost = NodeCost(size, triNum);
            
            if (depth < maxDepth && triNum >= 3) {
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

                //if (childA.getTriangleCount() > 0 && childB.getTriangleCount() > 0) {
                    split(childIndexLeft, triGlobalStart, numOnLeft, depth + 1);
                    split(childIndexRight, triGlobalStart + numOnLeft, numOnRight, depth + 1);
                //} 
            } else {
                allNodes[parentIndex].setTriangleIndex(triGlobalStart);
                allNodes[parentIndex].setTriangleCount(triNum);
            }
        }
};




