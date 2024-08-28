#pragma once

template<typename T>
class Array {
    private:
        T* data;
        uint data_size;
        uint spaceUsed = 0;
        
    public:
        __host__ __device__ Array() : data(nullptr), data_size(0) {};

        __host__ __device__ Array(const uint data_size) : data_size(data_size) {
            data = new T[data_size];
        };

        __host__ __device__ Array(const T& tri) : Array(1) {
            push_back(tri);
        };

        __host__ __device__ uint push_back(const T item) {
            if (spaceUsed == data_size) {
                data_size++;
                T* tri_tmp = new T[data_size];
                for (uint i = 0; i < spaceUsed; i++) {
                    tri_tmp[i] = data[i];
                }
                if (data != nullptr)
                    delete[] data;
                data = tri_tmp;
            }
            data[spaceUsed++] = item;
            return spaceUsed-1;
        }

        __host__ __device__ uint size() const {
            return data_size;
        }

        template<typename I>
        __host__ __device__ T operator[](const I i) const {
            if ((uint)i < data_size) {
                return data[i];
            } else {
                exit(1);
            }
        }

        template<typename I>
        __host__ __device__ T& operator[](const I i) {
            if ((uint)i < data_size) {
                return data[i];
            } else {
                exit(1);
            }
        }

        __host__ __device__ ~Array() {
            //if (data != nullptr)
                //delete[] data;
        }
};
