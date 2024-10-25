#pragma once

#include <type_traits>
#include "cuda_ready.hpp"

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

static std::string human_rep(const int num_bytes) {
    if (num_bytes < 1000) return std::to_string(num_bytes) + " B (" + std::to_string(num_bytes) + " B)";
    if (num_bytes < 1000 * 1000) return std::to_string((float)num_bytes/1000).substr(0, 5) + " KB (" + std::to_string((float)num_bytes/1024).substr(0, 5) + " KiB)";
    if (num_bytes < 1000 * 1000 * 1000) return std::to_string((float)num_bytes/(1000*1000)).substr(0, 5) + " MB (" + std::to_string((float)num_bytes/(1024*1024)).substr(0, 5) + " MiB)";
    return std::to_string((float)num_bytes/(1000*1000*1000)).substr(0, 5) + " GB (" + std::to_string((float)num_bytes/(1024*1024*1024)).substr(0, 5) + " GiB)";;
}

template<typename T>
class Array : public CudaReady {
    private:
        T* data;
        T* data_cpu;
        T* data_gpu = nullptr;
        uint data_size;

    protected:
        uint spaceUsed = 0;
        
    public:
        __host__ __device__ Array() : data(nullptr), data_cpu(nullptr), data_size(0) {};

        __host__ Array(const uint data_size) : data_size(data_size) {
            data_cpu = new T[data_size];
            data = data_cpu;
        };

        __host__ Array(const T& item) : Array(1) {
            push_back(item);
        };

        __host__ uint push_back(const T item) {
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
            data_cpu = data;
            return spaceUsed-1;
        }

        __host__ __device__ uint size() const {
            return spaceUsed;
        }

        __host__ void clear() {
            spaceUsed = 0;
        }

        template<typename I>
        __host__ __device__ T operator[](const I i) const {
            //printf("%d, %u, %d\n", spaceUsed + i, spaceUsed, i);
            if constexpr (std::is_signed_v<I>) {
                if (i < 0)
                    return data[(int)spaceUsed + i];
            }
            return data[i];
        }

        template<typename I>
        __host__ __device__ T& operator[](const I i) {
            //printf("%d, %u, %d\n", spaceUsed + i, spaceUsed, i);
            if constexpr (std::is_signed_v<I>) {
                if (i < 0)
                    return data[(int)spaceUsed + i];
            }
            return data[i];
        }

        template<typename I>
        __host__ __device__ T getValueFromCPU(const I i) const {
            if constexpr (std::is_signed_v<I>) {
                if (i < 0)
                    return data_cpu[(int)spaceUsed + i];
            }
            return data_cpu[i];
        }
        
        __host__ void cuda() override {
            if constexpr (std::is_base_of<CudaReady, T>::value) {
                for (uint i=0; i<size(); i++) {
                    data[i].cuda();
                }
            }
            if (data_gpu == nullptr) {
                //std::cout << "Allocating : " << data_size*sizeof(T) << " bytes" << std::endl;
                cudaErrorCheck(cudaMalloc(&data_gpu, data_size*sizeof(T)));
                cudaErrorCheck(cudaMemcpy(data_gpu, data_cpu, data_size*sizeof(T), cudaMemcpyHostToDevice));
            }
            data = data_gpu;
        }

        __host__ void cpu() override {
            if (data_gpu != nullptr) {
                cudaErrorCheck(cudaMemcpy(data_cpu, data_gpu, data_size*sizeof(T), cudaMemcpyDeviceToHost));
            }
            data = data_cpu;
            if constexpr (std::is_base_of<CudaReady, T>::value) {
                for (uint i=0; i<size(); i++) {
                    data[i].cpu();
                }
            }
        }

        __host__ void sync_to_cpu() override {
            if (data_cpu != nullptr && data_gpu != nullptr) {
                cudaErrorCheck(cudaMemcpy(data_cpu, data_gpu, data_size*sizeof(T), cudaMemcpyDeviceToHost));
            }
            if constexpr (std::is_base_of<CudaReady, T>::value) {
                for (uint i=0; i<size(); i++) {
                    data[i].sync_to_cpu();
                }
            }
        }

        __host__ void free() override {
            if constexpr (std::is_base_of<CudaReady, T>::value) {
                for (uint i=0; i<size(); i++) {
                    data[i].free();
                }
            }
            if (data != nullptr) {
                delete[] data;
                data = nullptr;
                data_cpu = nullptr;
            }
            if (data_gpu != nullptr) {
                cudaErrorCheck(cudaFree(data_gpu));
                data_gpu = nullptr;
            }
        }
};
