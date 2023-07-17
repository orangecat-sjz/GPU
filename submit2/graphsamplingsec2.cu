#include <iostream>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stdio.h>
#include <sys/time.h>
#define HISTNUM 1000

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

/*return a random offset in [0,range). if returning -1, it means the range is not bigger than 0.*/
__device__ int random_offset(int range, int cycle)
{
    if (range <= 0)
    {
        return -1;
    }
    thrust::minstd_rand engine;
    // remove replication
    engine.discard(threadIdx.x * cycle);
    thrust::uniform_int_distribution<> dist(0, range - 1);
    int32_t offset = dist(engine);
    return offset;
}

__global__ void RandomSampling(int *d_indptr, int *d_indice,
                               int *d_src_ids, int *d_dst_ids,
                               int src_num, int sampling_count) // src_num = 232965;
{
    int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = gridDim.x * blockDim.x;

    for (int i = indexWithinTheGrid; i < src_num; i += gridStride)
    {
        int src_id = d_src_ids[i];
        int start = d_indptr[src_id];
        int end = d_indptr[src_id + 1];
        for (int j = 0; j < sampling_count; j++)
        {
            int offset = random_offset(end - start, j);
            if (offset < 0)
            {
                d_dst_ids[i * sampling_count + j] = -1;
            }
            else
            {
                d_dst_ids[i * sampling_count + j] = d_indice[start + offset];
            }
        }
    }
}

// __device__ int random_offset(int range){
//     if(range <= 0){
//         return -1;
//     }
//     thrust::minstd_rand engine;
//     //remove replication
//     engine.discard(threadIdx.x);
//     thrust::uniform_int_distribution<> dist(0, range - 1);
//     int32_t offset = dist(engine);
//     return offset;
// }
// __global__ void RandomSampling(int* d_indptr, int* d_indice, int* d_src_ids, int* d_dst_ids, int src_num, int sampling_count)
// {
// 	for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < src_num * sampling_count; thread_idx += gridDim.x * blockDim.x){
//         int src_id = d_src_ids[thread_idx/sampling_count];

//         int neighbor_count = d_indptr[src_id + 1] - d_indptr[src_id];
//         int start_offset = d_indptr[src_id];

//         int offset = random_offset(neighbor_count);
//         if(offset < 0){
//             d_dst_ids[thread_idx] = -1;
//         }else{
//             d_dst_ids[thread_idx] = d_indice[start_offset + offset];
//         }
//     }
// }

// __global__ void Histogram(int* d_dst_ids, int src_num, int sampling_count, int* global_hist){
// 	for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < src_num * sampling_count; thread_idx += gridDim.x * blockDim.x){
//         int dst_id = d_dst_ids[thread_idx];
//         if((dst_id / 40) < HISTNUM){
//             atomicAdd(global_hist + dst_id / 40, 1);
//         }
//     }
// }

//  opt1
__global__ void Histogram(int *d_dst_ids, int src_num, int sampling_count, int *global_hist)
{
    __shared__ int shared_hist[HISTNUM];

    for (int i = threadIdx.x; i < HISTNUM; i += blockDim.x)
    {
        shared_hist[i] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < src_num * sampling_count; i += gridDim.x * blockDim.x)
    {
        int dst_id = d_dst_ids[i];
        int bin_idx = dst_id / 40;

        if (bin_idx < HISTNUM)
        {
            atomicAdd(&shared_hist[bin_idx], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < HISTNUM; i += blockDim.x)
    {
        atomicAdd(&global_hist[i], shared_hist[i]);
    }
}

// opt2
//  __global__ void Histogram(int* d_dst_ids, int src_num, int sampling_count, int* global_hist) {
//      __shared__ int shared_hist[HISTNUM];
//      __shared__ int shared_ids[1024];

//     for (int i = threadIdx.x; i < HISTNUM; i += blockDim.x) {
//         shared_hist[i] = 0;
//     }
//     __syncthreads();

//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     // 计算输入数组中的数据在GPU内存中的地址

//     // 将对应块内的输入数据读入共享内存
//     if(tid < src_num * sampling_count){
//         shared_ids[threadIdx.x] = d_dst_ids[idx];
//     }
//     else{
//         shared_ids[threadIdx.x] = 0;
//     }

//     // 等待所有线程同步，确保共享内存全部准备好
//     __syncthreads();

//     for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < src_num * sampling_count; i += gridDim.x * blockDim.x) {
//         // int dst_id = d_dst_ids[i];
//         int dst_id = shared_ids[threadIdx.x];
//         int bin_idx = dst_id / 40;

//         if (bin_idx < HISTNUM) {
//             atomicAdd(&shared_hist[bin_idx], 1);
//         }
//     }
//     __syncthreads();

//     for (int i = threadIdx.x; i < HISTNUM; i += blockDim.x) {
//         atomicAdd(&global_hist[i], shared_hist[i]);
//     }
// }

void mmap_read(std::string &file_name, int32_t *ret)
{
    int64_t index = 0;
    int32_t fd = open(file_name.c_str(), O_RDONLY);
    if (fd == -1)
    {
        std::cout << "cannout open file: " << file_name << "\n";
        return;
    }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int32_t *buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t *buf_end = buf + buf_len / sizeof(int32_t);
    int32_t temp;
    while (buf < buf_end)
    {
        temp = *buf;
        ret[index++] = temp;
        buf++;
    }
    close(fd);
    return;
}

int main(int argc, char **argv)
{
    /*load arxiv dataset*/
    int vertex_num = 232965;
    int edge_num = 114615892;
    int src_num = 232965;
    int sampling_count = 1000;

    int *h_src_ids = (int *)malloc(src_num * sizeof(int));
    int *h_dst_ids = (int *)malloc(src_num * sampling_count * sizeof(int));
    int *h_indptr = (int *)malloc((vertex_num + 1) * sizeof(int));
    int *h_indice = (int *)malloc(edge_num * sizeof(int));

    std::string src_id_file = "src_ids"; // make sure these files in your current working directory.
    std::string indptr_file = "indptr";
    std::string indice_file = "indice";

    mmap_read(src_id_file, h_src_ids);
    mmap_read(indptr_file, h_indptr);
    mmap_read(indice_file, h_indice);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*initialize graph storage on GPU*/
    int *d_src_ids;
    cudaMalloc(&d_src_ids, src_num * sizeof(int));
    int *d_dst_ids;
    cudaMalloc(&d_dst_ids, src_num * sampling_count * sizeof(int));
    int *d_indptr;
    cudaMalloc(&d_indptr, (vertex_num + 1) * sizeof(int));
    int *d_indice;
    cudaMalloc(&d_indice, edge_num * sizeof(int));
    int *global_hist;
    cudaMalloc(&global_hist, HISTNUM * sizeof(int));
    cudaMemset(global_hist, 0, HISTNUM * sizeof(int));
    cudaCheckError();

    cudaMemcpy(d_src_ids, h_src_ids, src_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, h_indptr, (vertex_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indice, h_indice, edge_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 block_num(72, 1);
    dim3 thread_num(1024, 1);
    RandomSampling<<<block_num, thread_num>>>(d_indptr, d_indice, d_src_ids, d_dst_ids, src_num, sampling_count);

    cudaEventRecord(start, 0);

    Histogram<<<block_num, thread_num>>>(d_dst_ids, src_num, sampling_count, global_hist);

    cudaCheckError();
    /*return result*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaDeviceSynchronize();
    // cudaMemcpy(h_dst_ids, d_dst_ids, src_num * sampling_count * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 10; i++){
    //     for(int j = 0; j < 4; j++){
    //         std::cout<<i<<" "<<j<<" "<<h_dst_ids[i * 4 + j]<<"\n";
    //     }
    // }
    int *h_hist = (int *)malloc(HISTNUM * sizeof(int));
    cudaMemcpy(h_hist, global_hist, HISTNUM * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; i++)
    {
        std::cout << i << " " << h_hist[i] << "\n";
    }
    std::cout << "time cost: " << time << " ms\n";
}
