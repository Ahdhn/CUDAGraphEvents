#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "helper.h"


__global__ static void write_id(int* mem, int id)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        mem[0] = id;
    }
}

int main(int argc, char** argv)
{
    std::vector<int> gpu_ids{0, 0, 0};
    // std::vector<int> gpu_ids{0, 1, 2};

    std::vector<int*>            d_buf(gpu_ids.size());
    std::vector<int>             h_buf(2 * gpu_ids.size());
    std::vector<cudaStream_t>    streams(gpu_ids.size());
    std::vector<cudaGraph_t>     graphs(gpu_ids.size());
    std::vector<cudaGraphExec_t> exec_graphs(gpu_ids.size());
    std::vector<cudaGraphNode_t> kernel_nodes(gpu_ids.size());
    std::vector<cudaGraphNode_t> event_nodes(gpu_ids.size());
    std::vector<cudaEvent_t>     events(gpu_ids.size());

    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));
        CUDA_ERROR(cudaMalloc((void**)gpu_ids[i], sizeof(int)));
        CUDA_ERROR(cudaGraphCreate(&graphs[i], 0));
        CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }


    // 1st node is to write the GPU id to d_buf
    // 2nd node is a record event after the 1st node
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));

        // 1st
        void*                kernelArgs[2] = {&d_buf[i], &gpu_ids[i]};
        cudaKernelNodeParams kernelNodeParams = {0};
        kernelNodeParams.func = (void*)write_id;
        kernelNodeParams.gridDim = dim3(1, 1, 1);
        kernelNodeParams.blockDim = dim3(1, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void**)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_ERROR(cudaGraphAddKernelNode(&kernel_nodes[i], graphs[i], NULL, 0,
                                          &kernelNodeParams));

        // 2nd
        CUDA_ERROR(cudaEventCreate(&events[i]));
        CUDA_ERROR(cudaGraphAddEventRecordNode(&event_nodes[i], graphs[i],
                                               &kernel_nodes[i], 1, events[i]));
    }

    // 3rd nodes are the wait node on the event nodes from the two other graphs
    // 4th nodes are the memcpy from device to host where we copy the other two
    // graph data
    for (size_t i = 0; i < gpu_ids.size(); ++i) {

        std::vector<cudaGraphNode_t> wait_nodes(2);

        size_t i_next = (i + 1) % gpu_ids.size();
        size_t i_prev = (i == 0) ? gpu_ids.size() - 1 : i - 1;

        // 3rd
        CUDA_ERROR(cudaGraphAddEventWaitNode(
            &wait_nodes[0], graphs[i], &event_nodes[i], 1, events[i_next]));

        CUDA_ERROR(cudaGraphAddEventWaitNode(
            &wait_nodes[1], graphs[i], &event_nodes[i], 1, events[i_prev]));

        // 4th
        cudaGraphNode_t n_next, n_prev;
        CUDA_ERROR(cudaGraphAddMemcpyNode1D(
            &n_next, graphs[i], wait_nodes.data(), wait_nodes.size(),
            &h_buf[i * 2], d_buf[i_next], sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaGraphAddMemcpyNode1D(
            &n_prev, graphs[i], wait_nodes.data(), wait_nodes.size(),
            &h_buf[i * 2 + 1], d_buf[i_prev], sizeof(int),
            cudaMemcpyDeviceToHost));


        // make the graph executable
        cudaGraphNode_t pErrorNode = nullptr;
        const size_t    bufferSize = 1024;
        char            pLogBuffer[bufferSize];
        cudaError_t     res = ::cudaGraphInstantiate(
            &exec_graphs[i], graphs[i], &pErrorNode, pLogBuffer, bufferSize);
        bool trucatedErrorMessage = (pLogBuffer[bufferSize - 1] == '\0');
        pLogBuffer[bufferSize - 1] = '\0';
        if (res != cudaSuccess) {
            std::cout << "\n\t Error: " << cudaGetErrorString(res);
            std::cout << "\n\t Error: " << pLogBuffer;
            std::cout << "\n\t Error: Related Graph Node ->"
                      << reinterpret_cast<char*>(pErrorNode);
            if (trucatedErrorMessage) {
                std::cout << "\n Error: previous error message was truncated";
            }
        }
    }


    // launch the graphs
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaGraphLaunch(exec_graphs[i], streams[i]));
    }


    // sync and check the output
    for (int i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));
        CUDA_ERROR(cudaDeviceSynchronize());

        int i_next = (i + 1) % gpu_ids.size();
        int i_prev = (i == 0) ? gpu_ids.size() - 1 : i - 1;

        printf("\n*** ID = %d\n", i);
        printf("\n i_next = %d, buf= %d", i_next, h_buf[2 * i]);
        printf("\n i_prev = %d, buf= %d\n", i_prev, h_buf[2 * i + 1]);
    }


    // clean up
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaGraphExecDestroy(exec_graphs[i]));
        CUDA_ERROR(cudaGraphDestroy(graphs[i]));
        CUDA_ERROR(cudaStreamDestroy(streams[i]));
        CUDA_ERROR(cudaFree(d_buf[i]));
        CUDA_ERROR(cudaEventDestroy(events[i]));
    }
}
