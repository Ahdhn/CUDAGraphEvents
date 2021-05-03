#include <assert.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "helper.h"


__global__ static void write_value(int* mem, int value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        mem[0] = value;
    }
}

int main(int argc, char** argv)
{
    std::vector<int> gpu_ids{0, 0, 0};
    // std::vector<int> gpu_ids{0, 1, 2};

    std::vector<int> values{11, 22, 33};

    std::vector<int*>            d_buf(gpu_ids.size());
    std::vector<int>             h_result(2 * gpu_ids.size(), 99);
    std::vector<cudaStream_t>    streams(gpu_ids.size());
    std::vector<cudaGraph_t>     graphs(gpu_ids.size());
    std::vector<cudaGraphExec_t> exec_graphs(gpu_ids.size());
    std::vector<cudaGraphNode_t> kernel_nodes(gpu_ids.size());
    std::vector<cudaGraphNode_t> event_nodes(gpu_ids.size());
    std::vector<cudaEvent_t>     events(gpu_ids.size());

    // allocate device buffer and create graph, streams, and events
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));
        int* buf = NULL;
        CUDA_ERROR(cudaMalloc((void**)&buf, sizeof(int)));
        d_buf[i] = buf;
        CUDA_ERROR(cudaGraphCreate(&graphs[i], 0));
        CUDA_ERROR(cudaStreamCreate(&streams[i]));
        CUDA_ERROR(cudaEventCreate(&events[i]));
    }

    // initialize the device buffer with something
    for (int i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));
        CUDA_ERROR(cudaMemcpy(d_buf[i], &h_result[i], sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // 1st node is to write the GPU id to d_buf
    // 2nd node is a record event after the 1st node
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));

        // 1st
        void*                kernelArgs[2] = {&d_buf[i], &values[i]};
        cudaKernelNodeParams kernelNodeParams = {0};
        kernelNodeParams.func = (void*)write_value;
        kernelNodeParams.gridDim = dim3(1, 1, 1);
        kernelNodeParams.blockDim = dim3(1, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void**)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_ERROR(cudaGraphAddKernelNode(&kernel_nodes[i], graphs[i], NULL, 0,
                                          &kernelNodeParams));

        // 2nd
        CUDA_ERROR(cudaGraphAddEventRecordNode(&event_nodes[i], graphs[i],
                                               &kernel_nodes[i], 1, events[i]));
    }

    // 3rd nodes are the wait node on the event nodes from the two other graphs
    // 4th nodes are the memcpy from device to host where we copy the other two
    // graph data
    for (size_t i = 0; i < gpu_ids.size(); ++i) {

        std::vector<cudaGraphNode_t> wait_nodes(2);

        size_t i_next = (i + 1) % gpu_ids.size();
        size_t i_prev = (i + gpu_ids.size() - 1) % gpu_ids.size();

        // 3rd
        CUDA_ERROR(cudaGraphAddEventWaitNode(
            &wait_nodes[0], graphs[i], &event_nodes[i], 1, events[i_next]));

        CUDA_ERROR(cudaGraphAddEventWaitNode(
            &wait_nodes[1], graphs[i], &event_nodes[i], 1, events[i_prev]));

        // 4th
        cudaGraphNode_t n_next, n_prev;
        CUDA_ERROR(cudaGraphAddMemcpyNode1D(
            &n_next, graphs[i], wait_nodes.data(), wait_nodes.size(),
            &h_result[i * 2], d_buf[i_next], sizeof(int),
            cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaGraphAddMemcpyNode1D(
            &n_prev, graphs[i], wait_nodes.data(), wait_nodes.size(),
            &h_result[i * 2 + 1], d_buf[i_prev], sizeof(int),
            cudaMemcpyDeviceToHost));


        // make the graph executable and check for errors
        cudaGraphNode_t pErrorNode = nullptr;
        const size_t    bufferSize = 1024;
        char            pLogBuffer[bufferSize];
        cudaError_t     res = ::cudaGraphInstantiate(
            &exec_graphs[i], graphs[i], &pErrorNode, pLogBuffer, bufferSize);
        bool trucatedErrorMessage = (pLogBuffer[bufferSize - 1] == '\0');
        pLogBuffer[bufferSize - 1] = '\0';
        if (res != cudaSuccess) {
            std::cout << "\n Error: " << cudaGetErrorString(res);
            std::cout << "\n Error: " << pLogBuffer;
            std::cout << "\n Error: Related Graph Node ->"
                      << reinterpret_cast<char*>(pErrorNode);
            if (trucatedErrorMessage) {
                std::cout << "\n Error: previous error message was truncated";
            }
        }
    }


// launch the graphs
#pragma omp parallel for num_threads(gpu_ids.size())
    for (int i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaGraphLaunch(exec_graphs[i], streams[i]));
        CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }


    // copy the ground truth (values written to the device buffer) to the host
    std::vector<int> truth(gpu_ids.size());
    for (int i = 0; i < gpu_ids.size(); ++i) {
        CUDA_ERROR(cudaSetDevice(gpu_ids[i]));
        CUDA_ERROR(cudaMemcpy(&truth[i], d_buf[i], sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    // sync and check the output
    for (int i = 0; i < gpu_ids.size(); ++i) {
        int i_next = (i + 1) % gpu_ids.size();
        int i_prev = (i == 0) ? gpu_ids.size() - 1 : i - 1;

        printf("\n*** ID = %d", i);
        printf(
            "\n i_next = %d, correct_value= %d, value_from_device= %d, "
            "h_result= %d",
            i_next, values[i_next], truth[i_next], h_result[2 * i]);

        printf(
            "\n i_prev = %d, correct_value= %d, value_from_device= %d, "
            "h_result= %d\n",
            i_prev, values[i_prev], truth[i_prev], h_result[2 * i + 1]);
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
