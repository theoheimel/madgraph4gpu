#include "api.h"

#include "CPPProcess.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "GpuRuntime.h"

#include <cmath>


#ifdef MGONGPUCPP_GPUIMPL
using namespace mg5amcGpu;
#else
using namespace mg5amcCpu;
#endif

namespace {

void* initialize_impl(
    const fptype* momenta,
    const fptype* couplings,
    fptype* matrix_elements,
    fptype* numerators,
    fptype* denominators,
    std::size_t count
) {
    bool is_good_hel[CPPProcess::ncomb];
#ifdef MGONGPUCPP_GPUIMPL
    std::size_t n_threads = 256;
    std::size_t n_blocks = count / n_threads;
    bool *is_good_hel_device;
    checkGpu(cudaMalloc(&is_good_hel_device, CPPProcess::ncomb));
    sigmaKin_getGoodHel<<<n_blocks, n_threads, 0>>>(
        momenta, couplings, matrix_elements, numerators, denominators, is_good_hel_device
    );
    checkGpu(cudaPeekAtLastError());
    checkGpu(cudaMemcpy(
        is_good_hel, is_good_hel_device, sizeof(is_good_hel), cudaMemcpyDefault
    ));
#else // MGONGPUCPP_GPUIMPL
    sigmaKin_getGoodHel(
        momenta, couplings, matrix_elements, numerators, denominators, is_good_hel, count
    );
#endif // MGONGPUCPP_GPUIMPL
    sigmaKin_setGoodHel(is_good_hel);
    return nullptr;
}

void initialize(
    const fptype* momenta,
    const fptype* couplings,
    fptype* matrix_elements,
    fptype* numerators,
    fptype* denominators,
    std::size_t count
) {
    // static local initialization is called exactly once in a thread-safe way
    static void* dummy = initialize_impl(
        momenta, couplings, matrix_elements, numerators, denominators, count
    );
}

#ifdef MGONGPUCPP_GPUIMPL
__device__
#endif
void transpose_momenta(
    const double* momenta_in, fptype* momenta_out, std::size_t i_event, std::size_t stride
) {
    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    std::size_t i_page = i_event / page_size;
    std::size_t i_vector = i_event % page_size;

    for (std::size_t i_part = 0; i_part < CPPProcess::npar; ++i_part) {
        for(std::size_t i_mom = 0; i_mom < 4; ++i_mom) {
            momenta_out[
                i_page * CPPProcess::npar * 4 * page_size +
                i_part * 4 * page_size + i_mom * page_size + i_vector
            ] = momenta_in[
                stride * (CPPProcess::npar * i_mom + i_part) + i_event
            ];
        }
    }
}

#ifdef MGONGPUCPP_GPUIMPL

__global__ void copy_inputs(
    const double* momenta_in,
    fptype* momenta,
    fptype* helicity_random,
    fptype* color_random,
    fptype* g_s,
    unsigned int* channel_index,
    std::size_t count,
    std::size_t stride
) {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    channel_index[i_event] = 2;
    if (i_event >= count) return;

    transpose_momenta(momenta_in, momenta, i_event, stride);
    helicity_random[i_event] = 0.5;
    color_random[i_event] = 0.5;
    g_s[i_event] = 1.2177157847767195;
}

__global__ void copy_inputs_multichannel(
    const double* momenta_in,
    const double* random_in,
    const double* alpha_s_in,
    fptype* momenta,
    fptype* helicity_random,
    fptype* color_random,
    fptype* g_s,
    unsigned int* channel_index,
    std::size_t count,
    std::size_t stride
) {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    channel_index[i_event] = 2;
    if (i_event >= count) return;

    transpose_momenta(momenta_in, momenta, i_event, stride);
    helicity_random[i_event] = random_in[i_event];
    color_random[i_event] = random_in[i_event + stride];
    g_s[i_event] = sqrt(4 * M_PI * alpha_s_in[i_event]);
}

__global__ void copy_outputs(
    fptype* matrix_elements,
    double* m2_out,
    std::size_t count
) {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_event >= count) return;
    m2_out[i_event] = matrix_elements[i_event];
}

__global__ void copy_outputs_multichannel(
    fptype* denominators,
    fptype* numerators,
    fptype* matrix_elements,
    int* color_index,
    int* helicity_index,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out,
    std::size_t count,
    std::size_t stride
) {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_event >= count) return;

    double denominator = denominators[i_event];
    m2_out[i_event] = matrix_elements[i_event];
    for (std::size_t i_diag = 0; i_diag < CPPProcess::ndiagrams; ++i_diag) {
        amp2_out[stride * i_diag + i_event] = numerators[
            i_event * CPPProcess::ndiagrams + i_diag
        ] / denominator;
    }
    diagram_out[i_event] = 0;
    color_out[i_event] = color_index[i_event] - 1;
    helicity_out[i_event] = helicity_index[i_event] - 1;
}

#endif // MGONGPUCPP_GPUIMPL

}

extern "C" {

const SubProcessInfo* subprocess_info() {
    static SubProcessInfo info = {
#ifdef MGONGPUCPP_GPUIMPL
        /* on_gpu          = */ true,
#else
        /* on_gpu          = */ false,
#endif
        /* particle_count  = */ CPPProcess::npar,
        /* diagram_count   = */ CPPProcess::ndiagrams,
        /* helicity_count  = */ CPPProcess::ncomb
    };
    return &info;
}

void* init_subprocess(const char* param_card_path) {
    CPPProcess process;
    process.initProc(param_card_path);
    // We don't actually need the CPPProcess instance for anything as it initializes a
    // global variable. So here we just return a boolean that is used to store whether
    // the good helicities are initialized
    return new bool(false);
}

void free_subprocess(void* subprocess) {
    delete static_cast<bool*>(subprocess);
}

void compute_matrix_element(
    void* subprocess,
    size_t count,
    size_t stride,
    const double* momenta_in,
    const int* flavor_in,
    double* m2_out,
    void* cuda_stream
) {
#ifdef MGONGPUCPP_GPUIMPL
    std::size_t n_threads = 256;
    std::size_t n_blocks = (count + n_threads - 1) / n_threads;
    std::size_t rounded_count = n_blocks * n_threads;

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    fptype *momenta, *couplings, *g_s, *helicity_random, *color_random;
    fptype *matrix_elements, *numerators, *denominators;
    int *helicity_index, *color_index;
    unsigned int *channel_index;

    std::size_t n_coup = mg5amcGpu::Parameters_sm_dependentCouplings::ndcoup;
    checkGpu(cudaMallocAsync(&momenta, rounded_count * CPPProcess::npar * 4 * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&couplings, rounded_count * n_coup * 2 * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&g_s, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&helicity_random, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&color_random, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&matrix_elements, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&channel_index, rounded_count * sizeof(unsigned int), stream));
    checkGpu(cudaMallocAsync(&numerators, rounded_count * CPPProcess::ndiagrams * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&denominators, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&helicity_index, rounded_count * sizeof(int), stream));
    checkGpu(cudaMallocAsync(&color_index, rounded_count * sizeof(int), stream));

    copy_inputs<<<n_blocks, n_threads, 0, stream>>>(
        momenta_in,
        momenta, helicity_random, color_random, g_s, channel_index,
        count, stride
    );
    computeDependentCouplings<<<n_blocks, n_threads, 0, stream>>>(g_s, couplings);
    checkGpu(cudaPeekAtLastError());

    bool& is_initialized = *static_cast<bool*>(subprocess);
    if (!is_initialized) {
        checkGpu(cudaStreamSynchronize(stream));
        initialize(
            momenta, couplings, matrix_elements, numerators, denominators, rounded_count
        );
        is_initialized = true;
    }

    sigmaKin<<<n_blocks, n_threads, 0, stream>>>(
        momenta,
        couplings,
        helicity_random,
        color_random,
        matrix_elements,
        channel_index,
        numerators,
        denominators,
        helicity_index,
        color_index
    );
    copy_outputs<<<n_blocks, n_threads, 0, stream>>>(
        matrix_elements, m2_out, count
    );
    checkGpu(cudaPeekAtLastError());

    checkGpu(cudaFreeAsync(momenta, stream));
    checkGpu(cudaFreeAsync(couplings, stream));
    checkGpu(cudaFreeAsync(g_s, stream));
    checkGpu(cudaFreeAsync(helicity_random, stream));
    checkGpu(cudaFreeAsync(color_random, stream));
    checkGpu(cudaFreeAsync(matrix_elements, stream));
    checkGpu(cudaFreeAsync(channel_index, stream));
    checkGpu(cudaFreeAsync(numerators, stream));
    checkGpu(cudaFreeAsync(denominators, stream));
    checkGpu(cudaFreeAsync(helicity_index, stream));
    checkGpu(cudaFreeAsync(color_index, stream));

#else // MGONGPUCPP_GPUIMPL
    // need to round to round to double page size for some reason
    std::size_t page_size2 = 2 * MemoryAccessMomentaBase::neppM;
    std::size_t rounded_count = (count + page_size2 - 1) / page_size2 * page_size2;

    HostBufferBase<fptype, false> momenta(rounded_count * CPPProcess::npar * 4);
    HostBufferBase<fptype, false> couplings(
        rounded_count * mg5amcCpu::Parameters_sm_dependentCouplings::ndcoup * 2
    );
    // alpha s from the paramcard is discarded, so we just use 0.118 for now
    HostBufferBase<fptype, false> g_s(rounded_count); // sqrt(4 pi alpha_s)
    HostBufferBase<fptype, false> helicity_random(rounded_count);
    HostBufferBase<fptype, false> color_random(rounded_count);
    HostBufferBase<fptype, false> matrix_elements(rounded_count);
    HostBufferBase<unsigned int, false> channel_index(rounded_count);
    HostBufferBase<fptype, false> numerators(rounded_count * CPPProcess::ndiagrams);
    HostBufferBase<fptype, false> denominators(rounded_count);
    HostBufferBase<int, false> helicity_index(rounded_count);
    HostBufferBase<int, false> color_index(rounded_count);

    for (std::size_t i_event = 0; i_event < rounded_count; ++i_event) {
        channel_index[i_event] = 2;
        helicity_random[i_event] = 0.5;
        color_random[i_event] = 0.5;
        g_s[i_event] = 1.2177157847767195;
    }
    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        transpose_momenta(momenta_in, momenta.data(), i_event, stride);
    }
    computeDependentCouplings(
        g_s.data(), couplings.data(), rounded_count
    );

    bool& is_initialized = *static_cast<bool*>(subprocess);
    if (!is_initialized) {
        initialize(
            momenta.data(),
            couplings.data(),
            matrix_elements.data(),
            numerators.data(),
            denominators.data(),
            rounded_count
        );
        is_initialized = true;
    }

    sigmaKin(
        momenta.data(),
        couplings.data(),
        helicity_random.data(),
        color_random.data(),
        matrix_elements.data(),
        channel_index.data(),
        numerators.data(),
        denominators.data(),
        color_index.data(),
        helicity_index.data(),
        rounded_count
    );

    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        std::size_t i_page = i_event / page_size;
        std::size_t i_vector = i_event % page_size;
        m2_out[i_event] = matrix_elements[i_event];
    }
#endif // MGONGPUCPP_GPUIMPL
}

void compute_matrix_element_multichannel(
    void* subprocess,
    size_t count,
    size_t stride,
    const double* momenta_in,
    const double* alpha_s_in,
    const double* random_in,
    const int* flavor_in,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out,
    void* cuda_stream
) {
#ifdef MGONGPUCPP_GPUIMPL
    std::size_t n_threads = 256;
    std::size_t n_blocks = (count + n_threads - 1) / n_threads;
    std::size_t rounded_count = n_blocks * n_threads;

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    fptype *momenta, *couplings, *g_s, *helicity_random, *color_random;
    fptype *matrix_elements, *numerators, *denominators;
    int *helicity_index, *color_index;
    unsigned int *channel_index;

    std::size_t n_coup = mg5amcGpu::Parameters_sm_dependentCouplings::ndcoup;
    checkGpu(cudaMallocAsync(&momenta, rounded_count * CPPProcess::npar * 4 * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&couplings, rounded_count * n_coup * 2 * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&g_s, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&helicity_random, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&color_random, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&matrix_elements, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&channel_index, rounded_count * sizeof(unsigned int), stream));
    checkGpu(cudaMallocAsync(&numerators, rounded_count * CPPProcess::ndiagrams * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&denominators, rounded_count * sizeof(fptype), stream));
    checkGpu(cudaMallocAsync(&helicity_index, rounded_count * sizeof(int), stream));
    checkGpu(cudaMallocAsync(&color_index, rounded_count * sizeof(int), stream));

    copy_inputs_multichannel<<<n_blocks, n_threads, 0, stream>>>(
        momenta_in, random_in, alpha_s_in,
        momenta, helicity_random, color_random, g_s, channel_index,
        count, stride
    );
    computeDependentCouplings<<<n_blocks, n_threads, 0, stream>>>(g_s, couplings);
    checkGpu(cudaPeekAtLastError());

    bool& is_initialized = *static_cast<bool*>(subprocess);
    if (!is_initialized) {
        checkGpu(cudaStreamSynchronize(stream));
        initialize(
            momenta, couplings, matrix_elements, numerators, denominators, rounded_count
        );
        is_initialized = true;
    }

    sigmaKin<<<n_blocks, n_threads, 0, stream>>>(
        momenta,
        couplings,
        helicity_random,
        color_random,
        matrix_elements,
        channel_index,
        numerators,
        denominators,
        helicity_index,
        color_index
    );
    copy_outputs_multichannel<<<n_blocks, n_threads, 0, stream>>>(
        denominators, numerators, matrix_elements, color_index, helicity_index,
        m2_out, amp2_out, diagram_out, color_out, helicity_out,
        count, stride
    );
    checkGpu(cudaPeekAtLastError());

    checkGpu(cudaFreeAsync(momenta, stream));
    checkGpu(cudaFreeAsync(couplings, stream));
    checkGpu(cudaFreeAsync(g_s, stream));
    checkGpu(cudaFreeAsync(helicity_random, stream));
    checkGpu(cudaFreeAsync(color_random, stream));
    checkGpu(cudaFreeAsync(matrix_elements, stream));
    checkGpu(cudaFreeAsync(channel_index, stream));
    checkGpu(cudaFreeAsync(numerators, stream));
    checkGpu(cudaFreeAsync(denominators, stream));
    checkGpu(cudaFreeAsync(helicity_index, stream));
    checkGpu(cudaFreeAsync(color_index, stream));

#else // MGONGPUCPP_GPUIMPL
    // need to round to round to double page size for some reason
    std::size_t page_size2 = 2 * MemoryAccessMomentaBase::neppM;
    std::size_t rounded_count = (count + page_size2 - 1) / page_size2 * page_size2;

    HostBufferBase<fptype, false> momenta(rounded_count * CPPProcess::npar * 4);
    HostBufferBase<fptype, false> couplings(
        rounded_count * mg5amcCpu::Parameters_sm_dependentCouplings::ndcoup * 2
    );
    HostBufferBase<fptype, false> g_s(rounded_count);
    HostBufferBase<fptype, false> helicity_random(rounded_count);
    HostBufferBase<fptype, false> color_random(rounded_count);
    HostBufferBase<fptype, false> matrix_elements(rounded_count);
    HostBufferBase<unsigned int, false> channel_index(rounded_count);
    HostBufferBase<fptype, false> numerators(rounded_count * CPPProcess::ndiagrams);
    HostBufferBase<fptype, false> denominators(rounded_count);
    HostBufferBase<int, false> helicity_index(rounded_count);
    HostBufferBase<int, false> color_index(rounded_count);

    for (std::size_t i_event = 0; i_event < rounded_count; ++i_event) {
        channel_index[i_event] = 2;
    }
    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        transpose_momenta(momenta_in, momenta.data(), i_event, stride);
        helicity_random[i_event] = random_in[i_event];
        color_random[i_event] = random_in[i_event + stride];
        g_s[i_event] = sqrt(4 * M_PI * alpha_s_in[i_event]);
    }
    computeDependentCouplings(
        g_s.data(), couplings.data(), rounded_count
    );

    bool& is_initialized = *static_cast<bool*>(subprocess);
    if (!is_initialized) {
        initialize(
            momenta.data(),
            couplings.data(),
            matrix_elements.data(),
            numerators.data(),
            denominators.data(),
            rounded_count
        );
        is_initialized = true;
    }

    sigmaKin(
        momenta.data(),
        couplings.data(),
        helicity_random.data(),
        color_random.data(),
        matrix_elements.data(),
        channel_index.data(),
        numerators.data(),
        denominators.data(),
        helicity_index.data(),
        color_index.data(),
        rounded_count
    );

    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        std::size_t i_page = i_event / page_size;
        std::size_t i_vector = i_event % page_size;

        double denominator = denominators[i_event];
        m2_out[i_event] = matrix_elements[i_event];
        for (std::size_t i_diag = 0; i_diag < CPPProcess::ndiagrams; ++i_diag) {
            amp2_out[stride * i_diag + i_event] = numerators[
                i_page * page_size * CPPProcess::ndiagrams +
                i_diag * page_size + i_vector
            ] / denominator;
        }
        diagram_out[i_event] = 0;
        color_out[i_event] = color_index[i_event] - 1;
        helicity_out[i_event] = helicity_index[i_event] - 1;
    }
#endif // MGONGPUCPP_GPUIMPL
}

}
