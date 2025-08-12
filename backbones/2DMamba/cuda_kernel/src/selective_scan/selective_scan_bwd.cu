/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


#include "scan/commons.h"
#include "selective_scan/global.cuh"
#include "selective_scan/selective_scan.cuh"


#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }


#define DISPATCH_ITYPE_FLOAT_AND_HALF(ITYPE, NAME, ...)                             \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }



#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half) {                                             \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::Float)  {                                    \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }


#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == at::ScalarType::Float) {                                            \
       using weight_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::ComplexFloat) {                              \
        using weight_t = c10::complex<float>;                                        \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }


template <typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_cuda(SSMParamsBwd & params, cudaStream_t stream);


void set_ssm_params_fwd(SSMParamsBase & params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t height,
                        const size_t width,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t maxDimPerBlock,

                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor out,
                        const at::Tensor z,
                        const at::Tensor out_z,
                        void * D_ptr,
                        void * delta_bias_ptr,
                        void * x_ptr,
                        bool has_z,
                        bool delta_softplus);


void set_ssm_params_bwd(SSMParamsBwd & params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t height,
                        const size_t width,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t maxDimPerBlock,

                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor z,
                        const at::Tensor out,
                        const at::Tensor out_z,
                        void * D_ptr,
                        void * delta_bias_ptr,
                        void * x_ptr,
                        const at::Tensor dout,
                        const at::Tensor du,
                        const at::Tensor ddelta,
                        const at::Tensor dA,
                        const at::Tensor dB,
                        const at::Tensor dC,
                        const at::Tensor dz,
                        void * dD_ptr,
                        void * ddelta_bias_ptr,
                        bool has_z,
                        bool delta_softplus,
                        bool recompute_out_z,
                        void * rev_shift_tmp_ptr)
{
    // Pass in "dout" instead of "out", we're not gonna use "out" unless we have z
    set_ssm_params_fwd(params,
                       batch,
                       dim,
                       height,
                       width,
                       dstate,
                       n_groups,
                       maxDimPerBlock,
                       is_variable_B,
                       is_variable_C,
                       u,
                       delta,
                       A,
                       B,
                       C,
                       // If not recompute_out_z, pass dout instead of out_z.
                       // This won't be used by the bwd kernel
                       has_z ? out : dout,
                       has_z ? z : dout,
                       recompute_out_z ? out_z : dout,
                       D_ptr,
                       delta_bias_ptr,
                       x_ptr,
                       has_z,
                       delta_softplus);

    if (!recompute_out_z)
    {
        params.out_z_ptr = nullptr;
    }

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;
    params.dz_ptr = has_z ? dz.data_ptr() : nullptr;

    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.dA_d_stride = dA.stride(0);
    params.dA_dstate_stride = dA.stride(1);

    if (!is_variable_B)
    {
        params.dB_d_stride = dB.stride(0);
    }
    else
    {
        params.dB_batch_stride = dB.stride(0);
        params.dB_group_stride = dB.stride(1);
    }

    params.dB_dstate_stride = !is_variable_B ? dB.stride(1) : dB.stride(2);

    if (!is_variable_C)
    {
        params.dC_d_stride = dC.stride(0);
    }
    else
    {
        params.dC_batch_stride = dC.stride(0);
        params.dC_group_stride = dC.stride(1);
    }

    params.dC_dstate_stride = !is_variable_C ? dC.stride(1) : dC.stride(2);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_d_stride = ddelta.stride(1);

    if (has_z)
    {
        params.dz_batch_stride = dz.stride(0);
        params.dz_d_stride = dz.stride(1);
    }

    params.rev_shift_tmp_ptr = rev_shift_tmp_ptr;
}


std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor & u,
                   const at::Tensor & delta,
                   const at::Tensor & A,
                   const at::Tensor & B,
                   const at::Tensor & C,
                   const c10::optional<at::Tensor> & D_,
                   const c10::optional<at::Tensor> & z_,
                   const c10::optional<at::Tensor> & delta_bias_,
                   const at::Tensor & dout,
                   const c10::optional<at::Tensor> & x_,
                   const c10::optional<at::Tensor> & out_,
                   c10::optional<at::Tensor> & dz_,
                   bool delta_softplus,
                   bool recompute_out_z,
                   int height,
                   int width)
{
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    auto output_type = dout.scalar_type();

    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half);
    TORCH_CHECK(weight_type == at::ScalarType::Float);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));
    // TORCH_CHECK(dout.scalar_type() == input_type);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());
    TORCH_CHECK(dout.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);
    TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(seqlen == height * width, "flattened input last dimention size mismatch with specified width and height");
    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);

    if (!is_variable_B)
    {
        CHECK_SHAPE(B, dim, dstate);
    }
    else
    {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }

    if (!is_variable_C)
    {
        CHECK_SHAPE(C, dim, dstate);
    }
    else
    {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }

    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    if (D_.has_value())
    {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value())
    {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    at::Tensor z, out, dz, out_z;

    const bool has_z = z_.has_value();

    if (has_z)
    {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);

        TORCH_CHECK(out_.has_value());
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == output_type);
        TORCH_CHECK(out.is_cuda());
        TORCH_CHECK(out.stride(-1) == 1 || out.size(-1) == 1);
        CHECK_SHAPE(out, batch_size, dim, seqlen);

        if (dz_.has_value())
        {
            dz = dz_.value();
            TORCH_CHECK(dz.scalar_type() == input_type);
            TORCH_CHECK(dz.is_cuda());
            TORCH_CHECK(dz.stride(-1) == 1 || dz.size(-1) == 1);
            CHECK_SHAPE(dz, batch_size, dim, seqlen);
        }
        else
        {
            dz = torch::empty_like(z);
        }

        if (recompute_out_z)
        {
            out_z = torch::empty_like(out);
        }
    }

    // const int n_chunks = (seqlen + 2048 - 1) / 2048;
    const int numChunksDimX = (width + ndmamba::kMaxDimPerBlock - 1) / ndmamba::kMaxDimPerBlock;
    const int numChunksDimY = (height + ndmamba::kMaxDimPerBlock - 1) / ndmamba::kMaxDimPerBlock;

    if (1 < numChunksDimX || 1 < numChunksDimY)
    {
        TORCH_CHECK(x_.has_value());
    }

    if (x_.has_value())
    {
        auto x = x_.value();
        TORCH_CHECK(x.scalar_type() == weight_type);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());

        // See L261 of selective_scan_fwd.cu
        CHECK_SHAPE(
                x,
                batch_size,
                dim,
                numChunksDimY,
                numChunksDimX,
                dstate,
                2,  // horizontal, vertical
                ndmamba::kMaxDimPerBlock,
                2   // A, Bx
        );
    }

    at::Tensor delta_a_exp_shifts = torch::empty(
            {
                    batch_size,
                    dim,
                    numChunksDimY,
                    numChunksDimX,
                    dstate,
                    2,  // horizontal, vertical
                    ndmamba::kMaxDimPerBlock,
            },
//            2333.6666f,
            u.options().dtype(weight_type)
    );

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);
    at::Tensor dA = torch::zeros_like(A);
    at::Tensor dB = !is_variable_B ? torch::zeros_like(B) : torch::zeros_like(B, B.options().dtype(torch::kFloat32));
    at::Tensor dC = !is_variable_C ? torch::zeros_like(C) : torch::zeros_like(C, C.options().dtype(torch::kFloat32));
    at::Tensor dD;

    if (D_.has_value())
    {
        dD = torch::zeros_like(D_.value());
    }

    at::Tensor ddelta_bias;

    if (delta_bias_.has_value())
    {
        ddelta_bias = torch::zeros_like(delta_bias_.value());
    }

    SSMParamsBwd params;
    set_ssm_params_bwd(params, batch_size, dim, height, width, dstate, n_groups, ndmamba::kMaxDimPerBlock, is_variable_B, is_variable_C,
                       u, delta, A, B, C, z, out, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x_.has_value() ? x_.value().data_ptr() : nullptr,
                       dout, du, ddelta, dA, dB, dC, dz,
                       D_.has_value() ? dD.data_ptr() : nullptr,
                       delta_bias_.has_value() ? ddelta_bias.data_ptr() : nullptr,
                       has_z, delta_softplus, recompute_out_z,
                       delta_a_exp_shifts.data_ptr());

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard {(char) u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using weight_t = float;

    DISPATCH_ITYPE_FLOAT_AND_HALF(u.scalar_type(), "selective_scan_bwd", [&]
    {
        if (output_type == input_type)
        {
            selective_scan_bwd_cuda<input_t, weight_t, input_t>(params, stream);
        }
        else
        {
            selective_scan_bwd_cuda<input_t, weight_t, float>(params, stream);
        }
    });

    std::vector<at::Tensor> result = {du, ddelta, dA, dB.to(B.dtype()), dC.to(C.dtype()), dD, ddelta_bias};

    if (has_z)
    {
        result.push_back(dz);
    }

    if (recompute_out_z)
    {
        result.push_back(out_z);
    }

    return result;
}