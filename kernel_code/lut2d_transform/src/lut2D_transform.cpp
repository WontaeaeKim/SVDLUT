// #include "trilinear_kernel.h"
// #include <torch/extension.h>
// #include <THC/THC.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAEvent.h>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void TriLinear2DForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, torch::Tensor output);
void TriLinear2DBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &lut, const torch::Tensor &input,
                                const torch::Tensor &weights, const torch::Tensor &bias,
                                torch::Tensor grad_lut, torch::Tensor grad_image, torch::Tensor grad_weights, torch::Tensor grad_bias);

void TriLinear2DCPUForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, torch::Tensor output);
void TriLinear2DCPUBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &lut, const torch::Tensor &input,
                                   const torch::Tensor &weights, const torch::Tensor &bias,
                                   torch::Tensor grad_lut, torch::Tensor grad_image, torch::Tensor grad_weights, torch::Tensor grad_bias);

void trilinear2D_forward_cuda(const torch::Tensor &lut,
                              const torch::Tensor &input,
                              const torch::Tensor &weights,
                              const torch::Tensor &bias,
                              torch::Tensor output)
{
    if (input.device().is_cuda())
    {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(weights);
        CHECK_INPUT(bias);
        CHECK_INPUT(output);

        TriLinear2DForwardLaucher(lut, input, weights, bias, output);
    }
    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(weights);
        CHECK_CONTIGUOUS(bias);
        CHECK_CONTIGUOUS(output);

        TriLinear2DCPUForwardLaucher(lut, input, weights, bias, output);
    }
}

void trilinear2D_backward_cuda(const torch::Tensor &grad_output,
                               const torch::Tensor &lut,
                               const torch::Tensor &input,
                               const torch::Tensor &weights,
                               const torch::Tensor &bias,
                               torch::Tensor grad_lut,
                               torch::Tensor grad_inp,
                               torch::Tensor grad_weights,
                               torch::Tensor grad_bias)
{
    if (input.device().is_cuda())
    {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);

        CHECK_INPUT(weights);
        CHECK_INPUT(bias);
        CHECK_INPUT(grad_weights);
        CHECK_INPUT(grad_bias);

        TriLinear2DBackwardLaucher(grad_output, lut, input, weights, bias, grad_lut, grad_inp, grad_weights, grad_bias);
    }

    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);

        CHECK_CONTIGUOUS(weights);
        CHECK_CONTIGUOUS(bias);
        CHECK_CONTIGUOUS(grad_weights);
        CHECK_CONTIGUOUS(grad_bias);

        TriLinear2DCPUBackwardLaucher(grad_output, lut, input, weights, bias, grad_lut, grad_inp, grad_weights, grad_bias);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tri2D_forward", &trilinear2D_forward_cuda, "Trilinear 2D forward");
    m.def("tri2D_backward", &trilinear2D_backward_cuda, "Trilinear 2D backward");
}
