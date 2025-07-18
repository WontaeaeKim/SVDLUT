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

void TriLinear2DSliceAndLUTTransformForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &grid_weights, const torch::Tensor &grid_bias,
                                                   const torch::Tensor &lut, const torch::Tensor &lut_weights, const torch::Tensor &lut_bias, torch::Tensor output);
void TriLinear2DSliceAndLUTTransformBackwardLaucher(const torch::Tensor &grad_output,
                                                    const torch::Tensor &grid, const torch::Tensor &input,
                                                    const torch::Tensor &grid_weights, const torch::Tensor &grid_bias,
                                                    const torch::Tensor &lut,
                                                    const torch::Tensor &lut_weights, const torch::Tensor &lut_bias,
                                                    torch::Tensor grad_grid, torch::Tensor grad_image,
                                                    torch::Tensor grad_grid_weights, torch::Tensor grad_grid_bias,
                                                    torch::Tensor grad_lut,
                                                    torch::Tensor grad_lut_weights, torch::Tensor grad_lut_bias);

void TriLinearCPU2DSliceAndLUTTransformForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &grid_weights, const torch::Tensor &grid_bias,
                                                      const torch::Tensor &lut, const torch::Tensor &lut_weights, const torch::Tensor &lut_bias, torch::Tensor output);

void TriLinearCPU2DSliceAndLUTTransformBackwardLaucher(const torch::Tensor &grad_output,
                                                       const torch::Tensor &grid, const torch::Tensor &input,
                                                       const torch::Tensor &grid_weights, const torch::Tensor &grid_bias,
                                                       const torch::Tensor &lut,
                                                       const torch::Tensor &lut_weights, const torch::Tensor &lut_bias,
                                                       torch::Tensor grad_grid, torch::Tensor grad_image,
                                                       torch::Tensor grad_grid_weights, torch::Tensor grad_grid_bias,
                                                       torch::Tensor grad_lut,
                                                       torch::Tensor grad_lut_weights, torch::Tensor grad_lut_bias);

void trilinear2Dslice_and_LUTTransform_forward_cuda(const torch::Tensor &grid,
                                                    const torch::Tensor &input,
                                                    const torch::Tensor &grid_weights,
                                                    const torch::Tensor &grid_bias,
                                                    const torch::Tensor &lut,
                                                    const torch::Tensor &lut_weights,
                                                    const torch::Tensor &lut_bias,
                                                    torch::Tensor output)
{
    if (input.device().is_cuda())
    {
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(output);

        CHECK_INPUT(grid_weights);
        CHECK_INPUT(grid_bias);

        CHECK_INPUT(lut);
        CHECK_INPUT(lut_weights);
        CHECK_INPUT(lut_bias);

        TriLinear2DSliceAndLUTTransformForwardLaucher(grid, input, grid_weights, grid_bias, lut, lut_weights, lut_bias, output);
    }

    else
    {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(output);

        CHECK_CONTIGUOUS(grid_weights);
        CHECK_CONTIGUOUS(grid_bias);

        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(lut_weights);
        CHECK_CONTIGUOUS(lut_bias);

        TriLinearCPU2DSliceAndLUTTransformForwardLaucher(grid, input, grid_weights, grid_bias, lut, lut_weights, lut_bias, output);
    }
}

void trilinear2Dslice_and_LUTTransform_backward_cuda(const torch::Tensor &grad_output,
                                                     const torch::Tensor &grid,
                                                     const torch::Tensor &input,
                                                     const torch::Tensor &grid_weights,
                                                     const torch::Tensor &grid_bias,
                                                     const torch::Tensor &lut,
                                                     const torch::Tensor &lut_weights,
                                                     const torch::Tensor &lut_bias,
                                                     torch::Tensor grad_grid,
                                                     torch::Tensor grad_inp,
                                                     torch::Tensor grad_grid_weights,
                                                     torch::Tensor grad_grid_bias,
                                                     torch::Tensor grad_lut,
                                                     torch::Tensor grad_lut_weights,
                                                     torch::Tensor grad_lut_bias)
{

    if (input.device().is_cuda())
    {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(grid_weights);
        CHECK_INPUT(grid_bias);

        CHECK_INPUT(lut);
        CHECK_INPUT(lut_weights);
        CHECK_INPUT(lut_bias);

        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_grid);
        CHECK_INPUT(grad_grid_weights);
        CHECK_INPUT(grad_grid_bias);

        CHECK_INPUT(grad_lut);
        CHECK_INPUT(grad_lut_weights);
        CHECK_INPUT(grad_lut_bias);

        TriLinear2DSliceAndLUTTransformBackwardLaucher(grad_output,
                                                       grid, input,
                                                       grid_weights, grid_bias,
                                                       lut,
                                                       lut_weights, lut_bias,
                                                       grad_grid, grad_inp,
                                                       grad_grid_weights, grad_grid_bias,
                                                       grad_lut,
                                                       grad_lut_weights, grad_lut_bias);
    }
    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(grid_weights);
        CHECK_CONTIGUOUS(grid_bias);

        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(lut_weights);
        CHECK_CONTIGUOUS(lut_bias);

        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_grid);
        CHECK_CONTIGUOUS(grad_grid_weights);
        CHECK_CONTIGUOUS(grad_grid_bias);

        CHECK_CONTIGUOUS(grad_lut);
        CHECK_CONTIGUOUS(grad_lut_weights);
        CHECK_CONTIGUOUS(grad_lut_bias);

        TriLinearCPU2DSliceAndLUTTransformBackwardLaucher(grad_output,
                                                          grid, input,
                                                          grid_weights, grid_bias,
                                                          lut,
                                                          lut_weights, lut_bias,
                                                          grad_grid, grad_inp,
                                                          grad_grid_weights, grad_grid_bias,
                                                          grad_lut,
                                                          grad_lut_weights, grad_lut_bias);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tri_forward", &trilinear2Dslice_and_LUTTransform_forward_cuda, "Trilinear Slice and LUT transform forward");
    m.def("tri_backward", &trilinear2Dslice_and_LUTTransform_backward_cuda, "Trilinear Slice and LUT transform backward");
}
