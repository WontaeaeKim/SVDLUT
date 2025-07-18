//#include "trilinear_kernel.h"
//#include <torch/extension.h>
//#include <THC/THC.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAEvent.h>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void TriLinear2DSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, torch::Tensor output); 
void TriLinear2DSliceBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, 
     torch::Tensor grad_grid, torch::Tensor grad_image, torch::Tensor grad_weights, torch::Tensor grad_bias);


void TriLinear2DCPUSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, torch::Tensor output);
void TriLinear2DCPUSliceBackwardLaucher(const torch::Tensor &grad_output, const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, 
     torch::Tensor grad_grid, torch::Tensor grad_image, torch::Tensor grad_weights, torch::Tensor grad_bias);

void trilinear2Dslice_forward_cuda(const torch::Tensor &grid,
    const torch::Tensor &input,
    const torch::Tensor &weights,
    const torch::Tensor &bias, 
    torch::Tensor output)
{
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(output);
        
        CHECK_INPUT(weights);
        CHECK_INPUT(bias);

        TriLinear2DSliceForwardLaucher(grid, input, weights, bias, output);
    }
    
    else
    {
  
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(output);
        
        CHECK_CONTIGUOUS(weights);
        CHECK_CONTIGUOUS(bias);
        
        TriLinear2DCPUSliceForwardLaucher(grid, input, weights, bias, output);
    }
    
}

void trilinear2Dslice_backward_cuda(const torch::Tensor &grad_output,
    const torch::Tensor &grid,
    const torch::Tensor &input,
    const torch::Tensor &weights,
    const torch::Tensor &bias,
    torch::Tensor grad_grid,
    torch::Tensor grad_inp,
    torch::Tensor grad_weights,
    torch::Tensor grad_bias)
{
    

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(grid);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_grid);

        CHECK_INPUT(weights);
        CHECK_INPUT(bias);
        CHECK_INPUT(grad_weights);
        CHECK_INPUT(grad_bias);
        
        TriLinear2DSliceBackwardLaucher(grad_output, grid, input, weights, bias, grad_grid, grad_inp, grad_weights, grad_bias);
    }

    else
    {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(grid);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_grid);

        CHECK_CONTIGUOUS(weights);
        CHECK_CONTIGUOUS(bias);
        CHECK_CONTIGUOUS(grad_weights);
        CHECK_CONTIGUOUS(grad_bias);
        
        TriLinear2DCPUSliceBackwardLaucher(grad_output, grid, input, weights, bias, grad_grid, grad_inp, grad_weights, grad_bias);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_forward", &trilinear2Dslice_forward_cuda, "Trilinear Slice forward");
  m.def("tri_backward", &trilinear2Dslice_backward_cuda, "Trilinear Slice backward");
}

