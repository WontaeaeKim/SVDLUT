import torch

from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Tuple
'''
#for test
import lut2D_transform
import bilateral2D_slicing_wo_conv
'''
import bilateral2D_slicing_LUTTransform


class Bilinear2DSliceAndLUTTransformFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                grid: torch.Tensor, 
                x: torch.Tensor,
                grid_weight: torch.Tensor, 
                grid_bias: torch.Tensor,
                lut: torch.Tensor,
                lut_weight: torch.Tensor, 
                lut_bias: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        grid = grid.contiguous()
        
        grid_weight = grid_weight.contiguous()
        grid_bias = grid_bias.contiguous()
        
        lut = lut.contiguous()
        lut_weight = lut_weight.contiguous()
        lut_bias = lut_bias.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert grid.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), lut.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        bilateral2D_slicing_LUTTransform.tri_forward(grid, x, grid_weight, grid_bias, lut, lut_weight, lut_bias, output)
        
        ctx.save_for_backward(grid, x, grid_weight, grid_bias, lut, lut_weight, lut_bias)
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        grid, x, grid_weight, grid_bias, lut, lut_weight, lut_bias = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_grid = torch.zeros_like(grid) 
        
        grad_grid_weights = torch.zeros_like(grid_weight)
        grad_grid_bias = torch.zeros_like(grid_bias)
        
        grad_lut = torch.zeros_like(lut) 
        
        grad_lut_weights = torch.zeros_like(lut_weight)
        grad_lut_bias = torch.zeros_like(lut_bias)
                  
        
        bilateral2D_slicing_LUTTransform.tri_backward(grad_output, grid, x, grid_weight, grid_bias, lut, lut_weight, lut_bias, grad_grid, grad_img, grad_grid_weights, grad_grid_bias, grad_lut, grad_lut_weights, grad_lut_bias)

        return grad_grid, grad_img, grad_grid_weights, grad_grid_bias, grad_lut, grad_lut_weights, grad_lut_bias
    

def bilinear_2Dslicing_lut_transform(
    grid: torch.Tensor,
    img: torch.Tensor,
    grid_weights: torch.Tensor,
    grid_bias: torch.Tensor,
    lut: torch.Tensor,
    lut_weights: torch.Tensor,
    lut_bias: torch.Tensor) -> torch.Tensor:
    r"""Trilinear 3D Lookup Table Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return Bilinear2DSliceAndLUTTransformFunction.apply(grid, img, grid_weights, grid_bias, lut, lut_weights, lut_bias)


# For test
'''
class Trilinear2DLUTTransformFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                lut: torch.Tensor, 
                x: torch.Tensor,
                lut_weights: torch.Tensor,
                lut_bias: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        lut = lut.contiguous()
        lut_weights = lut_weights.contiguous()
        lut_bias = lut_bias.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), lut.size(1), x.size(2), x.size(3)))
        output.contiguous()
               
        lut2D_transform.tri2D_forward(lut, x, lut_weights, lut_bias, output)
        
        ctx.save_for_backward(lut, x, lut_weights, lut_bias)
        
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        lut, x , lut_weights, lut_bias  = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_lut = torch.zeros_like(lut)
        
        grad_weights = torch.zeros_like(lut_weights)
        grad_bias = torch.zeros_like(lut_bias)           
        
        lut2D_transform.tri2D_backward(grad_output, lut, x, lut_weights, lut_bias, grad_lut, grad_img, grad_weights, grad_bias)

        return grad_lut, grad_img, grad_weights, grad_bias
    
# For test    
class Trilinear2DSliceFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, 
                grid: torch.Tensor, 
                x: torch.Tensor,
                grid_weight: torch.Tensor, 
                grid_bias: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        grid = grid.contiguous()
        
        grid_weight = grid_weight.contiguous()
        grid_bias = grid_bias.contiguous()

        assert x.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert grid.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        
        output = x.new_zeros((x.size(0), grid.size(1), x.size(2), x.size(3)))
        output.contiguous()
                
        bilateral2D_slicing_wo_conv.tri_forward(grid, x, grid_weight, grid_bias, output)
        
        ctx.save_for_backward(grid, x, grid_weight, grid_bias)
        
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        
        grad_output = grad_output.contiguous()
        
        grid, x, grid_weight, grid_bias = ctx.saved_tensors
                
        grad_img = torch.zeros_like(x)
        grad_grid = torch.zeros_like(grid) 
        
        grad_weights = torch.zeros_like(grid_weight)
        grad_bias = torch.zeros_like(grid_bias)          
        
        bilateral2D_slicing_wo_conv.tri_backward(grad_output, grid, x, grid_weight, grid_bias, grad_grid, grad_img, grad_weights, grad_bias)

        return grad_grid, grad_img, grad_weights, grad_bias


def trilinear2D_slice_function(
    grid: torch.Tensor,
    img: torch.Tensor,
    grid_weight: torch.Tensor, 
    grid_bias: torch.Tensor) -> torch.Tensor:
    r"""Trilinear Bilateral Grid Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        grid (torch.Tensor): output values of the Bilateral grid, shape (b, 3*N, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3*(N + 1), h, w).
    """
    return Trilinear2DSliceFunction.apply(grid, img, grid_weight, grid_bias)
'''