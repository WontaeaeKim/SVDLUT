#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return min(optimal_block_num, max_block_num);
}

/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline __device__ constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void TriLinear2DSliceForward(const int nthreads, 
                                const scalar_t* __restrict__ grid, 
                                const scalar_t* __restrict__ image,
                                const scalar_t* __restrict__ weights,
                                const scalar_t* __restrict__ bias, 
                                scalar_t* __restrict__ output, 
                                const int dim, 
                                const int shift, 
                                const scalar_t binsize, 
                                const int width, 
                                const int height,
								const int num_channels,
                                const int grid_per_ch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        const int x_ = index % width;
	    const int y_ = index / width;

        const scalar_t x = x_ / (width -1);
	    const scalar_t y = y_ / (height -1);

	    const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];
        
        const int32_t x_id = clamp((int32_t)floor(x * (dim-1)),0, dim-2);
        const int32_t y_id = clamp((int32_t)floor(y * (dim-1)),0, dim-2);

	    const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
	    const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
	    const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  x_d = (x - binsize * x_id) / binsize;
        const scalar_t  y_d = (y - binsize * y_id) / binsize;

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id00_xy = (x_id    ) + (y_id    ) * dim;
        const int id10_xy = (x_id + 1) + (y_id    ) * dim;
        const int id01_xy = (x_id    ) + (y_id + 1) * dim;
        const int id11_xy = (x_id + 1) + (y_id + 1) * dim;


		const int id00_xr = (x_id    ) + (r_id    ) * dim; 
        const int id10_xr = (x_id + 1) + (r_id    ) * dim;
        const int id01_xr = (x_id    ) + (r_id + 1) * dim;
        const int id11_xr = (x_id + 1) + (r_id + 1) * dim;

        const int id00_yr = (y_id    ) + (r_id    ) * dim; 
        const int id10_yr = (y_id + 1) + (r_id    ) * dim;
        const int id01_yr = (y_id    ) + (r_id + 1) * dim;
        const int id11_yr = (y_id + 1) + (r_id + 1) * dim;


        const int id00_xg = (x_id    ) + (g_id    ) * dim; 
        const int id10_xg = (x_id + 1) + (g_id    ) * dim;
        const int id01_xg = (x_id    ) + (g_id + 1) * dim;
        const int id11_xg = (x_id + 1) + (g_id + 1) * dim;

        const int id00_yg = (y_id    ) + (g_id    ) * dim; 
        const int id10_yg = (y_id + 1) + (g_id    ) * dim;
        const int id01_yg = (y_id    ) + (g_id + 1) * dim;
        const int id11_yg = (y_id + 1) + (g_id + 1) * dim;

        const int id00_xb = (x_id    ) + (b_id    ) * dim; 
        const int id10_xb = (x_id + 1) + (b_id    ) * dim;
        const int id01_xb = (x_id    ) + (b_id + 1) * dim;
        const int id11_xb = (x_id + 1) + (b_id + 1) * dim;

        const int id00_yb = (y_id    ) + (b_id    ) * dim; 
        const int id10_yb = (y_id + 1) + (b_id    ) * dim;
        const int id01_yb = (y_id    ) + (b_id + 1) * dim;
        const int id11_yb = (y_id + 1) + (b_id + 1) * dim;

		
        const scalar_t  w00_xy = (1 - x_d) * (1 - y_d);
        const scalar_t  w10_xy = (    x_d) * (1 - y_d);
        const scalar_t  w01_xy = (1 - x_d) * (    y_d);
        const scalar_t  w11_xy = (    x_d) * (    y_d);


        const scalar_t  w00_xr = (1 - x_d) * (1 - r_d);
        const scalar_t  w10_xr = (    x_d) * (1 - r_d);
        const scalar_t  w01_xr = (1 - x_d) * (    r_d);
        const scalar_t  w11_xr = (    x_d) * (    r_d);

        const scalar_t  w00_yr = (1 - y_d) * (1 - r_d);
        const scalar_t  w10_yr = (    y_d) * (1 - r_d);
        const scalar_t  w01_yr = (1 - y_d) * (    r_d);
        const scalar_t  w11_yr = (    y_d) * (    r_d);


		const scalar_t  w00_xg = (1 - x_d) * (1 - g_d);
        const scalar_t  w10_xg = (    x_d) * (1 - g_d);
        const scalar_t  w01_xg = (1 - x_d) * (    g_d);
        const scalar_t  w11_xg = (    x_d) * (    g_d);

        const scalar_t  w00_yg = (1 - y_d) * (1 - g_d);
        const scalar_t  w10_yg = (    y_d) * (1 - g_d);
        const scalar_t  w01_yg = (1 - y_d) * (    g_d);
        const scalar_t  w11_yg = (    y_d) * (    g_d);


        const scalar_t  w00_xb = (1 - x_d) * (1 - b_d);
        const scalar_t  w10_xb = (    x_d) * (1 - b_d);
        const scalar_t  w01_xb = (1 - x_d) * (    b_d);
        const scalar_t  w11_xb = (    x_d) * (    b_d);

        const scalar_t  w00_yb = (1 - y_d) * (1 - b_d);
        const scalar_t  w10_yb = (    y_d) * (1 - b_d);
        const scalar_t  w01_yb = (1 - y_d) * (    b_d);
        const scalar_t  w11_yb = (    y_d) * (    b_d);

        scalar_t int_img[3] = {0,};

        for(int i = 0; i < grid_per_ch; ++i){
            int_img[0] += weights[3*(i + grid_per_ch * 0)    ] * (w00_xy * grid[id00_xy + shift * (3 * (i + grid_per_ch * 0) + 0)] + 
                                                                                                               w10_xy * grid[id10_xy + shift * (3 * (i + grid_per_ch * 0) + 0)] +
                                                                                                               w01_xy * grid[id01_xy + shift * (3 * (i + grid_per_ch * 0) + 0)] + 
                                                                                                               w11_xy * grid[id11_xy + shift * (3 * (i + grid_per_ch * 0) + 0)]) +
                                                                       
                                                                       weights[3*(i + grid_per_ch * 0) + 1] * (w00_xr * grid[id00_xr + shift * (3 * (i + grid_per_ch * 0) + 1)] + 
                                                                                                               w10_xr * grid[id10_xr + shift * (3 * (i + grid_per_ch * 0) + 1)] +
                                                                                                               w01_xr * grid[id01_xr + shift * (3 * (i + grid_per_ch * 0) + 1)] + 
                                                                                                               w11_xr * grid[id11_xr + shift * (3 * (i + grid_per_ch * 0) + 1)]) +

                                                                       weights[3*(i + grid_per_ch * 0) + 2] * (w00_yr * grid[id00_yr + shift * (3 * (i + grid_per_ch * 0) + 2)] + 
                                                                                                               w10_yr * grid[id10_yr + shift * (3 * (i + grid_per_ch * 0) + 2)] +
                                                                                                               w01_yr * grid[id01_yr + shift * (3 * (i + grid_per_ch * 0) + 2)] + 
                                                                                                               w11_yr * grid[id11_yr + shift * (3 * (i + grid_per_ch * 0) + 2)]) +                                      
                                                                        bias[(i + grid_per_ch * 0)];

																	 
			int_img[1] += weights[3*(i + grid_per_ch * 1)    ] * (w00_xy * grid[id00_xy + shift * (3 * (i + grid_per_ch * 1) + 0)] + 
                                                                                                                w10_xy * grid[id10_xy + shift * (3 * (i + grid_per_ch * 1) + 0)] +
                                                                                                                w01_xy * grid[id01_xy + shift * (3 * (i + grid_per_ch * 1) + 0)] + 
                                                                                                                w11_xy * grid[id11_xy + shift * (3 * (i + grid_per_ch * 1) + 0)]) +
                                                                       
                                                                        weights[3*(i + grid_per_ch * 1) + 1] * (w00_xg * grid[id00_xg + shift * (3 * (i + grid_per_ch * 1) + 1)] + 
                                                                                                                w10_xg * grid[id10_xg + shift * (3 * (i + grid_per_ch * 1) + 1)] +
                                                                                                                w01_xg * grid[id01_xg + shift * (3 * (i + grid_per_ch * 1) + 1)] + 
                                                                                                                w11_xg * grid[id11_xg + shift * (3 * (i + grid_per_ch * 1) + 1)]) +

                                                                        weights[3*(i + grid_per_ch * 1) + 2] * (w00_yg * grid[id00_yg + shift * (3 * (i + grid_per_ch * 1) + 2)] + 
                                                                                                                w10_yg * grid[id10_yg + shift * (3 * (i + grid_per_ch * 1) + 2)] +
                                                                                                                w01_yg * grid[id01_yg + shift * (3 * (i + grid_per_ch * 1) + 2)] + 
                                                                                                                w11_yg * grid[id11_yg + shift * (3 * (i + grid_per_ch * 1) + 2)]) + 
                                                                        bias[(i + grid_per_ch * 1)];
																			
            int_img[2] += weights[3*(i + grid_per_ch * 2)    ] * (w00_xy * grid[id00_xy + shift * (3 * (i + grid_per_ch * 2) + 0)] + 
                                                                                                                w10_xy * grid[id10_xy + shift * (3 * (i + grid_per_ch * 2) + 0)] +
                                                                                                                w01_xy * grid[id01_xy + shift * (3 * (i + grid_per_ch * 2) + 0)] + 
                                                                                                                w11_xy * grid[id11_xy + shift * (3 * (i + grid_per_ch * 2) + 0)]) +
                                                                       
                                                                        weights[3*(i + grid_per_ch * 2) + 1] * (w00_xb * grid[id00_xb + shift * (3 * (i + grid_per_ch * 2) + 1)] + 
                                                                                                                w10_xb * grid[id10_xb + shift * (3 * (i + grid_per_ch * 2) + 1)] +
                                                                                                                w01_xb * grid[id01_xb + shift * (3 * (i + grid_per_ch * 2) + 1)] + 
                                                                                                                w11_xb * grid[id11_xb + shift * (3 * (i + grid_per_ch * 2) + 1)]) +

                                                                        weights[3*(i + grid_per_ch * 2) + 2] * (w00_yb * grid[id00_yb + shift * (3 * (i + grid_per_ch * 2) + 2)] + 
                                                                                                                w10_yb * grid[id10_yb + shift * (3 * (i + grid_per_ch * 2) + 2)] +
                                                                                                                w01_yb * grid[id01_yb + shift * (3 * (i + grid_per_ch * 2) + 2)] + 
                                                                                                                w11_yb * grid[id11_yb + shift * (3 * (i + grid_per_ch * 2) + 2)]) + 
                                                                        bias[(i + grid_per_ch * 2)];
        }
        output[index + width * height * 0] = int_img[0];
		output[index + width * height * 1] = int_img[1];
		output[index + width * height * 2] = int_img[2];
    }
}


void TriLinear2DSliceForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, torch::Tensor output) {
    c10::cuda::CUDAGuard device_guard(input.device());
    
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = input.size(1);
	int grid_channels = grid.size(1);
    int dim   = grid.size(3);
    int shift   = dim * dim;
	
	int grid_per_ch = grid_channels / num_channels;
   
    int num_kernels = height * width;
    
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cuda_forward", ([&] {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                const scalar_t *data_weights = weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_bias = bias[elt].data_ptr<scalar_t>();

                TriLinear2DSliceForward<<<GET_BLOCKS(num_kernels),
                                              THREADS_PER_BLOCK, 0,
                                              at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_grid, data_image, data_weights, data_bias, data_output,
                    dim, shift, binsize,
                    width, height, num_channels, grid_per_ch);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void TriLinear2DSliceBackward(const int nthreads,
                                  const scalar_t* __restrict__  output_grad, 
                                  const scalar_t* __restrict__ grid, 
                                  const scalar_t* __restrict__ image,
                                  const scalar_t* __restrict__ weights,
                                  const scalar_t* __restrict__ bias,                                                                     
                                  scalar_t* __restrict__  grid_grad,
                                  scalar_t* __restrict__  image_grad,
                                  scalar_t* __restrict__  wights_grad,
                                  scalar_t* __restrict__  bias_grad, 
                                  const int dim, 
                                  const int shift, 
                                  const scalar_t binsize, 
                                  const int width, 
                                  const int height, 
                                  const int num_channels, 
                                  const int grid_per_ch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        const int x_ = index % width;
	    const int y_ = index / width;

        const scalar_t x = x_ / (width -1);
	    const scalar_t y = y_ / (height -1);

	    const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];
        
        const int32_t x_id = clamp((int32_t)floor(x * (dim-1)),0, dim-2);
        const int32_t y_id = clamp((int32_t)floor(y * (dim-1)),0, dim-2);

	    const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
	    const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
	    const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  x_d = (x - binsize * x_id) / binsize;
        const scalar_t  y_d = (y - binsize * y_id) / binsize;

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id00_xy = (x_id    ) + (y_id    ) * dim;
        const int id10_xy = (x_id + 1) + (y_id    ) * dim;
        const int id01_xy = (x_id    ) + (y_id + 1) * dim;
        const int id11_xy = (x_id + 1) + (y_id + 1) * dim;


		const int id00_xr = (x_id    ) + (r_id    ) * dim; 
        const int id10_xr = (x_id + 1) + (r_id    ) * dim;
        const int id01_xr = (x_id    ) + (r_id + 1) * dim;
        const int id11_xr = (x_id + 1) + (r_id + 1) * dim;

        const int id00_yr = (y_id    ) + (r_id    ) * dim; 
        const int id10_yr = (y_id + 1) + (r_id    ) * dim;
        const int id01_yr = (y_id    ) + (r_id + 1) * dim;
        const int id11_yr = (y_id + 1) + (r_id + 1) * dim;


        const int id00_xg = (x_id    ) + (g_id    ) * dim; 
        const int id10_xg = (x_id + 1) + (g_id    ) * dim;
        const int id01_xg = (x_id    ) + (g_id + 1) * dim;
        const int id11_xg = (x_id + 1) + (g_id + 1) * dim;

        const int id00_yg = (y_id    ) + (g_id    ) * dim; 
        const int id10_yg = (y_id + 1) + (g_id    ) * dim;
        const int id01_yg = (y_id    ) + (g_id + 1) * dim;
        const int id11_yg = (y_id + 1) + (g_id + 1) * dim;
        

        const int id00_xb = (x_id    ) + (b_id    ) * dim; 
        const int id10_xb = (x_id + 1) + (b_id    ) * dim;
        const int id01_xb = (x_id    ) + (b_id + 1) * dim;
        const int id11_xb = (x_id + 1) + (b_id + 1) * dim;

        const int id00_yb = (y_id    ) + (b_id    ) * dim; 
        const int id10_yb = (y_id + 1) + (b_id    ) * dim;
        const int id01_yb = (y_id    ) + (b_id + 1) * dim;
        const int id11_yb = (y_id + 1) + (b_id + 1) * dim;

		
        const scalar_t  w00_xy = (1 - x_d) * (1 - y_d);
        const scalar_t  w10_xy = (    x_d) * (1 - y_d);
        const scalar_t  w01_xy = (1 - x_d) * (    y_d);
        const scalar_t  w11_xy = (    x_d) * (    y_d);


        const scalar_t  w00_xr = (1 - x_d) * (1 - r_d);
        const scalar_t  w10_xr = (    x_d) * (1 - r_d);
        const scalar_t  w01_xr = (1 - x_d) * (    r_d);
        const scalar_t  w11_xr = (    x_d) * (    r_d);

        const scalar_t  w00_yr = (1 - y_d) * (1 - r_d);
        const scalar_t  w10_yr = (    y_d) * (1 - r_d);
        const scalar_t  w01_yr = (1 - y_d) * (    r_d);
        const scalar_t  w11_yr = (    y_d) * (    r_d);


		const scalar_t  w00_xg = (1 - x_d) * (1 - g_d);
        const scalar_t  w10_xg = (    x_d) * (1 - g_d);
        const scalar_t  w01_xg = (1 - x_d) * (    g_d);
        const scalar_t  w11_xg = (    x_d) * (    g_d);

        const scalar_t  w00_yg = (1 - y_d) * (1 - g_d);
        const scalar_t  w10_yg = (    y_d) * (1 - g_d);
        const scalar_t  w01_yg = (1 - y_d) * (    g_d);
        const scalar_t  w11_yg = (    y_d) * (    g_d);


        const scalar_t  w00_xb = (1 - x_d) * (1 - b_d);
        const scalar_t  w10_xb = (    x_d) * (1 - b_d);
        const scalar_t  w01_xb = (1 - x_d) * (    b_d);
        const scalar_t  w11_xb = (    x_d) * (    b_d);

        const scalar_t  w00_yb = (1 - y_d) * (1 - b_d);
        const scalar_t  w10_yb = (    y_d) * (1 - b_d);
        const scalar_t  w01_yb = (1 - y_d) * (    b_d);
        const scalar_t  w11_yb = (    y_d) * (    b_d);
		
 		/* derivatives: w to rd, gd, bd */
		const scalar_t w00_xd = - (1 - x_d);
		const scalar_t w10_xd = - (    x_d);
		const scalar_t w01_xd =   (1 - x_d);
		const scalar_t w11_xd =   (    x_d);

		const scalar_t w00_yd = - (1 - y_d);
		const scalar_t w10_yd = - (    y_d);
		const scalar_t w01_yd =   (1 - y_d);
		const scalar_t w11_yd =   (    y_d);
		
        scalar_t grad_o_r = output_grad[index + width * height * 0];
        scalar_t grad_o_g = output_grad[index + width * height * 1];
        scalar_t grad_o_b = output_grad[index + width * height * 2];

		for(int i=0;i<grid_per_ch;++i)
		{

			atomicAdd(grid_grad + id00_xy + shift * (3 * (i + grid_per_ch * 0) + 0), grad_o_r * weights[3*(i + grid_per_ch * 0) + 0] * w00_xy);
			atomicAdd(grid_grad + id10_xy + shift * (3 * (i + grid_per_ch * 0) + 0), grad_o_r * weights[3*(i + grid_per_ch * 0) + 0] * w10_xy);
			atomicAdd(grid_grad + id01_xy + shift * (3 * (i + grid_per_ch * 0) + 0), grad_o_r * weights[3*(i + grid_per_ch * 0) + 0] * w01_xy);
			atomicAdd(grid_grad + id11_xy + shift * (3 * (i + grid_per_ch * 0) + 0), grad_o_r * weights[3*(i + grid_per_ch * 0) + 0] * w11_xy);

            atomicAdd(grid_grad + id00_xr + shift * (3 * (i + grid_per_ch * 0) + 1), grad_o_r * weights[3*(i + grid_per_ch * 0) + 1] * w00_xr);
			atomicAdd(grid_grad + id10_xr + shift * (3 * (i + grid_per_ch * 0) + 1), grad_o_r * weights[3*(i + grid_per_ch * 0) + 1] * w10_xr);
			atomicAdd(grid_grad + id01_xr + shift * (3 * (i + grid_per_ch * 0) + 1), grad_o_r * weights[3*(i + grid_per_ch * 0) + 1] * w01_xr);
			atomicAdd(grid_grad + id11_xr + shift * (3 * (i + grid_per_ch * 0) + 1), grad_o_r * weights[3*(i + grid_per_ch * 0) + 1] * w11_xr);

            atomicAdd(grid_grad + id00_yr + shift * (3 * (i + grid_per_ch * 0) + 2), grad_o_r * weights[3*(i + grid_per_ch * 0) + 2] * w00_yr);
			atomicAdd(grid_grad + id10_yr + shift * (3 * (i + grid_per_ch * 0) + 2), grad_o_r * weights[3*(i + grid_per_ch * 0) + 2] * w10_yr);
			atomicAdd(grid_grad + id01_yr + shift * (3 * (i + grid_per_ch * 0) + 2), grad_o_r * weights[3*(i + grid_per_ch * 0) + 2] * w01_yr);
			atomicAdd(grid_grad + id11_yr + shift * (3 * (i + grid_per_ch * 0) + 2), grad_o_r * weights[3*(i + grid_per_ch * 0) + 2] * w11_yr);


            atomicAdd(grid_grad + id00_xy + shift * (3 * (i + grid_per_ch * 1) + 0), grad_o_g * weights[3*(i + grid_per_ch * 1) + 0] * w00_xy);
			atomicAdd(grid_grad + id10_xy + shift * (3 * (i + grid_per_ch * 1) + 0), grad_o_g * weights[3*(i + grid_per_ch * 1) + 0] * w10_xy);
			atomicAdd(grid_grad + id01_xy + shift * (3 * (i + grid_per_ch * 1) + 0), grad_o_g * weights[3*(i + grid_per_ch * 1) + 0] * w01_xy);
			atomicAdd(grid_grad + id11_xy + shift * (3 * (i + grid_per_ch * 1) + 0), grad_o_g * weights[3*(i + grid_per_ch * 1) + 0] * w11_xy);

            atomicAdd(grid_grad + id00_xg + shift * (3 * (i + grid_per_ch * 1) + 1), grad_o_g * weights[3*(i + grid_per_ch * 1) + 1] * w00_xg);
			atomicAdd(grid_grad + id10_xg + shift * (3 * (i + grid_per_ch * 1) + 1), grad_o_g * weights[3*(i + grid_per_ch * 1) + 1] * w10_xg);
			atomicAdd(grid_grad + id01_xg + shift * (3 * (i + grid_per_ch * 1) + 1), grad_o_g * weights[3*(i + grid_per_ch * 1) + 1] * w01_xg);
			atomicAdd(grid_grad + id11_xg + shift * (3 * (i + grid_per_ch * 1) + 1), grad_o_g * weights[3*(i + grid_per_ch * 1) + 1] * w11_xg);

            atomicAdd(grid_grad + id00_yg + shift * (3 * (i + grid_per_ch * 1) + 2), grad_o_g * weights[3*(i + grid_per_ch * 1) + 2] * w00_yg);
			atomicAdd(grid_grad + id10_yg + shift * (3 * (i + grid_per_ch * 1) + 2), grad_o_g * weights[3*(i + grid_per_ch * 1) + 2] * w10_yg);
			atomicAdd(grid_grad + id01_yg + shift * (3 * (i + grid_per_ch * 1) + 2), grad_o_g * weights[3*(i + grid_per_ch * 1) + 2] * w01_yg);
			atomicAdd(grid_grad + id11_yg + shift * (3 * (i + grid_per_ch * 1) + 2), grad_o_g * weights[3*(i + grid_per_ch * 1) + 2] * w11_yg);


            atomicAdd(grid_grad + id00_xy + shift * (3 * (i + grid_per_ch * 2) + 0), grad_o_b * weights[3*(i + grid_per_ch * 2) + 0] * w00_xy);
			atomicAdd(grid_grad + id10_xy + shift * (3 * (i + grid_per_ch * 2) + 0), grad_o_b * weights[3*(i + grid_per_ch * 2) + 0] * w10_xy);
			atomicAdd(grid_grad + id01_xy + shift * (3 * (i + grid_per_ch * 2) + 0), grad_o_b * weights[3*(i + grid_per_ch * 2) + 0] * w01_xy);
			atomicAdd(grid_grad + id11_xy + shift * (3 * (i + grid_per_ch * 2) + 0), grad_o_b * weights[3*(i + grid_per_ch * 2) + 0] * w11_xy);

            atomicAdd(grid_grad + id00_xb + shift * (3 * (i + grid_per_ch * 2) + 1), grad_o_b * weights[3*(i + grid_per_ch * 2) + 1] * w00_xb);
			atomicAdd(grid_grad + id10_xb + shift * (3 * (i + grid_per_ch * 2) + 1), grad_o_b * weights[3*(i + grid_per_ch * 2) + 1] * w10_xb);
			atomicAdd(grid_grad + id01_xb + shift * (3 * (i + grid_per_ch * 2) + 1), grad_o_b * weights[3*(i + grid_per_ch * 2) + 1] * w01_xb);
			atomicAdd(grid_grad + id11_xb + shift * (3 * (i + grid_per_ch * 2) + 1), grad_o_b * weights[3*(i + grid_per_ch * 2) + 1] * w11_xb);

            atomicAdd(grid_grad + id00_yb + shift * (3 * (i + grid_per_ch * 2) + 2), grad_o_b * weights[3*(i + grid_per_ch * 2) + 2] * w00_yb);
			atomicAdd(grid_grad + id10_yb + shift * (3 * (i + grid_per_ch * 2) + 2), grad_o_b * weights[3*(i + grid_per_ch * 2) + 2] * w10_yb);
			atomicAdd(grid_grad + id01_yb + shift * (3 * (i + grid_per_ch * 2) + 2), grad_o_b * weights[3*(i + grid_per_ch * 2) + 2] * w01_yb);
			atomicAdd(grid_grad + id11_yb + shift * (3 * (i + grid_per_ch * 2) + 2), grad_o_b * weights[3*(i + grid_per_ch * 2) + 2] * w11_yb);


			scalar_t grad_d = 0;
			scalar_t grid00_x = grid[id00_xr + shift * (3 * (i + grid_per_ch * 0) + 1)];
			scalar_t grid10_x = grid[id10_xr + shift * (3 * (i + grid_per_ch * 0) + 1)];
			scalar_t grid01_x = grid[id01_xr + shift * (3 * (i + grid_per_ch * 0) + 1)];
			scalar_t grid11_x = grid[id11_xr + shift * (3 * (i + grid_per_ch * 0) + 1)];
			
            scalar_t grid00_y = grid[id00_yr + shift * (3 * (i + grid_per_ch * 0) + 2)];
			scalar_t grid10_y = grid[id10_yr + shift * (3 * (i + grid_per_ch * 0) + 2)];
			scalar_t grid01_y = grid[id01_yr + shift * (3 * (i + grid_per_ch * 0) + 2)];
			scalar_t grid11_y = grid[id11_yr + shift * (3 * (i + grid_per_ch * 0) + 2)];
			// r
			grad_d = grad_o_r *
					(weights[3*(i + grid_per_ch * 0) + 1] * (w00_xd * grid00_x + w10_xd * grid10_x + w01_xd * grid01_x + w11_xd * grid11_x) +
					 weights[3*(i + grid_per_ch * 0) + 2] * (w00_yd * grid00_y + w10_yd * grid10_y + w01_yd * grid01_y + w11_yd * grid11_y));
			atomicAdd(image_grad + index, grad_d * 1 / binsize);
			// g
			grid00_x = grid[id00_xg + shift * (3 * (i + grid_per_ch * 1) + 1)];
			grid10_x = grid[id10_xg + shift * (3 * (i + grid_per_ch * 1) + 1)];
			grid01_x = grid[id01_xg + shift * (3 * (i + grid_per_ch * 1) + 1)];
			grid11_x = grid[id11_xg + shift * (3 * (i + grid_per_ch * 1) + 1)];
			
            grid00_y = grid[id00_yg + shift * (3 * (i + grid_per_ch * 1) + 2)];
			grid10_y = grid[id10_yg + shift * (3 * (i + grid_per_ch * 1) + 2)];
			grid01_y = grid[id01_yg + shift * (3 * (i + grid_per_ch * 1) + 2)];
			grid11_y = grid[id11_yg + shift * (3 * (i + grid_per_ch * 1) + 2)];
			grad_d = grad_o_g *
					(weights[3*(i + grid_per_ch * 1) + 1] * (w00_xd * grid00_x + w10_xd * grid10_x + w01_xd * grid01_x + w11_xd * grid11_x) +
					 weights[3*(i + grid_per_ch * 1) + 2] * (w00_yd * grid00_y + w10_yd * grid10_y + w01_yd * grid01_y + w11_yd * grid11_y));
			atomicAdd(image_grad + index + height * width, grad_d * 1 / binsize);
			// b
			grid00_x = grid[id00_xb + shift * (3 * (i + grid_per_ch * 2) + 1)];
			grid10_x = grid[id10_xb + shift * (3 * (i + grid_per_ch * 2) + 1)];
			grid01_x = grid[id01_xb + shift * (3 * (i + grid_per_ch * 2) + 1)];
			grid11_x = grid[id11_xb + shift * (3 * (i + grid_per_ch * 2) + 1)];
			
            grid00_y = grid[id00_yb + shift * (3 * (i + grid_per_ch * 2) + 2)];
			grid10_y = grid[id10_yb + shift * (3 * (i + grid_per_ch * 2) + 2)];
			grid01_y = grid[id01_yb + shift * (3 * (i + grid_per_ch * 2) + 2)];
			grid11_y = grid[id11_yb + shift * (3 * (i + grid_per_ch * 2) + 2)];
			grad_d = grad_o_b *
					(weights[3*(i + grid_per_ch * 2) + 1] * (w00_xd * grid00_x + w10_xd * grid10_x + w01_xd * grid01_x + w11_xd * grid11_x) +
					 weights[3*(i + grid_per_ch * 2) + 2] * (w00_yd * grid00_y + w10_yd * grid10_y + w01_yd * grid01_y + w11_yd * grid11_y));
			atomicAdd(image_grad + index + height * width * 2, grad_d * 1 / binsize);
            
            scalar_t out_xy = (w00_xy * grid[id00_xy + shift * (3 * (i + grid_per_ch * 0) + 0)] + 
                               w10_xy * grid[id10_xy + shift * (3 * (i + grid_per_ch * 0) + 0)] +
                               w01_xy * grid[id01_xy + shift * (3 * (i + grid_per_ch * 0) + 0)] + 
                               w11_xy * grid[id11_xy + shift * (3 * (i + grid_per_ch * 0) + 0)]);

            scalar_t out_x  = (w00_xr * grid[id00_xr + shift * (3 * (i + grid_per_ch * 0) + 1)] + 
                               w10_xr * grid[id10_xr + shift * (3 * (i + grid_per_ch * 0) + 1)] +
                               w01_xr * grid[id01_xr + shift * (3 * (i + grid_per_ch * 0) + 1)] + 
                               w11_xr * grid[id11_xr + shift * (3 * (i + grid_per_ch * 0) + 1)]);

            scalar_t out_y  = (w00_yr * grid[id00_yr + shift * (3 * (i + grid_per_ch * 0) + 2)] + 
                               w10_yr * grid[id10_yr + shift * (3 * (i + grid_per_ch * 0) + 2)] +
                               w01_yr * grid[id01_yr + shift * (3 * (i + grid_per_ch * 0) + 2)] + 
                               w11_yr * grid[id11_yr + shift * (3 * (i + grid_per_ch * 0) + 2)]);
            
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 0)    , grad_o_r * out_xy);
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 0) + 1, grad_o_r * out_x);
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 0) + 2, grad_o_r * out_y);

            atomicAdd(bias_grad + (i + grid_per_ch * 0), grad_o_r);

            out_xy = (w00_xy * grid[id00_xy + shift * (3 * (i + grid_per_ch * 1) + 0)] + 
                      w10_xy * grid[id10_xy + shift * (3 * (i + grid_per_ch * 1) + 0)] +
                      w01_xy * grid[id01_xy + shift * (3 * (i + grid_per_ch * 1) + 0)] + 
                      w11_xy * grid[id11_xy + shift * (3 * (i + grid_per_ch * 1) + 0)]);

            out_x  = (w00_xg * grid[id00_xg + shift * (3 * (i + grid_per_ch * 1) + 1)] + 
                      w10_xg * grid[id10_xg + shift * (3 * (i + grid_per_ch * 1) + 1)] +
                      w01_xg * grid[id01_xg + shift * (3 * (i + grid_per_ch * 1) + 1)] + 
                      w11_xg * grid[id11_xg + shift * (3 * (i + grid_per_ch * 1) + 1)]);

            out_y  = (w00_yg * grid[id00_yg + shift * (3 * (i + grid_per_ch * 1) + 2)] + 
                      w10_yg * grid[id10_yg + shift * (3 * (i + grid_per_ch * 1) + 2)] +
                      w01_yg * grid[id01_yg + shift * (3 * (i + grid_per_ch * 1) + 2)] + 
                      w11_yg * grid[id11_yg + shift * (3 * (i + grid_per_ch * 1) + 2)]);
            
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 1)    , grad_o_g * out_xy);
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 1) + 1, grad_o_g * out_x);
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 1) + 2, grad_o_g * out_y);

            atomicAdd(bias_grad + (i + grid_per_ch * 1), grad_o_g);


            out_xy = (w00_xy * grid[id00_xy + shift * (3 * (i + grid_per_ch * 2) + 0)] + 
                      w10_xy * grid[id10_xy + shift * (3 * (i + grid_per_ch * 2) + 0)] +
                      w01_xy * grid[id01_xy + shift * (3 * (i + grid_per_ch * 2) + 0)] + 
                      w11_xy * grid[id11_xy + shift * (3 * (i + grid_per_ch * 2) + 0)]);

            out_x  = (w00_xb * grid[id00_xb + shift * (3 * (i + grid_per_ch * 2) + 1)] + 
                      w10_xb * grid[id10_xb + shift * (3 * (i + grid_per_ch * 2) + 1)] +
                      w01_xb * grid[id01_xb + shift * (3 * (i + grid_per_ch * 2) + 1)] + 
                      w11_xb * grid[id11_xb + shift * (3 * (i + grid_per_ch * 2) + 1)]);

            out_y  = (w00_yb * grid[id00_yb + shift * (3 * (i + grid_per_ch * 2) + 2)] + 
                      w10_yb * grid[id10_yb + shift * (3 * (i + grid_per_ch * 2) + 2)] +
                      w01_yb * grid[id01_yb + shift * (3 * (i + grid_per_ch * 2) + 2)] + 
                      w11_yb * grid[id11_yb + shift * (3 * (i + grid_per_ch * 2) + 2)]);
            
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 2)    , grad_o_b * out_xy);
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 2) + 1, grad_o_b * out_x);
            atomicAdd(wights_grad + 3*(i + grid_per_ch * 2) + 2, grad_o_b * out_y);

            atomicAdd(bias_grad + (i + grid_per_ch * 2), grad_o_b);

		}
    }
    }

void TriLinear2DSliceBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &grid, const torch::Tensor &input,
    const torch::Tensor &weights, const torch::Tensor &bias, 
    torch::Tensor grad_grid, torch::Tensor grad_image,
    torch::Tensor grad_weights, torch::Tensor grad_bias) {
    
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = input.size(1);
	int grid_channels = grid.size(1);
    int dim   = grid.size(3);
    int shift   = dim * dim;
	
	int grid_per_ch = grid_channels / num_channels;
   
    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cuda_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_grid_ = grad_grid[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                const scalar_t *data_weights = weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_bias = bias[elt].data_ptr<scalar_t>();
                scalar_t *grad_weights_  = grad_weights[elt].data_ptr<scalar_t>();
                scalar_t *grad_bias_ = grad_bias[elt].data_ptr<scalar_t>();

                TriLinear2DSliceBackward<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_grid, data_image, data_weights, data_bias,
                    grad_grid_, grad_image_, grad_weights_, grad_bias_,
                    dim, shift, binsize,
                    width, height, num_channels, grid_per_ch);
            }));
        AT_CUDA_CHECK(cudaGetLastError());
    }
}
