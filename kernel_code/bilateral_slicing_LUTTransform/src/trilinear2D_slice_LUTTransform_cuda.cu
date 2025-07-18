#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N)
{
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
    return min(optimal_block_num, max_block_num);
}

/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline __device__ constexpr const scalar_t &clamp(
    const scalar_t &v, const scalar_t &lo, const scalar_t &hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
    __global__ void TriLinear2DSliceAndLUTTransformForward(const int nthreads,
                                                           const scalar_t *__restrict__ grid,
                                                           const scalar_t *__restrict__ image,
                                                           const scalar_t *__restrict__ grid_weights,
                                                           const scalar_t *__restrict__ grid_bias,
                                                           const scalar_t *__restrict__ lut,
                                                           const scalar_t *__restrict__ lut_weights,
                                                           const scalar_t *__restrict__ lut_bias,
                                                           scalar_t *__restrict__ output,
                                                           const int grid_dim,
                                                           const int grid_shift,
                                                           const scalar_t grid_binsize,
                                                           const int lut_dim,
                                                           const int lut_shift,
                                                           const scalar_t lut_binsize,
                                                           const int width,
                                                           const int height,
                                                           const int num_channels,
                                                           const int grid_per_ch)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        const int x_ = index % width;
        const int y_ = index / width;

        const scalar_t x = x_ / (width - 1);
        const scalar_t y = y_ / (height - 1);

        const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];

        const int32_t x_id = clamp((int32_t)floor(x * (grid_dim - 1)), 0, grid_dim - 2);
        const int32_t y_id = clamp((int32_t)floor(y * (grid_dim - 1)), 0, grid_dim - 2);

        int32_t r_id = clamp((int32_t)floor(r * (grid_dim - 1)), 0, grid_dim - 2);
        int32_t g_id = clamp((int32_t)floor(g * (grid_dim - 1)), 0, grid_dim - 2);
        int32_t b_id = clamp((int32_t)floor(b * (grid_dim - 1)), 0, grid_dim - 2);

        const scalar_t x_d = (x - grid_binsize * x_id) / grid_binsize;
        const scalar_t y_d = (y - grid_binsize * y_id) / grid_binsize;

        scalar_t r_d = (r - grid_binsize * r_id) / grid_binsize;
        scalar_t g_d = (g - grid_binsize * g_id) / grid_binsize;
        scalar_t b_d = (b - grid_binsize * b_id) / grid_binsize;

        const int id00_xy = (x_id) + (y_id)*grid_dim;
        const int id10_xy = (x_id + 1) + (y_id)*grid_dim;
        const int id01_xy = (x_id) + (y_id + 1) * grid_dim;
        const int id11_xy = (x_id + 1) + (y_id + 1) * grid_dim;

        const int id00_xr = (x_id) + (r_id)*grid_dim;
        const int id10_xr = (x_id + 1) + (r_id)*grid_dim;
        const int id01_xr = (x_id) + (r_id + 1) * grid_dim;
        const int id11_xr = (x_id + 1) + (r_id + 1) * grid_dim;

        const int id00_yr = (y_id) + (r_id)*grid_dim;
        const int id10_yr = (y_id + 1) + (r_id)*grid_dim;
        const int id01_yr = (y_id) + (r_id + 1) * grid_dim;
        const int id11_yr = (y_id + 1) + (r_id + 1) * grid_dim;

        const int id00_xg = (x_id) + (g_id)*grid_dim;
        const int id10_xg = (x_id + 1) + (g_id)*grid_dim;
        const int id01_xg = (x_id) + (g_id + 1) * grid_dim;
        const int id11_xg = (x_id + 1) + (g_id + 1) * grid_dim;

        const int id00_yg = (y_id) + (g_id)*grid_dim;
        const int id10_yg = (y_id + 1) + (g_id)*grid_dim;
        const int id01_yg = (y_id) + (g_id + 1) * grid_dim;
        const int id11_yg = (y_id + 1) + (g_id + 1) * grid_dim;

        const int id00_xb = (x_id) + (b_id)*grid_dim;
        const int id10_xb = (x_id + 1) + (b_id)*grid_dim;
        const int id01_xb = (x_id) + (b_id + 1) * grid_dim;
        const int id11_xb = (x_id + 1) + (b_id + 1) * grid_dim;

        const int id00_yb = (y_id) + (b_id)*grid_dim;
        const int id10_yb = (y_id + 1) + (b_id)*grid_dim;
        const int id01_yb = (y_id) + (b_id + 1) * grid_dim;
        const int id11_yb = (y_id + 1) + (b_id + 1) * grid_dim;

        const scalar_t w00_xy = (1 - x_d) * (1 - y_d);
        const scalar_t w10_xy = (x_d) * (1 - y_d);
        const scalar_t w01_xy = (1 - x_d) * (y_d);
        const scalar_t w11_xy = (x_d) * (y_d);

        const scalar_t w00_xr = (1 - x_d) * (1 - r_d);
        const scalar_t w10_xr = (x_d) * (1 - r_d);
        const scalar_t w01_xr = (1 - x_d) * (r_d);
        const scalar_t w11_xr = (x_d) * (r_d);

        const scalar_t w00_yr = (1 - y_d) * (1 - r_d);
        const scalar_t w10_yr = (y_d) * (1 - r_d);
        const scalar_t w01_yr = (1 - y_d) * (r_d);
        const scalar_t w11_yr = (y_d) * (r_d);

        const scalar_t w00_xg = (1 - x_d) * (1 - g_d);
        const scalar_t w10_xg = (x_d) * (1 - g_d);
        const scalar_t w01_xg = (1 - x_d) * (g_d);
        const scalar_t w11_xg = (x_d) * (g_d);

        const scalar_t w00_yg = (1 - y_d) * (1 - g_d);
        const scalar_t w10_yg = (y_d) * (1 - g_d);
        const scalar_t w01_yg = (1 - y_d) * (g_d);
        const scalar_t w11_yg = (y_d) * (g_d);

        const scalar_t w00_xb = (1 - x_d) * (1 - b_d);
        const scalar_t w10_xb = (x_d) * (1 - b_d);
        const scalar_t w01_xb = (1 - x_d) * (b_d);
        const scalar_t w11_xb = (x_d) * (b_d);

        const scalar_t w00_yb = (1 - y_d) * (1 - b_d);
        const scalar_t w10_yb = (y_d) * (1 - b_d);
        const scalar_t w01_yb = (1 - y_d) * (b_d);
        const scalar_t w11_yb = (y_d) * (b_d);

        scalar_t int_img[3] = {
            0,
        };

        for (int i = 0; i < grid_per_ch; ++i)
        {
            int_img[0] = int_img[0] + grid_weights[3 * (i + grid_per_ch * 0)] * (w00_xy * grid[id00_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)] + w10_xy * grid[id10_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)] + w01_xy * grid[id01_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)] + w11_xy * grid[id11_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)]) +

                         grid_weights[3 * (i + grid_per_ch * 0) + 1] * (w00_xr * grid[id00_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)] +
                                                                        w10_xr * grid[id10_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)] +
                                                                        w01_xr * grid[id01_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)] +
                                                                        w11_xr * grid[id11_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)]) +

                         grid_weights[3 * (i + grid_per_ch * 0) + 2] * (w00_yr * grid[id00_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)] +
                                                                        w10_yr * grid[id10_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)] +
                                                                        w01_yr * grid[id01_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)] +
                                                                        w11_yr * grid[id11_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)]) +
                         grid_bias[(i + grid_per_ch * 0)];

            int_img[1] = int_img[1] + grid_weights[3 * (i + grid_per_ch * 1)] * (w00_xy * grid[id00_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)] + w10_xy * grid[id10_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)] + w01_xy * grid[id01_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)] + w11_xy * grid[id11_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)]) +

                         grid_weights[3 * (i + grid_per_ch * 1) + 1] * (w00_xg * grid[id00_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)] +
                                                                        w10_xg * grid[id10_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)] +
                                                                        w01_xg * grid[id01_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)] +
                                                                        w11_xg * grid[id11_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)]) +

                         grid_weights[3 * (i + grid_per_ch * 1) + 2] * (w00_yg * grid[id00_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)] +
                                                                        w10_yg * grid[id10_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)] +
                                                                        w01_yg * grid[id01_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)] +
                                                                        w11_yg * grid[id11_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)]) +
                         grid_bias[(i + grid_per_ch * 1)];

            int_img[2] = int_img[2] + grid_weights[3 * (i + grid_per_ch * 2)] * (w00_xy * grid[id00_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)] + w10_xy * grid[id10_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)] + w01_xy * grid[id01_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)] + w11_xy * grid[id11_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)]) +

                         grid_weights[3 * (i + grid_per_ch * 2) + 1] * (w00_xb * grid[id00_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)] +
                                                                        w10_xb * grid[id10_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)] +
                                                                        w01_xb * grid[id01_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)] +
                                                                        w11_xb * grid[id11_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)]) +

                         grid_weights[3 * (i + grid_per_ch * 2) + 2] * (w00_yb * grid[id00_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)] +
                                                                        w10_yb * grid[id10_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)] +
                                                                        w01_yb * grid[id01_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)] +
                                                                        w11_yb * grid[id11_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)]) +
                         grid_bias[(i + grid_per_ch * 2)];
        }

        r_id = clamp((int32_t)floor(r * (lut_dim - 1)), 0, lut_dim - 2);
        g_id = clamp((int32_t)floor(g * (lut_dim - 1)), 0, lut_dim - 2);
        b_id = clamp((int32_t)floor(b * (lut_dim - 1)), 0, lut_dim - 2);

        r_d = (r - lut_binsize * r_id) / lut_binsize;
        g_d = (g - lut_binsize * g_id) / lut_binsize;
        b_d = (b - lut_binsize * b_id) / lut_binsize;

        const int id00_rg = r_id + g_id * lut_dim;
        const int id10_rg = r_id + 1 + g_id * lut_dim;
        const int id01_rg = r_id + (g_id + 1) * lut_dim;
        const int id11_rg = r_id + 1 + (g_id + 1) * lut_dim;

        const int id00_rb = r_id + b_id * lut_dim;
        const int id10_rb = r_id + 1 + b_id * lut_dim;
        const int id01_rb = r_id + (b_id + 1) * lut_dim;
        const int id11_rb = r_id + 1 + (b_id + 1) * lut_dim;

        const int id00_gb = g_id + b_id * lut_dim;
        const int id10_gb = g_id + 1 + b_id * lut_dim;
        const int id01_gb = g_id + (b_id + 1) * lut_dim;
        const int id11_gb = g_id + 1 + (b_id + 1) * lut_dim;

        const scalar_t w00_rg = (1 - r_d) * (1 - g_d);
        const scalar_t w10_rg = (r_d) * (1 - g_d);
        const scalar_t w01_rg = (1 - r_d) * (g_d);
        const scalar_t w11_rg = (r_d) * (g_d);

        const scalar_t w00_rb = (1 - r_d) * (1 - b_d);
        const scalar_t w10_rb = (r_d) * (1 - b_d);
        const scalar_t w01_rb = (1 - r_d) * (b_d);
        const scalar_t w11_rb = (r_d) * (b_d);

        const scalar_t w00_gb = (1 - g_d) * (1 - b_d);
        const scalar_t w10_gb = (g_d) * (1 - b_d);
        const scalar_t w01_gb = (1 - g_d) * (b_d);
        const scalar_t w11_gb = (g_d) * (b_d);

        for (int i = 0; i < num_channels; ++i)
        {
            scalar_t output_rg = w00_rg * lut[id00_rg + lut_shift * 3 * i] + w10_rg * lut[id10_rg + lut_shift * 3 * i] +
                                 w01_rg * lut[id01_rg + lut_shift * 3 * i] + w11_rg * lut[id11_rg + lut_shift * 3 * i];

            scalar_t output_rb = w00_rb * lut[id00_rb + lut_shift * (3 * i + 1)] + w10_rb * lut[id10_rb + lut_shift * (3 * i + 1)] +
                                 w01_rb * lut[id01_rb + lut_shift * (3 * i + 1)] + w11_rb * lut[id11_rb + lut_shift * (3 * i + 1)];

            scalar_t output_gb = w00_gb * lut[id00_gb + lut_shift * (3 * i + 2)] + w10_gb * lut[id10_gb + lut_shift * (3 * i + 2)] +
                                 w01_gb * lut[id01_gb + lut_shift * (3 * i + 2)] + w11_gb * lut[id11_gb + lut_shift * (3 * i + 2)];

            output[index + width * height * i] = int_img[i] + lut_weights[3 * i] * output_rg + lut_weights[3 * i + 1] * output_rb + lut_weights[3 * i + 2] * output_gb + lut_bias[i];
        }
    }
}

void TriLinear2DSliceAndLUTTransformForwardLaucher(const torch::Tensor &grid, const torch::Tensor &input, const torch::Tensor &grid_weights, const torch::Tensor &grid_bias,
                                                   const torch::Tensor &lut, const torch::Tensor &lut_weights, const torch::Tensor &lut_bias, torch::Tensor output)
{
    c10::cuda::CUDAGuard device_guard(input.device());

    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);

    int num_channels = input.size(1);
    int grid_channels = grid.size(1);

    int grid_dim = grid.size(3);
    int grid_shift = grid_dim * grid_dim;

    int grid_per_ch = grid_channels / num_channels;

    int num_kernels = height * width;

    int lut_dim = lut.size(3);
    int lut_shift = lut_dim * lut_dim;

    for (int elt = 0; elt < batch_size; ++elt)
    {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cuda_forward", ([&]
                                                            {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t grid_binsize = 1.0 / (grid_dim - 1);;

                const scalar_t *data_grid_weights = grid_weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid_bias = grid_bias[elt].data_ptr<scalar_t>();

                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut_weights = lut_weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut_bias = lut_bias[elt].data_ptr<scalar_t>();
                scalar_t lut_binsize = 1.0 / (lut_dim - 1);

                TriLinear2DSliceAndLUTTransformForward<<<GET_BLOCKS(num_kernels),
                                              THREADS_PER_BLOCK, 0,
                                              at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_grid, data_image, data_grid_weights, data_grid_bias, data_lut, data_lut_weights, data_lut_bias, data_output,
                    grid_dim, grid_shift, grid_binsize, lut_dim, lut_shift, lut_binsize,
                    width, height, num_channels, grid_per_ch); }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
    __global__ void TriLinear2DSliceAndLUTTransformBackward(const int nthreads,
                                                            const scalar_t *__restrict__ output_grad,
                                                            const scalar_t *__restrict__ grid,
                                                            const scalar_t *__restrict__ image,
                                                            const scalar_t *__restrict__ grid_weights,
                                                            const scalar_t *__restrict__ grid_bias,
                                                            const scalar_t *__restrict__ lut,
                                                            const scalar_t *__restrict__ lut_weights,
                                                            const scalar_t *__restrict__ lut_bias,
                                                            scalar_t *__restrict__ grid_grad,
                                                            scalar_t *__restrict__ image_grad,
                                                            scalar_t *__restrict__ grid_wights_grad,
                                                            scalar_t *__restrict__ grid_bias_grad,
                                                            scalar_t *__restrict__ lut_grad,
                                                            scalar_t *__restrict__ lut_weights_grad,
                                                            scalar_t *__restrict__ lut_bias_grad,
                                                            const int grid_dim,
                                                            const int grid_shift,
                                                            const scalar_t grid_binsize,
                                                            const int lut_dim,
                                                            const int lut_shift,
                                                            const scalar_t lut_binsize,
                                                            const int width,
                                                            const int height,
                                                            const int num_channels,
                                                            const int grid_per_ch)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        const int x_ = index % width;
        const int y_ = index / width;

        const scalar_t x = x_ / (width - 1);
        const scalar_t y = y_ / (height - 1);

        const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];

        const int32_t x_id = clamp((int32_t)floor(x * (grid_dim - 1)), 0, grid_dim - 2);
        const int32_t y_id = clamp((int32_t)floor(y * (grid_dim - 1)), 0, grid_dim - 2);

        int32_t r_id = clamp((int32_t)floor(r * (grid_dim - 1)), 0, grid_dim - 2);
        int32_t g_id = clamp((int32_t)floor(g * (grid_dim - 1)), 0, grid_dim - 2);
        int32_t b_id = clamp((int32_t)floor(b * (grid_dim - 1)), 0, grid_dim - 2);

        const scalar_t x_d = (x - grid_binsize * x_id) / grid_binsize;
        const scalar_t y_d = (y - grid_binsize * y_id) / grid_binsize;

        scalar_t r_d = (r - grid_binsize * r_id) / grid_binsize;
        scalar_t g_d = (g - grid_binsize * g_id) / grid_binsize;
        scalar_t b_d = (b - grid_binsize * b_id) / grid_binsize;

        const int id00_xy = (x_id) + (y_id)*grid_dim;
        const int id10_xy = (x_id + 1) + (y_id)*grid_dim;
        const int id01_xy = (x_id) + (y_id + 1) * grid_dim;
        const int id11_xy = (x_id + 1) + (y_id + 1) * grid_dim;

        const int id00_xr = (x_id) + (r_id)*grid_dim;
        const int id10_xr = (x_id + 1) + (r_id)*grid_dim;
        const int id01_xr = (x_id) + (r_id + 1) * grid_dim;
        const int id11_xr = (x_id + 1) + (r_id + 1) * grid_dim;

        const int id00_yr = (y_id) + (r_id)*grid_dim;
        const int id10_yr = (y_id + 1) + (r_id)*grid_dim;
        const int id01_yr = (y_id) + (r_id + 1) * grid_dim;
        const int id11_yr = (y_id + 1) + (r_id + 1) * grid_dim;

        const int id00_xg = (x_id) + (g_id)*grid_dim;
        const int id10_xg = (x_id + 1) + (g_id)*grid_dim;
        const int id01_xg = (x_id) + (g_id + 1) * grid_dim;
        const int id11_xg = (x_id + 1) + (g_id + 1) * grid_dim;

        const int id00_yg = (y_id) + (g_id)*grid_dim;
        const int id10_yg = (y_id + 1) + (g_id)*grid_dim;
        const int id01_yg = (y_id) + (g_id + 1) * grid_dim;
        const int id11_yg = (y_id + 1) + (g_id + 1) * grid_dim;

        const int id00_xb = (x_id) + (b_id)*grid_dim;
        const int id10_xb = (x_id + 1) + (b_id)*grid_dim;
        const int id01_xb = (x_id) + (b_id + 1) * grid_dim;
        const int id11_xb = (x_id + 1) + (b_id + 1) * grid_dim;

        const int id00_yb = (y_id) + (b_id)*grid_dim;
        const int id10_yb = (y_id + 1) + (b_id)*grid_dim;
        const int id01_yb = (y_id) + (b_id + 1) * grid_dim;
        const int id11_yb = (y_id + 1) + (b_id + 1) * grid_dim;

        const scalar_t w00_xy = (1 - x_d) * (1 - y_d);
        const scalar_t w10_xy = (x_d) * (1 - y_d);
        const scalar_t w01_xy = (1 - x_d) * (y_d);
        const scalar_t w11_xy = (x_d) * (y_d);

        const scalar_t w00_xr = (1 - x_d) * (1 - r_d);
        const scalar_t w10_xr = (x_d) * (1 - r_d);
        const scalar_t w01_xr = (1 - x_d) * (r_d);
        const scalar_t w11_xr = (x_d) * (r_d);

        const scalar_t w00_yr = (1 - y_d) * (1 - r_d);
        const scalar_t w10_yr = (y_d) * (1 - r_d);
        const scalar_t w01_yr = (1 - y_d) * (r_d);
        const scalar_t w11_yr = (y_d) * (r_d);

        const scalar_t w00_xg = (1 - x_d) * (1 - g_d);
        const scalar_t w10_xg = (x_d) * (1 - g_d);
        const scalar_t w01_xg = (1 - x_d) * (g_d);
        const scalar_t w11_xg = (x_d) * (g_d);

        const scalar_t w00_yg = (1 - y_d) * (1 - g_d);
        const scalar_t w10_yg = (y_d) * (1 - g_d);
        const scalar_t w01_yg = (1 - y_d) * (g_d);
        const scalar_t w11_yg = (y_d) * (g_d);

        const scalar_t w00_xb = (1 - x_d) * (1 - b_d);
        const scalar_t w10_xb = (x_d) * (1 - b_d);
        const scalar_t w01_xb = (1 - x_d) * (b_d);
        const scalar_t w11_xb = (x_d) * (b_d);

        const scalar_t w00_yb = (1 - y_d) * (1 - b_d);
        const scalar_t w10_yb = (y_d) * (1 - b_d);
        const scalar_t w01_yb = (1 - y_d) * (b_d);
        const scalar_t w11_yb = (y_d) * (b_d);

        /* derivatives: w to rd, gd, bd */
        const scalar_t w00_xd = -(1 - x_d);
        const scalar_t w10_xd = -(x_d);
        const scalar_t w01_xd = (1 - x_d);
        const scalar_t w11_xd = (x_d);

        const scalar_t w00_yd = -(1 - y_d);
        const scalar_t w10_yd = -(y_d);
        const scalar_t w01_yd = (1 - y_d);
        const scalar_t w11_yd = (y_d);

        scalar_t grad_o[3] = {output_grad[index], output_grad[index + width * height * 1], output_grad[index + width * height * 2]};

        for (int i = 0; i < grid_per_ch; ++i)
        {

            atomicAdd(grid_grad + id00_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 0] * w00_xy);
            atomicAdd(grid_grad + id10_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 0] * w10_xy);
            atomicAdd(grid_grad + id01_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 0] * w01_xy);
            atomicAdd(grid_grad + id11_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 0] * w11_xy);

            atomicAdd(grid_grad + id00_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 1] * w00_xr);
            atomicAdd(grid_grad + id10_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 1] * w10_xr);
            atomicAdd(grid_grad + id01_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 1] * w01_xr);
            atomicAdd(grid_grad + id11_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 1] * w11_xr);

            atomicAdd(grid_grad + id00_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 2] * w00_yr);
            atomicAdd(grid_grad + id10_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 2] * w10_yr);
            atomicAdd(grid_grad + id01_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 2] * w01_yr);
            atomicAdd(grid_grad + id11_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2), grad_o[0] * grid_weights[3 * (i + grid_per_ch * 0) + 2] * w11_yr);

            atomicAdd(grid_grad + id00_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 0] * w00_xy);
            atomicAdd(grid_grad + id10_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 0] * w10_xy);
            atomicAdd(grid_grad + id01_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 0] * w01_xy);
            atomicAdd(grid_grad + id11_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 0] * w11_xy);

            atomicAdd(grid_grad + id00_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 1] * w00_xg);
            atomicAdd(grid_grad + id10_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 1] * w10_xg);
            atomicAdd(grid_grad + id01_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 1] * w01_xg);
            atomicAdd(grid_grad + id11_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 1] * w11_xg);

            atomicAdd(grid_grad + id00_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 2] * w00_yg);
            atomicAdd(grid_grad + id10_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 2] * w10_yg);
            atomicAdd(grid_grad + id01_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 2] * w01_yg);
            atomicAdd(grid_grad + id11_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2), grad_o[1] * grid_weights[3 * (i + grid_per_ch * 1) + 2] * w11_yg);

            atomicAdd(grid_grad + id00_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 0] * w00_xy);
            atomicAdd(grid_grad + id10_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 0] * w10_xy);
            atomicAdd(grid_grad + id01_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 0] * w01_xy);
            atomicAdd(grid_grad + id11_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 0] * w11_xy);

            atomicAdd(grid_grad + id00_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 1] * w00_xb);
            atomicAdd(grid_grad + id10_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 1] * w10_xb);
            atomicAdd(grid_grad + id01_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 1] * w01_xb);
            atomicAdd(grid_grad + id11_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 1] * w11_xb);

            atomicAdd(grid_grad + id00_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 2] * w00_yb);
            atomicAdd(grid_grad + id10_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 2] * w10_yb);
            atomicAdd(grid_grad + id01_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 2] * w01_yb);
            atomicAdd(grid_grad + id11_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2), grad_o[2] * grid_weights[3 * (i + grid_per_ch * 2) + 2] * w11_yb);

            // r
            scalar_t grad_d = 0;
            scalar_t grid00_x = grid[id00_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)];
            scalar_t grid10_x = grid[id10_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)];
            scalar_t grid01_x = grid[id01_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)];
            scalar_t grid11_x = grid[id11_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)];

            scalar_t grid00_y = grid[id00_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)];
            scalar_t grid10_y = grid[id10_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)];
            scalar_t grid01_y = grid[id01_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)];
            scalar_t grid11_y = grid[id11_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)];
            grad_d = grad_o[0] *
                     (grid_weights[3 * (i + grid_per_ch * 0) + 1] * (w00_xd * grid00_x + w10_xd * grid10_x + w01_xd * grid01_x + w11_xd * grid11_x) +
                      grid_weights[3 * (i + grid_per_ch * 0) + 2] * (w00_yd * grid00_y + w10_yd * grid10_y + w01_yd * grid01_y + w11_yd * grid11_y));
            atomicAdd(image_grad + index, grad_d * 1 / grid_binsize);

            // g
            grid00_x = grid[id00_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)];
            grid10_x = grid[id10_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)];
            grid01_x = grid[id01_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)];
            grid11_x = grid[id11_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)];

            grid00_y = grid[id00_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)];
            grid10_y = grid[id10_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)];
            grid01_y = grid[id01_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)];
            grid11_y = grid[id11_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)];
            grad_d = grad_o[1] *
                     (grid_weights[3 * (i + grid_per_ch * 1) + 1] * (w00_xd * grid00_x + w10_xd * grid10_x + w01_xd * grid01_x + w11_xd * grid11_x) +
                      grid_weights[3 * (i + grid_per_ch * 1) + 2] * (w00_yd * grid00_y + w10_yd * grid10_y + w01_yd * grid01_y + w11_yd * grid11_y));
            atomicAdd(image_grad + index + height * width, grad_d * 1 / grid_binsize);

            // b
            grid00_x = grid[id00_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)];
            grid10_x = grid[id10_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)];
            grid01_x = grid[id01_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)];
            grid11_x = grid[id11_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)];

            grid00_y = grid[id00_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)];
            grid10_y = grid[id10_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)];
            grid01_y = grid[id01_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)];
            grid11_y = grid[id11_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)];
            grad_d = grad_o[2] *
                     (grid_weights[3 * (i + grid_per_ch * 2) + 1] * (w00_xd * grid00_x + w10_xd * grid10_x + w01_xd * grid01_x + w11_xd * grid11_x) +
                      grid_weights[3 * (i + grid_per_ch * 2) + 2] * (w00_yd * grid00_y + w10_yd * grid10_y + w01_yd * grid01_y + w11_yd * grid11_y));
            atomicAdd(image_grad + index + height * width * 2, grad_d * 1 / grid_binsize);

            scalar_t out_xy = (w00_xy * grid[id00_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)] +
                               w10_xy * grid[id10_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)] +
                               w01_xy * grid[id01_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)] +
                               w11_xy * grid[id11_xy + grid_shift * (3 * (i + grid_per_ch * 0) + 0)]);

            scalar_t out_x = (w00_xr * grid[id00_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)] +
                              w10_xr * grid[id10_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)] +
                              w01_xr * grid[id01_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)] +
                              w11_xr * grid[id11_xr + grid_shift * (3 * (i + grid_per_ch * 0) + 1)]);

            scalar_t out_y = (w00_yr * grid[id00_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)] +
                              w10_yr * grid[id10_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)] +
                              w01_yr * grid[id01_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)] +
                              w11_yr * grid[id11_yr + grid_shift * (3 * (i + grid_per_ch * 0) + 2)]);

            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 0), grad_o[0] * out_xy);
            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 0) + 1, grad_o[0] * out_x);
            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 0) + 2, grad_o[0] * out_y);

            atomicAdd(grid_bias_grad + (i + grid_per_ch * 0), grad_o[0]);

            out_xy = (w00_xy * grid[id00_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)] +
                      w10_xy * grid[id10_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)] +
                      w01_xy * grid[id01_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)] +
                      w11_xy * grid[id11_xy + grid_shift * (3 * (i + grid_per_ch * 1) + 0)]);

            out_x = (w00_xg * grid[id00_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)] +
                     w10_xg * grid[id10_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)] +
                     w01_xg * grid[id01_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)] +
                     w11_xg * grid[id11_xg + grid_shift * (3 * (i + grid_per_ch * 1) + 1)]);

            out_y = (w00_yg * grid[id00_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)] +
                     w10_yg * grid[id10_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)] +
                     w01_yg * grid[id01_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)] +
                     w11_yg * grid[id11_yg + grid_shift * (3 * (i + grid_per_ch * 1) + 2)]);

            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 1), grad_o[1] * out_xy);
            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 1) + 1, grad_o[1] * out_x);
            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 1) + 2, grad_o[1] * out_y);

            atomicAdd(grid_bias_grad + (i + grid_per_ch * 1), grad_o[1]);

            out_xy = (w00_xy * grid[id00_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)] +
                      w10_xy * grid[id10_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)] +
                      w01_xy * grid[id01_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)] +
                      w11_xy * grid[id11_xy + grid_shift * (3 * (i + grid_per_ch * 2) + 0)]);

            out_x = (w00_xb * grid[id00_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)] +
                     w10_xb * grid[id10_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)] +
                     w01_xb * grid[id01_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)] +
                     w11_xb * grid[id11_xb + grid_shift * (3 * (i + grid_per_ch * 2) + 1)]);

            out_y = (w00_yb * grid[id00_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)] +
                     w10_yb * grid[id10_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)] +
                     w01_yb * grid[id01_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)] +
                     w11_yb * grid[id11_yb + grid_shift * (3 * (i + grid_per_ch * 2) + 2)]);

            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 2), grad_o[2] * out_xy);
            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 2) + 1, grad_o[2] * out_x);
            atomicAdd(grid_wights_grad + 3 * (i + grid_per_ch * 2) + 2, grad_o[2] * out_y);

            atomicAdd(grid_bias_grad + (i + grid_per_ch * 2), grad_o[2]);
        }

        r_id = clamp((int32_t)floor(r * (lut_dim - 1)), 0, lut_dim - 2);
        g_id = clamp((int32_t)floor(g * (lut_dim - 1)), 0, lut_dim - 2);
        b_id = clamp((int32_t)floor(b * (lut_dim - 1)), 0, lut_dim - 2);

        r_d = (r - lut_binsize * r_id) / lut_binsize;
        g_d = (g - lut_binsize * g_id) / lut_binsize;
        b_d = (b - lut_binsize * b_id) / lut_binsize;

        const int id00_rg = r_id + g_id * lut_dim;
        const int id10_rg = r_id + 1 + g_id * lut_dim;
        const int id01_rg = r_id + (g_id + 1) * lut_dim;
        const int id11_rg = r_id + 1 + (g_id + 1) * lut_dim;

        const int id00_rb = r_id + b_id * lut_dim;
        const int id10_rb = r_id + 1 + b_id * lut_dim;
        const int id01_rb = r_id + (b_id + 1) * lut_dim;
        const int id11_rb = r_id + 1 + (b_id + 1) * lut_dim;

        const int id00_gb = g_id + b_id * lut_dim;
        const int id10_gb = g_id + 1 + b_id * lut_dim;
        const int id01_gb = g_id + (b_id + 1) * lut_dim;
        const int id11_gb = g_id + 1 + (b_id + 1) * lut_dim;

        const scalar_t w00_rg = (1 - r_d) * (1 - g_d);
        const scalar_t w10_rg = (r_d) * (1 - g_d);
        const scalar_t w01_rg = (1 - r_d) * (g_d);
        const scalar_t w11_rg = (r_d) * (g_d);

        const scalar_t w00_rb = (1 - r_d) * (1 - b_d);
        const scalar_t w10_rb = (r_d) * (1 - b_d);
        const scalar_t w01_rb = (1 - r_d) * (b_d);
        const scalar_t w11_rb = (r_d) * (b_d);

        const scalar_t w00_gb = (1 - g_d) * (1 - b_d);
        const scalar_t w10_gb = (g_d) * (1 - b_d);
        const scalar_t w01_gb = (1 - g_d) * (b_d);
        const scalar_t w11_gb = (g_d) * (b_d);

        /* derivatives: w to rd of rg grid */
        const scalar_t w00_rg_rd = -(1 - g_d);
        const scalar_t w10_rg_rd = (1 - g_d);
        const scalar_t w01_rg_rd = -(g_d);
        const scalar_t w11_rg_rd = (g_d);
        /* derivatives: w to gd of rg grid */
        const scalar_t w00_rg_gd = -(1 - r_d);
        const scalar_t w10_rg_gd = -(r_d);
        const scalar_t w01_rg_gd = (1 - r_d);
        const scalar_t w11_rg_gd = (r_d);

        /* derivatives: w to rd of rb grid */
        const scalar_t w00_rb_rd = -(1 - b_d);
        const scalar_t w10_rb_rd = (1 - b_d);
        const scalar_t w01_rb_rd = -(b_d);
        const scalar_t w11_rb_rd = (b_d);
        /* derivatives: w to bd of rb grid */
        const scalar_t w00_rb_bd = -(1 - r_d);
        const scalar_t w10_rb_bd = -(r_d);
        const scalar_t w01_rb_bd = (1 - r_d);
        const scalar_t w11_rb_bd = (r_d);

        /* derivatives: w to gd of gb grid */
        const scalar_t w00_gb_gd = -(1 - b_d);
        const scalar_t w10_gb_gd = (1 - b_d);
        const scalar_t w01_gb_gd = -(b_d);
        const scalar_t w11_gb_gd = (b_d);
        /* derivatives: w to bd of gb grid */
        const scalar_t w00_gb_bd = -(1 - g_d);
        const scalar_t w10_gb_bd = -(g_d);
        const scalar_t w01_gb_bd = (1 - g_d);
        const scalar_t w11_gb_bd = (g_d);

        for (int i = 0; i < num_channels; ++i)
        {
            /* derivatives: lut grad */
            atomicAdd(lut_grad + id00_rg + lut_shift * 3 * i, grad_o[i] * lut_weights[3 * i] * w00_rg);
            atomicAdd(lut_grad + id10_rg + lut_shift * 3 * i, grad_o[i] * lut_weights[3 * i] * w10_rg);
            atomicAdd(lut_grad + id01_rg + lut_shift * 3 * i, grad_o[i] * lut_weights[3 * i] * w01_rg);
            atomicAdd(lut_grad + id11_rg + lut_shift * 3 * i, grad_o[i] * lut_weights[3 * i] * w11_rg);

            atomicAdd(lut_grad + id00_rb + lut_shift * (3 * i + 1), grad_o[i] * lut_weights[3 * i + 1] * w00_rb);
            atomicAdd(lut_grad + id10_rb + lut_shift * (3 * i + 1), grad_o[i] * lut_weights[3 * i + 1] * w10_rb);
            atomicAdd(lut_grad + id01_rb + lut_shift * (3 * i + 1), grad_o[i] * lut_weights[3 * i + 1] * w01_rb);
            atomicAdd(lut_grad + id11_rb + lut_shift * (3 * i + 1), grad_o[i] * lut_weights[3 * i + 1] * w11_rb);

            atomicAdd(lut_grad + id00_gb + lut_shift * (3 * i + 2), grad_o[i] * lut_weights[3 * i + 2] * w00_gb);
            atomicAdd(lut_grad + id10_gb + lut_shift * (3 * i + 2), grad_o[i] * lut_weights[3 * i + 2] * w10_gb);
            atomicAdd(lut_grad + id01_gb + lut_shift * (3 * i + 2), grad_o[i] * lut_weights[3 * i + 2] * w01_gb);
            atomicAdd(lut_grad + id11_gb + lut_shift * (3 * i + 2), grad_o[i] * lut_weights[3 * i + 2] * w11_gb);

            scalar_t grad_d = 0;
            const scalar_t lut00_rg = lut[id00_rg + lut_shift * 3 * i];
            const scalar_t lut10_rg = lut[id10_rg + lut_shift * 3 * i];
            const scalar_t lut01_rg = lut[id01_rg + lut_shift * 3 * i];
            const scalar_t lut11_rg = lut[id11_rg + lut_shift * 3 * i];

            const scalar_t lut00_rb = lut[id00_rb + lut_shift * (3 * i + 1)];
            const scalar_t lut10_rb = lut[id10_rb + lut_shift * (3 * i + 1)];
            const scalar_t lut01_rb = lut[id01_rb + lut_shift * (3 * i + 1)];
            const scalar_t lut11_rb = lut[id11_rb + lut_shift * (3 * i + 1)];

            const scalar_t lut00_gb = lut[id00_gb + lut_shift * (3 * i + 2)];
            const scalar_t lut10_gb = lut[id10_gb + lut_shift * (3 * i + 2)];
            const scalar_t lut01_gb = lut[id01_gb + lut_shift * (3 * i + 2)];
            const scalar_t lut11_gb = lut[id11_gb + lut_shift * (3 * i + 2)];

            // r grad
            grad_d = grad_o[i] *
                     (lut_weights[3 * i] * (w00_rg_rd * lut00_rg + w10_rg_rd * lut10_rg + w01_rg_rd * lut01_rg + w11_rg_rd * lut11_rg) +
                      lut_weights[3 * i + 1] * (w00_rb_rd * lut00_rb + w10_rb_rd * lut10_rb + w01_rb_rd * lut01_rb + w11_rb_rd * lut11_rb));
            atomicAdd(image_grad + index, grad_d * 1 / lut_binsize);
            // g grad
            grad_d = grad_o[i] *
                     (lut_weights[3 * i] * (w00_rg_gd * lut00_rg + w10_rg_gd * lut10_rg + w01_rg_gd * lut01_rg + w11_rg_gd * lut11_rg) +
                      lut_weights[3 * i + 2] * (w00_gb_gd * lut00_gb + w10_gb_gd * lut10_gb + w01_gb_gd * lut01_gb + w11_gb_gd * lut11_gb));
            atomicAdd(image_grad + index + height * width, grad_d * 1 / lut_binsize);
            // b grad
            grad_d = grad_o[i] *
                     (lut_weights[3 * i + 1] * (w00_rb_bd * lut00_rb + w10_rb_bd * lut10_rb + w01_rb_bd * lut01_rb + w11_rb_bd * lut11_rb) +
                      lut_weights[3 * i + 2] * (w00_gb_bd * lut00_gb + w10_gb_bd * lut10_gb + w01_gb_bd * lut01_gb + w11_gb_bd * lut11_gb));
            atomicAdd(image_grad + index + height * width * 2, grad_d * 1 / lut_binsize);

            scalar_t output_rg = w00_rg * lut[id00_rg + lut_shift * 3 * i] + w10_rg * lut[id10_rg + lut_shift * 3 * i] +
                                 w01_rg * lut[id01_rg + lut_shift * 3 * i] + w11_rg * lut[id11_rg + lut_shift * 3 * i];

            scalar_t output_rb = w00_rb * lut[id00_rb + lut_shift * (3 * i + 1)] + w10_rb * lut[id10_rb + lut_shift * (3 * i + 1)] +
                                 w01_rb * lut[id01_rb + lut_shift * (3 * i + 1)] + w11_rb * lut[id11_rb + lut_shift * (3 * i + 1)];

            scalar_t output_gb = w00_gb * lut[id00_gb + lut_shift * (3 * i + 2)] + w10_gb * lut[id10_gb + lut_shift * (3 * i + 2)] +
                                 w01_gb * lut[id01_gb + lut_shift * (3 * i + 2)] + w11_gb * lut[id11_gb + lut_shift * (3 * i + 2)];
            // weight grad
            atomicAdd(lut_weights_grad + 3 * i, output_rg * grad_o[i]);
            atomicAdd(lut_weights_grad + 3 * i + 1, output_rb * grad_o[i]);
            atomicAdd(lut_weights_grad + 3 * i + 2, output_gb * grad_o[i]);
            // bias grad
            atomicAdd(lut_bias_grad + i, grad_o[i]);
        }
    }
}

void TriLinear2DSliceAndLUTTransformBackwardLaucher(const torch::Tensor &grad_output,
                                                    const torch::Tensor &grid, const torch::Tensor &input,
                                                    const torch::Tensor &grid_weights, const torch::Tensor &grid_bias,
                                                    const torch::Tensor &lut,
                                                    const torch::Tensor &lut_weights, const torch::Tensor &lut_bias,
                                                    torch::Tensor grad_grid, torch::Tensor grad_image,
                                                    torch::Tensor grad_grid_weights, torch::Tensor grad_grid_bias,
                                                    torch::Tensor grad_lut,
                                                    torch::Tensor grad_lut_weights, torch::Tensor grad_lut_bias)
{

    c10::cuda::CUDAGuard device_guard(grad_output.device());

    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);

    int num_channels = input.size(1);
    int grid_channels = grid.size(1);
    int grid_dim = grid.size(3);
    int grid_shift = grid_dim * grid_dim;

    int grid_per_ch = grid_channels / num_channels;

    int num_kernels = height * width;

    int lut_dim = lut.size(3);
    int lut_shift = lut_dim * lut_dim;

    for (int elt = 0; elt < batch_size; ++elt)
    {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cuda_backward", ([&]
                                                             {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid = grid[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_grid_ = grad_grid[elt].data_ptr<scalar_t>();
                scalar_t grid_binsize = 1.0 / (grid_dim - 1);

                const scalar_t *data_grid_weights = grid_weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_grid_bias = grid_bias[elt].data_ptr<scalar_t>();
                scalar_t *grad_grid_weights_  = grad_grid_weights[elt].data_ptr<scalar_t>();
                scalar_t *grad_grid_bias_ = grad_grid_bias[elt].data_ptr<scalar_t>();

                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut_weights = lut_weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut_bias = lut_bias[elt].data_ptr<scalar_t>();
                scalar_t lut_binsize = 1.0 / (lut_dim - 1);
                
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_weight_  = grad_lut_weights[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_bias_ = grad_lut_bias[elt].data_ptr<scalar_t>();


                TriLinear2DSliceAndLUTTransformBackward<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, grad_out, data_grid, data_image, data_grid_weights, data_grid_bias,
                    data_lut, data_lut_weights, data_lut_bias,
                    grad_grid_, grad_image_, grad_grid_weights_, grad_grid_bias_, grad_lut_, grad_lut_weight_, grad_lut_bias_,
                    grid_dim, grid_shift, grid_binsize, lut_dim, lut_shift, lut_binsize,
                    width, height, num_channels, grid_per_ch); }));
        AT_CUDA_CHECK(cudaGetLastError());
    }
}
