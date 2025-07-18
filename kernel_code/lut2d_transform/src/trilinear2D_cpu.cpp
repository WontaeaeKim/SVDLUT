#include <torch/extension.h>

#include <ATen/ATen.h>

template <typename scalar_t>
inline constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template <typename scalar_t>
void TriLinear2DCPUForward(const int nthreads, 
                        const scalar_t*  lut, 
                        const scalar_t*  image,                                 
                        const scalar_t*  weights,
                        const scalar_t*  bias,
                        scalar_t*  output,
                        const int dim, 
                        const int shift, 
                        const scalar_t binsize, 
                        const int width, 
                        const int height, 
                        const int num_channels) {
        for (int index=0;index<nthreads;index++) {

        const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];

        const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
        const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
        const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id00_rg = r_id     + g_id       * dim;
        const int id10_rg = r_id + 1 + g_id       * dim;
        const int id01_rg = r_id     + (g_id + 1) * dim;
        const int id11_rg = r_id + 1 + (g_id + 1) * dim;

        const int id00_rb = r_id     + b_id       * dim;
        const int id10_rb = r_id + 1 + b_id       * dim;
        const int id01_rb = r_id     + (b_id + 1) * dim;
        const int id11_rb = r_id + 1 + (b_id + 1) * dim;

        const int id00_gb = g_id     + b_id       * dim;
        const int id10_gb = g_id + 1 + b_id       * dim;
        const int id01_gb = g_id     + (b_id + 1) * dim;
        const int id11_gb = g_id + 1 + (b_id + 1) * dim;

        const scalar_t  w00_rg = (1-r_d)*(1-g_d);
        const scalar_t  w10_rg = (  r_d)*(1-g_d);
        const scalar_t  w01_rg = (1-r_d)*(  g_d);
        const scalar_t  w11_rg = (  r_d)*(  g_d);

        const scalar_t  w00_rb = (1-r_d)*(1-b_d);
        const scalar_t  w10_rb = (  r_d)*(1-b_d);
        const scalar_t  w01_rb = (1-r_d)*(  b_d);
        const scalar_t  w11_rb = (  r_d)*(  b_d);

        const scalar_t  w00_gb = (1-g_d)*(1-b_d);
        const scalar_t  w10_gb = (  g_d)*(1-b_d);
        const scalar_t  w01_gb = (1-g_d)*(  b_d);
        const scalar_t  w11_gb = (  g_d)*(  b_d);

        
        for(int i =0; i<num_channels;++i){
            scalar_t output_rg = w00_rg * lut[id00_rg + shift * 3 * i] + w10_rg * lut[id10_rg + shift * 3 * i] + 
                                w01_rg * lut[id01_rg + shift * 3 * i] + w11_rg * lut[id11_rg + shift * 3 * i];

            scalar_t output_rb = w00_rb * lut[id00_rb + shift *(3 * i + 1)] + w10_rb * lut[id10_rb + shift * (3 * i + 1)] + 
                                w01_rb * lut[id01_rb + shift * (3 * i + 1)] + w11_rb * lut[id11_rb + shift * (3 * i + 1)];

            scalar_t output_gb = w00_gb * lut[id00_gb + shift * (3 * i + 2)] + w10_gb * lut[id10_gb + shift * (3 * i + 2)] + 
                                w01_gb * lut[id01_gb + shift * (3 * i + 2)] + w11_gb * lut[id11_gb + shift * (3 * i + 2)];
                                                                        
            output[index + width * height * (i%3)] += weights[3 * i] * output_rg + weights[3 * i + 1] * output_rb + weights[3 * i + 2] * output_gb + bias[i];
        }
    }
}

void TriLinear2DCPUForwardLaucher(const torch::Tensor &lut, const torch::Tensor &input, const torch::Tensor &weights, const torch::Tensor &bias, torch::Tensor output) {
    
    
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int dim   = lut.size(3);
    int shift   = dim * dim;
   
    int num_kernels = height * width;
    
    for (int elt = 0; elt < batch_size; ++elt) {

        
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cpu_forward", ([&] {
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_output = output[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                const scalar_t *data_weights = weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_bias = bias[elt].data_ptr<scalar_t>();

                TriLinear2DCPUForward(
                    num_kernels, data_lut, data_image, data_weights, data_bias, data_output, 
                    dim, shift, binsize,
                    width, height, num_channels);
            }));
    }
}

template <typename scalar_t>
void TriLinear2DCPUBackward(const int nthreads,
                                  const scalar_t*   output_grad, 
                                  const scalar_t*  lut, 
                                  const scalar_t*  image,
                                  const scalar_t*  weights,
                                  const scalar_t*  bias,                                                                    
                                  scalar_t*   lut_grad,
                                  scalar_t*   image_grad,
                                  scalar_t*   wights_grad,
                                  scalar_t*   bias_grad, 
                                  const int dim, 
                                  const int shift, 
                                  const scalar_t binsize, 
                                  const int width, 
                                  const int height, 
                                  const int num_channels) {
    for (int index=0;index<nthreads;index++) {
        const scalar_t r = image[index];
        const scalar_t g = image[index + width * height];
        const scalar_t b = image[index + width * height * 2];

        const int32_t r_id = clamp((int32_t)floor(r * (dim-1)),0, dim-2);
        const int32_t g_id = clamp((int32_t)floor(g * (dim-1)),0, dim-2);
        const int32_t b_id = clamp((int32_t)floor(b * (dim-1)),0, dim-2);

        const scalar_t  r_d = (r - binsize * r_id) / binsize;
        const scalar_t  g_d = (g - binsize * g_id) / binsize;
        const scalar_t  b_d = (b - binsize * b_id) / binsize;

        const int id00_rg = r_id     + g_id       * dim;
        const int id10_rg = r_id + 1 + g_id       * dim;
        const int id01_rg = r_id     + (g_id + 1) * dim;
        const int id11_rg = r_id + 1 + (g_id + 1) * dim;

        const int id00_rb = r_id     + b_id       * dim;
        const int id10_rb = r_id + 1 + b_id       * dim;
        const int id01_rb = r_id     + (b_id + 1) * dim;
        const int id11_rb = r_id + 1 + (b_id + 1) * dim;

        const int id00_gb = g_id     + b_id       * dim;
        const int id10_gb = g_id + 1 + b_id       * dim;
        const int id01_gb = g_id     + (b_id + 1) * dim;
        const int id11_gb = g_id + 1 + (b_id + 1) * dim;

        const scalar_t  w00_rg = (1-r_d)*(1-g_d);
        const scalar_t  w10_rg = (  r_d)*(1-g_d);
        const scalar_t  w01_rg = (1-r_d)*(  g_d);
        const scalar_t  w11_rg = (  r_d)*(  g_d);

        const scalar_t  w00_rb = (1-r_d)*(1-b_d);
        const scalar_t  w10_rb = (  r_d)*(1-b_d);
        const scalar_t  w01_rb = (1-r_d)*(  b_d);
        const scalar_t  w11_rb = (  r_d)*(  b_d);

        const scalar_t  w00_gb = (1-g_d)*(1-b_d);
        const scalar_t  w10_gb = (  g_d)*(1-b_d);
        const scalar_t  w01_gb = (1-g_d)*(  b_d);
        const scalar_t  w11_gb = (  g_d)*(  b_d);

        
        /* derivatives: w to rd of rg grid */
        const scalar_t w00_rg_rd = - (1 - g_d);
        const scalar_t w10_rg_rd =   (1 - g_d);
        const scalar_t w01_rg_rd = - (    g_d);
        const scalar_t w11_rg_rd =   (    g_d);
        /* derivatives: w to gd of rg grid */
        const scalar_t w00_rg_gd = - (1 - r_d);
        const scalar_t w10_rg_gd = - (    r_d);
        const scalar_t w01_rg_gd =   (1 - r_d);
        const scalar_t w11_rg_gd =   (    r_d);

        /* derivatives: w to rd of rb grid */
        const scalar_t w00_rb_rd = - (1 - b_d);
        const scalar_t w10_rb_rd =   (1 - b_d);
        const scalar_t w01_rb_rd = - (    b_d);
        const scalar_t w11_rb_rd =   (    b_d);
        /* derivatives: w to bd of rb grid */
        const scalar_t w00_rb_bd = - (1 - r_d);
        const scalar_t w10_rb_bd = - (    r_d);
        const scalar_t w01_rb_bd =   (1 - r_d);
        const scalar_t w11_rb_bd =   (    r_d);

        /* derivatives: w to gd of gb grid */
        const scalar_t w00_gb_gd = - (1 - b_d);
        const scalar_t w10_gb_gd =   (1 - b_d);
        const scalar_t w01_gb_gd = - (    b_d);
        const scalar_t w11_gb_gd =   (    b_d);
        /* derivatives: w to bd of gb grid */
        const scalar_t w00_gb_bd = - (1 - g_d);
        const scalar_t w10_gb_bd = - (    g_d);
        const scalar_t w01_gb_bd =   (1 - g_d);
        const scalar_t w11_gb_bd =   (    g_d);


        for(int i=0;i<num_channels;++i)
        {
            scalar_t grad_o_ = output_grad[index + width * height * (i%3)];
            /* derivatives: lut grad */
            lut_grad[id00_rg + shift * 3 * i] += grad_o_ * weights[3 * i] * w00_rg;
            lut_grad[id10_rg + shift * 3 * i] +=  grad_o_ * weights[3 * i] * w10_rg;
            lut_grad[id01_rg + shift * 3 * i] +=  grad_o_ * weights[3 * i] * w01_rg;
            lut_grad[id11_rg + shift * 3 * i] +=  grad_o_ * weights[3 * i] * w11_rg;

            lut_grad[id00_rb + shift * (3 * i + 1)] +=  grad_o_ * weights[3 * i + 1] * w00_rb;
            lut_grad[id10_rb + shift * (3 * i + 1)] +=  grad_o_ * weights[3 * i + 1] * w10_rb;
            lut_grad[id01_rb + shift * (3 * i + 1)] +=  grad_o_ * weights[3 * i + 1] * w01_rb;
            lut_grad[id11_rb + shift * (3 * i + 1)] +=  grad_o_ * weights[3 * i + 1] * w11_rb;

            lut_grad[id00_gb + shift * (3 * i + 2)] +=  grad_o_ * weights[3 * i + 2] * w00_gb;
            lut_grad[id10_gb + shift * (3 * i + 2)] +=  grad_o_ * weights[3 * i + 2] * w10_gb;
            lut_grad[id01_gb + shift * (3 * i + 2)] +=  grad_o_ * weights[3 * i + 2] * w01_gb;
            lut_grad[id11_gb + shift * (3 * i + 2)] +=  grad_o_ * weights[3 * i + 2] * w11_gb;

           
            scalar_t grad_d = 0;
            const scalar_t lut00_rg = lut[id00_rg + shift * 3 * i];
            const scalar_t lut10_rg = lut[id10_rg + shift * 3 * i];
            const scalar_t lut01_rg = lut[id01_rg + shift * 3 * i];
            const scalar_t lut11_rg = lut[id11_rg + shift * 3 * i];

            const scalar_t lut00_rb = lut[id00_rb + shift * (3 * i + 1)];
            const scalar_t lut10_rb = lut[id10_rb + shift * (3 * i + 1)];
            const scalar_t lut01_rb = lut[id01_rb + shift * (3 * i + 1)];
            const scalar_t lut11_rb = lut[id11_rb + shift * (3 * i + 1)];

            const scalar_t lut00_gb = lut[id00_gb + shift * (3 * i + 2)];
            const scalar_t lut10_gb = lut[id10_gb + shift * (3 * i + 2)];
            const scalar_t lut01_gb = lut[id01_gb + shift * (3 * i + 2)];
            const scalar_t lut11_gb = lut[id11_gb + shift * (3 * i + 2)];

            // r grad
            grad_d = grad_o_ * 
                    (weights[3 * i] * (w00_rg_rd * lut00_rg +  w10_rg_rd * lut10_rg + w01_rg_rd * lut01_rg +  w11_rg_rd * lut11_rg) +
                     weights[3 * i + 1] * (w00_rb_rd * lut00_rb +  w10_rb_rd * lut10_rb + w01_rb_rd * lut01_rb +  w11_rb_rd * lut11_rb));
            image_grad[index] += grad_d * 1 / binsize;
            // g grad
            grad_d = grad_o_ *
                    (weights[3 * i] * (w00_rg_gd * lut00_rg +  w10_rg_gd * lut10_rg + w01_rg_gd * lut01_rg +  w11_rg_gd * lut11_rg) +
                     weights[3 * i +  2] * (w00_gb_gd * lut00_gb +  w10_gb_gd * lut10_gb + w01_gb_gd * lut01_gb +  w11_gb_gd * lut11_gb));
            image_grad[index + height * width] +=  grad_d * 1 / binsize;
            // b grad
            grad_d = grad_o_ *
                    (weights[3 * i + 1] * (w00_rb_bd * lut00_rb +  w10_rb_bd * lut10_rb + w01_rb_bd * lut01_rb +  w11_rb_bd * lut11_rb) +
                     weights[3 * i +  2] * (w00_gb_bd * lut00_gb +  w10_gb_bd * lut10_gb + w01_gb_bd * lut01_gb +  w11_gb_bd * lut11_gb));
            image_grad[index + height * width * 2] +=  grad_d * 1 / binsize;


            scalar_t output_rg = w00_rg * lut[id00_rg + shift * 3 * i] + w10_rg * lut[id10_rg + shift * 3 * i] + 
                                 w01_rg * lut[id01_rg + shift * 3 * i] + w11_rg * lut[id11_rg + shift * 3 * i];

            scalar_t output_rb = w00_rb * lut[id00_rb + shift * (3 * i + 1)] + w10_rb * lut[id10_rb + shift * (3 * i + 1)] + 
                                 w01_rb * lut[id01_rb + shift * (3 * i + 1)] + w11_rb * lut[id11_rb + shift * (3 * i + 1)];

            scalar_t output_gb = w00_gb * lut[id00_gb + shift * (3 * i + 2)] + w10_gb * lut[id10_gb + shift * (3 * i + 2)] + 
                                 w01_gb * lut[id01_gb + shift * (3 * i + 2)] + w11_gb * lut[id11_gb + shift * (3 * i + 2)];
            //weight grad                                                            
            wights_grad[3 * i] += output_rg * grad_o_;
            wights_grad[3 * i + 1] += output_rb * grad_o_;
            wights_grad[3 * i + 2] += output_gb * grad_o_;
            //bias grad 
            bias_grad[i] += grad_o_;       
        }

    }
}

void TriLinear2DCPUBackwardLaucher(const torch::Tensor &grad_output, 
    const torch::Tensor &lut, const torch::Tensor &input,
    const torch::Tensor &weights, const torch::Tensor &bias,  
    torch::Tensor grad_lut, torch::Tensor grad_image,
    torch::Tensor grad_weights, torch::Tensor grad_bias) {

    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut.size(1);
    int dim   = lut.size(3);
    int shift   = dim * dim;
   
    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "trilinear_cpu_backward", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_image = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_image_  = grad_image[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();
                scalar_t binsize = 1.0 / (dim - 1);

                const scalar_t *data_weights = weights[elt].data_ptr<scalar_t>();
                const scalar_t *data_bias = bias[elt].data_ptr<scalar_t>();

                scalar_t *grad_weight_  = grad_weights[elt].data_ptr<scalar_t>();
                scalar_t *grad_bias_ = grad_bias[elt].data_ptr<scalar_t>();

                TriLinear2DCPUBackward(
                    num_kernels, grad_out, data_lut, data_image,
                    data_weights, data_bias,
                    grad_lut_, grad_image_,
                    grad_weight_, grad_bias_,
                    dim, shift, binsize,
                    width, height, num_channels);
            }));
    }
}
