#include "batchnorm.h"

#include <assert.h>
#include <stdlib.h>

void batchnorm_init(layer_t* layer, int dim, bigfx_t mom, bigfx_t eps,
                    int ndim) {
    batchnorm_t* f = (batchnorm_t*)layer;
    f->type = BATCHNORM;
    f->dim = dim;
    f->ndim = ndim;
    f->mom = mom;
    f->eps = eps;
    f->gamma = full(bigfx_one(), dim);
    f->dgamma = zeros(dim);
    f->beta = zeros(dim);
    f->dbeta = zeros(dim);
    f->mov_mean = zeros(dim);
    f->mov_std = full(bigfx_one(), dim);
    f->x = empty();
}

layer_t* batchnorm(int dim, bigfx_t mom, bigfx_t eps, int ndim) {
    layer_t* layer = (layer_t*)malloc(sizeof(batchnorm_t));
    batchnorm_init(layer, dim, mom, eps, ndim);
    return layer;
}

void batchnorm_free(layer_t* layer) {
    batchnorm_t* f = (batchnorm_t*)layer;
    tensor_free(f->gamma);
    tensor_free(f->dgamma);
    tensor_free(f->beta);
    tensor_free(f->dbeta);
    tensor_free(f->mov_mean);
    tensor_free(f->mov_std);
    tensor_free(f->x);
    free(f->gamma);
    free(f->dgamma);
    free(f->beta);
    free(f->dbeta);
    free(f->mov_mean);
    free(f->mov_std);
    free(f->x);
}

void batchnorm_zero_grad(layer_t* layer) {
    batchnorm_t* f = (batchnorm_t*)layer;
    tensor_zero(f->dgamma);
    tensor_zero(f->dbeta);
}

tensor_t* batchnorm_forward(layer_t* layer, tensor_t* x, bool is_t) {
    batchnorm_t* f = (batchnorm_t*)layer;
    assert(f->ndim + 2 == x->ndim);
    assert(x->shape[1] == f->dim);
    int batch = x->shape[0];
    int dim = x->shape[1];
    int space = x->size / batch / dim;
    bigfx_t values_per_dim = bigfx_from_int(x->size / dim);
    tensor_t* y = tensor_clone(x);
    if (is_t) {
        tensor_set(f->x, x);
        for (int d = 0; d < dim; ++d) {
            bigfx_t x_mean = bigfx_zero();
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    x_mean = bigfx_add(x_mean, x->data[idx]);
                }
            }
            x_mean = bigint_div(x_mean, values_per_dim);
            x_mean = bigint_div_log2(x_mean, FX_BITS);

            bigfx_t x_var = bigfx_zero();
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    bigfx_t x_ctr = bigfx_sub(x->data[idx], x_mean);
                    x_var = bigfx_add(x_var, bigfx_sq(x_ctr));
                }
            }
            x_var = bigint_div(x_var, values_per_dim);
            x_var = bigint_div_log2(x_var, FX_BITS);
            x_var = bigfx_add(x_var, f->eps);
            bigfx_t x_std = bigfx_sqrt(x_var);
            x_std = bigfx_add(x_std, f->eps);

            bigfx_t gamma = f->gamma->data[d];
            bigfx_t beta = f->beta->data[d];
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    bigfx_t x_ctr = bigfx_sub(x->data[idx], x_mean);
                    bigfx_t a = bigfx_mul(gamma, x_ctr);
                    a = bigint_div(a, x_std);
                    a = bigint_div_log2(a, FX_BITS);
                    y->data[idx] = bigfx_add(a, beta);
                }
            }

            f->mov_mean->data[d] = bigfx_div_log2(
                bigfx_add(
                    bigint_mul(f->mom, f->mov_mean->data[d]),
                    bigint_mul(bigfx_sub(bigfx_one(), f->mom), x_mean)),
                FX_BITS);
            f->mov_std->data[d] = bigfx_div_log2(
                bigfx_add(
                    bigint_mul(f->mom, f->mov_std->data[d]),
                    bigint_mul(bigfx_sub(bigfx_one(), f->mom), x_std)),
                FX_BITS);
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            bigfx_t mov_mean = f->mov_mean->data[d];
            bigfx_t mov_std = f->mov_std->data[d];
            bigfx_t gamma = f->gamma->data[d];
            bigfx_t beta = f->beta->data[d];
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    bigfx_t x_ctr = bigfx_sub(x->data[idx], mov_mean);
                    bigfx_t a = bigint_mul(gamma, x_ctr);
                    a = bigint_div(a, mov_std);
                    a = bigint_div_log2(a, FX_BITS);
                    y->data[idx] = bigint_add(a, beta);
                }
            }
        }
    }
    return y;
}

tensor_t* batchnorm_backward(layer_t* layer, tensor_t* dy) {
    batchnorm_t* f = (batchnorm_t*)layer;
    assert(f->ndim + 2 == dy->ndim);
    assert(dy->shape[1] == f->dim);
    int batch = dy->shape[0];
    int dim = dy->shape[1];
    int space = dy->size / batch / dim;
    bigint_t values_per_dim = bigfx_from_int(dy->size / dim);
    tensor_t* dx = tensor_clone(dy);
    for (int d = 0; d < dim; ++d) {
        bigfx_t x_mean = bigfx_zero();
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                x_mean = bigfx_add(x_mean, f->x->data[idx]);
            }
        }
        x_mean = bigint_div(x_mean, values_per_dim);
        x_mean = bigint_div_log2(x_mean, FX_BITS);

        bigfx_t x_var = bigfx_zero();
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                bigfx_t x_ctr = bigfx_sub(f->x->data[idx], x_mean);
                x_var = bigfx_add(x_var, bigfx_sq(x_ctr));
            }
        }
        x_var = bigint_div(x_var, values_per_dim);
        x_var = bigint_div_log2(x_var, FX_BITS);
        x_var = bigfx_add(x_var, f->eps);
        bigfx_t x_std = bigfx_sqrt(x_var);
        x_std = bigint_mul_log2(x_std, FX_BITS);
        x_std = bigfx_add(x_std, f->eps);

        bigfx_t dgamma = bigfx_zero();
        bigfx_t dbeta = bigfx_zero();
        bigfx_t dy_mean = bigfx_zero();
        bigfx_t dy_x_ctr_mean = bigfx_zero();
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                bigfx_t dy_val = dy->data[idx];
                bigfx_t x_ctr = bigfx_sub(f->x->data[idx], x_mean);
                dgamma = bigfx_add(dgamma, bigfx_mul(dy_val, x_ctr));
                dbeta = bigfx_add(dbeta, dy_val);
                dy_mean = bigfx_add(dy_mean, dy_val);
                dy_x_ctr_mean = bigfx_add(dy_x_ctr_mean,
                                          bigfx_mul(dy_val, x_ctr));
            }
        }
        bigfx_t a = bigint_div(dgamma, x_std);
        a = bigint_div_log2(a, FX_BITS);
        f->dgamma->data[d] = bigfx_add(f->dgamma->data[d], a);

        f->dbeta->data[d] = bigfx_add(f->dbeta->data[d], dbeta);

        a = bigint_div(dy_mean, values_per_dim);
        dy_mean = bigint_div_log2(a, FX_BITS);

        a = bigint_div(dy_x_ctr_mean, values_per_dim);
        dy_x_ctr_mean = bigint_div_log2(a, FX_BITS);

        bigfx_t gamma = f->gamma->data[d];
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                bigfx_t dy_val = dy->data[idx];
                bigfx_t x_ctr = bigfx_sub(f->x->data[idx], x_mean);

                a = bigint_mul(x_ctr, dy_x_ctr_mean);
                a = bigint_div(a, x_var);
                a = bigint_div_log2(a, FX_BITS);
                a = bigint_div_log2(a, FX_BITS);
                bigfx_t dx_val = bigfx_sub(dy_val, bigfx_add(dy_mean,  a));
                a = bigint_mul(dx_val, gamma);
                a = bigint_div(a, x_std);
                dx->data[idx] = bigint_div_log2(a, FX_BITS);
/*
                dx->data[idx] = bigint_div(bigint_mul(dx_val, gamma), x_std);
                dx->data[idx] = bigint_div_log2(dx->data[idx], FX_BITS);
                dx->data[idx] = bigint_div_log2(dx->data[idx], SQRT_FX_BITS);
*/
            }
        }
    }
    return dx;
}

void batchnorm_update_step(layer_t* layer, bigfx_t lr) {
    batchnorm_t* f = (batchnorm_t*)layer;
    tensor_update_step(f->gamma, f->dgamma, lr);
    tensor_update_step(f->beta, f->dbeta, lr);
}
