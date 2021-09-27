#include "batchnorm.h"

#include <assert.h>
#include <stdlib.h>

#include <stdio.h>

void batchnorm_init(layer_t* layer, int dim, fx_t mom, fx_t eps,
                    int ndim) {
    batchnorm_t* f = (batchnorm_t*)layer;
    f->type = BATCHNORM;
    f->dim = dim;
    f->ndim = ndim;
    f->mom = mom;
    f->eps = eps;
    f->gamma = full(FX_ONE, dim);
    f->dgamma = zeros(dim);
    f->beta = zeros(dim);
    f->dbeta = zeros(dim);
    f->mov_mean = zeros(dim);
    f->mov_std = full(FX_ONE, dim);
    f->x = empty();
}

layer_t* batchnorm(int dim, fx_t mom, fx_t eps, int ndim) {
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
    int values_per_dim = x->size / dim;
    tensor_t* y = tensor_clone(x);
    if (is_t) {
        tensor_set(f->x, x);
        for (int d = 0; d < dim; ++d) {
            fx_t x_mean = 0;
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    x_mean += x->data[idx];
                }
            }
            x_mean = fx_div(x_mean, values_per_dim);

            fx_t x_var = 0;
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    fx_t x_ctr = x->data[idx] - x_mean;
                    x_var += fx_div(x_ctr * x_ctr, FX_ONE);
                }
            }
            x_var = fx_div(x_var, values_per_dim) + f->eps;
            fx_t x_std = fx_sqrt(x_var) + f->eps;

            fx_t gamma = f->gamma->data[d];
            fx_t beta = f->beta->data[d];
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    fx_t x_ctr = x->data[idx] - x_mean;
                    y->data[idx] = fx_div(gamma * x_ctr, x_std) + beta;
                }
            }

            f->mov_mean->data[d] = fx_div(
                (f->mom * f->mov_mean->data[d] +
                (FX_ONE - f->mom) * x_mean), FX_ONE);
            f->mov_std->data[d] = fx_div(
                (f->mom * f->mov_std->data[d] +
                (FX_ONE - f->mom) * x_std), FX_ONE);
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            fx_t mov_mean = f->mov_mean->data[d];
            fx_t mov_std = f->mov_std->data[d];
            fx_t gamma = f->gamma->data[d];
            fx_t beta = f->beta->data[d];
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    fx_t x_ctr = x->data[idx] - mov_mean;
                    y->data[idx] = fx_div(gamma * x_ctr, mov_std) + beta;
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
    int values_per_dim = dy->size / dim;
    tensor_t* dx = tensor_clone(dy);
    for (int d = 0; d < dim; ++d) {
        fx_t x_mean = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                x_mean += f->x->data[idx];
            }
        }
        x_mean = fx_div(x_mean, values_per_dim);

        fx_t x_var = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                fx_t x_ctr = f->x->data[idx] - x_mean;
                x_var += fx_div(x_ctr * x_ctr, FX_ONE);
            }
        }
        x_var = fx_div(x_var, values_per_dim) + f->eps;
        fx_t x_std = fx_sqrt(x_var) + f->eps;

        fx_t dgamma = 0;
        fx_t dbeta = 0;
        fx_t dy_mean = 0;
        fx_t dy_x_ctr_mean = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                fx_t dy_val = dy->data[idx];
                fx_t x_ctr = f->x->data[idx] - x_mean;
                dgamma += dy_val * x_ctr;
                dbeta += dy_val;
                dy_mean += dy_val;
                dy_x_ctr_mean += dy_val * x_ctr;
            }
        }
        f->dgamma->data[d] += fx_div(dgamma, x_std);
        f->dbeta->data[d] += dbeta;
        dy_mean = fx_div(dy_mean, values_per_dim);
        dy_x_ctr_mean = fx_div(dy_x_ctr_mean, values_per_dim);

        fx_t gamma = f->gamma->data[d];
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                fx_t dy_val = dy->data[idx];
                fx_t x_ctr = f->x->data[idx] - x_mean;
                fx_t dx_val = dy_val - dy_mean - fx_div(x_ctr * dy_x_ctr_mean,
                                                        x_var * FX_ONE);
                dx->data[idx] = fx_div(dx_val * gamma, x_std);
                }
        }
    }
    return dx;
}

void batchnorm_update_step(layer_t* layer, fx_t lr) {
    batchnorm_t* f = (batchnorm_t*)layer;
    tensor_update_step(f->gamma, f->dgamma, lr);
    tensor_update_step(f->beta, f->dbeta, lr);
}
