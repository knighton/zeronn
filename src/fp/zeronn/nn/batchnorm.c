#include "batchnorm.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

void batchnorm_init(layer_t* layer, int dim, float mom, float eps,
                    int ndim) {
    batchnorm_t* f = (batchnorm_t*)layer;
    f->type = BATCHNORM;
    f->dim = dim;
    f->ndim = ndim;
    f->mom = mom;
    f->eps = eps;
    f->gamma = ones(dim);
    f->dgamma = zeros(dim);
    f->beta = zeros(dim);
    f->dbeta = zeros(dim);
    f->mov_mean = zeros(dim);
    f->mov_std = ones(dim);
    f->x = empty();
}

layer_t* batchnorm(int dim, float mom, float eps, int ndim) {
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
            float x_mean = 0;
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    x_mean += x->data[idx];
                }
            }
            x_mean /= values_per_dim;

            float x_var = 0;
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    float val = x->data[idx] - x_mean;
                    x_var += val * val;
                }
            }
            x_var /= values_per_dim;
            x_var += f->eps;
            float x_std = sqrt(x_var);

            float gamma = f->gamma->data[d];
            float beta = f->beta->data[d];
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    float val = (x->data[idx] - x_mean) / x_std;
                    y->data[idx] = val * gamma + beta;
                }
            }

            f->mov_mean->data[d] = f->mom * f->mov_mean->data[d] +
                (1 - f->mom) * x_mean;
            f->mov_std->data[d] = f->mom * f->mov_std->data[d] +
                (1 - f->mom) * x_std;
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            float mov_mean = f->mov_mean->data[d];
            float mov_std = f->mov_std->data[d];
            float gamma = f->gamma->data[d];
            float beta = f->beta->data[d];
            for (int n = 0; n < batch; ++n) {
                for (int s = 0; s < space; ++s) {
                    int idx = (n * dim + d) * space + s;
                    float val = (x->data[idx] - mov_mean) / mov_std;
                    y->data[idx] = val * gamma * beta;
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
        float x_mean = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                x_mean += f->x->data[idx];
            }
        }
        x_mean /= values_per_dim;

        float x_var = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                float val = f->x->data[idx] - x_mean;
                x_var += val * val;
            }
        }
        x_var /= values_per_dim;
        x_var += f->eps;
        float x_std = sqrt(x_var);

        float dgamma = 0;
        float dbeta = 0;
        float dy_mean = 0;
        float dy_x_ctr_mean = 0;
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                float dy_val = dy->data[idx];
                float x_ctr = f->x->data[idx] - x_mean;
                float x_norm = x_ctr / x_std;
                dgamma += dy_val * x_norm;
                dbeta += dy_val;
                dy_mean += dy_val;
                dy_x_ctr_mean += dy_val * x_ctr;
            }
        }
        f->dgamma->data[d] += dgamma;
        f->dbeta->data[d] += dbeta;
        dy_mean /= values_per_dim;
        dy_x_ctr_mean /= values_per_dim;

        float gamma = f->gamma->data[d];
        for (int n = 0; n < batch; ++n) {
            for (int s = 0; s < space; ++s) {
                int idx = (n * dim + d) * space + s;
                float dy_val = dy->data[idx];
                float x_ctr = f->x->data[idx] - x_mean;
                float val = dy_val - dy_mean - x_ctr / x_var * dy_x_ctr_mean;
                dx->data[idx] = val * gamma / x_std;
            }
        }
    }
    return dx;
}

void batchnorm_update_step(layer_t* layer, float lr) {
    batchnorm_t* f = (batchnorm_t*)layer;
    tensor_update_step(f->gamma, f->dgamma, lr);
    tensor_update_step(f->beta, f->dbeta, lr);
}
