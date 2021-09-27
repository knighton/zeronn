#include "linear.h"

#include <assert.h>
#include <stdlib.h>

void linear_init(layer_t* layer, int x_dim, int y_dim) {
    assert(0 < x_dim);
    assert(0 < y_dim);
    linear_t* f = (linear_t*)layer;
    f->type = LINEAR;
    f->x_dim = x_dim;
    f->y_dim = y_dim;
    f->weight = uniform_log2(bigfx_from_fx(-128), 8, x_dim, y_dim);
    f->dweight = zeros(x_dim, y_dim);
    f->bias = uniform_log2(bigfx_from_fx(-128), 8, y_dim);
    f->dbias = zeros(y_dim);
    f->x = empty();
}

layer_t* linear(int x_dim, int y_dim) {
    layer_t* layer = (layer_t*)malloc(sizeof(linear_t));
    linear_init(layer, x_dim, y_dim);
    return layer;
}

void linear_free(layer_t* layer) {
    linear_t* f = (linear_t*)layer;
    tensor_free(f->weight);
    tensor_free(f->dweight);
    tensor_free(f->bias);
    tensor_free(f->dbias);
    tensor_free(f->x);
    free(f->weight);
    free(f->dweight);
    free(f->bias);
    free(f->dbias);
    free(f->x);
}

void linear_zero_grad(layer_t* layer) {
    linear_t* f = (linear_t*)layer;
    tensor_zero(f->dweight);
    tensor_zero(f->dbias);
}

tensor_t* linear_forward(layer_t* layer, tensor_t* x, bool is_t) {
    linear_t* f = (linear_t*)layer;
    if (is_t) {
        tensor_set(f->x, x);
    }
    int count = x->shape[0];
    assert(f->x_dim == x->shape[1]);
    tensor_t* y = zeros(count, f->y_dim);
    for (int n = 0; n < count; ++n) {
        for (int yd = 0; yd < f->y_dim; ++yd) {
            int y_idx = n * f->y_dim + yd;
            bigfx_t y_val = bigfx_zero();
            for (int xd = 0; xd < f->x_dim; ++xd) {
                int x_idx = n * f->x_dim + xd;
                int w_idx = xd * f->y_dim + yd;
                y_val = bigfx_add(y_val, bigint_mul(x->data[x_idx],
                                                    f->weight->data[w_idx]));
            }
            y->data[y_idx] = bigfx_add(bigfx_div_log2(y_val, FX_BITS),
                                       f->bias->data[yd]);
        }
    }
    return y;
}

tensor_t* linear_backward(layer_t* layer, tensor_t* dy) {
    linear_t* f = (linear_t*)layer;
    int count = dy->shape[0];
    assert(f->y_dim == dy->shape[1]);
    tensor_t* dx = zeros(count, f->x_dim);
    tensor_t* dweight = zeros(f->x_dim, f->y_dim);
    for (int n = 0; n < count; ++n) {
        for (int yd = 0; yd < f->y_dim; ++yd) {
            int y_idx = n * f->y_dim + yd;
            bigfx_t dy_val = dy->data[y_idx];
            for (int xd = 0; xd < f->x_dim; ++xd) {
                int x_idx = n * f->x_dim + xd;
                int w_idx = xd * f->y_dim + yd;
                dweight->data[w_idx] = bigfx_add(
                    dweight->data[w_idx],
                    bigint_mul(dy_val, f->x->data[x_idx]));
                dx->data[x_idx] = bigfx_add(
                    dx->data[x_idx],
                    bigint_mul(dy_val, f->weight->data[w_idx]));
            }
            f->dbias->data[yd] = bigfx_add(f->dbias->data[yd], dy_val);
        }
    }
    for (int i = 0; i < dweight->size; ++i) {
        f->dweight->data[i] = bigfx_add(
            f->dweight->data[i], bigfx_div_log2(dweight->data[i], FX_BITS));
    }
    for (int i = 0; i < dx->size; ++i) {
        dx->data[i] = bigfx_div_log2(dx->data[i], FX_BITS);
    }
    tensor_free(dweight);
    free(dweight);
    return dx;
}

void linear_update_step(layer_t* layer, bigfx_t lr) {
    linear_t* f = (linear_t*)layer;
    tensor_update_step(f->weight, f->dweight, lr);
    tensor_update_step(f->bias, f->dbias, lr);
}
