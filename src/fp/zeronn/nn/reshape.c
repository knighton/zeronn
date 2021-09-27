#include "reshape.h"

#include <assert.h>
#include <stdlib.h>

#include <stdio.h>

#include "zeronn/lint.h"

void reshape_init_common(layer_t* layer, int num_args, va_list* args) {
    reshape_t* f = (reshape_t*)layer;
    f->type = RESHAPE;
    assert(num_args + 1 <= NDIM_MAX);
    f->y.size = 1;
    f->y.ndim = num_args + 1;
    f->y.shape[0] = 0;
    bool has_wildcard = false;
    for (int i = 0; i < num_args; ++i) {
        int arg = va_arg(*args, int);
        if (has_wildcard) {
            assert(1 <= arg);
        } else {
            if (arg == -1) {
                has_wildcard = true;
            } else {
                assert(1 <= arg);
            }
        }
        f->y.size *= arg;
        f->y.shape[i + 1] = arg;
    }
    f->x.size = 0;
    f->x.ndim = 0;
}

void pp_reshape_init(layer_t* layer, int num_args, ...) {
    va_list args;
    va_start(args, num_args);
    reshape_init_common(layer, num_args, &args);
    va_end(args);
}

layer_t* pp_reshape(int num_args, ...) {
    layer_t* layer = (layer_t*)malloc(sizeof(reshape_t));
    va_list args;
    va_start(args, num_args);
    reshape_init_common(layer, num_args, &args);
    va_end(args);
    return layer;
}

void reshape_free(layer_t* layer) {
    UNUSED(layer);
}

void reshape_zero_grad(layer_t* layer) {
    UNUSED(layer);
}

tensor_t* reshape_forward(layer_t* layer, tensor_t* x, bool is_t) {
    UNUSED(is_t);
    reshape_t* f = (reshape_t*)layer;
    tensor_shape(x, &f->x);
    int batch_size = x->shape[0];
    f->y.size *= batch_size;
    f->y.shape[0] = batch_size;
    tensor_t* y = tensor_reshape(x, &f->y);
    f->y.size /= batch_size;
    return y;
}

tensor_t* reshape_backward(layer_t* layer, tensor_t* dy) {
    reshape_t* f = (reshape_t*)layer;
    return tensor_reshape(dy, &f->x);
}

void reshape_update_step(layer_t* layer, float lr) {
    UNUSED(layer);
    UNUSED(lr);
}
