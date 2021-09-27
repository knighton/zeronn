#include "softmax.h"

#include <assert.h>
#include <stdlib.h>

#include "zeronn/lint.h"

void softmax_init(layer_t* layer) {
    softmax_t* f = (softmax_t*)layer;
    f->type = SOFTMAX;
}

layer_t* softmax(void) {
    layer_t* layer = (layer_t*)malloc(sizeof(softmax_t));
    softmax_init(layer);
    return layer;
}

void softmax_free(layer_t* layer) {
    UNUSED(layer);
}

void softmax_zero_grad(layer_t* layer) {
    UNUSED(layer);
}

tensor_t* softmax_forward(layer_t* layer, tensor_t* x, bool is_t) {
    UNUSED(layer);
    assert(x->ndim == 2);
    int batch_size = x->shape[0];
    int dim = x->shape[1];
    tensor_t* y = zeros(batch_size, dim);
    for (int n = 0; n < batch_size; ++n) {
        fx_t max = x->data[n * dim];
        for (int d = 1; d < dim; ++d) {
            int idx = n * dim + d;
            fx_t val = x->data[idx];
            if (max < val) {
                max = val;
            }
        }

        fx_t sum = 0;
        for (int d = 0; d < dim; ++d) {
            int idx = n * dim + d;
            fx_t val = fx_exp(x->data[idx] - max);
            y->data[idx] = val;
            sum += val;
        }

        for (int d = 0; d < dim; ++d) {
            int idx = n * dim + d;
            y->data[idx] = fx_div(y->data[idx] * FX_ONE, sum);
        }
    }
    return y;
}

tensor_t* softmax_backward(layer_t* layer, tensor_t* dy) {
    UNUSED(layer);
    return tensor_clone(dy);
}

void softmax_update_step(layer_t* layer, fx_t lr) {
    UNUSED(layer);
    UNUSED(lr);
}
