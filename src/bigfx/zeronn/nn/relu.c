#include "relu.h"

#include <stdlib.h>

#include "zeronn/lint.h"

void relu_init(layer_t* layer) {
    relu_t* f = (relu_t*)layer;
    f->type = RELU;
    f->x = empty();
}

layer_t* relu(void) {
    layer_t* layer = (layer_t*)malloc(sizeof(relu_t));
    relu_init(layer);
    return layer;
}

void relu_free(layer_t* layer) {
    relu_t* f = (relu_t*)layer;
    tensor_free(f->x);
    free(f->x);
}

void relu_zero_grad(layer_t* layer) {
    UNUSED(layer);
}

tensor_t* relu_forward(layer_t* layer, tensor_t* x, bool is_t) {
    relu_t* f = (relu_t*)layer;
    if (is_t) {
        tensor_set(f->x, x);
    }
    tensor_t* y = tensor_clone(x);
    for (int i = 0; i < y->size; ++i) {
        if (y->data[i].sign == BIGFX_SIGN_NEG) {
            y->data[i] = bigfx_zero();
        }
    }
    return y;
}

tensor_t* relu_backward(layer_t* layer, tensor_t* dy) {
    relu_t* f = (relu_t*)layer;
    tensor_t* dx = tensor_clone(dy);
    for (int i = 0; i < f->x->size; ++i) {
        if (f->x->data[i].sign == BIGFX_SIGN_NEG) {
            dx->data[i] = bigfx_zero();
        }
    }
    return dx;
}

void relu_update_step(layer_t* layer, bigfx_t lr) {
    UNUSED(layer);
    UNUSED(lr);
}
