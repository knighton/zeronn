#include "dropout.h"

#include <assert.h>
#include <stdlib.h>

#include "zeronn/lint.h"
#include "zeronn/prng.h"

void dropout_init(layer_t* layer, fx_t rate) {
    assert(0 <= rate);
    assert(rate <= FX_ONE);
    dropout_t* f = (dropout_t*)layer;
    f->type = DROPOUT;
    f->rate = rate;
    f->mask = empty();
}

layer_t* dropout(fx_t rate) {
    layer_t* layer = (layer_t*)malloc(sizeof(dropout_t));
    dropout_init(layer, rate);
    return layer;
}

void dropout_free(layer_t* layer) {
    dropout_t* f = (dropout_t*)layer;
    tensor_free(f->mask);
    free(f->mask);
}

void dropout_zero_grad(layer_t* layer) {
    UNUSED(layer);
}

tensor_t* dropout_forward(layer_t* layer, tensor_t* x, bool is_t) {
    dropout_t* f = (dropout_t*)layer;
    tensor_t* y = tensor_clone(x);
    if (is_t) {
        tensor_set(f->mask, x);
        for (int i = 0; i < y->size; ++i) {
            if (f->rate <= prng_randint(0, FX_ONE)) {
                f->mask->data[i] = 1;
                y->data[i] = fx_div(y->data[i] * FX_ONE, f->rate);
            } else {
                f->mask->data[i] = 0;
                y->data[i] = 0;
            }
        }
    }
    return y;
}

tensor_t* dropout_backward(layer_t* layer, tensor_t* dy) {
    dropout_t* f = (dropout_t*)layer;
    tensor_t* dx = tensor_clone(dy);
    for (int i = 0; i < dx->size; ++i) {
        if (f->mask->data[i]) {
            dx->data[i] = fx_div(dx->data[i] * f->rate, FX_ONE);
        } else {
            dx->data[i] = 0;
        }
    }
    return dx;
}

void dropout_update_step(layer_t* layer, fx_t lr) {
    UNUSED(layer);
    UNUSED(lr);
}
