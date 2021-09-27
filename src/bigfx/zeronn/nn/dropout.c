#include "dropout.h"

#include <assert.h>
#include <stdlib.h>

#include "zeronn/lint.h"
#include "zeronn/prng.h"

void dropout_init(layer_t* layer, bigfx_t rate) {
    assert(bigfx_le(bigfx_zero(), rate));
    assert(bigfx_le(rate, bigfx_one()));
    dropout_t* f = (dropout_t*)layer;
    f->type = DROPOUT;
    f->rate = rate;
    f->mask = empty();
}

layer_t* dropout(bigfx_t rate) {
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
            if (bigfx_le(f->rate, prng_randint_log2(bigfx_zero(), FX_BITS))) {
                f->mask->data[i] = bigint_one();
                y->data[i] = bigint_div(y->data[i], f->rate);
                y->data[i] = bigint_div_log2(y->data[i], FX_BITS);
            } else {
                f->mask->data[i] = bigint_zero();
                y->data[i] = bigfx_zero();
            }
        }
    }
    return y;
}

tensor_t* dropout_backward(layer_t* layer, tensor_t* dy) {
    dropout_t* f = (dropout_t*)layer;
    tensor_t* dx = tensor_clone(dy);
    for (int i = 0; i < dx->size; ++i) {
        if (f->mask->data[i].sign == BIGINT_SIGN_ZERO) {
            dx->data[i] = bigfx_zero();
        }
    }
/*
        dx->data[i] = bigint_mul(dx->data[i], f->rate);
        dx->data[i] = bigint_div_log2(dx->data[i], FX_BITS);
        dx->data[i] = bigint_mul(dx->data[i], f->mask->data[i]);
    }
*/
    return dx;
}

void dropout_update_step(layer_t* layer, bigfx_t lr) {
    UNUSED(layer);
    UNUSED(lr);
}
