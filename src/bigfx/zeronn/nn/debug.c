#include "debug.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "zeronn/lint.h"

void debug_init(layer_t* layer, const char* text) {
    debug_t* f = (debug_t*)layer;
    f->type = DEBUG;
    f->text = text;
}

layer_t* debug(const char* text) {
    layer_t* layer = (layer_t*)malloc(sizeof(debug_t));
    debug_init(layer, text);
    return layer;
}

void debug_free(layer_t* layer) {
    UNUSED(layer);
}

void debug_zero_grad(layer_t* layer) {
    UNUSED(layer);
}

void debug_print(const char* text, tensor_t* x, FILE* out) {
    bigfx_t mean = bigfx_zero();
    for (int i = 0; i < x->size; ++i) {
        mean = bigfx_add(mean, x->data[i]);
    }
    mean = bigfx_div(mean, bigint_from_int(x->size));
    mean = bigfx_div_log2(mean, FX_BITS);

    bigfx_t std = bigfx_zero();
    for (int i = 0; i < x->size; ++i) {
        bigfx_t a = bigfx_sub(x->data[i], mean);
        a = bigint_mul(a, a);
        a = bigint_div_log2(a, FX_BITS);
        std = bigfx_add(std, a);
    }
    std = bigfx_div(std, bigint_from_int(x->size));
    std = bigfx_sqrt(std);
    std = bigfx_div_log2(std, SQRT_FX_BITS);

    bigfx_t min = x->data[0];
    bigfx_t max = x->data[0];
    for (int i = 1; i < x->size; ++i) {
        if (bigfx_lt(x->data[i], min)) {
            min = x->data[i];
        }
        if (bigfx_lt(max, x->data[i])) {
            max = x->data[i];
        }
    }

    int i_mean = int_from_bigint(mean);
    int i_std = int_from_bigint(std);
    int i_min = int_from_bigint(min);
    int i_max = int_from_bigint(max);

    fprintf(out, "[%10d : %10d +/- %10d : %10d] ", i_min, i_mean, i_std, i_max);
    fprintf(out, "%s %d @ (", text, x->size);
    if (x->ndim) {
        fprintf(out, "%d", x->shape[0]);
    }
    for (int i = 1; i < x->ndim; ++i) {
        fprintf(out, ", %d", x->shape[i]);
    }
    fprintf(out, ")\n");
    fflush(out);
}

tensor_t* debug_forward(layer_t* layer, tensor_t* x, bool is_t) {
    UNUSED(is_t);
    debug_t* f = (debug_t*)layer;
    debug_print(f->text, x, stdout);
    return tensor_clone(x);
}

tensor_t* debug_backward(layer_t* layer, tensor_t* dy) {
    debug_t* f = (debug_t*)layer;
    debug_print(f->text, dy, stdout);
    return tensor_clone(dy);
}

void debug_update_step(layer_t* layer, bigfx_t lr) {
    UNUSED(layer);
    UNUSED(lr);
}
