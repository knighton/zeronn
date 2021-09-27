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
    UNUSED(layer);
    return tensor_clone(dy);
}

void debug_update_step(layer_t* layer, float lr) {
    UNUSED(layer);
    UNUSED(lr);
}
