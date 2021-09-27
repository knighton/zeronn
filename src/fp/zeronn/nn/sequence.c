#include "sequence.h"

#include <stdlib.h>

void sequence_init(layer_t* layer, int num_layers, layer_t** layers) {
    sequence_t* f = (sequence_t*)layer;
    f->type = SEQUENCE;
    f->num_layers = num_layers;
    f->layers = layers;
}

layer_t* pp_sequence(int num_layers, ...) {
    va_list args;
    va_start(args, num_layers);
    layer_t** layers = (layer_t**)malloc(num_layers * sizeof(layer_t*));
    for (int i = 0; i < num_layers; ++i) {
        layers[i] = va_arg(args, layer_t*);
    }
    va_end(args);
    layer_t* f = (layer_t*)malloc(sizeof(sequence_t));
    sequence_init(f, num_layers, layers);
    return f;
}

void sequence_free(layer_t* layer) {
    sequence_t* f = (sequence_t*)layer;
    for (int i = 0; i < f->num_layers; ++i) {
        layer_t* sub_layer = f->layers[i];
        layer_free(sub_layer);
        free(sub_layer);
    }
    free(f->layers);
}

void sequence_zero_grad(layer_t* layer) {
    sequence_t* f = (sequence_t*)layer;
    for (int i = 0; i < f->num_layers; ++i) {
        layer_t* sub_layer = f->layers[i];
        layer_zero_grad(sub_layer);
    }
}

tensor_t* sequence_forward(layer_t* layer, tensor_t* x, bool is_t) {
    sequence_t* f = (sequence_t*)layer;
    for (int i = 0; i < f->num_layers; ++i) {
        layer_t* sub_layer = f->layers[i];
        tensor_t* y = layer_forward(sub_layer, x, is_t);
        if (i) {
            tensor_free(x);
            free(x);
        }
        x = y;
    }
    return x;
}

tensor_t* sequence_backward(layer_t* layer, tensor_t* dy) {
    sequence_t* f = (sequence_t*)layer;
    for (int i = 0; i < f->num_layers; ++i) {
        layer_t* sub_layer = f->layers[f->num_layers - i - 1];
        tensor_t* dx = layer_backward(sub_layer, dy);
        if (i) {
            tensor_free(dy);
            free(dy);
        }
        dy = dx;
    }
    return dy;
}

void sequence_update_step(layer_t* layer, float lr) {
    sequence_t* f = (sequence_t*)layer;
    for (int i = 0; i < f->num_layers; ++i) {
        layer_t* sub_layer = f->layers[i];
        layer_update_step(sub_layer, lr);
    }
}
