#include "layer.h"

#include <stddef.h>

#include "zeronn/nn/nn.h"

typedef void (*layer_free_t)(layer_t* layer);
typedef void (*layer_zero_grad_t)(layer_t* layer);
typedef tensor_t* (*layer_forward_t)(layer_t* layer, tensor_t* x,
                                     bool is_t);
typedef tensor_t* (*layer_backward_t)(layer_t* layer, tensor_t* dy);
typedef void (*layer_update_step_t)(layer_t* layer, fx_t lr);

typedef struct layer_api_t {
    layer_free_t free;
    layer_zero_grad_t zero_grad;
    layer_forward_t forward;
    layer_backward_t backward;
    layer_update_step_t update_step;
} layer_api_t;

#define TODO(layer) { NULL, NULL, NULL, NULL, NULL }

#define LAYER(layer) { \
    layer##_free, \
    layer##_zero_grad, \
    layer##_forward, \
    layer##_backward, \
    layer##_update_step \
}

static layer_api_t LAYERS[] = {
    LAYER(batchnorm),
    LAYER(conv1d),
    LAYER(conv2d),
    LAYER(conv3d),
    LAYER(conv4d),
    LAYER(debug),
    LAYER(dropout),
    LAYER(linear),
    LAYER(relu),
    LAYER(reshape),
    LAYER(sequence),
    LAYER(softmax),
};

#undef TODO

void layer_free(layer_t* layer) {
    LAYERS[layer->type].free(layer);
}

void layer_zero_grad(layer_t* layer) {
    LAYERS[layer->type].zero_grad(layer);
}

tensor_t* layer_forward(layer_t* layer, tensor_t* x, bool is_t) {
    return LAYERS[layer->type].forward(layer, x, is_t);
}

tensor_t* layer_backward(layer_t* layer, tensor_t* dy) {
    return LAYERS[layer->type].backward(layer, dy);
}

void layer_update_step(layer_t* layer, fx_t lr) {
    LAYERS[layer->type].update_step(layer, lr);
}
