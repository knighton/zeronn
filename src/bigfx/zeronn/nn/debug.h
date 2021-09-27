#ifndef ZERONN_NN_DEBUG_H_
#define ZERONN_NN_DEBUG_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    const char* text;
} debug_t;

void debug_init(layer_t* layer, const char* text);
layer_t* debug(const char* text);
void debug_free(layer_t* layer);
void debug_zero_grad(layer_t* layer);
tensor_t* debug_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* debug_backward(layer_t* layer, tensor_t* dy);
void debug_update_step(layer_t* layer, bigfx_t lr);

#endif  /* ZERONN_NN_DEBUG_H_ */
