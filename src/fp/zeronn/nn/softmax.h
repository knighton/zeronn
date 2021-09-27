#ifndef ZERONN_NN_SOFTMAX_H_
#define ZERONN_NN_SOFTMAX_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
} softmax_t;

void softmax_init(layer_t* layer);
layer_t* softmax(void);
void softmax_free(layer_t* layer);
void softmax_zero_grad(layer_t* layer);
tensor_t* softmax_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* softmax_backward(layer_t* layer, tensor_t* dy);
void softmax_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_SOFTMAX_H_ */
