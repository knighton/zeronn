#ifndef ZERONN_NN_DROPOUT_H_
#define ZERONN_NN_DROPOUT_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    float rate;
    tensor_t* mask;
    tensor_t* x;
} dropout_t;

void dropout_init(layer_t* layer, float rate);
layer_t* dropout(float rate);
void dropout_free(layer_t* layer);
void dropout_zero_grad(layer_t* layer);
tensor_t* dropout_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* dropout_backward(layer_t* layer, tensor_t* dy);
void dropout_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_DROPOUT_H_ */
