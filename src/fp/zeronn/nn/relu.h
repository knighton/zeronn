#ifndef ZERONN_NN_RELU_H_
#define ZERONN_NN_RELU_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    tensor_t* x;
} relu_t;

void relu_init(layer_t* layer);
layer_t* relu(void);
void relu_free(layer_t* layer);
void relu_zero_grad(layer_t* layer);
tensor_t* relu_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* relu_backward(layer_t* layer, tensor_t* dy);
void relu_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_RELU_H_ */
