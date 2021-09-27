#ifndef ZERONN_NN_LINEAR_H_
#define ZERONN_NN_LINEAR_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    int x_dim;
    int y_dim;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} linear_t;

void linear_init(layer_t* layer, int x_dim, int y_dim);
layer_t* linear(int x_dim, int y_dim);
void linear_free(layer_t* layer);
void linear_zero_grad(layer_t* layer);
tensor_t* linear_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* linear_backward(layer_t* layer, tensor_t* dy);
void linear_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_LINEAR_H_ */
