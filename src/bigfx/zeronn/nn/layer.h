#ifndef ZERONN_NN_MODULE_H_
#define ZERONN_NN_MODULE_H_

#include <stdbool.h>

#include "zeronn/tensor.h"
#include "zeronn/type/bigfx.h"

typedef enum layer_type_id_t {
    BATCHNORM,
    CONV1D,
    CONV2D,
    CONV3D,
    CONV4D,
    DEBUG,
    DROPOUT,
    LINEAR,
    RELU,
    RESHAPE,
    SEQUENCE,
    SOFTMAX,
} layer_type_id_t;

typedef struct {
    layer_type_id_t type;
} layer_t;

void layer_free(layer_t* layer);
void layer_zero_grad(layer_t* layer);
tensor_t* layer_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* layer_backward(layer_t* layer, tensor_t* dy);
void layer_update_step(layer_t* layer, bigfx_t lr);

#endif  /* ZERONN_NN_MODULE_H_ */
