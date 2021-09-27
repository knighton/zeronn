#ifndef ZERONN_NN_CONV2D_H_
#define ZERONN_NN_CONV2D_H_

#include "zeronn/coord.h"
#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    int x_channels;
    int y_channels;
    coord2d_t face;
    coord2d_t stride;
    coord2d_t pad;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} conv2d_t;

void conv2d_init(layer_t* layer, int x_channels, int y_channels, int face,
                 int stride, int pad);
layer_t* conv2d(int x_channels, int y_channels, int face, int stride,
                 int pad);
void conv2d_free(layer_t* layer);
void conv2d_zero_grad(layer_t* layer);
tensor_t* conv2d_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* conv2d_backward(layer_t* layer, tensor_t* dy);
void conv2d_update_step(layer_t* layer, bigfx_t lr);

#endif  /* ZERONN_NN_CONV2D_H_ */
