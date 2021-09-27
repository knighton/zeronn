#ifndef ZERONN_NN_CONV3D_H_
#define ZERONN_NN_CONV3D_H_

#include "zeronn/coord.h"
#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    int x_channels;
    int y_channels;
    coord3d_t face;
    coord3d_t stride;
    coord3d_t pad;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} conv3d_t;

void conv3d_init(layer_t* layer, int x_channels, int y_channels, int face,
                 int stride, int pad);
layer_t* conv3d(int x_channels, int y_channels, int face, int stride,
                 int pad);
void conv3d_free(layer_t* layer);
void conv3d_zero_grad(layer_t* layer);
tensor_t* conv3d_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* conv3d_backward(layer_t* layer, tensor_t* dy);
void conv3d_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_CONV3D_H_ */
