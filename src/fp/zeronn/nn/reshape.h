#ifndef ZERONN_NN_RESHAPE_H_
#define ZERONN_NN_RESHAPE_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    shape_t y;
    shape_t x;
} reshape_t;

void pp_reshape_init(layer_t* layer, int num_args, ...);
#define reshape_init(layer, ...) \
    pp_reshape_init(layer, PP_NARG(__VA_ARGS__), __VA_ARGS__)
layer_t* pp_reshape(int num_args, ...);
#define reshape(...) pp_reshape(PP_NARG(__VA_ARGS__), __VA_ARGS__)
#define flatten() reshape(-1)
void reshape_free(layer_t* layer);
void reshape_zero_grad(layer_t* layer);
tensor_t* reshape_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* reshape_backward(layer_t* layer, tensor_t* dy);
void reshape_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_RESHAPE_H_ */
