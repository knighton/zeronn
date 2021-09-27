#ifndef ZERONN_NN_SEQUENCE_H_
#define ZERONN_NN_SEQUENCE_H_

#include "zeronn/nn/layer.h"
#include "zeronn/preproc.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    int num_layers;
    layer_t** layers;
} sequence_t;

void sequence_init(layer_t* layer, int num_layers, layer_t** layers);
layer_t* pp_sequence(int num_layers, ...);
#define sequence(...) pp_sequence(PP_NARG(__VA_ARGS__), __VA_ARGS__)
void sequence_free(layer_t* layer);
void sequence_zero_grad(layer_t* layer);
tensor_t* sequence_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* sequence_backward(layer_t* layer, tensor_t* dy);
void sequence_update_step(layer_t* layer, bigfx_t lr);

#endif  /* ZERONN_NN_SEQUENCE_H_ */
