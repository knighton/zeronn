#ifndef ZERONN_NN_BATCHNORM_H_
#define ZERONN_NN_BATCHNORM_H_

#include "zeronn/nn/layer.h"
#include "zeronn/tensor.h"

typedef struct {
    layer_type_id_t type;
    int dim;
    float mom;
    float eps;
    int ndim;
    tensor_t* gamma;
    tensor_t* dgamma;
    tensor_t* beta;
    tensor_t* dbeta;
    tensor_t* mov_mean;
    tensor_t* mov_std;
    tensor_t* x;
} batchnorm_t;

void batchnorm_init(layer_t* layer, int dim, float mom, float eps, int ndim);
layer_t* batchnorm(int dim, float mom, float eps, int ndim);
#define batchnorm0d(...) batchnorm(__VA_ARGS__, 0)
#define batchnorm1d(...) batchnorm(__VA_ARGS__, 1)
#define batchnorm2d(...) batchnorm(__VA_ARGS__, 2)
#define batchnorm3d(...) batchnorm(__VA_ARGS__, 3)
#define batchnorm4d(...) batchnorm(__VA_ARGS__, 4)
void batchnorm_free(layer_t* layer);
void batchnorm_zero_grad(layer_t* layer);
tensor_t* batchnorm_forward(layer_t* layer, tensor_t* x, bool is_t);
tensor_t* batchnorm_backward(layer_t* layer, tensor_t* dy);
void batchnorm_update_step(layer_t* layer, float lr);

#endif  /* ZERONN_NN_BATCHNORM_H_ */
