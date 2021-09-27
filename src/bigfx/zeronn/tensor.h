#ifndef ZERONN_TENSOR_H_
#define ZERONN_TENSOR_H_

#include <stdarg.h>
#include <stdint.h>

#include "zeronn/preproc.h"
#include "zeronn/type/bigfx.h"

#define NDIM_MAX 6

typedef struct {
    int size;
    int ndim;
    int shape[NDIM_MAX];
    bigfx_t* data;
} tensor_t;

typedef struct {
    int size;
    int ndim;
    int shape[NDIM_MAX];
} shape_t;

void pp_tensor_init(tensor_t* x, int num_args, ...);
#define tensor_init(x, ...) \
    pp_tensor_init(x, PP_NARG(__VA_ARGS__), __VA_ARGS__)
tensor_t* empty(void);
tensor_t* pp_tensor(int num_args, ...);
#define tensor(...) pp_tensor(PP_NARG(__VA_ARGS__), __VA_ARGS__)
void tensor_free(tensor_t* x);

void tensor_fill(tensor_t* x, bigfx_t a);
#define tensor_one(x) tensor_fill(x, bigfx_one())
#define tensor_zero(x) tensor_fill(x, bigfx_zero())
tensor_t* pp_full(bigfx_t a, int num_args, ...);
#define full(a, ...) pp_full(a, PP_NARG(__VA_ARGS__), __VA_ARGS__)
#define ones(...) full(bigfx_one(), __VA_ARGS__)
#define zeros(...) full(bigfx_zero(), __VA_ARGS__)

void tensor_arange(tensor_t* x);
tensor_t* pp_arange(int num_args, ...);
#define arange(...) pp_arange(PP_NARG(__VA_ARGS__), __VA_ARGS__)

void tensor_uniform_log2(tensor_t* x, bigfx_t low, uint8_t log2_range);
tensor_t* pp_uniform_log2(bigfx_t low, uint8_t log2_range, int num_args, ...);
#define uniform_log2(low, log2_range, ...) \
    pp_uniform_log2(low, log2_range, PP_NARG(__VA_ARGS__), __VA_ARGS__)

void tensor_set(tensor_t* x, tensor_t* a);
tensor_t* tensor_clone(tensor_t* a);

void tensor_update_step(tensor_t* x, tensor_t* dx, bigfx_t lr);

void tensor_shape(tensor_t* t, shape_t* s);
tensor_t* tensor_reshape(tensor_t* t, shape_t* s);

#endif  /* ZERONN_TENSOR_H_ */
