#include "tensor.h"

#include <assert.h>
#include <stdlib.h>

#include "zeronn/prng.h"

void tensor_init_common(tensor_t* x, int num_args, va_list* args) {
    assert(num_args <= NDIM_MAX);
    x->size = 1;
    x->ndim = num_args;
    for (int i = 0; i < num_args; ++i) {
        int arg = va_arg(*args, int);
        x->size *= arg;
        x->shape[i] = arg;
    }
    x->data = (fx_t*)malloc(x->size * sizeof(fx_t));
}

void pp_tensor_init(tensor_t* x, int num_args, ...) {
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
}

tensor_t* empty(void) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    x->size = 0;
    x->ndim = 0;
    x->data = NULL;
    return x;
}

tensor_t* pp_tensor(int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    return x;
}

void tensor_free(tensor_t* x) {
    if (x->data) {
        free(x->data);
    }
}

void tensor_fill(tensor_t* x, fx_t a) {
    for (int i = 0; i < x->size; ++i) {
        x->data[i] = a;
    }
}

tensor_t* pp_full(fx_t a, int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    tensor_fill(x, a);
    return x;
}

void tensor_arange(tensor_t* x) {
    for (int i = 0; i < x->size; ++i) {
        x->data[i] = i;
    }
}

tensor_t* pp_arange(int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    tensor_arange(x);
    return x;
}

void tensor_uniform(tensor_t* x, fx_t low, fx_t high) {
    for (int i = 0; i < x->size; ++i) {
        x->data[i] = prng_randint(low, high);
    }
}

tensor_t* pp_uniform(fx_t low, fx_t high, int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    tensor_uniform(x, low, high);
    return x;
}

void tensor_set(tensor_t* x, tensor_t* a) {
    if (x->size != a->size) {
        x->size = a->size;
        if (x->data) {
            free(x->data);
        }
        x->data = (fx_t*)malloc(a->size * sizeof(fx_t));
    }
    x->ndim = a->ndim;
    for (int i = 0; i < a->ndim; ++i) {
        x->shape[i] = a->shape[i];
    }
    for (int i = 0; i < a->size; ++i) {
        x->data[i] = a->data[i];
    }
}

tensor_t* tensor_clone(tensor_t* a) {
    tensor_t* x = empty();
    tensor_set(x, a);
    return x;
}

void tensor_update_step(tensor_t* x, tensor_t* dx, fx_t lr) {
    assert(x->size == dx->size);
    for (int i = 0; i < x->size; ++i) {
        x->data[i] -= fx_div(lr * dx->data[i], FX_ONE);
    }
}

void tensor_shape(tensor_t* t, shape_t* s) {
    s->size = t->size;
    s->ndim = t->ndim;
    for (int i = 0; i < t->ndim; ++i) {
        s->shape[i] = t->shape[i];
    }
}

tensor_t* tensor_reshape(tensor_t* x, shape_t* s) {
    tensor_t* y = tensor_clone(x);
    if (0 < s->size) {
        assert(y->size == s->size);
        y->ndim = s->ndim;
        for (int i = 0; i < s->ndim; ++i) {
            int dim = s->shape[i];
            assert(1 <= dim);
            y->shape[i] = dim;
        }
    } else {
        assert(s->size < 0);
        int wildcard_size = -s->size;
        assert(y->size % wildcard_size == 0);
        int wildcard_dim = y->size / wildcard_size;
        y->ndim = s->ndim;
        for (int i = 0; i < s->ndim; ++i) {
            int dim = s->shape[i];
            if (dim == -1) {
                dim = wildcard_dim;
            } else {
                assert(1 <= dim);
            }
            y->shape[i] = dim;
        }
    }
    return y;
}
