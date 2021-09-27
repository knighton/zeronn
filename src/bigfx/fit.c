#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "zeronn/zeronn.h"

#define DROP 0.5
#define MOM (float)(7.0 / 8)
#define EPS (float)(1.0 / 1024)

layer_t* get_block(int in_dim, int out_dim) {
    return sequence(
        relu(),
        dropout(bigfx_from_float(DROP)),
        linear(in_dim, out_dim),
        batchnorm0d(out_dim, bigfx_from_float(MOM), bigfx_from_float(EPS))
    );
}

layer_t* get_model(int in_dim, int out_dim) {
    int a = 64;
    int b = 32;
    int c = 16;
    return sequence(
        linear(in_dim, a),
        batchnorm0d(a, bigfx_from_float(MOM), bigfx_from_float(EPS)),
        get_block(a, b),
        get_block(b, c),
        relu(),
        linear(c, out_dim),
        softmax()
    );
}

void get_batch(int out_dim, int log2_out_dim, bigfx_t signal, tensor_t* x, int* y_gold) {
    int batch_size = x->shape[0];
    int in_dim = x->shape[1];

    for (int i = 0; i < batch_size; ++i) {
        y_gold[i] = int_from_bigint(prng_randint_log2(bigfx_zero(), log2_out_dim));
    }

    tensor_uniform_log2(x, bigfx_zero(), FX_BITS - int_from_bigint(signal));

    assert(in_dim % out_dim == 0);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = y_gold[i]; j < in_dim; j += out_dim) {
            x->data[i * in_dim + j] = bigfx_add(x->data[i * in_dim + j], signal);
        }
    }
}

tensor_t* get_loss(tensor_t* y_pred, int* y_gold) {
    int batch_size = y_pred->shape[0];
    int dim = y_pred->shape[1];
    tensor_t* dy = tensor_clone(y_pred);
    for (int i = 0; i < batch_size; ++i) {
        int idx = i * dim + y_gold[i];
        dy->data[idx] = bigfx_sub(dy->data[idx], bigfx_one());
    }
    return dy;
}

int get_correct(tensor_t* y_pred, int* y_gold) {
    int batch_size = y_pred->shape[0];
    int dim = y_pred->shape[1];
    int num_correct = 0;
    for (int i = 0; i < batch_size; ++i) {
        int j_max = 0;
        bigfx_t y_max = y_pred->data[i * dim];
        for (int j = 1; j < dim; ++j) {
            bigfx_t y = y_pred->data[i * dim + j];
            if (bigfx_lt(y_max, y)) {
                j_max = j;
                y_max = y;
            }
        }
        if (j_max == y_gold[i]) {
            ++num_correct;
        }
    }
    return num_correct;
}

int main(void) {
    int num_epochs = 10;
    int rounds_per_epoch = 20;
    int train_per_round = 5;
    int eval_per_round = 1;
    int batch_size = 32;
    int in_dim = 128;
    int out_dim = 8;
    int log2_out_dim = 3;
    bigfx_t signal = bigfx_from_float(0.25);
    bigfx_t lr = bigfx_from_float(1.0 / 128);

    layer_t* f = get_model(in_dim, out_dim);

    tensor_t* x = zeros(batch_size, in_dim);
    int* y_gold = (int*)malloc(batch_size * sizeof(int));

    for (int i = 0; i < num_epochs; ++i) {
        int ta = 0;
        int va = 0;
        for (int j = 0; j < rounds_per_epoch; ++j) {
            for (int k = 0; k < train_per_round; ++k) {
                get_batch(out_dim, log2_out_dim, signal, x, y_gold);
                layer_zero_grad(f);
                tensor_t* y_pred = layer_forward(f, x, true);
                tensor_t* dy = get_loss(y_pred, y_gold);
                tensor_t* dx = layer_backward(f, dy);
                layer_update_step(f, lr);
                ta += get_correct(y_pred, y_gold);
                tensor_free(dx);
                free(dx);
                tensor_free(dy);
                free(dy);
                tensor_free(y_pred);
                free(y_pred);
            }
            for (int k = 0; k < eval_per_round; ++k) {
                get_batch(out_dim, log2_out_dim, signal, x, y_gold);
                tensor_t* y_pred = layer_forward(f, x, true);
                va += get_correct(y_pred, y_gold);
                tensor_free(y_pred);
                free(y_pred);
            }
        }
        int t = (ta * 10000) / (rounds_per_epoch * train_per_round *
                                batch_size);
        int v = (va * 10000) / (rounds_per_epoch * eval_per_round *
                                batch_size);
        printf("%7d %3d.%03d %3d.%03d\n", i, t / 100, t % 100, v / 100,
               v % 100);
    }

    free(y_gold);
    tensor_free(x);
    free(x);
    layer_free(f);
    free(f);
}
