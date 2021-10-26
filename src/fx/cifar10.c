#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "zeronn/zeronn.h"

#define DROP (1 * FX_ONE / 2)
#define MOM (7 * FX_ONE / 8)
#define EPS (1 * FX_ONE / 128)

typedef struct {
    int size;
    int channels;
    int height;
    int width;
    int classes;
    uint8_t* x;
    uint8_t* y;
} dataset_t;

void load(dataset_t* data, const char* p) {
    FILE* f = fopen(p, "r");
    assert(f);

    fseek(f, 0, SEEK_END);
    int fsize = ftell(f);
    rewind(f);

    data->channels = 3;
    data->height = 32;
    data->width = 32;
    data->classes = 10;
    int image_dim = data->channels * data->height * data->width;
    int sample_dim = 1 + image_dim;
    assert(fsize % sample_dim == 0);

    int n = fsize / sample_dim;
    data->size = n;
    data->x = (uint8_t*)malloc(n * image_dim * sizeof(uint8_t));
    data->y = (uint8_t*)malloc(n * 1 * sizeof(uint8_t));
    assert(data->x);
    assert(data->y);

    for (int i = 0; i < n; ++i) {
        assert(fread(&data->y[i], 1, 1, f) == 1);
        assert(fread(&data->x[i * image_dim], 1, image_dim, f) ==
               (unsigned)image_dim);
    }
}

void sample(dataset_t* data, tensor_t* x, int* y) {
    int n = x->shape[0];
    int d = x->size / n;
    for (int i = 0; i < n; ++i) {
        int s = prng_randint(0, data->size - 1);
        for (int j = 0; j < d; ++j) {
            x->data[i * d + j] = fx_div(data->x[s * d + j] * FX_ONE, 255);
        }
        y[i] = data->y[s];
    }
}

layer_t* cblock(int in_c, int out_c) {
    return sequence(
        relu(),
        conv2d(in_c, out_c, 3, 2, 1),
        batchnorm2d(out_c, MOM, EPS)
    );
}

layer_t* dblock(int in_d, int out_d) {
    return sequence(
        relu(),
        linear(in_d, out_d),
        batchnorm0d(out_d, MOM, EPS)
    );
}

layer_t* model(int in_c, int c, int out_d) {
    return sequence(
        conv2d(in_c, c, 3, 1, 1),
        batchnorm2d(c, MOM, EPS),
        cblock(c, c),
        cblock(c, c),
        cblock(c, c),
        cblock(c, c),
        flatten(),
        dblock(c * 4, c * 2),
        dblock(c * 2, c),
        relu(),
        linear(c, out_d),
        softmax()
    );
}

tensor_t* loss(tensor_t* y_pred, int* y_gold) {
    int batch_size = y_pred->shape[0];
    int dim = y_pred->shape[1];
    tensor_t* dy = tensor_clone(y_pred);
    for (int i = 0; i < batch_size; ++i) {
        dy->data[i * dim + y_gold[i]] -= FX_ONE;
    }
    for (int i = 0; i < dy->size; ++i) {
        dy->data[i] = fx_div(dy->data[i], dim);
    }
    return dy;
}

int accuracy(tensor_t* y_pred, int* y_gold) {
    int batch_size = y_pred->shape[0];
    int dim = y_pred->shape[1];
    int r = 0;
    for (int i = 0; i < batch_size; ++i) {
        int j_max = 0;
        fx_t y_max = y_pred->data[i * dim];
        for (int j = 1; j < dim; ++j) {
            fx_t y = y_pred->data[i * dim + j];
            if (y_max < y) {
                j_max = j;
                y_max = y;
            }
        }
        if (j_max == y_gold[i]) {
            ++r;
        }
    }
    return r;
}

int main(void) {
    const char* t_path = "data/cifar10/train.bin";
    const char* v_path = "data/cifar10/val.bin";
    int num_epochs = 20;
    int laps_per_epoch = 50;
    int train_per_lap = 4;
    int val_per_lap = 1;
    int batch_size = 10;
    int model_dim = 32;
    fx_t lr = fx_from_fp(1.0 / 256);

    dataset_t t_data;
    dataset_t v_data;
    load(&t_data, t_path);
    load(&v_data, v_path);

    layer_t* f = model(t_data.channels, model_dim, t_data.classes);
    tensor_t* x = zeros(batch_size, t_data.channels, t_data.height,
                        t_data.width);
    int* y = (int*)malloc(batch_size * sizeof(int));

    int ts_per_epoch = laps_per_epoch * train_per_lap * batch_size;
    int vs_per_epoch = laps_per_epoch * val_per_lap * batch_size;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        int ta = 0;
        int va = 0;
        for (int lap = 0; lap < laps_per_epoch; ++lap) {
            for (int batch = 0; batch < train_per_lap; ++batch) {
                sample(&t_data, x, y);
                layer_zero_grad(f);
                tensor_t* y_pred = layer_forward(f, x, true);
                tensor_t* dy = loss(y_pred, y);
                tensor_t* dx = layer_backward(f, dy);
                layer_update_step(f, lr);
                ta += accuracy(y_pred, y);
                tensor_free(dx);
                free(dx);
                tensor_free(dy);
                free(dy);
                tensor_free(y_pred);
                free(y_pred);
            }
            for (int batch = 0; batch < val_per_lap; ++batch) {
                sample(&v_data, x, y);
                tensor_t* y_pred = layer_forward(f, x, true);
                va += accuracy(y_pred, y);
                tensor_free(y_pred);
                free(y_pred);
            }
        }
        int t = (ta * 10000) / ts_per_epoch;
        int v = (va * 10000) / vs_per_epoch;
        printf("%3d %3d.%03d %3d.%03d\n", epoch, t / 100, t % 100, v / 100,
               v % 100);
    }

    free(y);
    tensor_free(x);
    free(x);
    layer_free(f);
    free(f);
}
