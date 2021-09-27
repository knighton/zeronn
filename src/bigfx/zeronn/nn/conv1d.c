#include "conv1d.h"

#include <assert.h>
#include <stdlib.h>

void conv1d_init(layer_t* layer, int x_channels, int y_channels, int face,
                 int stride, int pad) {
    assert(0 < x_channels);
    assert(0 < y_channels);
    conv1d_t* f = (conv1d_t*)layer;
    f->type = CONV1D;
    f->x_channels = x_channels;
    f->y_channels = y_channels;
    coord1d_init(face, &f->face, 1);
    coord1d_init(stride, &f->stride, 1);
    coord1d_init(pad, &f->pad, 0);
    f->weight = uniform_log2(bigfx_from_fx(-(1 << 7)), 7 + 1,
                             x_channels, y_channels, f->face.width);
    f->dweight = zeros(x_channels, y_channels, f->face.width);
    f->bias = uniform_log2(bigfx_from_fx(-(1 << 7)), 7 + 1, y_channels);
    f->dbias = zeros(y_channels);
    f->x = empty();
}

layer_t* conv1d(int x_channels, int y_channels, int face, int stride,
                 int pad) {
    layer_t* layer = (layer_t*)malloc(sizeof(conv1d_t));
    conv1d_init(layer, x_channels, y_channels, face, stride, pad);
    return layer;
}

void conv1d_free(layer_t* layer) {
    conv1d_t* f = (conv1d_t*)layer;
    tensor_free(f->weight);
    tensor_free(f->dweight);
    tensor_free(f->bias);
    tensor_free(f->dbias);
    tensor_free(f->x);
    free(f->weight);
    free(f->dweight);
    free(f->bias);
    free(f->dbias);
    free(f->x);
}

void conv1d_zero_grad(layer_t* layer) {
    conv1d_t* f = (conv1d_t*)layer;
    tensor_zero(f->dweight);
    tensor_zero(f->dbias);
}

tensor_t* conv1d_forward(layer_t* layer, tensor_t* x, bool is_t) {
    conv1d_t* f = (conv1d_t*)layer;
    if (is_t) {
        tensor_set(f->x, x);
    }

    int batch_size = x->shape[0];
    assert(f->x_channels == x->shape[1]);
    int x_width = x->shape[2];
    int y_width = (x_width + 2 * f->pad.width - f->face.width) /
        f->stride.width + 1;
    tensor_t* y = zeros(batch_size, f->y_channels, y_width);

    for (int y_w = 0, x_w_offset = -f->pad.width; y_w < y_width;
            ++y_w, x_w_offset += f->stride.width) {
        for (int face_w = 0; face_w < f->face.width; ++face_w) {
            int x_w = x_w_offset + face_w;
            if (x_w < 0 || x_width <= x_w) {
                continue;
            }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            bigfx_t y_val = bigfx_zero();
            for (int x_c = 0; x_c < f->x_channels; ++x_c) {
                int x_idx = (n * f->x_channels + x_c) * x_width + x_w;
                int w_idx = (x_c * f->y_channels + y_c) * f->face.width +
                    face_w;
                y_val = bigfx_add(y_val, bigint_mul(x->data[x_idx],
                                                    f->weight->data[w_idx]));
            }
            int y_idx = (n * f->y_channels + y_c) * y_width + y_w;
            y->data[y_idx] = bigfx_add(y->data[y_idx], y_val);
        }
    }

        }
    }

    for (int i = 0; i < y->size; ++i) {
        y->data[i] = bigfx_div_log2(y->data[i], FX_BITS);
    }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            for (int y_w = 0; y_w < y_width; ++y_w) {
                int y_idx = (n * f->y_channels + y_c) * y_width + y_w;
                y->data[y_idx] = bigfx_add(y->data[y_idx], f->bias->data[y_c]);
            }
        }
    }

    return y;
}

tensor_t* conv1d_backward(layer_t* layer, tensor_t* dy) {
    conv1d_t* f = (conv1d_t*)layer;

    int batch_size = dy->shape[0];
    int x_width = f->x->shape[2];
    assert(f->y_channels == dy->shape[1]);
    int y_width = dy->shape[2];
    tensor_t* dx = zeros(batch_size, f->x_channels, x_width);
    tensor_t* dweight = zeros(f->x_channels, f->y_channels, f->face.width);

    for (int y_w = 0, x_w_offset = -f->pad.width; y_w < y_width;
            ++y_w, x_w_offset += f->stride.width) {
        for (int face_w = 0; face_w < f->face.width; ++face_w) {
            int x_w = x_w_offset + face_w;
            if (x_w < 0 || x_width <= x_w) {
                continue;
            }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            int y_idx = (n * f->y_channels + y_c) * y_width + y_w;
            bigfx_t dy_val = dy->data[y_idx];
            for (int x_c = 0; x_c < f->x_channels; ++x_c) {
                int x_idx = (n * f->x_channels + x_c) * x_width + x_w;
                int w_idx = (x_c * f->y_channels + y_c) * f->face.width +
                    face_w;
                dweight->data[w_idx] = bigfx_add(
                    dweight->data[w_idx], bigint_mul(dy_val, f->x->data[x_idx]));
                dx->data[x_idx] = bigfx_add(
                    dx->data[x_idx], bigint_mul(dy_val, f->weight->data[w_idx]));
            }
        }
    }

        }
    }

    for (int i = 0; i < dweight->size; ++i) {
        f->dweight->data[i] = bigfx_add(
            f->dweight->data[i], bigfx_div_log2(dweight->data[i], FX_BITS));
    }
    for (int i = 0; i < dx->size; ++i) {
        dx->data[i] = bigfx_div_log2(dx->data[i], FX_BITS);
    }
    tensor_free(dweight);
    free(dweight);

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            for (int y_w = 0; y_w < y_width; ++y_w) {
                int y_idx = (n * f->y_channels + y_c) * y_width + y_w;
                f->dbias->data[y_c] = bigfx_add(f->dbias->data[y_c],
                                                dy->data[y_idx]);
            }
        }
    }

    return dx;
}

void conv1d_update_step(layer_t* layer, bigfx_t lr) {
    conv1d_t* f = (conv1d_t*)layer;
    tensor_update_step(f->weight, f->dweight, lr);
    tensor_update_step(f->bias, f->dbias, lr);
}
