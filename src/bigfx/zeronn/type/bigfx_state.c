#include "bigfx_state.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "bigfx_base.h"

#define BFX_COLS 8

bigfx_state_t BFX_STATE = {
    false,
    NULL,
    {0, {0, 0, 0}},
    {0, {0, 0, 0}},
};

void bigfx_state_init(void) {
    if (BFX_STATE.has_init) {
        return;
    }

    int size = BIGFX_MAG_SIZE * 8 * 256 * BFX_COLS * sizeof(bigfx_t);
    BFX_STATE.table = (bigfx_t*)malloc(size);
    assert(BFX_STATE.table);
    for (int i = 0; i < BIGFX_MAG_SIZE * 8; ++i) {
        float unit = (float)(1 << i) / (1 << FX_BITS);
        for (int j = 0; j < 256; ++j) {
            float x = (1 + j / 256.0) * unit;
            int row = i * 256 + j;
            BFX_STATE.table[row * BFX_COLS] = bigfx_from_float(x);
            BFX_STATE.table[row * BFX_COLS + 1] = bigfx_from_float(exp(-x));
            BFX_STATE.table[row * BFX_COLS + 2] = bigfx_from_float(exp(x));
            BFX_STATE.table[row * BFX_COLS + 3] = bigfx_from_float(log(x));
            BFX_STATE.table[row * BFX_COLS + 4] = bigfx_from_float(1 / x);
            BFX_STATE.table[row * BFX_COLS + 5] = bigfx_from_float(x * x);
            BFX_STATE.table[row * BFX_COLS + 6] = bigfx_from_float(sqrt(x));
            BFX_STATE.table[row * BFX_COLS + 7] = bigfx_from_float(tanh(x));
        }
    }

    BFX_STATE.recip_log2 = bigfx_from_float(1 / log(2));
    BFX_STATE.recip_log10 = bigfx_from_float(1 / log(10));

    BFX_STATE.has_init = true;
}

int row_from_mag(bigfx_t a) {
    int i = int_from_bigint(a);
    if (i < 0) {
        i = -i;
    }

    int log2 = 0;
    while (i) {
        ++log2;
        i >>= 1;
    }
    --log2;

    i = int_from_bigint(a);
    if (i < 0) {
        i = -i;
    }
    int bits;
    if (log2 < 8) {
        bits = i << (8 - log2);
    } else {
        bits = i >> (log2 - 8);
    }
    bits %= 256;

    return log2 * 256 + bits;
}

bigfx_t bigfx_state_get(bigfx_t a, uint8_t col) {
    bigfx_state_init();
    int row = row_from_mag(a);
    return BFX_STATE.table[row * BFX_COLS + col];
}

bigfx_t bigfx_exp(bigfx_t a) {
    if (a.sign == BIGINT_SIGN_ZERO) {
        return bigfx_one();
    }
    uint8_t col = a.sign == BIGINT_SIGN_NEG ? 1 : 2;
    return bigfx_state_get(a, col);
}

bigfx_t bigfx_log(bigfx_t a) {
    assert(a.sign == BIGINT_SIGN_POS);
    uint8_t col = 3;
    return bigfx_state_get(a, col);
}

bigfx_t bigfx_recip(bigfx_t a) {
    assert(a.sign != BIGINT_SIGN_ZERO);
    uint8_t col = 4;
    bigint_t r = bigfx_state_get(a, col);
    r.sign = a.sign;
    return r;
}

bigfx_t bigfx_sq(bigfx_t a) {
    if (a.sign == BIGINT_SIGN_ZERO) {
        return a;
    }
    uint8_t col = 5;
    return bigfx_state_get(a, col);
}

bigfx_t bigfx_sqrt(bigfx_t a) {
    if (a.sign == BIGINT_SIGN_NEG) {
        assert(false);
    } else if (a.sign == BIGINT_SIGN_ZERO) {
        return a;
    }
    uint8_t col = 6;
    return bigfx_state_get(a, col);
}

bigfx_t bigfx_tanh(bigfx_t a) {
    if (a.sign == BIGINT_SIGN_ZERO) {
        return a;
    }
    uint8_t col = 7;
    bigint_t r = bigfx_state_get(a, col);
    r.sign = a.sign;
    return r;
}
