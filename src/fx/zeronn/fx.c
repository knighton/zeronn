#include "fx.h"

#include <assert.h>
#include <stdio.h>

#include "zeronn/prng.h"

float fp_from_fx(fx_t x) {
    return (float)x / FX_ONE;
}

fx_t fx_from_fp(float x) {
    return (int32_t)(x * FX_ONE);
}

fx_t fx_div(fx_t a, fx_t b) {
    fx_t r = a / b;
    if ((0 <= a) != (0 <= b)) {
        --r;
        if (-(a % b) <= (prng_get() % b)) {
            ++r;
        }
    } else {
        if ((prng_get() % b) < (a % b)) {
            ++r;
        }
    }
    return r;
}

int32_t i_sqrt(int32_t x) {
    if (x <= 0) {
        return 0;
    }

    int32_t r = 0;
    int32_t a = 1 << 30;

    while (x < a) {
        a >>= 2;
    }

    while (a != 0) {
        if (r + a <= x) {
            x -= r + a;
            r = (r >> 1) + a;
        } else {
            r >>= 1;
        }
        a >>= 2;
    }

    return r;
}

fx_t fx_sqrt(fx_t x) {
    if (x <= 0) {
        return 0;
    }

    return i_sqrt(x) * FX_ONE_SQRT;
}

fx_t fx_sq(fx_t x) {
    return fx_div(x * x, FX_ONE);
}

fx_t fx_exp(fx_t x) {
    x = FX_ONE + fx_div(x, 64);
    for (int i = 0; i < 6; ++i) {
        x = fx_div(x * x, FX_ONE);
    }
    return x;
}
