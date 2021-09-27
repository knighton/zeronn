#include "bigfx_base.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define BIGFX_FROM_FX_LIMIT ((1L << MIN(BIGFX_MAG_SIZE * 8, 31)) - 1)
#define BIGFX_FROM_INT_LIMIT ((1 << ((BIGFX_MAG_SIZE * 8) - FX_BITS)) - 1)
#define BIGFX_FROM_FLOAT_LIMIT ((1 << ((BIGFX_MAG_SIZE * 8) - FX_BITS)) - 1)

bigfx_t bigfx_from_fx(int32_t a) {
    if (a <= -BIGFX_FROM_FX_LIMIT) {
        a = -BIGFX_FROM_FX_LIMIT;
    } else if (BIGFX_FROM_FX_LIMIT <= a) {
        a = BIGFX_FROM_FX_LIMIT;
    }
    return bigint_from_int(a);
}

int32_t fx_from_bigfx(bigfx_t a) {
    return int_from_bigint(a);
}

bigfx_t bigfx_from_int(int32_t a) {
    if (a < -BIGFX_FROM_INT_LIMIT) {
        a = -BIGFX_FROM_INT_LIMIT;
    } else if (BIGFX_FROM_INT_LIMIT < a) {
        a = BIGFX_FROM_INT_LIMIT;
    }
    return bigint_from_int(a * FX_ONE);
}

int32_t int_from_bigfx(bigfx_t a) {
    return int_from_bigint(a) / FX_ONE;
}

bigfx_t bigfx_zero(void) {
    return bigint_zero();
}

bigfx_t bigfx_one(void) {
    return bigint_from_int(FX_ONE);
}

bigfx_t bigfx_from_bool(bool a) {
    if (a) {
        return bigfx_one();
    } else {
        return bigfx_zero();
    }
}

bool bool_from_bigfx(bigfx_t a) {
    return a.sign;
}

float float_from_bigfx(bigfx_t a) {
    int32_t x = int_from_bigint(a);
    return (float)x / FX_ONE;
}

bigfx_t bigfx_from_float(float a) {
    if (a < -BIGFX_FROM_FLOAT_LIMIT) {
        a = -BIGFX_FROM_FLOAT_LIMIT;
    } else if (BIGFX_FROM_FLOAT_LIMIT < a) {
        a = BIGFX_FROM_FLOAT_LIMIT;
    }
    int32_t x = (int32_t)(a * FX_ONE);
    return bigfx_from_fx(x);
}

bigfx_cmp_t bigfx_cmp(bigfx_t a, bigfx_t b) {
    return bigint_cmp(a, b);
}

bool bigfx_lt(bigfx_t a, bigfx_t b) {
    return bigint_lt(a, b);
}

bool bigfx_le(bigfx_t a, bigfx_t b) {
    return bigint_le(a, b);
}

bool bigfx_eq(bigfx_t a, bigfx_t b) {
    return bigint_eq(a, b);
}

bool bigfx_ne(bigfx_t a, bigfx_t b) {
    return bigint_ne(a, b);
}

bool bigfx_ge(bigfx_t a, bigfx_t b) {
    return bigint_ge(a, b);
}

bool bigfx_gt(bigfx_t a, bigfx_t b) {
    return bigint_gt(a, b);
}
