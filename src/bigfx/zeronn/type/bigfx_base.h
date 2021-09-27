#ifndef ZERONN_TYPE_BIGFX_BASE_H_
#define ZERONN_TYPE_BIGFX_BASE_H_

#include <stdbool.h>
#include <stdint.h>

#include "zeronn/type/bigint.h"

#define SQRT_FX_BITS 5
#define FX_BITS (SQRT_FX_BITS * 2)
#define SQRT_FX_ONE (1 << SQRT_FX_BITS)
#define FX_ONE (1 << FX_BITS)

typedef enum {
    BIGFX_SIGN_ZERO,
    BIGFX_SIGN_POS,
    BIGFX_SIGN_NEG,
} bigfx_sign_t;

#define BIGFX_MAG_SIZE 3

#define bigfx_t bigint_t

bigfx_t bigfx_from_int(int32_t a);
int32_t int_from_bigfx(bigfx_t a);
bigfx_t bigfx_from_fx(int32_t a);
int32_t fx_from_bigfx(bigfx_t a);
bigfx_t bigfx_zero(void);
bigfx_t bigfx_one(void);
bigfx_t bigfx_from_bool(bool a);
bool bool_from_bigfx(bigfx_t x);
bigfx_t bigfx_from_float(float a);
float float_from_bigfx(bigfx_t a);

typedef enum {
    BIGFX_CMP_LT,
    BIGFX_CMP_EQ,
    BIGFX_CMP_GT,
} bigfx_cmp_t;

bigfx_cmp_t bigfx_cmp(bigfx_t a, bigfx_t b);
bool bigfx_lt(bigfx_t a, bigfx_t b);
bool bigfx_le(bigfx_t a, bigfx_t b);
bool bigfx_eq(bigfx_t a, bigfx_t b);
bool bigfx_ne(bigfx_t a, bigfx_t b);
bool bigfx_ge(bigfx_t a, bigfx_t b);
bool bigfx_gt(bigfx_t a, bigfx_t b);

#endif /* ZERONN_TYPE_BIGFX_BASE_H_ */
