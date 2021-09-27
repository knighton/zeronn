#ifndef ZERONN_TYPE_BIGINT_H_
#define ZERONN_TYPE_BIGINT_H_

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    BIGINT_SIGN_ZERO,
    BIGINT_SIGN_POS,
    BIGINT_SIGN_NEG,
} bigint_sign_t;

#define BIGINT_MAG_SIZE 3

typedef struct {
    uint8_t sign;
    uint8_t mag[BIGINT_MAG_SIZE];
} bigint_t;

bigint_t bigint_from_int(int32_t a);
int32_t int_from_bigint(bigint_t a);
bigint_t bigint_zero(void);
bigint_t bigint_one(void);
bigint_t bigint_from_bool(bool a);
bool bool_from_bigint(bigint_t a);
bigint_t bigint_canonicalize(bigint_t a);

typedef enum {
    BIGINT_CMP_LT,
    BIGINT_CMP_EQ,
    BIGINT_CMP_GT,
} bigint_cmp_t;

bigint_cmp_t bigint_cmp(bigint_t a, bigint_t b);
bool bigint_lt(bigint_t a, bigint_t b);
bool bigint_le(bigint_t a, bigint_t b);
bool bigint_eq(bigint_t a, bigint_t b);
bool bigint_ne(bigint_t a, bigint_t b);
bool bigint_ge(bigint_t a, bigint_t b);
bool bigint_gt(bigint_t a, bigint_t b);

bigint_t bigint_add(bigint_t a, bigint_t b);
bigint_t bigint_sub(bigint_t a, bigint_t b);
bigint_t bigint_incr(bigint_t a);
bigint_t bigint_decr(bigint_t a);

bigint_t bigint_mul(bigint_t a, bigint_t b);
bigint_t bigint_mul_log2(bigint_t a, uint8_t b);
bigint_t bigint_div_log2(bigint_t a, uint8_t b);

#endif /* ZERONN_TYPE_BIGINT_H_ */
