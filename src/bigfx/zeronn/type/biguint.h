#ifndef ZERONN_TYPE_BIGUINT_H_
#define ZERONN_TYPE_BIGUINT_H_

#include <stdbool.h>
#include <stdint.h>

#define BIGUINT_MAG_SIZE 4

typedef struct {
    uint8_t mag[BIGUINT_MAG_SIZE];
} biguint_t;

biguint_t biguint_from_uint(uint32_t a);
uint32_t uint_from_biguint(biguint_t a);
biguint_t biguint_zero(void);
biguint_t biguint_one(void);
biguint_t biguint_from_bool(bool a);
bool bool_from_biguint(biguint_t x);

typedef enum {
    BIGUINT_CMP_LT,
    BIGUINT_CMP_EQ,
    BIGUINT_CMP_GT,
} biguint_cmp_t;

biguint_cmp_t biguint_cmp(biguint_t a, biguint_t b);
bool biguint_lt(biguint_t a, biguint_t b);
bool biguint_le(biguint_t a, biguint_t b);
bool biguint_eq(biguint_t a, biguint_t b);
bool biguint_ne(biguint_t a, biguint_t b);
bool biguint_ge(biguint_t a, biguint_t b);
bool biguint_gt(biguint_t a, biguint_t b);

biguint_t biguint_add(biguint_t a, biguint_t b);
biguint_t biguint_sub(biguint_t a, biguint_t b);
biguint_t biguint_incr(biguint_t a);
biguint_t biguint_decr(biguint_t a);

biguint_t biguint_mul(biguint_t a, biguint_t b);
biguint_t biguint_mul_log2(biguint_t a, uint8_t b);
biguint_t biguint_div_log2(biguint_t a, uint8_t b);

biguint_t biguint_mod_log2(biguint_t a, uint8_t b);

biguint_t biguint_xor(biguint_t a, biguint_t b);

#endif /* ZERONN_TYPE_BIGUINT_H_ */
