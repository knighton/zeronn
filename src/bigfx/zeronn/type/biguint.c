#include "biguint.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "zeronn/type/big_base.h"

biguint_t biguint_from_uint(uint32_t a) {
    biguint_t r;
    for (uint8_t i = BIGUINT_MAG_SIZE - 1; i < BIGUINT_MAG_SIZE; --i) {
        r.mag[i] = a & 0xFF;
        a >>= 8;
    }
    return r;
}

uint32_t uint_from_biguint(biguint_t a) {
    uint32_t r = 0;
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        r <<= 8;
        r |= (uint32_t)a.mag[i];
    }
    return r;
}

biguint_t biguint_zero(void) {
    biguint_t r;
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        r.mag[i] = 0;
    }
    return r;
}

biguint_t biguint_one(void) {
    biguint_t r;
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        r.mag[i] = 0;
    }
    r.mag[BIGUINT_MAG_SIZE - 1] = 1;
    return r;
}

biguint_t biguint_from_bool(bool a) {
    if (a) {
        return biguint_one();
    } else {
        return biguint_zero();
    }
}

bool bool_from_biguint(biguint_t a) {
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        if (a.mag[i]) {
            return true;
        }
    }
    return false;
}

biguint_cmp_t biguint_cmp(biguint_t a, biguint_t b) {
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        if (a.mag[i] != b.mag[i]) {
            if (a.mag[i] < b.mag[i]) {
                return BIGUINT_CMP_LT;
            } else {
                return BIGUINT_CMP_GT;
            }
        }
    }
    return BIGUINT_CMP_EQ;
}

bool biguint_lt(biguint_t a, biguint_t b) {
    return biguint_cmp(a, b) == BIGUINT_CMP_LT;
}

bool biguint_le(biguint_t a, biguint_t b) {
    return biguint_cmp(a, b) != BIGUINT_CMP_GT;
}

bool biguint_eq(biguint_t a, biguint_t b) {
    return biguint_cmp(a, b) == BIGUINT_CMP_EQ;
}

bool biguint_ne(biguint_t a, biguint_t b) {
    return biguint_cmp(a, b) != BIGUINT_CMP_EQ;
}

bool biguint_ge(biguint_t a, biguint_t b) {
    return biguint_cmp(a, b) != BIGUINT_CMP_LT;
}

bool biguint_gt(biguint_t a, biguint_t b) {
    return biguint_cmp(a, b) == BIGUINT_CMP_GT;
}

biguint_t biguint_add(biguint_t a, biguint_t b) {
    biguint_t r;
    uint8_t a_sum;
    uint8_t b_sum;
    uint8_t carry = 0;
    for (uint8_t i = BIGUINT_MAG_SIZE - 1; i < BIGUINT_MAG_SIZE; --i) {
        a_sum = a.mag[i] + carry;
        carry = a_sum < a.mag[i];
        b_sum = a_sum + b.mag[i];
        carry += b_sum < a_sum;
        r.mag[i] = b_sum;
    }
    return r;
}

biguint_t biguint_sub(biguint_t a, biguint_t b) {
    biguint_t r;
    uint8_t a_diff;
    uint8_t b_diff;
    uint8_t borrow = 0;
    for (uint8_t i = BIGUINT_MAG_SIZE - 1; i < BIGUINT_MAG_SIZE; --i) {
        a_diff = a.mag[i] - borrow;
        borrow = a.mag[i] < a_diff;
        b_diff = a_diff - b.mag[i];
        borrow += a_diff < b_diff;
        r.mag[i] = b_diff;
    }
    return r;
}

biguint_t biguint_incr(biguint_t a) {
    biguint_t b = biguint_one();
    return biguint_add(a, b);
}

biguint_t biguint_decr(biguint_t a) {
    biguint_t b = biguint_one();
    return biguint_sub(a, b);
}

biguint_t biguint_mul(biguint_t a, biguint_t b) {
    biguint_t r;
    uint8_t i;
    uint8_t j;
    uint8_t k;
    uint8_t prod_hi;
    uint8_t prod_lo;
    uint8_t carry;

    for (i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        r.mag[i] = 0;
    }

    for (i = BIGUINT_MAG_SIZE - 1; i < BIGUINT_MAG_SIZE; --i) {
        if (a.mag[i] == 0) {
            continue;
        }

        carry = 0;
        for (j = BIGUINT_MAG_SIZE - 1; j < BIGUINT_MAG_SIZE; --j) {
            mul_8_8_16(a.mag[i], b.mag[j], &prod_hi, &prod_lo);

            prod_lo += carry;
            carry = prod_lo < carry;

            k = i + j - BIGUINT_MAG_SIZE + 1;
            r.mag[k] += prod_lo;
            carry += r.mag[k] < prod_lo;

            carry += prod_hi;
        }
    }

    return r;
}

biguint_t biguint_mul_log2(biguint_t a, uint8_t b) {
    biguint_t r = a;
    shift_left(BIGUINT_MAG_SIZE, r.mag, b);
    return r;
}

biguint_t biguint_div_log2(biguint_t a, uint8_t b) {
    biguint_t r = a;
    shift_right(BIGUINT_MAG_SIZE, r.mag, b);
    return r;
}

biguint_t biguint_mod_log2(biguint_t a, uint8_t b) {
    biguint_t r;
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        uint8_t idx = BIGUINT_MAG_SIZE - 1 - i;
        if (b < 8) {
            r.mag[idx] = a.mag[idx] % (1 << b);
            b = 0;
        } else {
            r.mag[idx] = a.mag[idx];
            b -= 8;
        }
    }
    return r;
}

biguint_t biguint_xor(biguint_t a, biguint_t b) {
    biguint_t r;
    for (uint8_t i = 0; i < BIGUINT_MAG_SIZE; ++i) {
        r.mag[i] = a.mag[i] ^ b.mag[i];
    }
    return r;
}
