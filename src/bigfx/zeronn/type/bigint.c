#include "bigint.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "zeronn/type/big_base.h"

bigint_t bigint_from_int(int32_t a) {
    bigint_t r;

    if (a < 0) {
        r.sign = BIGINT_SIGN_NEG;
        a = -a;
    } else if (a == 0) {
        r.sign = BIGINT_SIGN_ZERO;
    } else {
        r.sign = BIGINT_SIGN_POS;
    }

    for (uint8_t i = BIGINT_MAG_SIZE - 1; i < BIGINT_MAG_SIZE; --i) {
        r.mag[i] = a & 0xFF;
        a >>= 8;
    }

    return r;
}

int32_t int_from_bigint(bigint_t a) {
    int32_t r = 0;
    for (uint8_t i = 0; i < BIGINT_MAG_SIZE; ++i) {
        r <<= 8;
        r |= (int32_t)a.mag[i];
    }
    if (a.sign == BIGINT_SIGN_NEG) {
        r = -r;
    } else if (a.sign == BIGINT_SIGN_ZERO) {
        r = 0;
    } else if (a.sign == BIGINT_SIGN_POS) {
        r &= 0x7FFFFFFF;
    }
    return r;
}

bigint_t bigint_zero(void) {
    bigint_t r;
    r.sign = BIGINT_SIGN_ZERO;
    for (uint8_t i = 0; i < BIGINT_MAG_SIZE; ++i) {
        r.mag[i] = 0;
    }
    return r;
}

bigint_t bigint_one(void) {
    bigint_t r;
    r.sign = BIGINT_SIGN_POS;
    for (uint8_t i = 0; i < BIGINT_MAG_SIZE; ++i) {
        r.mag[i] = 0;
    }
    r.mag[BIGINT_MAG_SIZE - 1] = 1;
    return r;
}

bigint_t bigint_from_bool(bool a) {
    if (a) {
        return bigint_one();
    } else {
        return bigint_zero();
    }
}

bool bool_from_bigint(bigint_t a) {
    return a.sign;
}

bigint_cmp_t bigint_cmp(bigint_t a, bigint_t b) {
    if (a.sign != b.sign) {
        if (a.sign == BIGINT_SIGN_POS) {
            return BIGINT_CMP_GT;
        } else if (b.sign == BIGINT_SIGN_POS) {
            return BIGINT_CMP_LT;
        }

        if (a.sign == BIGINT_SIGN_ZERO) {
            return BIGINT_CMP_GT;
        } else if (b.sign == BIGINT_SIGN_ZERO) {
            return BIGINT_CMP_LT;
        }

        assert(false);
    }

    for (uint8_t i = 0; i < BIGINT_MAG_SIZE; ++i) {
        if (a.mag[i] != b.mag[i]) {
            if (a.mag[i] < b.mag[i]) {
                if (a.sign == BIGINT_SIGN_POS) {
                    return BIGINT_CMP_LT;
                } else {
                    return BIGINT_CMP_GT;
                }
            } else {
                if (a.sign == BIGINT_SIGN_POS) {
                    return BIGINT_CMP_GT;
                } else {
                    return BIGINT_CMP_LT;
                }
            }
        }
    }

    return BIGINT_CMP_EQ;
}

bool bigint_lt(bigint_t a, bigint_t b) {
    return bigint_cmp(a, b) == BIGINT_CMP_LT;
}

bool bigint_le(bigint_t a, bigint_t b) {
    return bigint_cmp(a, b) != BIGINT_CMP_GT;
}

bool bigint_eq(bigint_t a, bigint_t b) {
    return bigint_cmp(a, b) == BIGINT_CMP_EQ;
}

bool bigint_ne(bigint_t a, bigint_t b) {
    return bigint_cmp(a, b) != BIGINT_CMP_EQ;
}

bool bigint_ge(bigint_t a, bigint_t b) {
    return bigint_cmp(a, b) != BIGINT_CMP_LT;
}

bool bigint_gt(bigint_t a, bigint_t b) {
    return bigint_cmp(a, b) == BIGINT_CMP_GT;
}

bigint_t bigint_canonicalize(bigint_t a) {
    if (a.sign == BIGINT_SIGN_ZERO) {
        return a;
    }

    for (uint8_t i = 0; i < BIGINT_MAG_SIZE; ++i) {
        if (a.mag[i]) {
            return a;
        }
    }

    bigint_t r = a;
    r.sign = BIGINT_SIGN_ZERO;
    return r;
}

bigint_sign_t cmp_mag(bigint_t a, bigint_t b) {
    for (uint8_t i = 0; i < BIGINT_MAG_SIZE; ++i) {
        if (a.mag[i] != b.mag[i]) {
            if (a.mag[i] < b.mag[i]) {
                return BIGINT_SIGN_NEG;
            } else {
                return BIGINT_SIGN_POS;
            }
        }
    }
    return BIGINT_SIGN_ZERO;
}

void add_mag(bigint_t a, bigint_t b, bigint_t* r) {
    uint8_t a_sum;
    uint8_t b_sum;
    uint8_t carry = 0;
    for (uint8_t i = BIGINT_MAG_SIZE - 1; i < BIGINT_MAG_SIZE; --i) {
        a_sum = a.mag[i] + carry;
        carry = a_sum < a.mag[i];
        b_sum = a_sum + b.mag[i];
        carry += b_sum < a_sum;
        r->mag[i] = b_sum;
    }
}

void sub_mag_ltr(bigint_t a, bigint_t b, bigint_t* r) {
    uint8_t a_diff;
    uint8_t b_diff;
    uint8_t borrow = 0;
    for (uint8_t i = BIGINT_MAG_SIZE - 1; i < BIGINT_MAG_SIZE; --i) {
        a_diff = a.mag[i] - borrow;
        borrow = a.mag[i] < a_diff;
        b_diff = a_diff - b.mag[i];
        borrow += a_diff < b_diff;
        r->mag[i] = b_diff;
    }
}

void sub_mag(bigint_t a, bigint_t b, bigint_t* r) {
    r->sign = cmp_mag(a, b);
    if (r->sign == BIGINT_SIGN_NEG) {
        sub_mag_ltr(b, a, r);
    } else {
        sub_mag_ltr(a, b, r);
    }
}

bigint_t bigint_add(bigint_t a, bigint_t b) {
    bigint_t r;

    if (a.sign == BIGINT_SIGN_NEG) {
        if (b.sign == BIGINT_SIGN_NEG) {
            /* -a + -b = -(a + b) */
            add_mag(a, b, &r);
            r.sign = BIGINT_SIGN_NEG;
        } else {
            /* -a + b = b - a */
            sub_mag(b, a, &r);
        }
    } else {
        if (b.sign == BIGINT_SIGN_NEG) {
            /* a + -b = a - b */
            sub_mag(a, b, &r);
        } else {
            /* a + b */
            add_mag(a, b, &r);
            r.sign = BIGINT_SIGN_POS;
        }
    }

    return bigint_canonicalize(r);
}

bigint_t bigint_sub(bigint_t a, bigint_t b) {
    bigint_t r;

    if (a.sign == BIGINT_SIGN_NEG) {
        if (b.sign == BIGINT_SIGN_NEG) {
            /* -a - -b = b - a */
            sub_mag(b, a, &r);
        } else {
            /* -a - b = -(a + b) */
            add_mag(a, b, &r);
            r.sign = BIGINT_SIGN_NEG;
        }
    } else {
        if (b.sign == BIGINT_SIGN_NEG) {
            /* a - -b = a + b */
            add_mag(a, b, &r);
            r.sign = BIGINT_SIGN_POS;
        } else {
            /* a - b */
            sub_mag(a, b, &r);
        }
    }

    return bigint_canonicalize(r);
}

bigint_t bigint_incr(bigint_t a) {
    bigint_t b = bigint_one();
    return bigint_add(a, b);
}

bigint_t bigint_decr(bigint_t a) {
    bigint_t b = bigint_one();
    return bigint_sub(a, b);
}

bigint_t bigint_mul(bigint_t a, bigint_t b) {
    bigint_t r;
    uint8_t i;
    uint8_t j;
    uint8_t k;
    uint8_t prod_hi;
    uint8_t prod_lo;
    uint8_t carry;

    for (i = 0; i < BIGINT_MAG_SIZE; ++i) {
        r.mag[i] = 0;
    }

    if (a.sign == BIGINT_SIGN_ZERO || b.sign == BIGINT_SIGN_ZERO) {
        r.sign = BIGINT_SIGN_ZERO;
        return r;
    }

    if (a.sign == b.sign) {
        r.sign = BIGINT_SIGN_POS;
    } else {
        r.sign = BIGINT_SIGN_NEG;
    }

    for (i = BIGINT_MAG_SIZE - 1; i < BIGINT_MAG_SIZE; --i) {
        if (a.mag[i] == 0) {
            continue;
        }

        carry = 0;
        for (j = BIGINT_MAG_SIZE - 1; j < BIGINT_MAG_SIZE; --j) {
            mul_8_8_16(a.mag[i], b.mag[j], &prod_hi, &prod_lo);

            prod_lo += carry;
            carry = prod_lo < carry;

            k = i + j - BIGINT_MAG_SIZE + 1;
            r.mag[k] += prod_lo;
            carry += r.mag[k] < prod_lo;

            carry += prod_hi;
        }
    }

    return bigint_canonicalize(r);
}

bigint_t bigint_mul_log2(bigint_t a, uint8_t b) {
    bigint_t r = a;
    shift_left(BIGINT_MAG_SIZE, r.mag, b);
    return bigint_canonicalize(r);
}

bigint_t bigint_div_log2(bigint_t a, uint8_t b) {
    bigint_t r = a;
    shift_right(BIGINT_MAG_SIZE, r.mag, b);
    return bigint_canonicalize(r);
}
