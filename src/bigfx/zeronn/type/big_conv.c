#include "big_conv.h"

#include <assert.h>
#include <stdint.h>

#define COMMON_MAG_SIZE \
    (BIGINT_MAG_SIZE < BIGUINT_MAG_SIZE ? BIGINT_MAG_SIZE : BIGUINT_MAG_SIZE)

bigint_t bigint_from_biguint(biguint_t a) {
    int r = (int)uint_from_biguint(a);
    return bigint_from_int(r);
}

biguint_t biguint_from_bigint(bigint_t a) {
    assert(a.sign != BIGINT_SIGN_NEG);
    biguint_t r;
    for (uint8_t i = 0; i < COMMON_MAG_SIZE; ++i) {
        uint8_t a_idx = BIGINT_MAG_SIZE - 1 - i;
        uint8_t r_idx = BIGUINT_MAG_SIZE - 1 - i;
        r.mag[r_idx] = a.mag[a_idx];
    }
    for (uint8_t i = COMMON_MAG_SIZE; i < BIGUINT_MAG_SIZE; ++i) {
        uint8_t r_idx = BIGUINT_MAG_SIZE - 1 - i;
        r.mag[r_idx] = 0;
    }
    for (uint8_t i = COMMON_MAG_SIZE; i < BIGINT_MAG_SIZE; ++i) {
        uint8_t r_idx = BIGINT_MAG_SIZE - 1 - i;
        assert(!a.mag[r_idx]);
    }
    return r;
}
