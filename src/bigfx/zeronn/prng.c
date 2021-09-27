#include "prng.h"

#include "zeronn/type/big_conv.h"
#include "zeronn/type/bigfx.h"
#include "zeronn/type/biguint.h"

static xorshift32_t PRNG_STATE = { 0x00, 0x00, 0x00, 0x07 };

void prng_set_seed(biguint_t seed) {
    PRNG_STATE.a = seed;
}

bigint_t prng_randint_log2(bigint_t low, uint8_t log2_range) {
    biguint_t a = PRNG_STATE.a;
    a = biguint_xor(a, biguint_mul_log2(a, 13));
    a = biguint_xor(a, biguint_div_log2(a, 17));
    a = biguint_xor(a, biguint_mul_log2(a, 5));
    PRNG_STATE.a = a;
    a = biguint_mod_log2(a, log2_range);
    bigint_t range = bigint_from_biguint(a);
    return bigint_add(low, range);
}
