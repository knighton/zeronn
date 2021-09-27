#include "bigfx_algo.h"

#include <stdint.h>

#include "zeronn/type/bigint.h"

bigfx_t bigfx_add(bigfx_t a, bigfx_t b) {
    return bigint_add(a, b);
}

bigfx_t bigfx_incr(bigfx_t a) {
    return bigint_incr(a);
}

bigfx_t bigfx_sub(bigfx_t a, bigfx_t b) {
    return bigint_sub(a, b);
}

bigfx_t bigfx_decr(bigfx_t a) {
    return bigint_decr(a);
}

bigfx_t bigfx_mul(bigfx_t a, bigfx_t b) {
    a = bigint_div_log2(a, SQRT_FX_BITS);
    b = bigint_div_log2(b, SQRT_FX_BITS);
    return bigint_mul(a, b);
}

bigfx_t bigfx_mul_log2(bigfx_t a, uint8_t b) {
    return bigint_mul_log2(a, b);
}

bigfx_t bigfx_div(bigfx_t a, bigfx_t b) {
    bigfx_t br = bigfx_recip(b);
    return bigfx_mul(a, br);
}

bigfx_t bigint_div(bigfx_t a, bigfx_t b) {
    bigfx_t br = bigfx_recip(b);
    return bigint_mul(a, br);
}

bigfx_t bigfx_div_log2(bigfx_t a, uint8_t b) {
    return bigint_div_log2(a, b);
}

bigfx_t bigfx_log2(bigfx_t a) {
    bigfx_state_init();
    return bigfx_mul(bigfx_log(a), BFX_STATE.recip_log2);
}

bigfx_t bigfx_log10(bigfx_t a) {
    bigfx_state_init();
    return bigfx_mul(bigfx_log(a), BFX_STATE.recip_log10);
}
