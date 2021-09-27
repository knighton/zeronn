#include "prng.h"

#include <stdint.h>
#include <stdlib.h>

#include "zeronn/lint.h"

static xorshift32_t PRNG_STATE = { 0x007 };

void xorshift32_free(xorshift32_t* state) {
    UNUSED(state);
}

void xorshift32_set_seed(xorshift32_t* state, uint32_t seed) {
    state->a = seed;
}

uint32_t xorshift32_get(xorshift32_t* state) {
    uint32_t x = state->a;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state->a = x;
    return x % ZERONN_RAND_MAX;
}

void prng_set_seed(int seed) {
    xorshift32_set_seed(&PRNG_STATE, (uint32_t)seed);
}

int prng_get(void) {
    return (int)xorshift32_get(&PRNG_STATE) & 0x1FFFFFFF;
}

int prng_randint(int low, int high) {
    int x = prng_get();
    return x % (1 + high - low) + low;
}
