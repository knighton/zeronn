#ifndef ZERONN_PRNG_H_
#define ZERONN_PRNG_H_

#include "zeronn/type/bigint.h"
#include "zeronn/type/biguint.h"

typedef struct {
    biguint_t a;
} xorshift32_t;

void prng_set_seed(biguint_t seed);
bigint_t prng_randint_log2(bigint_t low, uint8_t log2_range);

#endif  /* ZERONN_PRNG_H_ */
